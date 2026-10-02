"""Track B pipeline: PCA, few-shot fit and dev/lock scoring, one (scenario, seed) at a time.

`run_track_b` is the Track B `run_experiment` runner (see
`anometa.experiment.RUNNERS`). For each of `cfg.scenarios` it fits one PCA on
that scenario's train normals, then for each of `cfg.seeds` samples few-shot
shots (skipped for one-class classifiers), fits a scorer on train normals
plus those shots, times the fit and the timed prediction on the evaluation
rows, and collects per-image predictions. `fit_pca` and `design_matrix` are
its PCA and feature-assembly building blocks.
"""

import json
import multiprocessing
import time
from collections.abc import Generator, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from anometa.artifacts import LICENCES, TrackOutput
from anometa.config import (
    ONE_CLASS,
    ClassifierName,
    FeatureBlock,
    Scenario,
    TrackBConfig,
    config_hash,
    resolve_device,
    seed_parallelism,
)
from anometa.data.splits import BudgetError, eval_rows, load_split, sample_few_shot
from anometa.features.extract import Features, load_features
from anometa.metrics.aggregate import group_metrics, summarize
from anometa.metrics.image import prior_correct
from anometa.trackb.classifiers import Scorer, ThinkingScorer, make_scorer, tabpfn_revision

_EMBEDDING_BLOCKS: tuple[FeatureBlock, ...] = ("cls", "mean_patch")
_TABPFN_VERSIONS: dict[str, Literal["v3.5", "v3.5-fast"]] = {
    "tabpfn": "v3.5",
    "tabpfn_fast": "v3.5-fast",
    "tabpfn_outlier": "v3.5",
}
"""TabPFN checkpoint version per classifier whose scorer loads `tabpfn` (so its licence applies)."""

_THINKING_KEY_EXCLUDE: tuple[str, ...] = ("seeds", "scenarios", "device", "classifier_params")
"""Fields (besides `name`/`paths`) left out of the `tabpfn_thinking` cache key.

One cached prediction per (scenario, seed), whichever other seeds and
scenarios a run spans and whatever device it names; `classifier_params` is
always empty for `tabpfn_thinking`.
"""

_PRED_COLUMNS: list[str] = [
    "scenario",
    "seed",
    "image_id",
    "scene_id",
    "label",
    "lighting",
    "score",
    "score_balanced",
]


def fit_pca(embedding: NDArray[np.float32], dim: int) -> PCA:
    """Fit a full-SVD PCA on an embedding matrix.

    Args:
        embedding: `(n, d)` embedding rows to fit on; callers pass a
            scenario's train normals only, never dev or lock rows.
        dim: Number of principal components to keep.

    Returns:
        The fitted `PCA`.
    """
    return PCA(n_components=dim, svd_solver="full").fit(embedding)


def embedding(
    feats: Features, rows: NDArray[np.int64], blocks: tuple[FeatureBlock, ...]
) -> NDArray[np.float32] | None:
    """Concatenate the `cls`/`mean_patch` blocks requested in `blocks` for some rows.

    Public: also used by callers that need to fit a PCA before `design_matrix`
    can run, e.g. the GUI demo's `fit_and_score`.

    Args:
        feats: A scenario's cached features.
        rows: Row positions into `feats`.
        blocks: Feature blocks to include; `novelty` is ignored here.

    Returns:
        `(len(rows), d)` embedding in `_EMBEDDING_BLOCKS` order, or `None`
        when `blocks` holds neither `cls` nor `mean_patch`.
    """
    arrays = {"cls": feats.cls, "mean_patch": feats.mean_patch}
    parts = [arrays[b][rows] for b in _EMBEDDING_BLOCKS if b in blocks]
    return np.concatenate(parts, axis=1) if parts else None


def design_matrix(
    feats: Features,
    rows: NDArray[np.int64],
    blocks: tuple[FeatureBlock, ...],
    pca: PCA | None,
) -> NDArray[np.float32]:
    """Build a classifier-ready feature matrix for a set of feature rows.

    Concatenates whichever of `cls`/`mean_patch` are present in `blocks` and
    PCA-transforms that block, then appends the two `novelty` columns when
    requested.

    Args:
        feats: A scenario's cached features.
        rows: Row positions into `feats`, as returned by `Features.rows`.
        blocks: Feature blocks to include, e.g. `cfg.features`.
        pca: PCA fitted on this scenario's train-normal embedding
            (`fit_pca`); required when `blocks` holds `cls` or `mean_patch`.

    Returns:
        `(len(rows), d)` float32 feature matrix.

    Raises:
        ValueError: If `blocks` includes `cls` or `mean_patch` but `pca` is
            `None`.
    """
    parts: list[NDArray[np.float32]] = []
    emb = embedding(feats, rows, blocks)
    if emb is not None:
        if pca is None:
            raise ValueError("pca is required when blocks include cls or mean_patch")
        parts.append(pca.transform(emb).astype(np.float32))
    if "novelty" in blocks:
        parts.append(feats.novelty[rows])
    return np.concatenate(parts, axis=1)


def context_normals(
    train_rows: NDArray[np.int64], n_normals: int | None, *, seed: int, scenario: Scenario
) -> NDArray[np.int64]:
    """Pick the train normals a few-shot scorer fits on.

    TabPFN's predictions are better calibrated with fewer normals in its
    context (PLAN_1 § Track B), so `n_normals` draws a subsample, seeded by
    `(seed, scenario)` on a stream separate from shot sampling.

    Args:
        train_rows: Row positions of the scenario's train normals.
        n_normals: How many to keep; `None` or at least `len(train_rows)`
            keeps every row.
        seed: The run seed.
        scenario: The scenario, so scenarios draw independently.

    Returns:
        The kept row positions, sorted.
    """
    if n_normals is None or n_normals >= len(train_rows):
        return train_rows
    rng = np.random.default_rng([seed, list(Scenario).index(scenario), 7])
    return np.sort(rng.choice(train_rows, size=n_normals, replace=False))


def fit_and_score_shots(
    feats: Features,
    pca: PCA | None,
    train_rows: NDArray[np.int64],
    features: tuple[FeatureBlock, ...],
    shots: Sequence[str],
    split_df: pd.DataFrame,
    split: Literal["dev", "lock"],
    scorer: Scorer,
    device: str,
) -> tuple[pd.DataFrame, NDArray[np.float64], NDArray[np.float64], float, float]:
    """Fit `scorer` on train normals plus `shots`, score `split`'s eval rows, balance the scores.

    The per-seed fit/score/balance step shared by `run_track_b`'s scenario
    loop and the GUI demo's `fit_and_score`, so the two paths can't silently
    diverge. `train_rows` selection and the PCA fit happen once per scenario
    in the caller, since `run_track_b` reuses both across seeds.

    Args:
        feats: A scenario's cached features.
        pca: PCA fitted on the scenario's train-normal embedding, or `None`
            when `features` is novelty-only.
        train_rows: Row positions of the scenario's train-normal images.
        features: Feature blocks to concatenate, e.g. `cfg.features`.
        shots: Image ids labelled anomalous (the `y=1` fit rows).
        split_df: The scenario's dev/lock split, as returned by `load_split`.
        split: `"dev"` or `"lock"`: which rows `eval_rows` scores.
        scorer: An unfitted `Scorer`.
        device: Device `scorer` runs on; CUDA is synchronised around each
            timed call so the latencies reflect the GPU work, not just launch.

    Returns:
        `(eval_df, score, score_balanced, fit_ms, predict_ms)`. `eval_df` is
        `eval_rows(split_df, split=split, shots=shots)`; `score` and
        `score_balanced` (NaN when not `scorer.probabilistic`) align to its
        rows; `fit_ms` and `predict_ms` are wall-clock milliseconds.
    """
    shot_rows = feats.rows(shots)
    fit_positions = np.concatenate([train_rows, shot_rows])
    y_fit: NDArray[np.int64] = np.concatenate(
        [np.zeros(len(train_rows), dtype=np.int64), np.ones(len(shot_rows), dtype=np.int64)]
    )
    x_fit = design_matrix(feats, fit_positions, features, pca)

    start = time.perf_counter()
    scorer.fit(x_fit, y_fit)
    if device == "cuda":
        torch.cuda.synchronize()
    fit_ms = (time.perf_counter() - start) * 1000

    eval_df = eval_rows(split_df, split=split, shots=shots)
    eval_ids: list[str] = eval_df["image_id"].tolist()
    eval_positions = feats.rows(eval_ids)
    x_eval = design_matrix(feats, eval_positions, features, pca)

    start = time.perf_counter()
    score = scorer.anomaly_score(x_eval)
    if device == "cuda":
        torch.cuda.synchronize()
    predict_ms = (time.perf_counter() - start) * 1000

    if scorer.probabilistic:
        score_balanced = prior_correct(score, len(shots) / (len(train_rows) + len(shots)))
    else:
        score_balanced = np.full(score.shape, np.nan)

    return eval_df, score, score_balanced, fit_ms, predict_ms


@dataclass(frozen=True)
class _SeedTask:
    """One (scenario, seed)'s inputs to `_score_seed`; picklable for a process pool."""

    feats: Features
    pca: PCA | None
    train_rows: NDArray[np.int64]
    features: tuple[FeatureBlock, ...]
    shots: list[str]
    split_df: pd.DataFrame
    split: Literal["dev", "lock"]
    classifier: ClassifierName
    classifier_params: dict[str, int | float | str]
    seed: int
    device: str
    cache_key: str | None
    cache_dir: Path | None


def _score_seed(
    task: _SeedTask,
) -> tuple[pd.DataFrame, NDArray[np.float64], NDArray[np.float64], float, float, float]:
    """Build a fresh scorer for one seed and run `fit_and_score_shots` with it.

    Module-level so a process pool can pickle it. Every scorer seeds only its
    own generator, except `tabpfn_outlier`, which is one-class and so always
    runs a single seed.

    Args:
        task: The (scenario, seed)'s inputs.

    Returns:
        `fit_and_score_shots`' tuple plus this process's peak allocated CUDA
        memory in MB (NaN off CUDA), which a process pool's parent can't see.
    """
    scorer = make_scorer(
        task.classifier,
        task.classifier_params,
        seed=task.seed,
        device=task.device,
        cache_key=task.cache_key,
        cache_dir=task.cache_dir,
    )
    result = fit_and_score_shots(
        task.feats,
        task.pca,
        task.train_rows,
        task.features,
        task.shots,
        task.split_df,
        task.split,
        scorer,
        task.device,
    )
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if task.device == "cuda" else float("nan")
    return (*result, peak_mb)


@contextmanager
def _seed_pool(device: str) -> Generator[Executor | None]:
    """Open the executor `seed_parallelism` asks for, or `None` to run seeds serially.

    Processes start with `spawn`, the start method that is safe once CUDA is
    initialised.

    Args:
        device: The resolved device the seeds run on.

    Yields:
        A thread or process pool with `ANOMETA_SEED_WORKERS` workers, or
        `None` for one worker.

    Raises:
        ValueError: For threads on MPS, which aborts on concurrent use.
    """
    workers, kind = seed_parallelism()
    if workers == 1:
        yield None
        return
    if kind == "thread" and device == "mps":
        raise ValueError("MPS aborts on concurrent threads; set ANOMETA_SEED_EXECUTOR=process")
    pool: Executor = (
        ThreadPoolExecutor(max_workers=workers)
        if kind == "thread"
        else ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
    )
    with pool:
        yield pool


def _encoder_revision(feats: Features) -> str:
    """Format a `model_revisions` entry from a scenario's feature provenance.

    Args:
        feats: Cached features for one (encoder, scenario); its
            `provenance` records the backend, revision, device and dtype
            that produced it.

    Returns:
        `"<backend>@<revision> (device=<device>, dtype=<dtype>)"`.
    """
    p = feats.provenance
    return f"{p['backend']}@{p['revision']} (device={p['device']}, dtype={p['dtype']})"


def run_track_b(cfg: TrackBConfig, run_dir: Path) -> TrackOutput:
    """Run one Track B experiment: PCA, few-shot fit and scoring per scenario and seed.

    First loads every scenario's split and samples every (scenario, seed)'s
    few-shot shots, so a `BudgetError` surfaces before any PCA or scorer fit
    runs. Then, for each scenario: loads its cached features and fits one
    PCA on the scenario's train normals. For each seed: fits a fresh scorer
    on train normals (y=0; `classifier_params["n_normals"]` of them when
    set, see `context_normals`) plus that seed's shots (y=1, empty for one-class
    classifiers), times the fit and the timed `anomaly_score` call on the
    split's evaluation rows (CUDA synchronised), and prior-corrects
    probabilistic scores to a balanced 50/50 prior. `tabpfn_thinking`
    caches each prediction at
    `cfg.paths.cache / "thinking" / <key> / f"{scenario}-{seed}.npz"`,
    where `<key>` is `config_hash` without `_THINKING_KEY_EXCLUDE`, so a
    rerun, or another run sharing that (scenario, seed), never calls the API
    again. Its fit/predict latencies are recorded as NaN: they time the
    network, not the model.

    Args:
        cfg: The Track B experiment configuration.
        run_dir: Unused; the runner reports through `TrackOutput`, and
            `run_experiment` writes it to the run directory.

    Returns:
        Predictions for every (scenario, seed)'s evaluation rows, summary
        metrics (`summarize(group_metrics(...))` plus `fit_latency_ms`,
        `predict_latency_ms` and, on CUDA, `peak_vram_mb`), the model
        revisions and licences this run depended on (`tabpfn_thinking`
        records the Prior Labs API terms, not the local weights licence),
        and the sampled shots per (scenario, seed) as `shots.json`.

    Raises:
        BudgetError: If a seed's few-shot sample exceeds the shot pool, or
            (for `split="dev"`) exhausts every dev defect scene, leaving
            none to evaluate.
        ValueError: If the scenarios' features differ in provenance
            (backend, revision, device or dtype), since the manifest records
            one encoder revision per run.
    """
    device = resolve_device(cfg.device)
    probabilistic = cfg.classifier not in ONE_CLASS
    n_normals_param = cfg.classifier_params.get("n_normals")
    n_normals = n_normals_param if isinstance(n_normals_param, int) else None
    thinking_key = config_hash(cfg, exclude=_THINKING_KEY_EXCLUDE)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pred_parts: list[pd.DataFrame] = []
    fit_latencies_ms: list[float] = []
    predict_latencies_ms: list[float] = []
    worker_peaks_mb: list[float] = []
    model_revisions: dict[str, str] = {}
    licences: dict[str, str] = {"ad2": LICENCES["ad2"]}
    if cfg.classifier in _TABPFN_VERSIONS:
        licences["tabpfn"] = LICENCES["tabpfn"]
    elif cfg.classifier == "tabpfn_thinking":
        licences["tabpfn_api"] = LICENCES["tabpfn_api"]

    # Sample every (scenario, seed)'s shots first, so a BudgetError fires before any fit.
    splits: dict[Scenario, pd.DataFrame] = {}
    shots_by_scenario: dict[str, dict[str, list[str]]] = {}
    for scenario in cfg.scenarios:
        split_df = splits[scenario] = load_split(scenario, cfg.paths)
        seed_shots = shots_by_scenario.setdefault(str(scenario), {})
        for seed in cfg.seeds:
            shots: list[str] = []
            if cfg.classifier not in ONE_CLASS:
                shots = sample_few_shot(
                    split_df, scenario=scenario, k=cfg.k, seed=seed, lighting=cfg.shot_lighting
                )
                if (
                    cfg.split == "dev"
                    and not (eval_rows(split_df, split="dev", shots=shots)["label"] == 1).any()
                ):
                    dev_defects = split_df.query("split == 'dev' and label == 1")
                    raise BudgetError(
                        f"{scenario}: k={cfg.k} shots at seed={seed} cover all "
                        f"{dev_defects['scene_id'].nunique()} dev defect scenes, "
                        "leaving none to evaluate"
                    )
            seed_shots[str(seed)] = shots

    with _seed_pool(device) as pool:
        for scenario in cfg.scenarios:
            split_df = splits[scenario]
            seed_shots = shots_by_scenario[str(scenario)]
            feats = load_features(cfg.encoder, scenario, cfg.paths, cfg.encoder_backend)
            encoder_revision = _encoder_revision(feats)
            if model_revisions.setdefault(cfg.encoder, encoder_revision) != encoder_revision:
                raise ValueError(
                    f"{scenario}: {cfg.encoder} features come from {encoder_revision}, but "
                    f"earlier scenarios' from {model_revisions[cfg.encoder]}; re-extract them "
                    "with one setup"
                )
            family = "dinov3" if cfg.encoder.startswith("dinov3") else "siglip2"
            licences.setdefault(family, LICENCES[family])

            train_rows: NDArray[np.int64] = np.flatnonzero(
                np.char.startswith(feats.image_id, "train/good/")
            ).astype(np.int64)
            train_embedding = embedding(feats, train_rows, cfg.features)
            pca: PCA | None = None
            if train_embedding is not None:
                assert cfg.pca_dim is not None  # guaranteed by TrackBConfig validation
                pca = fit_pca(train_embedding, cfg.pca_dim)

            tasks = [
                _SeedTask(
                    feats=feats,
                    pca=pca,
                    train_rows=context_normals(train_rows, n_normals, seed=seed, scenario=scenario),
                    features=cfg.features,
                    shots=seed_shots[str(seed)],
                    split_df=split_df,
                    split=cfg.split,
                    classifier=cfg.classifier,
                    classifier_params=cfg.classifier_params,
                    seed=seed,
                    device=device,
                    cache_key=(
                        f"{thinking_key}/{scenario}-{seed}"
                        if cfg.classifier == "tabpfn_thinking"
                        else None
                    ),
                    cache_dir=(
                        cfg.paths.cache / "thinking"
                        if cfg.classifier == "tabpfn_thinking"
                        else None
                    ),
                )
                for seed in cfg.seeds
            ]
            results = map(_score_seed, tasks) if pool is None else pool.map(_score_seed, tasks)
            for task, (eval_df, score, score_balanced, fit_ms, predict_ms, peak_mb) in zip(
                tasks, results, strict=True
            ):
                fit_latencies_ms.append(fit_ms)
                predict_latencies_ms.append(predict_ms)
                worker_peaks_mb.append(peak_mb)

                part = eval_df[["image_id", "scene_id", "label", "lighting"]].copy()
                part["scenario"] = str(scenario)
                part["seed"] = task.seed
                part["score"] = score
                part["score_balanced"] = score_balanced
                pred_parts.append(part)

    if cfg.classifier in _TABPFN_VERSIONS:
        # After the fits: on a fresh host the first fit downloads the checkpoint.
        rev = tabpfn_revision(_TABPFN_VERSIONS[cfg.classifier])
        model_revisions["tabpfn"] = (
            f"{rev['tabpfn_version']}@{rev['checkpoint']}#{rev['sha256'][:12]}"
        )
    elif cfg.classifier == "tabpfn_thinking":
        model_revisions["tabpfn_thinking"] = (
            f"api:v3.5 effort={ThinkingScorer.effort} metric={ThinkingScorer.metric}"
        )

    predictions = pd.concat(pred_parts, ignore_index=True)[_PRED_COLUMNS]
    metrics = summarize(group_metrics(predictions, probabilistic))
    timed = cfg.classifier != "tabpfn_thinking"  # API round trips aren't model latency
    metrics["fit_latency_ms"] = float(np.median(fit_latencies_ms)) if timed else float("nan")
    metrics["predict_latency_ms"] = (
        float(np.median(predict_latencies_ms)) if timed else float("nan")
    )
    if device == "cuda":
        metrics["peak_vram_mb"] = max(torch.cuda.max_memory_allocated() / 2**20, *worker_peaks_mb)

    return TrackOutput(
        metrics=metrics,
        predictions=predictions,
        model_revisions=model_revisions,
        licences=licences,
        extra_files={
            "shots.json": json.dumps(shots_by_scenario, indent=2, sort_keys=True).encode()
        },
    )
