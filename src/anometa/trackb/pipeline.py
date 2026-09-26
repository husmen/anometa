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
import time
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
    FeatureBlock,
    Scenario,
    TrackBConfig,
    config_hash,
    resolve_device,
)
from anometa.data.splits import BudgetError, eval_rows, load_split, sample_few_shot
from anometa.features.extract import Features, load_features
from anometa.metrics.aggregate import group_metrics, summarize
from anometa.metrics.image import prior_correct
from anometa.trackb.classifiers import make_scorer, tabpfn_revision

_EMBEDDING_BLOCKS: tuple[FeatureBlock, ...] = ("cls", "mean_patch")
_TABPFN_VERSIONS: dict[str, Literal["v3.5", "v3.5-fast"]] = {
    "tabpfn": "v3.5",
    "tabpfn_fast": "v3.5-fast",
    "tabpfn_outlier": "v3.5",
}
"""TabPFN checkpoint version per classifier whose scorer loads `tabpfn` (so its licence applies)."""

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


def _embedding(
    feats: Features, rows: NDArray[np.int64], blocks: tuple[FeatureBlock, ...]
) -> NDArray[np.float32] | None:
    """Concatenate the `cls`/`mean_patch` blocks requested in `blocks` for some rows.

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
    embedding = _embedding(feats, rows, blocks)
    if embedding is not None:
        if pca is None:
            raise ValueError("pca is required when blocks include cls or mean_patch")
        parts.append(pca.transform(embedding).astype(np.float32))
    if "novelty" in blocks:
        parts.append(feats.novelty[rows])
    return np.concatenate(parts, axis=1)


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
    on train normals (y=0) plus that seed's shots (y=1, empty for one-class
    classifiers), times the fit and the timed `anomaly_score` call on the
    split's evaluation rows (CUDA synchronised), and prior-corrects
    probabilistic scores to a balanced 50/50 prior. `tabpfn_thinking` is
    keyed by `f"{config_hash(cfg)}/{scenario}-{seed}"` under
    `cfg.paths.cache / "thinking"`, so its predictions are cached across
    reruns of the same config.

    Args:
        cfg: The Track B experiment configuration.
        run_dir: Unused; the runner reports through `TrackOutput`, and
            `run_experiment` writes it to the run directory.

    Returns:
        Predictions for every (scenario, seed)'s evaluation rows, summary
        metrics (`summarize(group_metrics(...))` plus `fit_latency_ms`,
        `predict_latency_ms` and, on CUDA, `peak_vram_mb`), the model
        revisions and licences this run depended on, and the sampled shots
        per (scenario, seed) as `shots.json`.

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
    cfg_hash = config_hash(cfg)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pred_parts: list[pd.DataFrame] = []
    fit_latencies_ms: list[float] = []
    predict_latencies_ms: list[float] = []
    model_revisions: dict[str, str] = {}
    licences: dict[str, str] = {"ad2": LICENCES["ad2"]}
    if cfg.classifier in _TABPFN_VERSIONS or cfg.classifier == "tabpfn_thinking":
        licences["tabpfn"] = LICENCES["tabpfn"]

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

    for scenario in cfg.scenarios:
        split_df = splits[scenario]
        seed_shots = shots_by_scenario[str(scenario)]
        feats = load_features(cfg.encoder, scenario, cfg.paths, cfg.encoder_backend)
        encoder_revision = _encoder_revision(feats)
        if model_revisions.setdefault(cfg.encoder, encoder_revision) != encoder_revision:
            raise ValueError(
                f"{scenario}: {cfg.encoder} features come from {encoder_revision}, but earlier "
                f"scenarios' from {model_revisions[cfg.encoder]}; re-extract them with one setup"
            )
        family = "dinov3" if cfg.encoder.startswith("dinov3") else "siglip2"
        licences.setdefault(family, LICENCES[family])

        train_rows: NDArray[np.int64] = np.flatnonzero(
            np.char.startswith(feats.image_id, "train/good/")
        ).astype(np.int64)
        train_embedding = _embedding(feats, train_rows, cfg.features)
        pca: PCA | None = None
        if train_embedding is not None:
            assert cfg.pca_dim is not None  # guaranteed by TrackBConfig validation
            pca = fit_pca(train_embedding, cfg.pca_dim)

        for seed in cfg.seeds:
            shots = seed_shots[str(seed)]
            shot_rows = feats.rows(shots)
            fit_positions = np.concatenate([train_rows, shot_rows])
            y_fit: NDArray[np.int64] = np.concatenate(
                [
                    np.zeros(len(train_rows), dtype=np.int64),
                    np.ones(len(shot_rows), dtype=np.int64),
                ]
            )
            x_fit = design_matrix(feats, fit_positions, cfg.features, pca)

            if cfg.classifier == "tabpfn_thinking":
                scorer = make_scorer(
                    cfg.classifier,
                    cfg.classifier_params,
                    seed=seed,
                    device=device,
                    cache_key=f"{cfg_hash}/{scenario}-{seed}",
                    cache_dir=cfg.paths.cache / "thinking",
                )
            else:
                scorer = make_scorer(
                    cfg.classifier, cfg.classifier_params, seed=seed, device=device
                )

            start = time.perf_counter()
            scorer.fit(x_fit, y_fit)
            if device == "cuda":
                torch.cuda.synchronize()
            fit_latencies_ms.append((time.perf_counter() - start) * 1000)

            eval_df = eval_rows(split_df, split=cfg.split, shots=shots)
            eval_ids: list[str] = eval_df["image_id"].tolist()
            eval_positions = feats.rows(eval_ids)
            x_eval = design_matrix(feats, eval_positions, cfg.features, pca)

            start = time.perf_counter()
            score = scorer.anomaly_score(x_eval)
            if device == "cuda":
                torch.cuda.synchronize()
            predict_latencies_ms.append((time.perf_counter() - start) * 1000)

            if probabilistic:
                score_balanced = prior_correct(score, cfg.k / (len(train_rows) + cfg.k))
            else:
                score_balanced = np.full(score.shape, np.nan)

            part = eval_df[["image_id", "scene_id", "label", "lighting"]].copy()
            part["scenario"] = str(scenario)
            part["seed"] = seed
            part["score"] = score
            part["score_balanced"] = score_balanced
            pred_parts.append(part)

    if cfg.classifier in _TABPFN_VERSIONS:
        # After the fits: on a fresh host the first fit downloads the checkpoint.
        rev = tabpfn_revision(_TABPFN_VERSIONS[cfg.classifier])
        model_revisions["tabpfn"] = (
            f"{rev['tabpfn_version']}@{rev['checkpoint']}#{rev['sha256'][:12]}"
        )

    predictions = pd.concat(pred_parts, ignore_index=True)[_PRED_COLUMNS]
    metrics = summarize(group_metrics(predictions, probabilistic))
    metrics["fit_latency_ms"] = float(np.median(fit_latencies_ms))
    metrics["predict_latency_ms"] = float(np.median(predict_latencies_ms))
    if device == "cuda":
        metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20

    return TrackOutput(
        metrics=metrics,
        predictions=predictions,
        model_revisions=model_revisions,
        licences=licences,
        extra_files={
            "shots.json": json.dumps(shots_by_scenario, indent=2, sort_keys=True).encode()
        },
    )
