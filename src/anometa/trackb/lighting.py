"""Post-freeze lighting-adaptation study: TabPFN and controls with target-lit images in context.

`run_lighting` is the `run_experiment` runner for `LightingConfig` (track
`"L"`, PLAN_1 § Post-freeze study). Per scenario, fold seed and target
lighting it fits one scorer on the train normals, the fold's regular-lit
defect shots and the target-lit images of the first `adapt_normals`
adaptation scenes, then scores the target-lit images of every held-out
scene. Features, PCA and the design matrix are Track B's.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from anometa.artifacts import LICENCES, TrackOutput
from anometa.config import ONE_CLASS, LightingConfig, resolve_device
from anometa.data.splits import lighting_fold, load_split
from anometa.features.extract import load_features
from anometa.metrics.aggregate import group_metrics, summarize
from anometa.metrics.image import prior_correct
from anometa.trackb.classifiers import make_scorer, tabpfn_revision
from anometa.trackb.pipeline import design_matrix, embedding, fit_pca

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
_TABPFN_VERSIONS = {"tabpfn": "v3.5", "tabpfn_fast": "v3.5-fast", "tabpfn_outlier": "v3.5"}


def _summary(pred: pd.DataFrame, probabilistic: bool) -> dict[str, float]:
    """Summarise predictions with every (scenario, lighting) pair as its own group.

    Each pair has its own fitted context, so scores are only comparable
    within it; the summary is the mean over (scenario, lighting, seed)
    groups.

    Args:
        pred: Predictions of one or more target lightings.
        probabilistic: Whether `score_balanced` feeds calibration metrics.

    Returns:
        `summarize` output over the regrouped predictions.
    """
    regrouped = pred.assign(scenario=pred["scenario"] + "|" + pred["lighting"])
    return summarize(group_metrics(regrouped, probabilistic))


def run_lighting(cfg: LightingConfig, run_dir: Path) -> TrackOutput:
    """Run the lighting-adaptation study for one classifier, k and m.

    Args:
        cfg: The study configuration.
        run_dir: Unused; `run_experiment` writes the returned output.

    Returns:
        Per-image predictions (every fold and target lighting) and metrics:
        `regular/<m>` and `shifted/<m>` (means over regular and over shifted
        (scenario, lighting, fold) groups), `gap_auroc` (regular minus
        shifted AUROC), `<lighting>/auroc` per lighting, plus `nll_bal`
        variants for probabilistic classifiers.
    """
    device = resolve_device(cfg.device)
    probabilistic = cfg.classifier not in ONE_CLASS
    parts: list[pd.DataFrame] = []
    revisions: dict[str, str] = {}
    licences = {"ad2": LICENCES["ad2"]}
    family = "dinov3" if cfg.encoder.startswith("dinov3") else "siglip2"
    licences[family] = LICENCES[family]
    for scenario in cfg.scenarios:
        split_df = load_split(scenario, cfg.paths)
        feats = load_features(cfg.encoder, scenario, cfg.paths, cfg.encoder_backend)
        p = feats.provenance
        revisions[cfg.encoder] = f"{p['backend']}@{p['revision']}"
        train_rows = np.flatnonzero(np.char.startswith(feats.image_id, "train/good/")).astype(
            np.int64
        )
        train_embedding = embedding(feats, train_rows, cfg.features)
        pca = None
        if train_embedding is not None:
            assert cfg.pca_dim is not None  # guaranteed by LightingConfig validation
            pca = fit_pca(train_embedding, cfg.pca_dim)
        for seed in cfg.seeds:
            fold = lighting_fold(
                split_df,
                scenario=scenario,
                seed=seed,
                k=cfg.k,
                adapt_max=cfg.adapt_max,
                pool=cfg.scene_pool,
            )
            adapt = fold.adapt_scenes[: cfg.adapt_normals]
            shots = [] if cfg.classifier in ONE_CLASS else fold.shots
            shot_rows = feats.rows(shots)
            for light in sorted(split_df["lighting"].unique()):
                adapt_ids = split_df.loc[
                    split_df["scene_id"].isin(adapt) & (split_df["lighting"] == light), "image_id"
                ].tolist()
                normal_rows = np.concatenate([train_rows, feats.rows(adapt_ids)])
                fit_rows = np.concatenate([normal_rows, shot_rows])
                y = np.concatenate(
                    [np.zeros(len(normal_rows), np.int64), np.ones(len(shot_rows), np.int64)]
                )
                scorer = make_scorer(cfg.classifier, {}, seed=seed, device=device)
                scorer.fit(design_matrix(feats, fit_rows, cfg.features, pca), y)
                ev = fold.eval_rows[fold.eval_rows["lighting"] == light]
                x_eval = design_matrix(
                    feats, feats.rows(ev["image_id"].tolist()), cfg.features, pca
                )
                score = scorer.anomaly_score(x_eval)
                part = ev[["image_id", "scene_id", "label", "lighting"]].copy()
                part["scenario"] = str(scenario)
                part["seed"] = seed
                part["score"] = score
                part["score_balanced"] = (
                    prior_correct(score, len(shot_rows) / len(fit_rows))
                    if probabilistic
                    else np.full(len(score), np.nan)
                )
                parts.append(part)
    if cfg.classifier in _TABPFN_VERSIONS:
        rev = tabpfn_revision(_TABPFN_VERSIONS[cfg.classifier])
        revisions["tabpfn"] = f"{rev['tabpfn_version']}@{rev['checkpoint']}#{rev['sha256'][:12]}"
        licences["tabpfn"] = LICENCES["tabpfn"]

    pred = pd.concat(parts, ignore_index=True)[_PRED_COLUMNS]
    regular = pred["lighting"] == "regular"
    names = ["auroc", "nll_bal"] if probabilistic else ["auroc"]
    reg = _summary(pred[regular], probabilistic)
    shifted = _summary(pred[~regular], probabilistic)
    metrics = {f"regular/{n}": reg[n] for n in names} | {f"shifted/{n}": shifted[n] for n in names}
    metrics["gap_auroc"] = reg["auroc"] - shifted["auroc"]
    for light, g in pred.groupby("lighting", sort=True):
        metrics[f"{light}/auroc"] = _summary(g, probabilistic)["auroc"]
    if device == "cuda":
        metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return TrackOutput(
        metrics=metrics, predictions=pred, model_revisions=revisions, licences=licences
    )
