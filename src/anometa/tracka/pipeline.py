"""Track A pipeline: anomaly maps, validation threshold and pixel metrics per scenario.

`run_track_a` is the Track A `run_experiment` runner (see
`anometa.experiment.RUNNERS`). For each of `cfg.scenarios` it gets one raw
anomaly map per `validation` and `test_public` image (an anomalib model fit
by `fit_predict_anomalib`, or the cached DINOv3 patch distance map from
`distance_maps`), thresholds at the upsampled validation maps' mean + 3 std,
and scores the split's evaluation images at full resolution with the AD2
pixel metrics, on all, regular-lit and shifted-lit rows. Each scenario's
result is saved under the run directory as soon as it is computed, so an
interrupted run resumes at the first unfinished scenario.
"""

import importlib.metadata
import io
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from PIL import Image

from anometa.artifacts import LICENCES, TrackOutput, git_info
from anometa.config import Paths, Scenario, TrackAConfig, resolve_device
from anometa.data.ad2 import index_scenario
from anometa.data.splits import load_split
from anometa.features.extract import load_features
from anometa.metrics.aggregate import _nanmean, group_metrics, summarize
from anometa.metrics.pixel import pixel_metrics, upsample, validation_threshold

_PIXEL_METRICS: tuple[str, ...] = ("au_pro_005", "au_pro_030", "seg_f1", "class_f1", "auroc")
"""Keys `pixel_metrics` returns."""

_PRED_COLUMNS: list[str] = [
    "scenario",
    "seed",
    "image_id",
    "scene_id",
    "label",
    "lighting",
    "score",
]

_Map = NDArray[np.float16] | NDArray[np.float32]


def distance_maps(cfg: TrackAConfig, scenario: Scenario, paths: Paths) -> dict[str, _Map]:
    """Read a scenario's cached patch distance maps for its non-train images.

    Args:
        cfg: A `patch_distance` Track A config; names the encoder and backend.
        scenario: The scenario to read.
        paths: Run paths; see `load_features`.

    Returns:
        Image id to its `distmap` (patch-grid resolution, float16), for every
        `validation` and `test_public` image.
    """
    assert cfg.encoder is not None  # guaranteed by TrackAConfig validation
    feats = load_features(cfg.encoder, scenario, paths, cfg.encoder_backend)
    keep = ~np.char.startswith(feats.image_id, "train/")
    return {
        str(image_id): distmap
        for image_id, distmap in zip(feats.image_id[keep], feats.distmap[keep], strict=True)
    }


def _size_hw(path: str) -> tuple[int, int]:
    """Read an image's `(height, width)` from its header.

    Args:
        path: Image file.

    Returns:
        `(height, width)` in pixels.
    """
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def _mask(mask_path: object, size_hw: tuple[int, int]) -> NDArray[np.int64]:
    """Load a ground-truth mask as 0/1, or all zeros for a good image.

    Args:
        mask_path: The index's `mask_path`; `None`/NaN for good images.
        size_hw: The image's `(height, width)`.

    Returns:
        `size_hw`-shaped mask, 1 where the PNG is above 0.
    """
    if not isinstance(mask_path, str):
        return np.zeros(size_hw, dtype=np.int64)
    with Image.open(mask_path) as image:
        return (np.asarray(image.convert("L")) > 0).astype(np.int64)


def _subset_metrics(
    maps: Sequence[NDArray[np.float16]],
    masks: Sequence[NDArray[np.int64]],
    labels: NDArray[np.int64],
    keep: NDArray[np.bool_],
    thr: float,
) -> dict[str, float]:
    """Compute `pixel_metrics` on a subset of rows, NaN when it is single-class.

    Args:
        maps: Full-resolution maps, one per evaluation row.
        masks: Masks aligned with `maps`.
        labels: Image labels aligned with `maps`.
        keep: Which rows to score.
        thr: Validation threshold.

    Returns:
        `pixel_metrics` of the kept rows, or NaN for every metric when they
        hold only one class.
    """
    idx: list[int] = np.flatnonzero(keep).tolist()
    if len(np.unique(labels[idx])) < 2:
        return dict.fromkeys(_PIXEL_METRICS, float("nan"))
    return pixel_metrics([maps[i] for i in idx], [masks[i] for i in idx], labels[idx], thr)


@dataclass(frozen=True)
class _ScenarioResult:
    """One scenario's Track A outcome, as saved under `run_dir / "scenarios"`."""

    predictions: pd.DataFrame
    metrics: dict[str, float]
    gap: dict[str, float]
    fit_s: float | None
    model_revisions: dict[str, str]
    maps: dict[str, NDArray[np.float16]]


def _scenario_result(cfg: TrackAConfig, scenario: Scenario) -> _ScenarioResult:
    """Compute one scenario's maps, threshold, predictions and pixel metrics.

    Args:
        cfg: The Track A experiment configuration.
        scenario: The scenario to run.

    Returns:
        The scenario's predictions (`seed = cfg.seeds[0]`), pixel metrics on
        all rows, regular minus shifted gaps, `fit_s` (anomalib models only),
        model revisions, and model-resolution float16 maps of its evaluation
        images keyed by image id.
    """
    split_df = load_split(scenario, cfg.paths)
    index = index_scenario(cfg.paths.data, scenario).set_index("image_id")
    model_revisions: dict[str, str] = {}
    fit_s: float | None = None
    if cfg.model == "patch_distance":
        assert cfg.encoder is not None  # guaranteed by TrackAConfig validation
        maps = distance_maps(cfg, scenario, cfg.paths)
        p = load_features(cfg.encoder, scenario, cfg.paths, cfg.encoder_backend).provenance
        model_revisions[cfg.encoder] = (
            f"{p['backend']}@{p['revision']} (device={p['device']}, dtype={p['dtype']})"
        )
    else:
        from anometa.tracka.anomalib_models import fit_predict_anomalib  # loads anomalib

        start = time.perf_counter()
        maps = fit_predict_anomalib(cfg, scenario, cfg.paths)
        fit_s = time.perf_counter() - start

    val_ids = index.index[index["source"] == "validation"]
    thr = validation_threshold(
        upsample(maps[i], _size_hw(str(index.at[i, "path"]))) for i in val_ids
    )

    eval_df = split_df[split_df["split"] == cfg.split].reset_index(drop=True)
    eval_ids: list[str] = eval_df["image_id"].tolist()
    sizes = [_size_hw(str(index.at[i, "path"])) for i in eval_ids]
    full = [upsample(maps[i], size) for i, size in zip(eval_ids, sizes, strict=True)]
    masks = [_mask(index.at[i, "mask_path"], size) for i, size in zip(eval_ids, sizes, strict=True)]
    labels = eval_df["label"].to_numpy(dtype=np.int64)
    regular = (eval_df["lighting"] == "regular").to_numpy(dtype=np.bool_)

    everything = np.ones(len(eval_ids), dtype=np.bool_)
    on_all = _subset_metrics(full, masks, labels, everything, thr)
    on_regular = _subset_metrics(full, masks, labels, regular, thr)
    on_shifted = _subset_metrics(full, masks, labels, ~regular, thr)

    part = eval_df[["image_id", "scene_id", "label", "lighting"]].copy()
    part["scenario"] = str(scenario)
    part["seed"] = cfg.seeds[0]
    part["score"] = [float(np.max(m)) for m in full]
    return _ScenarioResult(
        predictions=part[_PRED_COLUMNS],
        metrics=on_all,
        gap={m: on_regular[m] - on_shifted[m] for m in _PIXEL_METRICS},
        fit_s=fit_s,
        model_revisions=model_revisions,
        maps={i: maps[i].astype(np.float16) for i in eval_ids},
    )


def _save_scenario(result: _ScenarioResult, path: Path) -> None:
    """Save a scenario result to `path` atomically (temporary directory, then rename).

    Args:
        result: The scenario's outcome.
        path: Directory to create; it must not exist yet.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    result.predictions.to_parquet(tmp / "predictions.parquet")
    meta = {
        "metrics": result.metrics,
        "gap": result.gap,
        "fit_s": result.fit_s,
        "model_revisions": result.model_revisions,
        "git_commit": git_info()[0],
    }
    (tmp / "result.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    np.savez_compressed(tmp / "maps.npz", allow_pickle=False, **result.maps)
    tmp.rename(path)


def _load_scenario(path: Path) -> _ScenarioResult | None:
    """Load a scenario result saved by `_save_scenario`.

    Args:
        path: The scenario's result directory.

    Returns:
        The saved outcome, or `None` when `path` doesn't exist (not computed yet).
    """
    if not path.is_dir():
        return None
    meta = json.loads((path / "result.json").read_text())
    with np.load(path / "maps.npz", allow_pickle=False) as saved:
        maps: dict[str, NDArray[np.float16]] = {k: saved[k] for k in saved.files}
    return _ScenarioResult(
        predictions=pd.read_parquet(path / "predictions.parquet"),
        metrics={k: float(v) for k, v in meta["metrics"].items()},
        gap={k: float(v) for k, v in meta["gap"].items()},
        fit_s=None if meta["fit_s"] is None else float(meta["fit_s"]),
        model_revisions={str(k): str(v) for k, v in meta["model_revisions"].items()},
        maps=maps,
    )


def run_track_a(cfg: TrackAConfig, run_dir: Path) -> TrackOutput:
    """Run one Track A experiment: anomaly maps and pixel metrics per scenario.

    Per scenario: maps from `distance_maps` (`patch_distance`) or a fitted
    anomalib model (`fit_predict_anomalib`, timed as `fit_s`); the threshold
    from `validation_threshold` on the validation maps upsampled to full
    resolution; evaluation rows are the split rows with `split == cfg.split`.
    Their maps are bilinearly upsampled to the image size (float16) and
    scored with `pixel_metrics` against their masks (zeros for good images)
    on all, regular-lit and shifted-lit rows. A single-class subset gives
    NaN metrics. The image score is the full-resolution map's maximum.

    Each scenario's result is saved to `run_dir / "scenarios" / <scenario>`
    as soon as it is computed (`_save_scenario`, atomic) and loaded instead of
    recomputed when present, so a rerun of an interrupted dev run resumes at
    the first unfinished scenario. `peak_vram_mb` then covers only the
    scenarios computed in this process.

    Args:
        cfg: The Track A experiment configuration.
        run_dir: The run directory; per-scenario results go under
            `run_dir / "scenarios"`. `run_experiment` writes the run's own
            artifacts there too.

    Returns:
        Predictions (`seed = cfg.seeds[0]`) for every evaluation image;
        metrics `summarize(group_metrics(pred, probabilistic=False))` merged
        with, per pixel metric `m`, `<m>` (mean over scenarios),
        `<scenario>/<m>` and `gap_<m>` (mean over scenarios of regular minus
        shifted), plus `fit_s` (median over scenarios, anomalib models only)
        and `peak_vram_mb` (CUDA only); and `maps.npz` holding each
        evaluation image's model-resolution float16 map under
        `"<scenario>/<image_id>"`.
    """
    device = resolve_device(cfg.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    licences: dict[str, str] = {"ad2": LICENCES["ad2"]}
    model_revisions: dict[str, str] = {}
    if cfg.model == "patch_distance":
        licences["dinov3"] = LICENCES["dinov3"]
    else:
        licences["anomalib"] = LICENCES["anomalib"]
        model_revisions[cfg.model] = f"anomalib {importlib.metadata.version('anomalib')}"

    pred_parts: list[pd.DataFrame] = []
    per_scenario: dict[str, dict[str, float]] = {}
    gaps: dict[str, dict[str, float]] = {}
    fit_seconds: list[float] = []
    saved_maps: dict[str, NDArray[np.float16]] = {}
    for scenario in cfg.scenarios:
        path = run_dir / "scenarios" / str(scenario)
        result = _load_scenario(path)
        if result is None:
            result = _scenario_result(cfg, scenario)
            _save_scenario(result, path)
        per_scenario[scenario] = result.metrics
        gaps[scenario] = result.gap
        if result.fit_s is not None:
            fit_seconds.append(result.fit_s)
        model_revisions |= result.model_revisions
        pred_parts.append(result.predictions)
        saved_maps |= {f"{scenario}/{i}": m for i, m in result.maps.items()}

    predictions = pd.concat(pred_parts, ignore_index=True)[_PRED_COLUMNS]
    metrics = summarize(group_metrics(predictions, probabilistic=False))
    for m in _PIXEL_METRICS:
        metrics[m] = _nanmean(np.array([v[m] for v in per_scenario.values()]))
        metrics[f"gap_{m}"] = _nanmean(np.array([v[m] for v in gaps.values()]))
        for scenario, values in per_scenario.items():
            metrics[f"{scenario}/{m}"] = values[m]
    if fit_seconds:
        metrics["fit_s"] = float(np.median(fit_seconds))
    if device == "cuda":
        metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20

    buffer = io.BytesIO()
    np.savez_compressed(buffer, allow_pickle=False, **saved_maps)
    return TrackOutput(
        metrics=metrics,
        predictions=predictions,
        model_revisions=model_revisions,
        licences=licences,
        extra_files={"maps.npz": buffer.getvalue()},
    )
