"""Rerun inspection recordings: images, masks, anomaly maps and embeddings of one run.

`write_rrd` reads a finished run directory and writes `inspect.rrd` next to
its artifacts, for the Rerun viewer (`rerun <path>`). It uses its own
`RecordingStream`, never the global one, so it can run inside other tools.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
from matplotlib import colormaps
from numpy.typing import NDArray
from PIL import Image
from pydantic import TypeAdapter
from sklearn.decomposition import PCA

from anometa.artifacts import read_manifest
from anometa.config import ExperimentConfig, Paths, Scenario
from anometa.data.ad2 import index_scenario, load_rgb
from anometa.features.extract import load_features

_MAX_SIDE = 1024
"""Longest image side logged, in pixels."""


def _downscale(image: Image.Image, resample: Image.Resampling) -> Image.Image:
    """Shrink an image so its longest side is at most `_MAX_SIDE`.

    Args:
        image: The image.
        resample: PIL resampling filter.

    Returns:
        The image, unchanged when it is already small enough.
    """
    scale = _MAX_SIDE / max(image.size)
    if scale >= 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), resample)


def _mask(mask_path: object, size: tuple[int, int]) -> NDArray[np.uint8]:
    """Load a ground-truth mask as class ids (0 good, 1 defect) at a display size.

    Args:
        mask_path: The index's `mask_path`; `None`/NaN for good images.
        size: Display `(width, height)`.

    Returns:
        `(height, width)` class-id array.
    """
    if not isinstance(mask_path, str):
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    with Image.open(mask_path) as image:
        resized = image.convert("L").resize(size, Image.Resampling.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def _overlay(
    map_: NDArray[np.float16], size: tuple[int, int], lo: float, hi: float
) -> NDArray[np.uint8]:
    """Colour an anomaly map with `turbo` as an RGBA overlay at a display size.

    Args:
        map_: Raw anomaly map at model resolution.
        size: Display `(width, height)`.
        lo: Score mapped to the colormap's low end.
        hi: Score mapped to the colormap's high end.

    Returns:
        `(height, width, 4)` uint8 RGBA image.
    """
    resized = Image.fromarray(map_.astype(np.float32)).resize(size, Image.Resampling.BILINEAR)
    norm = np.clip((np.asarray(resized) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    rgba: NDArray[np.float64] = colormaps["turbo"](norm)
    return (rgba * 255).astype(np.uint8)


def _log_embedding(
    rec: rr.RecordingStream, cfg: ExperimentConfig, pred: pd.DataFrame, paths: Paths
) -> None:
    """Log a 3-component PCA of each scenario's evaluation-image CLS features.

    Uses the run's encoder, or `dinov3_s` for an anomalib run; a scenario
    without cached features is skipped.

    Args:
        rec: The recording.
        cfg: The run's config.
        pred: The run's predictions (one seed).
        paths: Where the feature cache lives.
    """
    encoder = cfg.encoder or "dinov3_s"
    for scenario, rows in pred.groupby("scenario"):
        try:
            feats = load_features(encoder, Scenario(str(scenario)), paths, cfg.encoder_backend)
        except FileNotFoundError:
            continue
        cls = feats.cls[feats.rows(rows["image_id"].tolist())]
        points = PCA(n_components=min(3, *cls.shape)).fit_transform(cls)
        rec.log(
            f"embedding/{scenario}",
            rr.Points3D(
                np.pad(points, ((0, 0), (0, 3 - points.shape[1]))),
                class_ids=rows["label"].to_numpy(),
                labels=rows["lighting"].tolist(),
            ),
            static=True,
        )


def write_rrd(run_dir: Path, paths: Paths, max_images: int = 50) -> Path:
    """Write a Rerun recording that inspects one finished run.

    On the `image` timeline, one step per evaluation image of the run's first
    seed, highest score first, up to `max_images`: the RGB image (longest
    side at most 1024 px) at `image`, its mask as a `SegmentationImage` at
    `image/mask` (good/defect `AnnotationContext`, logged static), for Track
    A the anomaly map as a 50% opaque `turbo` overlay at `image/anomaly_map`
    (colour range shared across the logged maps), and the score at `score`.
    Once per run: a 3-component PCA of the evaluation images' CLS features
    per scenario at `embedding/<scenario>`, coloured by label and labelled
    with the lighting.

    Args:
        run_dir: A run directory with `manifest.json` and
            `predictions.parquet` (and `maps.npz` for Track A).
        paths: Where the AD2 images and the feature cache live.
        max_images: Most images logged.

    Returns:
        `run_dir / "inspect.rrd"`.

    Raises:
        FileNotFoundError: If `run_dir` has no manifest.
    """
    manifest = read_manifest(run_dir)
    if manifest is None:
        raise FileNotFoundError(f"{run_dir / 'manifest.json'} not found")
    cfg = TypeAdapter(ExperimentConfig).validate_python(manifest["config"])
    pred = pd.read_parquet(run_dir / "predictions.parquet")
    pred = pred[pred["seed"] == pred["seed"].min()]
    shown = pred.sort_values("score", ascending=False).head(max_images)
    maps: dict[str, NDArray[np.float16]] = {}
    if cfg.track == "A":
        with np.load(run_dir / "maps.npz") as data:
            maps = {
                f"{s}/{i}": data[f"{s}/{i}"]
                for s, i in zip(shown.scenario, shown.image_id, strict=True)
            }
    values = np.concatenate([m.ravel() for m in maps.values()]) if maps else np.zeros(1)
    lo, hi = float(values.min()), float(values.max())
    indexes = {
        str(s): index_scenario(paths.data, Scenario(str(s))).set_index("image_id")
        for s in shown["scenario"].unique()
    }

    out = run_dir / "inspect.rrd"
    rec = rr.RecordingStream("anometa")
    rec.save(out)
    rec.log(
        "image",
        rr.AnnotationContext([(0, "good", (0, 0, 0, 0)), (1, "defect", (255, 0, 0, 128))]),
        static=True,
    )
    for i, row in enumerate(shown.itertuples(index=False)):
        rec.set_time("image", sequence=i)
        entry = indexes[str(row.scenario)].loc[str(row.image_id)]
        rgb = _downscale(
            Image.fromarray(load_rgb(Path(str(entry["path"])))), Image.Resampling.BILINEAR
        )
        rec.log("image", rr.Image(np.asarray(rgb)))
        rec.log("image/mask", rr.SegmentationImage(_mask(entry["mask_path"], rgb.size)))
        key = f"{row.scenario}/{row.image_id}"
        if key in maps:
            rec.log(
                "image/anomaly_map",
                rr.Image(_overlay(maps[key], rgb.size, lo, hi), opacity=0.5, draw_order=1),
            )
        rec.log("score", rr.Scalars(float(row.score)))
    _log_embedding(rec, cfg, pred, paths)
    rec.flush()
    rec.disconnect()
    return out
