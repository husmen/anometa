"""DINOv3 transformers versus timm fallback parity: token agreement, speed and run comparison.

`token_agreement` compares two encodings of one image, `benchmark` measures
an encoder's throughput and VRAM, `run_parity` applies both to the two
backends of one DINOv3 model, and `compare_runs` compares the downstream
AUROC of two runs that differ only in their backend.
"""

import json
import math
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from anometa.config import Paths, Scenario
from anometa.data.ad2 import index_scenario, load_rgb
from anometa.features.encoders import Encoded, Encoder, load_encoder
from anometa.metrics.aggregate import bootstrap_ci, paired_bootstrap_diff

_AGREEMENT_FIELDS: tuple[str, ...] = (
    "cls_cosine",
    "patch_cosine_mean",
    "patch_cosine_min",
    "max_abs_diff",
)
_BENCHMARK_SHAPES: dict[str, tuple[int, int]] = {
    "sheet_metal": (1056, 4224),
    "fabric": (2048, 2448),
}
"""Zero images of the widest (Sheet Metal) and largest (Fabric) AD2 resolutions."""


def token_agreement(a: Encoded, b: Encoded) -> dict[str, float]:
    """Compare two encodings of the same image, in float32 on the CPU.

    Args:
        a: One backend's encoding.
        b: The other backend's encoding, on the same patch grid.

    Returns:
        `cls_cosine`, `patch_cosine_mean` and `patch_cosine_min` (cosine
        similarity of the pooled embeddings and of each patch pair), and
        `max_abs_diff` (largest absolute difference over the pooled and
        patch values).

    Raises:
        ValueError: If the two encodings have different patch grids.
    """
    if a.patches.shape != b.patches.shape:
        raise ValueError(
            f"patch grids differ: {tuple(a.patches.shape)} vs {tuple(b.patches.shape)}"
        )
    cls_a, cls_b = a.cls.float().cpu(), b.cls.float().cpu()
    dim = a.patches.shape[-1]
    patch_a = a.patches.float().cpu().reshape(-1, dim)
    patch_b = b.patches.float().cpu().reshape(-1, dim)
    patch_cos = torch.nn.functional.cosine_similarity(patch_a, patch_b, dim=1)
    return {
        "cls_cosine": float(torch.nn.functional.cosine_similarity(cls_a, cls_b, dim=0)),
        "patch_cosine_mean": float(patch_cos.mean()),
        "patch_cosine_min": float(patch_cos.min()),
        "max_abs_diff": max(
            float((cls_a - cls_b).abs().max()), float((patch_a - patch_b).abs().max())
        ),
    }


def benchmark(
    encoder: Encoder, image: NDArray[np.uint8], n: int = 20, warmup: int = 3
) -> dict[str, float]:
    """Measure an encoder's throughput and peak VRAM on one image.

    Args:
        encoder: The loaded encoder.
        image: HxWx3 uint8 image, encoded `warmup + n` times.
        n: Timed encodes.
        warmup: Untimed encodes first.

    Returns:
        `images_per_s` over the timed encodes (CUDA-synchronised when the
        encoder's output lives on CUDA) and `peak_vram_mb` (peak allocated
        CUDA memory during the timed encodes; NaN when the output is not on
        CUDA).
    """
    out = encoder.encode(image)
    for _ in range(warmup - 1):
        out = encoder.encode(image)
    cuda = out.cls.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(n):
        encoder.encode(image)
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else math.nan
    return {"images_per_s": n / elapsed, "peak_vram_mb": float(peak)}


def run_parity(
    name: Literal["dinov3_s", "dinov3_l"],
    scenario: Scenario,
    paths: Paths,
    *,
    n_images: int = 32,
    device: str = "auto",
) -> dict[str, float]:
    """Compare the transformers and timm backends of one DINOv3 model.

    Args:
        name: The DINOv3 model.
        scenario: Scenario whose first `n_images` indexed images (by
            `image_id`) are encoded by both backends.
        paths: Run paths; images are read from `paths.data`, the result is
            written to `paths.artifacts / f"parity_{name}.json"`.
        n_images: Images compared.
        device: Device for both backends (see `resolve_device`).

    Returns:
        `<field>_mean` and `<field>_min` over the images for every
        `token_agreement` field, `n_images`, and, per backend and benchmark
        shape, `<backend>_<shape>_images_per_s` and
        `<backend>_<shape>_peak_vram_mb`.
    """
    encoders = {
        "transformers": load_encoder(name, device, backend="transformers"),
        "timm": load_encoder(name, device, backend="timm"),
    }
    rows = index_scenario(paths.data, scenario).head(n_images)
    agreements = [
        token_agreement(encoders["transformers"].encode(image), encoders["timm"].encode(image))
        for image in (load_rgb(Path(p)) for p in rows["path"])
    ]
    frame = pd.DataFrame(agreements)
    result: dict[str, float] = {"n_images": float(len(frame))}
    for field in _AGREEMENT_FIELDS:
        result[f"{field}_mean"] = float(frame[field].mean())
        result[f"{field}_min"] = float(frame[field].min())
    for backend, encoder in encoders.items():
        for shape_name, (h, w) in _BENCHMARK_SHAPES.items():
            speed = benchmark(encoder, np.zeros((h, w, 3), np.uint8))
            for key, value in speed.items():
                result[f"{backend}_{shape_name}_{key}"] = value
    out = paths.artifacts / f"parity_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    return result


def compare_runs(run_a: Path, run_b: Path, metric: str = "auroc") -> dict[str, float]:
    """Compare one metric of two runs scored on the same evaluation rows.

    Args:
        run_a: Run directory holding `predictions.parquet`.
        run_b: Run directory over the same rows (e.g. the same config with
            the other backend).
        metric: A ranking `group_metrics` column, e.g. `"auroc"`.

    Returns:
        `a`, `a_lo`, `a_hi` and `b`, `b_lo`, `b_hi` (each run's value and
        95% bootstrap CI), and `diff`, `diff_lo`, `diff_hi` (A minus B with
        a 95% paired bootstrap CI).
    """
    pred_a = pd.read_parquet(run_a / "predictions.parquet")
    pred_b = pd.read_parquet(run_b / "predictions.parquet")
    a, a_lo, a_hi = bootstrap_ci(pred_a, metric, probabilistic=False)
    b, b_lo, b_hi = bootstrap_ci(pred_b, metric, probabilistic=False)
    diff, diff_lo, diff_hi, _ = paired_bootstrap_diff(pred_a, pred_b, metric, probabilistic=False)
    return {
        "a": a,
        "a_lo": a_lo,
        "a_hi": a_hi,
        "b": b,
        "b_lo": b_lo,
        "b_hi": b_hi,
        "diff": diff,
        "diff_lo": diff_lo,
        "diff_hi": diff_hi,
    }
