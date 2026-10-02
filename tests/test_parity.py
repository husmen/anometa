"""Tests for `anometa.features.parity`: token agreement, throughput and run comparison."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import FakeEncoder
from huggingface_hub import get_token

from anometa.config import Paths, Scenario
from anometa.features import parity
from anometa.features.encoders import Encoded, load_encoder
from anometa.features.parity import benchmark, compare_runs, token_agreement


def enc(seed: int, noise: float = 0.0) -> Encoded:
    """Build a 2x3-patch, 8-dim encoding; `noise` perturbs the patches only."""
    g = torch.Generator().manual_seed(seed)
    patches = torch.randn(2, 3, 8, generator=g)
    return Encoded(
        cls=patches.mean((0, 1)), patches=patches + noise * torch.randn(2, 3, 8, generator=g)
    )


def test_token_agreement_identical_and_perturbed() -> None:
    """Identical encodings agree exactly; perturbed patches lower the patch cosine."""
    same = token_agreement(enc(0), enc(0))
    assert same["cls_cosine"] == pytest.approx(1.0)
    assert same["patch_cosine_mean"] == pytest.approx(1.0)
    assert same["max_abs_diff"] == 0.0
    perturbed = token_agreement(enc(0), enc(0, noise=0.5))
    assert perturbed["patch_cosine_min"] < 1.0
    assert perturbed["max_abs_diff"] > 0.0


def test_token_agreement_rejects_different_grids() -> None:
    """Encodings on different patch grids cannot be compared patch by patch."""
    other = Encoded(cls=torch.zeros(8), patches=torch.zeros(3, 2, 8))
    with pytest.raises(ValueError, match="grid"):
        token_agreement(enc(0), other)


def test_benchmark_with_fake_encoder() -> None:
    """The benchmark reports throughput, and NaN VRAM for an encoder that stays on the CPU."""
    out = benchmark(FakeEncoder(), np.zeros((16, 24, 3), np.uint8), n=3, warmup=1)
    assert out["images_per_s"] > 0
    assert np.isnan(out["peak_vram_mb"])


def _write_run(run_dir: Path, scores: np.ndarray) -> Path:
    """Write a one-scenario, two-seed predictions.parquet with 10 good then 10 bad images."""
    rows = [
        dict(
            scenario="vial",
            seed=s,
            scene_id=f"{kind}/{i:03d}",
            image_id=f"{kind}/{i:03d}",
            label=label,
            lighting="regular",
            score=float(scores[label * 10 + i]),
        )
        for s in range(2)
        for label, kind in [(0, "good"), (1, "bad")]
        for i in range(10)
    ]
    run_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(run_dir / "predictions.parquet")
    return run_dir


def test_compare_runs_perfect_against_inverted(tmp_path: Path) -> None:
    """A perfect run against its inverse: AUROC 1 and 0, paired difference exactly 1."""
    label = np.r_[np.zeros(10), np.ones(10)]
    out = compare_runs(_write_run(tmp_path / "a", label), _write_run(tmp_path / "b", 1 - label))
    assert (out["a"], out["a_lo"], out["a_hi"]) == (1.0, 1.0, 1.0)
    assert (out["b"], out["b_lo"], out["b_hi"]) == (0.0, 0.0, 0.0)
    assert (out["diff"], out["diff_lo"], out["diff_hi"]) == (1.0, 1.0, 1.0)


@pytest.mark.models
@pytest.mark.skipif(get_token() is None, reason="needs gated DINOv3 access (HF token)")
def test_backends_agree_on_one_image() -> None:
    """The transformers and timm backends of DINOv3-S give nearly identical tokens."""
    img = np.random.default_rng(0).integers(0, 255, (2048, 2448, 3), dtype=np.uint8)
    a = load_encoder("dinov3_s", "cpu", backend="transformers").encode(img)
    b = load_encoder("dinov3_s", "cpu", backend="timm").encode(img)
    assert token_agreement(a, b)["cls_cosine"] >= 0.99


def test_run_parity_with_fake_backends(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two identical fake backends agree exactly; the JSON holds agreement and speed fields."""
    monkeypatch.setattr(parity, "load_encoder", lambda name, device, backend: FakeEncoder())
    result = parity.run_parity("dinov3_s", Scenario.VIAL, paths, n_images=3)
    assert result["n_images"] == 3
    assert result["cls_cosine_min"] == pytest.approx(1.0)
    assert result["max_abs_diff_mean"] == 0.0
    assert result["timm_fabric_images_per_s"] > 0
    saved = json.loads((paths.artifacts / "parity_dinov3_s.json").read_text())
    assert saved.keys() == result.keys()
