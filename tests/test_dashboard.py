"""Tests for `anometa.dashboard.app`: the Streamlit dashboard's pure functions."""

import json
from pathlib import Path

import numpy as np
import pytest

from anometa.config import Paths, Scenario
from anometa.data.splits import eval_rows, load_split
from anometa.features.extract import load_features

pytest.importorskip("streamlit")

from anometa.dashboard.app import (
    fit_and_score,
    heat_overlay,
    load_mask,
    map_range,
    map_runs,
    outline,
    thumbnail,
)


def test_thumbnail_keeps_aspect_and_size(ad2_root: Path) -> None:
    """Thumbnails are RGB uint8 arrays no larger than the requested side."""
    rgb = thumbnail(ad2_root / "vial/train/good/000_regular.png", 64)
    assert rgb.dtype == np.uint8
    assert rgb.ndim == 3
    assert rgb.shape[2] == 3
    assert max(rgb.shape[:2]) <= 64


def test_load_mask_is_empty_for_good_images() -> None:
    """A missing mask path (good image) gives an all-False mask of the requested size."""
    mask = load_mask(None, (5, 7))
    assert mask.shape == (5, 7)
    assert not mask.any()


def test_outline_paints_only_the_mask_boundary() -> None:
    """The boundary of a square mask is painted; its interior and the outside stay unchanged."""
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    out = outline(rgb, mask, color=(255, 0, 0), width=1)
    assert tuple(out[2, 2]) == (255, 0, 0)
    assert tuple(out[5, 5]) == (0, 0, 0)
    assert tuple(out[0, 0]) == (0, 0, 0)
    assert not rgb.any()  # input untouched


def test_heat_overlay_brightens_high_scores() -> None:
    """A map's high-score half comes out brighter than its low half, at the image's size."""
    rgb = np.full((20, 40, 3), 100, dtype=np.uint8)
    heat = np.array([[0.0, 1.0]], dtype=np.float32)
    out = heat_overlay(rgb, heat, lo=0.0, hi=1.0)
    assert out.shape == rgb.shape
    assert out[:, 30:].mean() > out[:, :10].mean()


def test_map_range_uses_robust_percentiles() -> None:
    """The display range ignores a single extreme value."""
    maps = [np.linspace(0, 1, 1000, dtype=np.float32), np.array([1000.0], dtype=np.float32)]
    lo, hi = map_range(maps)
    assert lo < 0.05
    assert hi < 10


def test_map_runs_finds_dev_track_a_maps(tmp_path: Path) -> None:
    """Only finished dev Track A runs with saved maps for the scenario are offered."""

    def run(name: str, **config: object) -> None:
        d = tmp_path / name
        d.mkdir()
        (d / "maps.npz").write_bytes(b"")
        base = {"track": "A", "split": "dev", "model": "patchcore", "scenarios": ["vial"]}
        manifest = {"status": "ok", "config": base | config}
        (d / "manifest.json").write_text(json.dumps(manifest))

    run("pc-512", image_size=[512, 512])
    run("lock-pc", split="lock")
    run("other-scenario", scenarios=["can"])
    run("distance", model="patch_distance")
    found = map_runs(tmp_path, Scenario.VIAL)
    assert list(found) == ["patchcore 512x512 (pc-512)"]


def test_fit_and_score_excludes_picked_scenes(prepared: Paths) -> None:
    """Scoring covers dev rows except every image of a picked scene."""
    split = load_split(Scenario.VIAL, prepared)
    picked = list(
        split.query("split == 'dev' and label == 1 and lighting == 'regular'").image_id[:2]
    )
    rows, ms = fit_and_score(
        load_features("dinov3_s", Scenario.VIAL, prepared),
        split,
        picked,
        features=("cls", "mean_patch"),
        pca_dim=4,
        classifier="logreg",
        device="cpu",
    )
    assert set(rows.image_id) == set(eval_rows(split, split="dev", shots=picked).image_id)
    assert ms > 0
