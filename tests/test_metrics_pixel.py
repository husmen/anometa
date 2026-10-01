"""Tests for `anometa.metrics.pixel`: AU-PRO, SegF1, ClassF1 and pixel metrics."""

import numpy as np
import pytest

from anometa.metrics.pixel import (
    au_pro,
    au_pros,
    class_f1,
    pixel_metrics,
    seg_f1,
    upsample,
    validation_threshold,
)


def test_au_pro_hand_example():
    """One 1x4 image, one component: AU-PRO is 0.5 up to FPR 0.3 and 0.75 up to 1.0."""
    maps = [np.array([[0.9, 0.2, 0.5, 0.1]])]
    masks = [np.array([[1, 1, 0, 0]])]
    assert au_pro(maps, masks, 0.3) == pytest.approx(0.5)
    assert au_pro(maps, masks, 1.0) == pytest.approx(0.75)


def test_au_pros_matches_au_pro_per_limit():
    """One PRO curve integrated at several limits equals separate au_pro calls."""
    rng = np.random.default_rng(0)
    maps = [rng.random((8, 8)) for _ in range(3)]
    masks = [(rng.random((8, 8)) > 0.8).astype(int) for _ in range(3)]
    limits = [0.05, 0.3, 1.0]
    assert au_pros(maps, masks, limits) == [au_pro(maps, masks, lim) for lim in limits]


def test_au_pro_perfect_map():
    """A map equal to the mask scores 1 at the server limit."""
    m = np.zeros((8, 8))
    m[2:4, 2:5] = 1
    assert au_pro([m], [m.astype(int)], 0.05) == pytest.approx(1.0)


def test_seg_f1_pooled_over_images():
    """Pixel counts are pooled: TP=2, FP=1, FN=1 gives F1 = 2/3."""
    maps = [np.array([[1.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])]
    masks = [np.array([[1, 1, 1]]), np.array([[0, 0, 0]])]
    assert seg_f1(maps, masks, 0.5) == pytest.approx(2 / 3)


def test_class_f1_any_pixel():
    """One pixel above the threshold makes an image anomalous."""
    maps = [np.array([[0.0, 0.9]]), np.array([[0.1, 0.2]]), np.array([[0.6, 0.0]])]
    assert class_f1(maps, np.array([1, 0, 0]), 0.5) == pytest.approx(2 / 3)


def test_validation_threshold_and_upsample():
    """Threshold is mean + 3 std; upsampling returns float16 at the requested size."""
    assert validation_threshold([np.array([[0.0, 2.0]]), np.array([[0.0, 2.0]])]) == pytest.approx(
        4.0
    )
    up = upsample(np.ones((4, 6), np.float32), (16, 24))
    assert up.shape == (16, 24)
    assert up.dtype == np.float16


def test_pixel_metrics_keys():
    """`pixel_metrics` reports the full MVTec AD 2 pixel metric set."""
    maps = [np.array([[0.9, 0.2, 0.5, 0.1]]), np.array([[0.05, 0.05, 0.05, 0.05]])]
    masks = [np.array([[1, 1, 0, 0]]), np.array([[0, 0, 0, 0]])]
    labels = np.array([1, 0])
    result = pixel_metrics(maps, masks, labels, thr=0.5)
    assert set(result) == {"au_pro_005", "au_pro_030", "seg_f1", "class_f1", "auroc"}
    assert all(isinstance(v, float) for v in result.values())
