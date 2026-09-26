"""Tests for `anometa.metrics.image`: image-level metrics and prior correction."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from anometa.metrics.image import auroc, ece, image_metrics, prior_correct


def test_auroc_matches_sklearn_with_ties():
    """Rank AUROC equals sklearn's, including tied scores."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200) + 0.3 * y, 1)
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_metrics_nan_on_single_class():
    """With one class present every metric is NaN, never an exception."""
    m = image_metrics(np.ones(5), np.full(5, 0.7), np.full(5, 0.7))
    assert all(np.isnan(v) for v in m.values())


def test_ece_extremes():
    """Perfect hard predictions give ECE 0; inverted ones give 1."""
    y = np.array([0, 1, 0, 1])
    assert ece(y, y.astype(float)) == 0.0
    assert ece(y, 1.0 - y) == 1.0


def test_prior_correct():
    """A 50/50 prior is the identity; p equal to the training prior maps to 0.5."""
    p = np.array([0.02, 0.3, 0.9])
    assert np.allclose(prior_correct(p, 0.5), p)
    assert prior_correct(np.array([0.05]), 0.05)[0] == pytest.approx(0.5)
    assert np.all(np.diff(prior_correct(p, 0.05)) > 0)


@pytest.mark.parametrize("pi", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_prior_correct_rejects_pi_outside_open_unit_interval(pi):
    """A prior of 0, 1, outside [0, 1] or NaN raises ValueError instead of returning NaN/inf."""
    with pytest.raises(ValueError, match="pi"):
        prior_correct(np.array([0.2, 0.8]), pi)


def test_image_metrics_keys():
    """Probabilistic scores get raw and balanced calibration metrics."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert set(image_metrics(y, p, p)) == {
        "auroc",
        "auprc",
        "nll",
        "ece",
        "brier",
        "nll_bal",
        "ece_bal",
        "brier_bal",
    }
    assert set(image_metrics(y, p)) == {"auroc", "auprc"}
