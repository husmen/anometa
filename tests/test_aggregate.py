"""Tests for `anometa.metrics.aggregate`: grouping, summaries and bootstrap CIs."""

import numpy as np
import pandas as pd
import pytest

from anometa.metrics.aggregate import (
    bootstrap_ci,
    group_metrics,
    paired_bootstrap_diff,
    summarize,
)
from anometa.metrics.image import image_metrics


def pred_frame(regular_ok=True, shifted_ok=False, shifted_labels=(0, 1)):
    """Build a predictions frame for scenario "vial", seed 0, four scenes.

    Two good (label 0) and two bad (label 1) scenes, each shot under
    "regular" lighting and, for labels in `shifted_labels`, also under
    "shift_1". A lighting's score is perfect (`label`) when its `ok` flag is
    true, inverted (`1 - label`) otherwise.
    """
    rows = []
    for scene, label in [("good/000", 0), ("good/001", 0), ("bad/000", 1), ("bad/001", 1)]:
        lightings = [("regular", regular_ok)] + (
            [("shift_1", shifted_ok)] if label in shifted_labels else []
        )
        for lighting, ok in lightings:
            rows.append(
                dict(
                    scenario="vial",
                    seed=0,
                    scene_id=scene,
                    image_id=f"{scene}_{lighting}",
                    label=label,
                    lighting=lighting,
                    score=float(label if ok else 1 - label),
                )
            )
    return pd.DataFrame(rows)


def test_group_metrics_gap():
    """Perfect on regular, inverted on shifted: AUROC gap is 1."""
    g = group_metrics(pred_frame(), probabilistic=False)
    assert g.loc[0, "gap_auroc"] == pytest.approx(1.0)


def test_single_class_subset_gives_nan_not_error():
    """A shifted subset with only anomalies yields a NaN gap that summarize counts."""
    s = summarize(group_metrics(pred_frame(shifted_labels=(1,)), probabilistic=False))
    assert np.isnan(s["gap_auroc"])
    assert s["n_nan_gap_auroc"] == 1
    assert s["auroc"] == pytest.approx(0.75)
    assert s["vial/auroc"] == pytest.approx(0.75)


def test_bootstrap_ci_perfect_and_deterministic():
    """A perfect scorer has CI (1, 1, 1); the same seed reproduces the same CI."""
    df = pred_frame(shifted_ok=True)
    assert bootstrap_ci(df, "auroc", probabilistic=False, n_boot=200) == (1.0, 1.0, 1.0)
    noisy = df.assign(score=np.random.default_rng(1).random(len(df)))
    a = bootstrap_ci(noisy, "auroc", probabilistic=False, n_boot=200, seed=3)
    assert a == bootstrap_ci(noisy, "auroc", probabilistic=False, n_boot=200, seed=3)
    assert a[1] <= a[0] <= a[2]


def test_bootstrap_ci_multi_scenario_multi_seed():
    """Two scenarios, uneven seed pools: perfect gives (1, 1, 1); noisy point = summarize auroc."""
    vial_seed1 = pred_frame(shifted_ok=True).assign(scenario="vial", seed=1)
    perfect = pd.concat(
        [
            pred_frame(shifted_ok=True).assign(scenario="vial", seed=0),
            vial_seed1[vial_seed1.scene_id != "bad/000"],  # a shot scene seed 1 never evaluates
            pred_frame(shifted_ok=True).assign(scenario="wallplugs", seed=0),
        ],
        ignore_index=True,
    )
    assert bootstrap_ci(perfect, "auroc", probabilistic=False, n_boot=200) == (1.0, 1.0, 1.0)

    noisy = perfect.assign(score=np.random.default_rng(1).random(len(perfect)))
    a = bootstrap_ci(noisy, "auroc", probabilistic=False, n_boot=200, seed=3)
    assert a == bootstrap_ci(noisy, "auroc", probabilistic=False, n_boot=200, seed=3)
    shuffled = noisy.sample(frac=1, random_state=0)
    assert a == bootstrap_ci(shuffled, "auroc", probabilistic=False, n_boot=200, seed=3)
    assert a[0] == summarize(group_metrics(noisy, probabilistic=False))["auroc"]
    assert a[1] <= a[0] <= a[2]


def test_bootstrap_ci_identical_seeds_keep_width():
    """Ten identical seed copies keep at least 0.8x the one-seed CI width.

    Seeds share each replicate's scene draw, so duplicating a seed adds no
    information and must not shrink the interval (independent per-seed scene
    draws would shrink it by about sqrt(10)).
    """
    rng = np.random.default_rng(0)
    rows = [
        dict(
            scenario="vial",
            seed=0,
            scene_id=f"{kind}/{i:03d}",
            image_id=f"{kind}/{i:03d}_{lighting}",
            label=label,
            lighting=lighting,
            score=0.5 * label + rng.random(),
        )
        for label, kind in [(0, "good"), (1, "bad")]
        for i in range(20)
        for lighting in ("regular", "shift_1")
    ]
    one = pd.DataFrame(rows)
    ten = pd.concat([one.assign(seed=s) for s in range(10)], ignore_index=True)
    _, lo1, hi1 = bootstrap_ci(one, "auroc", probabilistic=False, n_boot=500)
    _, lo10, hi10 = bootstrap_ci(ten, "auroc", probabilistic=False, n_boot=500)
    assert hi10 - lo10 >= 0.8 * (hi1 - lo1)


def test_group_metrics_probabilistic_known_values():
    """Perfect regular and inverted shifted scores with a constant 0.2 balanced score.

    Every calibration metric and its gap matches the hand-computed value, and
    the columns are the `image_metrics` keys plus their `gap_` counterparts.
    """
    df = pred_frame().assign(score_balanced=0.2)
    g = group_metrics(df, probabilistic=True)
    keys = set(image_metrics(np.array([0, 1]), np.array([0.2, 0.8]), np.array([0.2, 0.8])))
    assert set(g.columns) == {"scenario", "seed"} | keys | {f"gap_{k}" for k in keys}
    row = g.loc[0]
    assert row["auroc"] == pytest.approx(0.5)
    assert row["ece"] == pytest.approx(0.5)
    assert row["brier"] == pytest.approx(0.5)
    assert row["gap_brier"] == pytest.approx(-1.0)
    assert row["gap_ece"] == pytest.approx(-1.0)
    assert row["ece_bal"] == pytest.approx(0.3)
    assert row["brier_bal"] == pytest.approx(0.34)
    assert row["nll_bal"] == pytest.approx(-np.log(0.16) / 2)
    assert row["gap_ece_bal"] == pytest.approx(0.0)


def test_bootstrap_ci_ece_bal_known_value():
    """A constant 0.2 balanced score over a 50/50 label mix gives ece_bal CI (0.3, 0.3, 0.3).

    Each label's scene draw keeps its row count, so every replicate sees the
    same label mix and the same ECE.
    """
    df = pred_frame().assign(score_balanced=0.2)
    ci = bootstrap_ci(df, "ece_bal", probabilistic=True, n_boot=200)
    assert ci == pytest.approx((0.3, 0.3, 0.3))


def _two_scenario_frame(seed: int = 0) -> pd.DataFrame:
    """Build two scenarios x three seeds of 12 good and 12 bad scenes under two lightings.

    Scores are noisy but informative (`0.5 * label + uniform`).
    """
    rng = np.random.default_rng(seed)
    rows = [
        dict(
            scenario=scenario,
            seed=s,
            scene_id=f"{kind}/{i:03d}",
            image_id=f"{kind}/{i:03d}_{lighting}",
            label=label,
            lighting=lighting,
            score=0.5 * label + rng.random(),
        )
        for scenario in ("vial", "can")
        for s in range(3)
        for label, kind in [(0, "good"), (1, "bad")]
        for i in range(12)
        for lighting in ("regular", "shift_1")
    ]
    return pd.DataFrame(rows)


def test_paired_bootstrap_diff_known_values():
    """A perfect run against its inverse differs by exactly 1; a run against itself by 0."""
    df = pred_frame(shifted_ok=True)
    inverted = df.assign(score=1 - df["score"])
    assert paired_bootstrap_diff(df, inverted, "auroc", probabilistic=False, n_boot=100) == (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    noisy = _two_scenario_frame()
    same = paired_bootstrap_diff(noisy, noisy, "auroc", probabilistic=False, n_boot=100)
    assert same == (0.0, 0.0, 0.0, 0.0)


def test_paired_bootstrap_diff_is_narrower_than_unpaired():
    """Two runs that differ by small noise get a paired CI far narrower than either run's own CI.

    The point is the difference of the two runs' summarize means, input row
    order does not matter, and the same seed reproduces the result.
    """
    a = _two_scenario_frame()
    b = a.assign(score=a["score"] + np.random.default_rng(9).normal(0, 0.05, len(a)))
    point, lo, hi, share = paired_bootstrap_diff(a, b, "auroc", probabilistic=False, n_boot=300)
    expected = (
        summarize(group_metrics(a, False))["auroc"] - summarize(group_metrics(b, False))["auroc"]
    )
    assert point == pytest.approx(expected)
    assert lo <= point <= hi
    assert 0.0 <= share <= 1.0
    _, a_lo, a_hi = bootstrap_ci(a, "auroc", probabilistic=False, n_boot=300)
    assert hi - lo < 0.5 * (a_hi - a_lo)
    shuffled = b.sample(frac=1, random_state=0)
    assert paired_bootstrap_diff(a, shuffled, "auroc", probabilistic=False, n_boot=300) == (
        point,
        lo,
        hi,
        share,
    )


def test_paired_bootstrap_diff_rejects_unpaired_runs():
    """Runs over different rows, or with different labels, raise ValueError."""
    a = _two_scenario_frame()
    with pytest.raises(ValueError, match="same rows"):
        paired_bootstrap_diff(a, a.iloc[1:], "auroc", probabilistic=False, n_boot=10)
    with pytest.raises(ValueError, match="same rows"):
        paired_bootstrap_diff(a, a.assign(label=1 - a["label"]), "auroc", probabilistic=False)
