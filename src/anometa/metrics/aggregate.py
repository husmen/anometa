"""Aggregate per-image predictions into per-run metrics, gaps and bootstrap CIs.

The predictions frame these functions consume has one row per scored image,
with columns `scenario, seed, image_id, scene_id, label, lighting, score` and
an optional `score_balanced` (a probability under a balanced training prior,
see `anometa.metrics.image.image_metrics`). `group_metrics` reduces it to one
row per (scenario, seed); `summarize` reduces that further to scalars;
`bootstrap_ci` estimates a confidence interval for one column of it by
resampling seeds and scenes, and `paired_bootstrap_diff` does the same for the
difference between two runs scored on the same rows.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from anometa.metrics.image import auprc, auroc, brier, ece, nll

_CI_PERCENTILES: tuple[float, float] = (2.5, 97.5)

_RANKING_METRICS: tuple[str, ...] = ("auroc", "auprc")
_METRIC_FNS: dict[str, tuple[Callable[[NDArray[np.int64], NDArray[np.float64]], float], bool]] = {
    "auroc": (auroc, False),
    "auprc": (auprc, False),
    "nll": (nll, False),
    "ece": (ece, False),
    "brier": (brier, False),
    "nll_bal": (nll, True),
    "ece_bal": (ece, True),
    "brier_bal": (brier, True),
}
"""Metric name -> (function, whether it reads `score_balanced` instead of `score`).

Same names and order as `image_metrics`; only `_RANKING_METRICS` apply
without `score_balanced`.
"""

_Arrays = tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]
"""`(label, score, score_balanced, is_regular)` columns of a predictions frame."""


def _nanmean(values: NDArray[np.float64]) -> float:
    """Mean ignoring NaNs, without a "Mean of empty slice" warning.

    Args:
        values: 1-D array, possibly containing NaNs.

    Returns:
        The NaN-ignoring mean, or NaN if every value is NaN.
    """
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _metric_names(probabilistic: bool) -> list[str]:
    """List the base metrics `group_metrics` reports.

    Args:
        probabilistic: Whether calibration metrics are included.

    Returns:
        Metric names in `image_metrics` order.
    """
    return list(_METRIC_FNS) if probabilistic else list(_RANKING_METRICS)


def _arrays(pred: pd.DataFrame, probabilistic: bool) -> _Arrays:
    """Extract the columns metric computation reads from a predictions frame.

    Args:
        pred: Predictions frame (see module docstring).
        probabilistic: Whether to read `score_balanced`; when `False` it is
            all-NaN and never read, since only ranking metrics apply.

    Returns:
        `(label, score, score_balanced, is_regular)`, aligned with `pred`'s rows.
    """
    y = pred["label"].to_numpy(dtype=np.int64)
    score = pred["score"].to_numpy(dtype=np.float64)
    bal = (
        pred["score_balanced"].to_numpy(dtype=np.float64)
        if probabilistic
        else np.full(len(pred), np.nan)
    )
    regular = (pred["lighting"] == "regular").to_numpy(dtype=np.bool_)
    return y, score, bal, regular


def _base_value(name: str, arrays: _Arrays, idx: NDArray[np.intp]) -> float:
    """Compute one base metric over a set of rows.

    Args:
        name: A `_METRIC_FNS` key.
        arrays: `_arrays` output.
        idx: Row positions to score; repeats count as separate images.

    Returns:
        The metric, or NaN if the rows hold only one class (as
        `image_metrics` does).
    """
    y, score, bal, _regular = arrays
    labels = y[idx]
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.shape[0]:
        return float("nan")
    fn, balanced = _METRIC_FNS[name]
    return fn(labels, (bal if balanced else score)[idx])


def _value(metric: str, arrays: _Arrays, idx: NDArray[np.intp]) -> float:
    """Compute one `group_metrics` column over a set of rows.

    Args:
        metric: A base metric name, or `gap_<name>` for that metric on
            regular-lit rows minus the same metric on the rest.
        arrays: `_arrays` output.
        idx: Row positions to score.

    Returns:
        The metric or gap value; NaN when a scored subset is single-class.
    """
    base = metric.removeprefix("gap_")
    if base == metric:
        return _base_value(base, arrays, idx)
    regular = arrays[3][idx]
    return _base_value(base, arrays, idx[regular]) - _base_value(base, arrays, idx[~regular])


def group_metrics(pred: pd.DataFrame, probabilistic: bool) -> pd.DataFrame:
    """Reduce a predictions frame to one metrics row per (scenario, seed).

    For each (scenario, seed) group, computes the `image_metrics` set over
    every row plus a `gap_<m>` for every metric `m`: that metric on
    `lighting == "regular"` rows minus the same metric on the remaining rows.
    A positive `gap_auroc`/`gap_auprc` means regular lighting scores better;
    a positive `gap_nll`/`gap_ece`/`gap_brier` (and their `_bal` variants)
    means regular lighting scores worse, since lower is better for those.

    Args:
        pred: Predictions frame (see module docstring).
        probabilistic: Whether to read `score_balanced` so calibration
            metrics (`nll`, `ece`, `brier`, ...) are included.

    Returns:
        One row per (scenario, seed), with columns `scenario`, `seed`, the
        `image_metrics` keys and their `gap_<m>` counterparts.
    """
    pred = pred.reset_index(drop=True)
    arrays = _arrays(pred, probabilistic)
    names = _metric_names(probabilistic)
    columns = [*names, *(f"gap_{name}" for name in names)]
    rows: list[dict[str, float | int | str]] = []
    for (scenario, seed), g in pred.groupby(["scenario", "seed"], sort=True):
        assert isinstance(seed, int | np.integer)  # narrows the Hashable groupby key
        idx = g.index.to_numpy(dtype=np.intp)
        row: dict[str, float | int | str] = {"scenario": str(scenario), "seed": int(seed)}
        row |= {col: _value(col, arrays, idx) for col in columns}
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(groups: pd.DataFrame) -> dict[str, float]:
    """Reduce `group_metrics` output to scalar summaries.

    For every metric column (every column but `scenario` and `seed`),
    reports the NaN-ignoring mean over all rows (`<m>`), the NaN-ignoring
    mean over that scenario's rows alone (`<scenario>/<m>`), and the NaN
    count over all rows (`n_nan_<m>`).

    Args:
        groups: `group_metrics` output.

    Returns:
        A flat dict of the summaries described above.
    """
    metric_cols = [c for c in groups.columns if c not in ("scenario", "seed")]
    out: dict[str, float] = {}
    for col in metric_cols:
        values = groups[col].to_numpy(dtype=np.float64)
        out[col] = _nanmean(values)
        out[f"n_nan_{col}"] = float(np.isnan(values).sum())
    for scenario_key, g in groups.groupby("scenario", sort=True):
        scenario = str(scenario_key)
        for col in metric_cols:
            out[f"{scenario}/{col}"] = _nanmean(g[col].to_numpy(dtype=np.float64))
    return out


def _check_metric(metric: str, probabilistic: bool) -> None:
    """Reject a metric name that is not a `group_metrics` column.

    Args:
        metric: The metric name.
        probabilistic: Whether calibration metrics are available.

    Raises:
        KeyError: If `metric` is not a `group_metrics` column for this
            `probabilistic` setting.
    """
    if metric.removeprefix("gap_") not in _metric_names(probabilistic):
        raise KeyError(f"{metric!r} is not a group_metrics column (probabilistic={probabilistic})")


def _replicates(
    pred: pd.DataFrame, metric: str, scores: list[_Arrays], n_boot: int, seed: int
) -> NDArray[np.float64]:
    """Draw bootstrap replicates of a metric's mean for one or more score sets.

    The resampling is the one `bootstrap_ci` describes. Every score set is
    scored on the same draws, so replicates of different score sets over the
    same rows are paired.

    Args:
        pred: Predictions frame sorted by `scenario, seed, image_id` with a
            fresh index; only its structure columns are read.
        metric: A `group_metrics` column name.
        scores: `_arrays` outputs aligned with `pred`'s rows.
        n_boot: Number of bootstrap replicates.
        seed: RNG seed.

    Returns:
        Array of shape `(len(scores), n_boot)`: the replicate means.
    """
    # Codes follow sorted (scenario, label, scene_id): each (scenario, label) is one code range.
    scene_of_row = (
        pred.groupby(["scenario", "label", "scene_id"], sort=True).ngroup().to_numpy(dtype=np.intp)
    )
    blocks: dict[str, list[tuple[int, int]]] = {}
    for (scenario, _label), g in pred.groupby(["scenario", "label"], sort=True):
        codes = scene_of_row[g.index.to_numpy(dtype=np.intp)]
        first = int(codes.min())
        blocks.setdefault(str(scenario), []).append((first, int(codes.max()) - first + 1))
    seed_rows: dict[str, list[NDArray[np.intp]]] = {}
    for (scenario, _seed), g in pred.groupby(["scenario", "seed"], sort=True):
        seed_rows.setdefault(str(scenario), []).append(g.index.to_numpy(dtype=np.intp))

    rng = np.random.default_rng(seed)
    counts = np.zeros(int(scene_of_row.max()) + 1, dtype=np.intp)
    replicates = np.empty((len(scores), n_boot), dtype=np.float64)
    for b in range(n_boot):
        group_values: list[list[float]] = [[] for _ in scores]
        for scenario, pool in seed_rows.items():
            drawn = rng.integers(0, len(pool), size=len(pool))
            for start, size in blocks[scenario]:
                counts[start : start + size] = np.bincount(
                    rng.integers(0, size, size=size), minlength=size
                )
            for d in drawn:
                rows = pool[d]
                idx = np.repeat(rows, counts[scene_of_row[rows]])
                for values, arrays in zip(group_values, scores, strict=True):
                    values.append(_value(metric, arrays, idx))
        for i, values in enumerate(group_values):
            replicates[i, b] = _nanmean(np.asarray(values, dtype=np.float64))
    return replicates


def bootstrap_ci(
    pred: pd.DataFrame,
    metric: str,
    *,
    probabilistic: bool,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a confidence interval for one `group_metrics` column.

    Each replicate, per scenario (docs: protocol): resamples that
    scenario's seeds with replacement, and, per label, draws one multiset of
    scenes with replacement from the union of the scenario's evaluation
    scenes across seeds. Every drawn seed is scored on that same scene
    multiset, carrying along every lighting variant of a drawn scene and
    skipping scenes the seed never evaluated (its dev shot scenes). Sharing
    the scene draw across seeds keeps seed copies from averaging away the
    scene-sampling noise. The metric is recomputed per drawn seed (a `gap_`
    name recomputes the regular-vs-rest gap) and NaN-ignoring averaged over
    every drawn (scenario, seed) to give the replicate's value. Rows are
    sorted first, so the result does not depend on input row order.

    Args:
        pred: Predictions frame (see module docstring).
        metric: Any `group_metrics` column name, e.g. `"auroc"` or
            `"gap_auroc"`.
        probabilistic: Whether `score_balanced` feeds calibration metrics.
        n_boot: Number of bootstrap replicates.
        seed: RNG seed; the same seed reproduces the same output.

    Returns:
        `(point, ci_lo, ci_hi)`: `point` is the statistic on the original
        sample, the NaN-ignoring mean of `metric` over `group_metrics` rows
        (what `summarize` reports as `<metric>`); `ci_lo` and `ci_hi` are the
        bootstrap distribution's 2.5th/97.5th percentiles.

    Raises:
        KeyError: If `metric` is not a `group_metrics` column for this
            `probabilistic` setting.
    """
    _check_metric(metric, probabilistic)
    pred = pred.sort_values(["scenario", "seed", "image_id"], ignore_index=True)
    replicates = _replicates(pred, metric, [_arrays(pred, probabilistic)], n_boot, seed)[0]
    point = _nanmean(group_metrics(pred, probabilistic)[metric].to_numpy(dtype=np.float64))
    if np.all(np.isnan(replicates)):
        return point, float("nan"), float("nan")
    lo, hi = np.nanpercentile(replicates, _CI_PERCENTILES)
    return point, float(lo), float(hi)


_PAIR_KEYS: list[str] = ["scenario", "seed", "image_id"]


def paired_bootstrap_diff(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    metric: str,
    *,
    probabilistic: bool,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Bootstrap the difference of one `group_metrics` column between two runs.

    Both runs must score the same evaluation rows (same scenarios, seeds and
    images, as two classifiers at the same `k` and shot lighting do, since
    shots depend only on seed, scenario, `k` and lighting). Each replicate
    draws seeds and scenes once, as `bootstrap_ci` does, and scores both
    runs on that draw, so the interval reflects the noise in the difference
    rather than in each run alone.

    Args:
        pred_a: Predictions frame of run A (see module docstring).
        pred_b: Predictions frame of run B, over the same rows.
        metric: Any `group_metrics` column name, e.g. `"auroc"`.
        probabilistic: Whether `score_balanced` feeds calibration metrics;
            both frames must carry it when `True`.
        n_boot: Number of bootstrap replicates.
        seed: RNG seed; the same seed reproduces the same output.

    Returns:
        `(point, ci_lo, ci_hi, share_positive)`: `point` is A's metric minus
        B's, each the NaN-ignoring mean over `group_metrics` rows; `ci_lo`
        and `ci_hi` are the 2.5th/97.5th percentiles of the replicate
        differences; `share_positive` is the share of replicates where A
        minus B is above zero.

    Raises:
        KeyError: If `metric` is not a `group_metrics` column for this
            `probabilistic` setting.
        ValueError: If the two runs do not score the same rows with the same
            labels.
    """
    _check_metric(metric, probabilistic)
    a = pred_a.sort_values(_PAIR_KEYS, ignore_index=True)
    b = pred_b.sort_values(_PAIR_KEYS, ignore_index=True)
    if len(a) != len(b) or not a[[*_PAIR_KEYS, "label"]].equals(b[[*_PAIR_KEYS, "label"]]):
        raise ValueError("paired runs must score the same rows with the same labels")
    scores = [_arrays(a, probabilistic), _arrays(b, probabilistic)]
    replicates = _replicates(a, metric, scores, n_boot, seed)
    diffs = replicates[0] - replicates[1]
    point = _nanmean(group_metrics(a, probabilistic)[metric].to_numpy(dtype=np.float64))
    point -= _nanmean(group_metrics(b, probabilistic)[metric].to_numpy(dtype=np.float64))
    valid = diffs[~np.isnan(diffs)]
    if valid.size == 0:
        return point, float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(valid, _CI_PERCENTILES)
    return point, float(lo), float(hi), float(np.mean(valid > 0))
