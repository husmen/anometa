"""Image-level detection metrics and prior correction.

Every score-based metric takes 1-D arrays: `y` holds ground-truth labels in
`{0, 1}` (1 = anomalous), and `s`/`p`/`score` hold per-image scores. Ranking
metrics (`auroc`, `auprc`) and `image_metrics` return NaN rather than raising
when `y` has only one class, since small MVTec AD 2 lighting subsets can be
single-class (see PLAN_1 § Review focus).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

# _Labels arrive as either dtype across call sites (e.g. np.ones(5) vs. rng.integers(...)).
_Labels = NDArray[np.float64] | NDArray[np.int64]


def auroc(y: _Labels, s: NDArray[np.float64]) -> float:
    """Compute the area under the ROC curve via the Mann-Whitney U statistic.

    Ranking scores with `scipy.stats.rankdata` (average ties) is equivalent
    to sklearn's `roc_auc_score` but fast enough to call inside a bootstrap.

    Args:
        y: Ground-truth labels in `{0, 1}`, shape `(n,)`.
        s: Anomaly scores, shape `(n,)`.

    Returns:
        AUROC, or NaN if `y` has only one class.
    """
    n_pos = int(np.sum(y == 1))
    n_neg = y.shape[0] - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y: _Labels, s: NDArray[np.float64]) -> float:
    """Compute the area under the precision-recall curve.

    Args:
        y: Ground-truth labels in `{0, 1}`, shape `(n,)`.
        s: Anomaly scores, shape `(n,)`.

    Returns:
        Average precision, or NaN if `y` has only one class.
    """
    if np.unique(y).shape[0] < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def nll(y: _Labels, p: NDArray[np.float64], eps: float = 1e-7) -> float:
    """Compute the mean binary negative log-likelihood.

    Args:
        y: Ground-truth labels in `{0, 1}`, shape `(n,)`.
        p: Predicted probabilities of the anomalous class, shape `(n,)`.
        eps: Clip `p` to `[eps, 1 - eps]` before taking logs.

    Returns:
        Mean negative log-likelihood.
    """
    p_clipped = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))


def brier(y: _Labels, p: NDArray[np.float64]) -> float:
    """Compute the Brier score (mean squared probability error).

    Args:
        y: Ground-truth labels in `{0, 1}`, shape `(n,)`.
        p: Predicted probabilities of the anomalous class, shape `(n,)`.

    Returns:
        Mean squared error between `p` and `y`.
    """
    return float(np.mean((p - y) ** 2))


def ece(y: _Labels, p: NDArray[np.float64], n_bins: int = 10) -> float:
    """Compute the expected calibration error.

    Predictions are split into `n_bins` equal-width bins over `[0, 1]`; each
    bin contributes its size-weighted gap between mean label and mean
    prediction.

    Args:
        y: Ground-truth labels in `{0, 1}`, shape `(n,)`.
        p: Predicted probabilities of the anomalous class, shape `(n,)`.
        n_bins: Number of equal-width probability bins.

    Returns:
        Expected calibration error.
    """
    bin_idx = np.minimum(np.floor(p * n_bins).astype(np.int64), n_bins - 1)
    n = y.shape[0]
    total = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        total += mask.sum() / n * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def prior_correct(p: NDArray[np.float64], pi: float) -> NDArray[np.float64]:
    """Rescale probabilities fit under training prior `pi` to a 50/50 prior.

    Computes `(p/pi) / (p/pi + (1 - p)/(1 - pi))` (PLAN_1 § Track B), so a
    probability equal to `pi` maps to 0.5.

    Args:
        p: Predicted anomalous-class probabilities, shape `(n,)`.
        pi: Anomalous-class prior of the training set, e.g.
            `n_anomalous / n_train`.

    Returns:
        Balanced-prior probabilities, shape `(n,)`.

    Raises:
        ValueError: If `pi` is not strictly inside `(0, 1)`.
    """
    if not 0 < pi < 1:
        raise ValueError(f"pi must be strictly inside (0, 1), got {pi}")
    odds = p / pi
    odds_complement = (1 - p) / (1 - pi)
    result: NDArray[np.float64] = odds / (odds + odds_complement)
    return result


def image_metrics(
    y: _Labels,
    score: NDArray[np.float64],
    score_balanced: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Compute the standard image-level metric set for one scenario/subset.

    Always reports ranking metrics (`auroc`, `auprc`). When `score_balanced`
    is given (a probability under a balanced training prior), also reports
    calibration metrics on `score` (`nll`, `ece`, `brier`) and on
    `score_balanced` (`nll_bal`, `ece_bal`, `brier_bal`).

    Args:
        y: Ground-truth labels in `{0, 1}`, shape `(n,)`.
        score: Anomaly scores, shape `(n,)`.
        score_balanced: Optional balanced-prior probabilities, shape `(n,)`.

    Returns:
        A dict of the requested metric names to values; every value is NaN
        if `y` has only one class.
    """
    keys = ["auroc", "auprc"]
    if score_balanced is not None:
        keys += ["nll", "ece", "brier", "nll_bal", "ece_bal", "brier_bal"]
    if np.unique(y).shape[0] < 2:
        return dict.fromkeys(keys, float("nan"))
    metrics: dict[str, float] = {"auroc": auroc(y, score), "auprc": auprc(y, score)}
    if score_balanced is not None:
        metrics["nll"] = nll(y, score)
        metrics["ece"] = ece(y, score)
        metrics["brier"] = brier(y, score)
        metrics["nll_bal"] = nll(y, score_balanced)
        metrics["ece_bal"] = ece(y, score_balanced)
        metrics["brier_bal"] = brier(y, score_balanced)
    return metrics
