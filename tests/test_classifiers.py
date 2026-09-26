"""Tests for `anometa.trackb.classifiers`: Track B scorers and one-class controls."""

import re

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from anometa.config import ClassifierName
from anometa.trackb.classifiers import _model_specs, make_scorer, tabpfn_revision


def blobs(
    n0: int = 60, n1: int = 6, d: int = 5, seed: int = 0
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Draw two well-separated Gaussian blobs: `n0` normals, `n1` anomalies.

    `X` is cast to `float32`, matching the extracted-feature dtype every
    real `Scorer` fits on.

    Args:
        n0: Number of normal (label 0) rows, centred at 0.
        n1: Number of anomalous (label 1) rows, centred at 4.
        d: Number of features.
        seed: Seed for the random generator.

    Returns:
        `(X, y)`: stacked rows and their 0/1 labels.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n0, d)), rng.normal(4, 1, (n1, d))]).astype(np.float32)
    return X, np.r_[np.zeros(n0), np.ones(n1)].astype(int)


@pytest.mark.parametrize("name", ["logreg", "knn"])
def test_sklearn_scorers_rank_separable_data(name: ClassifierName) -> None:
    """Controls separate well-separated blobs and return probabilities."""
    X, y = blobs()
    Xt, yt = blobs(seed=1)
    s = make_scorer(name, {}, seed=0, device="cpu").fit(X, y).anomaly_score(Xt)
    assert s.shape == (len(yt),)
    assert ((s >= 0) & (s <= 1)).all()
    assert roc_auc_score(yt, s) > 0.95


def test_knn_caps_neighbours() -> None:
    """n_neighbors above the fit size is capped instead of failing."""
    X, y = blobs(n0=4, n1=2)
    make_scorer("knn", {"n_neighbors": 15}, seed=0, device="cpu").fit(X, y).anomaly_score(X)


def test_mahalanobis_ignores_anomalous_fit_rows() -> None:
    """One-class scoring uses normals only and ranks outliers higher."""
    X, y = blobs()
    Xt, yt = blobs(seed=1)
    sc = make_scorer("mahalanobis", {}, seed=0, device="cpu").fit(X, y)
    assert sc.probabilistic is False
    assert roc_auc_score(yt, sc.anomaly_score(Xt)) > 0.95


def test_make_scorer_rejects_unknown_name() -> None:
    """An unimplemented classifier name (e.g. Thinking) raises KeyError."""
    with pytest.raises(KeyError):
        make_scorer("tabpfn_thinking", {}, seed=0, device="cpu")


@pytest.mark.models
@pytest.mark.parametrize("name", ["tabpfn", "tabpfn_fast"])
def test_tabpfn_scorers(name: ClassifierName) -> None:
    """TabPFN-3.5 and 3.5-Fast fit a small binary task on CPU."""
    X, y = blobs()
    s = make_scorer(name, {}, seed=0, device="cpu").fit(X, y).anomaly_score(X)
    assert ((s >= 0) & (s <= 1)).all()
    assert re.fullmatch(r"[0-9a-f]{64}", tabpfn_revision("v3.5")["sha256"])


@pytest.mark.models
def test_tabpfn_outlier_scorer_small() -> None:
    """The TabPFN outlier score ranks blob outliers above normals."""
    X, y = blobs(d=3)
    Xt, yt = blobs(d=3, seed=1)
    sc = make_scorer("tabpfn_outlier", {"n_permutations": 2}, seed=0, device="cpu").fit(X, y)
    assert roc_auc_score(yt, sc.anomaly_score(Xt)) > 0.9


@pytest.mark.models
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("n_estimators", ["auto", 4])
def test_model_specs_match_fresh_checkpoint_load(fast: bool, n_estimators: int | str) -> None:
    """A classifier reusing cached model specs predicts exactly what a fresh load does."""
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = ModelVersion.V3_5_FAST if fast else ModelVersion.V3_5
    rng = np.random.default_rng(0)
    x, x_eval = rng.normal(size=(60, 6)), rng.normal(size=(20, 6))
    y = np.r_[np.zeros(55, dtype=np.int64), np.ones(5, dtype=np.int64)]
    specs = _model_specs(version, "classifier", device)
    for seed in (0, 1):
        kw: dict[str, object] = {
            "device": device,
            "random_state": seed,
            "n_estimators": n_estimators,
        }
        fresh = TabPFNClassifier.create_default_for_version(version, **kw).fit(x, y)
        reused = TabPFNClassifier.create_default_for_version(version, model_path=specs, **kw).fit(
            x, y
        )
        np.testing.assert_array_equal(reused.predict_proba(x_eval), fresh.predict_proba(x_eval))


@pytest.mark.models
def test_regressor_model_specs_match_fresh_checkpoint_load() -> None:
    """The `tabpfn_outlier` regressor on cached specs predicts exactly what a fresh load does."""
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    x, x_eval = rng.normal(size=(60, 4)), rng.normal(size=(20, 4))
    y = x @ np.array([1.0, -0.5, 0.2, 0.0]) + rng.normal(scale=0.1, size=60)
    specs = _model_specs(ModelVersion.V3_5, "regressor", device)
    kw: dict[str, object] = {"device": device, "random_state": 0}
    fresh = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, **kw).fit(x, y)
    reused = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5, model_path=specs, **kw
    ).fit(x, y)
    np.testing.assert_array_equal(reused.predict(x_eval), fresh.predict(x_eval))
