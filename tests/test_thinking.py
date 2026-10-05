"""Tests for `anometa.trackb.classifiers`: `ThinkingScorer`, `thinking_cost` and its cache key."""

import re
import sys
import types

import numpy as np
import pytest
from conftest import cfg_b

from anometa.trackb.classifiers import ThinkingScorer, make_scorer, thinking_cost
from anometa.trackb.pipeline import run_track_b


@pytest.fixture
def fake_client(monkeypatch):
    """Install a fake `tabpfn_client` module, so the real client is never imported.

    Its `TabPFNClassifier` predicts P(y == 1) = 0.7 for every row and logs the
    size of every fit in the module's `fits` list.
    """
    module = types.ModuleType("tabpfn_client")
    fits = []

    class FakeClient:
        """Stand-in for `tabpfn_client.TabPFNClassifier`: fixed class order, constant proba."""

        classes_ = np.array([0, 1])

        @classmethod
        def create_default_for_version(cls, *a, **kw):
            """Return a fresh fake client, ignoring every constructor argument."""
            return cls()

        def fit(self, X, y):
            """Log the number of fit rows; nothing is uploaded."""
            fits.append(len(X))
            return self

        def predict_proba(self, X):
            """Return a fixed 0.3/0.7 split for every row."""
            return np.tile([0.3, 0.7], (len(X), 1))

    monkeypatch.setattr(module, "TabPFNClassifier", FakeClient, raising=False)
    monkeypatch.setattr(module, "fits", fits, raising=False)
    monkeypatch.setitem(sys.modules, "tabpfn_client", module)
    return module


def test_thinking_uses_cache_without_api(tmp_path, fake_client):
    """A cached prediction is returned without fitting an API client."""
    f = tmp_path / "h/vial-0.npz"
    f.parent.mkdir()
    np.savez(f, p=np.array([0.2, 0.8]))
    sc = make_scorer(
        "tabpfn_thinking", {}, seed=0, device="cpu", cache_key="h/vial-0", cache_dir=tmp_path
    )
    x_fit = np.zeros((3, 2), dtype=np.float32)
    x_eval = np.zeros((2, 2), dtype=np.float32)
    assert np.allclose(sc.fit(x_fit, np.array([0, 0, 1])).anomaly_score(x_eval), [0.2, 0.8])
    assert fake_client.fits == []


def test_thinking_cache_length_mismatch_raises(tmp_path):
    """A cache that does not match the evaluation rows is rejected."""
    f = tmp_path / "k.npz"
    np.savez(f, p=np.array([0.5]))
    x_fit = np.zeros((2, 2), dtype=np.float32)
    x_eval = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="rows"):
        ThinkingScorer(0, f).fit(x_fit, np.array([0, 1])).anomaly_score(x_eval)


def test_thinking_writes_cache_after_call(tmp_path, fake_client):
    """A live call stores its probabilities for later reruns, leaving no temporary file."""
    f = tmp_path / "k.npz"
    x_fit = np.zeros((2, 2), dtype=np.float32)
    x_eval = np.zeros((4, 2), dtype=np.float32)
    ThinkingScorer(0, f).fit(x_fit, np.array([0, 1])).anomaly_score(x_eval)
    assert np.allclose(np.load(f)["p"], 0.7)
    assert list(tmp_path.iterdir()) == [f]


def test_thinking_cache_layout_shared_across_runs(prepared, tmp_path, fake_client):
    """Predictions land at cache/thinking/<key>/<scenario>-<seed>.npz, shared across runs.

    The key leaves out seeds, scenarios and device, so a second run with
    seeds (0, 1) on another device reads seed 0 from the first run's cache
    and fits the client only for seed 1.
    """
    run_track_b(cfg_b(prepared, classifier="tabpfn_thinking", seeds=(0,)), tmp_path)
    assert len(fake_client.fits) == 1
    run_track_b(cfg_b(prepared, classifier="tabpfn_thinking", seeds=(0, 1), device="cpu"), tmp_path)
    assert len(fake_client.fits) == 2
    (key_dir,) = (prepared.cache / "thinking").iterdir()
    assert re.fullmatch(r"[0-9a-f]{64}", key_dir.name)
    assert sorted(p.name for p in key_dir.iterdir()) == ["vial-0.npz", "vial-1.npz"]


def test_thinking_cost_sums_fit_and_predict(monkeypatch, fake_client):
    """`thinking_cost` sums the fit (medium effort) and predict token estimates."""

    class FakeEstimate:
        """Stand-in for `tabpfn_client`'s `EstimateCostResponse`: only `estimated_cost` is read."""

        def __init__(self, estimated_cost):
            """Store the fixed cost this fake estimate reports."""
            self.estimated_cost = estimated_cost

    calls = []

    def fake_estimate_cost(x_train, x_test, *, operation, thinking_effort=None):
        """Record which operation was estimated and return a distinct fixed cost.

        Mirrors the real client and server: `thinking_fit` takes no test rows
        and needs `thinking_effort`; `thinking_predict` rejects `thinking_effort`.
        """
        calls.append(operation)
        if operation == "thinking_fit":
            if x_test is not None:
                raise ValueError("thinking_fit does not use X_test")
            assert thinking_effort == "medium"
        elif thinking_effort is not None:
            raise ValueError("thinking_effort is only valid for thinking_fit")
        return FakeEstimate(10_000 if operation == "thinking_fit" else 15_000)

    monkeypatch.setattr(fake_client, "estimate_cost", fake_estimate_cost, raising=False)
    x_fit = np.zeros((3, 2), dtype=np.float32)
    x_eval = np.zeros((2, 2), dtype=np.float32)
    assert thinking_cost(x_fit, x_eval) == 25_000
    assert calls == ["thinking_fit", "thinking_predict"]


def test_thinking_retries_after_rate_limit(tmp_path, fake_client, monkeypatch):
    """A rate-limited fit is retried after the named wait; other errors and daily limits are not."""
    import anometa.trackb.classifiers as classifiers

    slept: list[float] = []
    monkeypatch.setattr(classifiers, "_sleep", slept.append)
    real_fit = fake_client.TabPFNClassifier.fit
    failures = ["Fail to call fit: [HTTP 429] Rate limit exceeded. Retry in 33s.."]

    def flaky_fit(self, X, y):
        """Fail once with the API's rate-limit message, then fit normally."""
        if failures:
            raise RuntimeError(failures.pop())
        return real_fit(self, X, y)

    monkeypatch.setattr(fake_client.TabPFNClassifier, "fit", flaky_fit)
    f = tmp_path / "k.npz"
    x = np.zeros((2, 2), dtype=np.float32)
    ThinkingScorer(0, f).fit(x, np.array([0, 1])).anomaly_score(x)
    assert slept == [34]
    assert f.exists()

    failures.append("Fail to call fit: [HTTP 500] boom")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        ThinkingScorer(0, tmp_path / "other.npz").fit(x, np.array([0, 1])).anomaly_score(x)

    failures.append("Fail to call fit: [HTTP 429] Daily usage limit reached.")
    with pytest.raises(RuntimeError, match="Daily usage limit"):
        ThinkingScorer(0, tmp_path / "third.npz").fit(x, np.array([0, 1])).anomaly_score(x)
    assert slept == [34]
