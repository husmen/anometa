"""Tests for `anometa.trackb.classifiers`: `ThinkingScorer`, `thinking_cost` and its cache key."""

import numpy as np
import pytest

from anometa.trackb.classifiers import ThinkingScorer, make_scorer, thinking_cost


def test_thinking_uses_cache_without_api(tmp_path, monkeypatch):
    """A cached prediction is returned without constructing an API client."""
    f = tmp_path / "h/vial-0.npz"
    f.parent.mkdir()
    np.savez(f, p=np.array([0.2, 0.8]))
    monkeypatch.setattr("tabpfn_client.TabPFNClassifier", None)  # any use would raise
    sc = make_scorer(
        "tabpfn_thinking", {}, seed=0, device="cpu", cache_key="h/vial-0", cache_dir=tmp_path
    )
    x_fit = np.zeros((3, 2), dtype=np.float32)
    x_eval = np.zeros((2, 2), dtype=np.float32)
    assert np.allclose(sc.fit(x_fit, np.array([0, 0, 1])).anomaly_score(x_eval), [0.2, 0.8])


def test_thinking_cache_length_mismatch_raises(tmp_path):
    """A cache that does not match the evaluation rows is rejected."""
    f = tmp_path / "k.npz"
    np.savez(f, p=np.array([0.5]))
    x_fit = np.zeros((2, 2), dtype=np.float32)
    x_eval = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="rows"):
        ThinkingScorer(0, f).fit(x_fit, np.array([0, 1])).anomaly_score(x_eval)


def test_thinking_writes_cache_after_call(tmp_path, monkeypatch):
    """A live call stores its probabilities for later reruns."""

    class FakeClient:
        """Stand-in for `tabpfn_client.TabPFNClassifier`: fixed class order, constant proba."""

        classes_ = np.array([0, 1])

        @classmethod
        def create_default_for_version(cls, *a, **kw):
            """Return a fresh fake client, ignoring every constructor argument."""
            return cls()

        def fit(self, X, y):
            """Do nothing; the fake never needs real fit rows."""
            return self

        def predict_proba(self, X):
            """Return a fixed 0.3/0.7 split for every row."""
            return np.tile([0.3, 0.7], (len(X), 1))

    monkeypatch.setattr("tabpfn_client.TabPFNClassifier", FakeClient)
    f = tmp_path / "k.npz"
    x_fit = np.zeros((2, 2), dtype=np.float32)
    x_eval = np.zeros((4, 2), dtype=np.float32)
    ThinkingScorer(0, f).fit(x_fit, np.array([0, 1])).anomaly_score(x_eval)
    assert np.allclose(np.load(f)["p"], 0.7)


def test_thinking_cost_sums_fit_and_predict(monkeypatch):
    """`thinking_cost` sums the fit and predict token estimates, both at medium effort."""

    class FakeEstimate:
        """Stand-in for `tabpfn_client`'s `EstimateCostResponse`: only `estimated_cost` is read."""

        def __init__(self, estimated_cost):
            """Store the fixed cost this fake estimate reports."""
            self.estimated_cost = estimated_cost

    calls = []

    def fake_estimate_cost(x_train, x_test, *, operation, thinking_effort):
        """Record which operation was estimated and return a distinct fixed cost."""
        calls.append(operation)
        assert thinking_effort == "medium"
        return FakeEstimate(10_000 if operation == "thinking_fit" else 15_000)

    monkeypatch.setattr("tabpfn_client.estimate_cost", fake_estimate_cost)
    x_fit = np.zeros((3, 2), dtype=np.float32)
    x_eval = np.zeros((2, 2), dtype=np.float32)
    assert thinking_cost(x_fit, x_eval) == 25_000
    assert calls == ["thinking_fit", "thinking_predict"]
