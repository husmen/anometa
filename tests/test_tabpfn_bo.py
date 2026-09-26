"""Tests for `anometa.search.tabpfn_bo`: the TabPFN-surrogate Bayesian optimization loop."""

import numpy as np
import pytest
import torch

from anometa.config import Paths, Scenario, resolve_device
from anometa.experiment import ExperimentResult
from anometa.search import tabpfn_bo
from anometa.search.optuna_search import initial_units
from anometa.search.tabpfn_bo import run_tabpfn_bo


def test_bo_loop_with_fake_proposer(monkeypatch, tmp_path):
    """The loop starts from the shared points, records failures as 0.5 and tracks the best."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tabpfn_bo, "propose_next_point", lambda *a, **kw: torch.full((7,), 0.5))

    def fake(cfg):
        """Fake `run_experiment`: `knn` configs fail with no metrics, else succeed at 0.7."""
        failed = cfg.classifier == "knn"
        return ExperimentResult(
            run_id="r",
            config_hash="h",
            status="failed" if failed else "ok",
            metrics={} if failed else {"auroc": 0.7},
            artifact_dir=tmp_path,
        )

    monkeypatch.setattr(tabpfn_bo, "run_experiment", fake)
    df = run_tabpfn_bo(5, n_init=3, seed=1, device="cpu")
    assert len(df) == 5
    assert np.allclose(np.stack(list(df.u[:3])), initial_units(3, 1))
    assert (df.loc[df.status == "failed", "auroc"] == 0.5).all()
    assert df.best_so_far.is_monotonic_increasing


def test_bo_resumes_without_repeating_trials(monkeypatch, tmp_path):
    """An interrupted BO run resumes from its per-trial parquet at the next trial.

    The fourth run raises `KeyboardInterrupt`; the parquet already holds the
    three finished trials, so the rerun only runs trials 3 and 4, and each
    proposal is seeded with `seed * 1000 + trial`.
    """
    calls = []

    def fake(cfg):
        """Fake `run_experiment`: interrupted on the 4th call, else ok at AUROC 0.6."""
        calls.append(cfg)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return ExperimentResult(
            run_id=f"r{len(calls)}",
            config_hash="h",
            status="ok",
            metrics={"auroc": 0.6},
            artifact_dir=tmp_path,
        )

    proposals = []

    def fake_propose(reg, train_x, train_y, **kw):
        """Record the training-set size and torch seed of each proposal."""
        proposals.append((len(train_x), torch.initial_seed()))
        return torch.full((7,), 0.5)

    monkeypatch.setattr(tabpfn_bo, "run_experiment", fake)
    monkeypatch.setattr(tabpfn_bo, "propose_next_point", fake_propose)
    paths = Paths(artifacts=tmp_path / "artifacts")
    with pytest.raises(KeyboardInterrupt):
        run_tabpfn_bo(5, n_init=2, seed=1, device="cpu", paths=paths)
    df = run_tabpfn_bo(5, n_init=2, seed=1, device="cpu", paths=paths)
    assert len(calls) == 6
    assert list(df.trial) == [0, 1, 2, 3, 4]
    assert list(df.run_id) == ["r1", "r2", "r3", "r5", "r6"]
    assert proposals == [(2, 1002), (3, 1003), (3, 1003), (4, 1004)]
    assert np.allclose(np.stack(list(df.u[:2])), initial_units(2, 1))


def test_bo_surrogate_sees_chance_for_failed_or_missing_auroc(monkeypatch, tmp_path):
    """A failed run, a missing AUROC and a NaN AUROC all reach the surrogate as 0.5."""

    def result(status, metrics):
        """Build one scripted run outcome."""
        return ExperimentResult(
            run_id="r", config_hash="h", status=status, metrics=metrics, artifact_dir=tmp_path
        )

    results = iter(
        [
            result("failed", {}),
            result("ok", {}),
            result("ok", {"auroc": float("nan")}),
            result("ok", {"auroc": 0.9}),
        ]
    )

    targets = []

    def fake_propose(reg, train_x, train_y, **kw):
        """Record the surrogate's training targets."""
        targets.append(train_y.tolist())
        return torch.full((7,), 0.5)

    monkeypatch.setattr(tabpfn_bo, "run_experiment", lambda cfg: next(results))
    monkeypatch.setattr(tabpfn_bo, "propose_next_point", fake_propose)
    run_tabpfn_bo(4, n_init=3, seed=0, device="cpu", paths=Paths(artifacts=tmp_path))
    assert targets == [[0.5, 0.5, 0.5]]


def test_bo_restricts_scenarios(monkeypatch, tmp_path):
    """Given scenarios reach every trial's config and get their own parquet file."""
    seen = []

    def fake(cfg):
        """Record the config; succeed at AUROC 0.6."""
        seen.append(cfg)
        return ExperimentResult(
            run_id="r", config_hash="h", status="ok", metrics={"auroc": 0.6}, artifact_dir=tmp_path
        )

    monkeypatch.setattr(tabpfn_bo, "run_experiment", fake)
    monkeypatch.setattr(tabpfn_bo, "propose_next_point", lambda *a, **kw: torch.full((7,), 0.5))
    paths = Paths(artifacts=tmp_path)
    scenarios = (Scenario.VIAL, Scenario.CAN)
    run_tabpfn_bo(3, n_init=2, device="cpu", paths=paths, scenarios=scenarios)
    assert {c.scenarios for c in seen} == {scenarios}
    assert (tmp_path / "search" / "bo-k2-seed0-vial+can.parquet").is_file()


@pytest.mark.models
def test_propose_next_point_stays_in_cube():
    """The real proposer returns a point in the unit cube on the host's best device."""
    from tabpfn import TabPFNRegressor
    from tabpfn_extensions.bayesian_optimization import propose_next_point

    device = resolve_device("auto")
    torch.manual_seed(0)
    x = torch.rand(10, 7, device=device)
    y = x.sum(1)
    reg = TabPFNRegressor(
        n_estimators=1,
        device=device,
        random_state=0,
        inference_precision=torch.float32,
        differentiable_input=True,
    )
    p = propose_next_point(reg, x, y)
    assert p.shape == (7,)
    assert ((p >= 0) & (p <= 1)).all()
