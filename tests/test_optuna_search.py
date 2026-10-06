"""Tests for `anometa.search.optuna_search`: NSGA-II and TPE dev-split searches."""

from pathlib import Path

import pandas as pd
import pytest
from optuna.trial import TrialState

from anometa.config import Paths, Scenario
from anometa.experiment import ExperimentResult
from anometa.search import optuna_search
from anometa.search.optuna_search import (
    initial_units,
    objectives,
    run_nsga2,
    run_tpe,
    trials_frame,
)
from anometa.search.space import PARAMS, decode_unit


@pytest.fixture
def fake_runs(monkeypatch, tmp_path):
    """Stub `run_experiment` so `knn` configs fail and everything else succeeds.

    Returns the list every stubbed call appends its config to, so a test can
    assert on the configs a search actually submitted.
    """
    seen = []

    def fake(cfg):
        """Fake `run_experiment`: `knn` fails with no metrics, else succeeds."""
        seen.append(cfg)
        ok = cfg.classifier != "knn"
        m = (
            {
                "auroc": 0.5 + 0.1 * (cfg.classifier == "tabpfn"),
                "ece_bal": 0.1,
                "predict_latency_ms": 5.0,
                "gap_auroc": 0.05,
            }
            if ok
            else {}
        )
        return ExperimentResult(
            run_id=f"r{len(seen)}",
            config_hash="h",
            status="ok" if ok else "failed",
            metrics=m,
            artifact_dir=tmp_path,
        )

    monkeypatch.setattr(optuna_search, "run_experiment", fake)
    return seen


def test_nsga2_records_every_trial(fake_runs, tmp_path):
    """Every trial is stored; knn runs fail as FAIL trials without stopping the study.

    A rerun on the same storage adds no trials: the 8-trial budget is already spent.
    """
    run_nsga2(8, storage=f"sqlite:///{tmp_path}/o.db", population_size=4)
    st = run_nsga2(8, storage=f"sqlite:///{tmp_path}/o.db", population_size=4)
    states = {t.state for t in st.trials}
    assert len(st.trials) == 8
    assert len(st.directions) == 4
    assert states <= {TrialState.COMPLETE, TrialState.FAIL}
    assert all(t.state == TrialState.FAIL for t in st.trials if t.params["classifier"] == "knn")


def test_tpe_starts_from_shared_points(fake_runs, tmp_path, monkeypatch):
    """TPE's first trials are exactly the shared initial points."""
    monkeypatch.chdir(tmp_path)
    st = run_tpe(3, n_init=3, seed=5, storage=f"sqlite:///{tmp_path}/o.db")
    expected = [decode_unit(u) for u in initial_units(3, 5)]
    assert [t.params["encoder"] for t in st.trials] == [e["encoder"] for e in expected]


def test_tpe_rerun_resumes_without_repeating(fake_runs, tmp_path):
    """Rerunning TPE on one storage tops the study up to `n_trials`, never past it.

    The default storage lives under `paths.artifacts`; the shared points are
    enqueued once, so the first `n_init` trials hold `n_init` distinct points,
    and the per-trial parquet holds every trial with the `trials_frame` columns.
    """
    paths = Paths(artifacts=tmp_path / "artifacts")
    run_tpe(2, n_init=3, seed=5, paths=paths)
    st = run_tpe(5, n_init=3, seed=5, paths=paths)
    assert (paths.artifacts / "optuna.db").is_file()
    assert len(st.trials) == 5
    assert len(fake_runs) == 5
    first = [tuple(sorted(t.params.items())) for t in st.trials[:3]]
    assert len(set(first)) == 3
    assert first == [tuple(sorted(decode_unit(u.tolist()).items())) for u in initial_units(3, 5)]
    columns = ["trial", "state", "run_id", "auroc", "best_so_far", *PARAMS]
    assert list(trials_frame(st).columns) == columns
    saved = pd.read_parquet(paths.artifacts / "search" / "tpe-k2-seed5.parquet")
    assert list(saved.columns) == columns
    assert list(saved.trial) == [0, 1, 2, 3, 4]


def test_objectives_minimise_absolute_gap():
    """NSGA-II's fourth objective is |gap_auroc|, so a negative gap counts as its size."""
    result = ExperimentResult(
        run_id="r",
        config_hash="h",
        status="ok",
        metrics={"auroc": 0.8, "ece_bal": 0.1, "predict_latency_ms": 5.0, "gap_auroc": -0.2},
        artifact_dir=Path("."),
    )
    assert objectives(result) == (0.8, 0.1, 5.0, 0.2)


def test_search_never_builds_lock_configs(fake_runs, tmp_path):
    """Search only ever evaluates dev configurations."""
    run_nsga2(4, storage=f"sqlite:///{tmp_path}/o.db", population_size=4)
    assert {c.split for c in fake_runs} == {"dev"}
    assert {c.name for c in fake_runs} == {"search"}


def test_search_restricts_scenarios(fake_runs, tmp_path):
    """Given scenarios reach every trial's config and get their own study name."""
    st = run_nsga2(
        4, storage=f"sqlite:///{tmp_path}/o.db", population_size=4, scenarios=(Scenario.VIAL,)
    )
    tpe = run_tpe(2, n_init=2, paths=Paths(artifacts=tmp_path / "a"), scenarios=(Scenario.VIAL,))
    assert {c.scenarios for c in fake_runs} == {(Scenario.VIAL,)}
    assert (st.study_name, tpe.study_name) == ("nsga2-k2-seed0-vial", "tpe-k2-seed0-vial")
    assert (tmp_path / "a" / "search" / "tpe-k2-seed0-vial.parquet").is_file()
