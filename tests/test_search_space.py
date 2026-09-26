"""Tests for `anometa.search`: unit-cube encoding and the committed grid."""

from pathlib import Path

import pytest
import yaml

from anometa.config import Scenario, run_id
from anometa.experiment import ExperimentResult
from anometa.search import grid
from anometa.search.grid import GridSpec, grid_configs, run_grid
from anometa.search.space import decode_unit, to_config


def test_decode_unit_bounds():
    """The cube's corners decode to the first and last choices and the bounds."""
    lo, hi = decode_unit([0.0] * 7), decode_unit([1.0] * 7)
    assert lo == {
        "encoder": "dinov3_s",
        "features": "cls+mean_patch",
        "pca_dim": 16,
        "classifier": "tabpfn",
        "n_estimators": 4,
        "C": pytest.approx(1e-3),
        "n_neighbors": 1,
    }
    assert hi["encoder"] == "siglip2"
    assert hi["C"] == pytest.approx(1e2)
    assert hi["n_neighbors"] == 15


def test_decode_unit_clamps_outside_cube():
    """Coordinates outside [0, 1] decode like the nearest cube face, never past a bound."""
    assert decode_unit([-0.5] * 7) == decode_unit([0.0] * 7)
    assert decode_unit([1.5] * 7) == decode_unit([1.0] * 7)


def test_to_config_is_dev_with_relevant_params():
    """Search configs are dev-only and carry only their classifier's hyperparameter."""
    p = decode_unit([0.9, 0.9, 0.0, 0.6, 0.0, 0.5, 0.5])  # siglip2, novelty, logreg
    cfg = to_config(p, k=2)
    assert cfg.split == "dev"
    assert cfg.pca_dim is None
    assert set(cfg.classifier_params) == {"C"}


def test_to_config_restricts_scenarios():
    """Given scenarios reach the config; without them it keeps every scenario."""
    p = decode_unit([0.0] * 7)
    assert to_config(p, k=2, scenarios=(Scenario.VIAL,)).scenarios == (Scenario.VIAL,)
    assert to_config(p, k=2).scenarios == tuple(Scenario)


def test_grid_size_and_uniqueness():
    """The committed grid expands to 366 unique dev configs."""
    spec = GridSpec(**yaml.safe_load(Path("configs/search/grid.yaml").read_text()))
    cfgs = grid_configs(spec)
    assert len(cfgs) == 288 + 36 + 24 + 12 + 6
    assert len({run_id(c) for c in cfgs}) == len(cfgs)
    assert {c.split for c in cfgs} == {"dev"}


def test_grid_configs_restrict_scenarios():
    """Every grid cell, few-shot and one-class, carries the given scenarios."""
    spec = GridSpec(**yaml.safe_load(Path("configs/search/grid.yaml").read_text()))
    cfgs = grid_configs(spec, scenarios=(Scenario.VIAL, Scenario.CAN))
    assert len(cfgs) == 366
    assert {c.scenarios for c in cfgs} == {(Scenario.VIAL, Scenario.CAN)}


def test_run_grid_reports_each_run_and_errors(monkeypatch, capsys, tmp_path):
    """Each run prints a flushed progress line; a raising config becomes an "error" row.

    The raising run's traceback goes to stderr and the grid continues past it.
    """
    spec = GridSpec(**yaml.safe_load(Path("configs/search/grid.yaml").read_text()))
    cfgs = grid_configs(spec)[:2]

    def fake(cfg):
        """Fake `run_experiment`: the first config raises, the second succeeds."""
        if cfg is cfgs[0]:
            raise RuntimeError("boom")
        return ExperimentResult(
            run_id="ok-run",
            config_hash="h",
            status="ok",
            metrics={"auroc": 0.75},
            artifact_dir=tmp_path,
        )

    monkeypatch.setattr(grid, "run_experiment", fake)
    df = run_grid(cfgs)
    out, err = capsys.readouterr()
    assert list(df.status) == ["error", "ok"]
    assert out.splitlines() == [
        f"[1/2] {run_id(cfgs[0])} error auroc=nan",
        "[2/2] ok-run ok auroc=0.75",
    ]
    assert "RuntimeError: boom" in err
