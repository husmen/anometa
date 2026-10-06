"""Tests for the `anometa lock` subcommand and frozen-config loading."""

from pathlib import Path

import pandas as pd
import pytest

import anometa.cli as cli
from anometa.cli import lock_configs, main
from anometa.config import ExperimentConfig, load_configs, run_id
from anometa.experiment import ExperimentResult, LockAlreadyEvaluatedError

_FROZEN = (
    "- {track: B, encoder: dinov3_s, features: [cls], pca_dim: 4, classifier: logreg, k: 2}\n"
    "- {track: A, model: patchcore}\n"
)


def test_load_configs_reads_a_mapping_or_a_list(tmp_path: Path) -> None:
    """A YAML file with one mapping gives one config; a list gives one config per item."""
    one = tmp_path / "one.yaml"
    one.write_text("track: A\nmodel: patchcore\n")
    many = tmp_path / "many.yaml"
    many.write_text(_FROZEN)
    assert [c.track for c in load_configs(one)] == ["A"]
    assert [c.track for c in load_configs(many)] == ["B", "A"]


def test_lock_configs_forces_the_lock_split(tmp_path: Path) -> None:
    """Every config of every YAML file in the frozen folder is revalidated with split lock."""
    (tmp_path / "a.yaml").write_text(_FROZEN)
    (tmp_path / "b.yaml").write_text("track: A\nmodel: patch_distance\nencoder: dinov3_s\n")
    configs = lock_configs(tmp_path)
    assert len(configs) == 3
    assert {c.split for c in configs} == {"lock"}


def test_lock_runs_each_config_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Frozen configs run on the lock split; a second invocation reruns nothing.

    Both invocations exit 0 and write `artifacts/lock_summary.csv`; the
    second marks every config as already evaluated.
    """
    monkeypatch.chdir(tmp_path)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "a.yaml").write_text(_FROZEN)
    ran: list[ExperimentConfig] = []

    def fake(cfg: ExperimentConfig) -> ExperimentResult:
        """Fake `run_experiment`: runs a config once, then refuses it like a lock run."""
        if cfg in ran:
            raise LockAlreadyEvaluatedError(run_id(cfg))
        ran.append(cfg)
        return ExperimentResult(
            run_id=run_id(cfg),
            config_hash="h",
            status="ok",
            metrics={"auroc": 0.5},
            artifact_dir=tmp_path,
        )

    monkeypatch.setattr(cli, "run_experiment", fake)
    assert main(["lock", str(frozen)]) == 0
    assert {c.split for c in ran} == {"lock"}
    assert len(ran) == 2
    first = pd.read_csv(tmp_path / "artifacts" / "lock_summary.csv")
    assert list(first["status"]) == ["ok", "ok"]
    assert main(["lock", str(frozen)]) == 0
    assert len(ran) == 2
    second = pd.read_csv(tmp_path / "artifacts" / "lock_summary.csv")
    assert list(second["status"]) == ["already evaluated"] * 2


def test_lock_returns_1_when_a_run_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed lock run makes `anometa lock` exit 1, after running every other config."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.yaml").write_text(_FROZEN)
    ran: list[str] = []

    def fake(cfg: ExperimentConfig) -> ExperimentResult:
        """Fake `run_experiment` whose Track B run fails."""
        ran.append(cfg.track)
        return ExperimentResult(
            run_id=run_id(cfg),
            config_hash="h",
            status="failed" if cfg.track == "B" else "ok",
            metrics={},
            artifact_dir=tmp_path,
            error="Traceback\nRuntimeError: boom\n" if cfg.track == "B" else None,
        )

    monkeypatch.setattr(cli, "run_experiment", fake)
    assert main(["lock", str(tmp_path)]) == 1
    assert ran == ["B", "A"]
