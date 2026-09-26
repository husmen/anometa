"""Tests for the `anometa` CLI entry point."""

from pathlib import Path

import pytest

import anometa.cli as cli
from anometa.cli import main
from anometa.config import ExperimentConfig, TrackBConfig
from anometa.experiment import ExperimentResult

_DEV_CONFIG = "track: B\nencoder: dinov3_s\nfeatures: [cls]\npca_dim: 4\nclassifier: logreg\nk: 1\n"


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """`anometa --help` exits 0 and names the program."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "anometa" in capsys.readouterr().out


def test_cli_no_command_prints_help_and_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    """Running with no subcommand prints help and returns exit code 1."""
    exit_code = main([])
    assert exit_code == 1
    assert "anometa" in capsys.readouterr().out


def test_cli_run_refuses_lock_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`anometa run` on a `split: lock` config exits 2, points to `anometa lock`, runs nothing."""

    def never_run(cfg: object) -> None:
        """Fake `run_experiment` that must not be reached."""
        raise AssertionError("lock config was run")

    monkeypatch.setattr(cli, "run_experiment", never_run)
    config = tmp_path / "lock.yaml"
    config.write_text(
        "track: B\nsplit: lock\nencoder: dinov3_s\nfeatures: [cls]\n"
        "pca_dim: 4\nclassifier: logreg\nk: 1\n"
    )
    assert main(["run", str(config)]) == 2
    assert "anometa lock" in capsys.readouterr().err


def test_cli_run_set_takes_several_pairs_per_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--set a=b c=d` and a repeated `--set` flag both apply every pair."""
    seen: list[ExperimentConfig] = []

    def fake_run(cfg: ExperimentConfig) -> ExperimentResult:
        """Record the config instead of running it."""
        seen.append(cfg)
        return ExperimentResult(
            run_id="r", config_hash="h", status="ok", metrics={}, artifact_dir=tmp_path
        )

    monkeypatch.setattr(cli, "run_experiment", fake_run)
    config = tmp_path / "dev.yaml"
    config.write_text(_DEV_CONFIG)
    argv = ["run", str(config), "--set", "k=2", "classifier=knn", "--set", "seeds=[0, 1]"]
    assert main(argv) == 0
    cfg = seen[0]
    assert isinstance(cfg, TrackBConfig)
    assert (cfg.k, cfg.classifier, cfg.seeds) == (2, "knn", (0, 1))


def test_cli_run_failure_prints_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed run exits 1 and prints its traceback's last line to stderr."""
    error = "Traceback (most recent call last):\n  ...\nBudgetError: vial: k=9 too many\n"

    def fake_run(cfg: ExperimentConfig) -> ExperimentResult:
        """Return a failed result without running anything."""
        return ExperimentResult(
            run_id="r",
            config_hash="h",
            status="failed",
            metrics={},
            artifact_dir=tmp_path,
            error=error,
        )

    monkeypatch.setattr(cli, "run_experiment", fake_run)
    config = tmp_path / "dev.yaml"
    config.write_text(_DEV_CONFIG)
    assert main(["run", str(config)]) == 1
    assert capsys.readouterr().err == "BudgetError: vial: k=9 too many\n"
