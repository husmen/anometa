"""Tests for the `anometa` CLI entry point."""

from pathlib import Path

import pytest

import anometa.cli as cli
from anometa.cli import main


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
