"""Tests for the `anometa` CLI entry point."""

import pytest

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
