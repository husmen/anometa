"""Tests for `anometa.experiment`: `run_experiment` and its run artifacts."""

import hashlib
import json
import re

import pandas as pd
import pytest

from anometa.artifacts import TrackOutput, dataset_hash, read_manifest
from anometa.config import Paths, Scenario, TrackBConfig, run_id
from anometa.data.splits import split_hash
from anometa.experiment import RUNNERS, LockAlreadyEvaluatedError, run_experiment

MANIFEST_KEYS = {
    "run_id",
    "config",
    "config_hash",
    "status",
    "error",
    "git_commit",
    "git_dirty",
    "dataset_hash",
    "split_hash",
    "model_revisions",
    "licences",
    "seeds",
    "hardware",
    "versions",
    "started_at",
    "duration_s",
}


def fake_output() -> TrackOutput:
    """Build a minimal successful `TrackOutput` for a fake Track B runner."""
    preds = pd.DataFrame({"scenario": ["vial"] * 2, "image_id": ["a", "b"], "score": [0.1, 0.9]})
    return TrackOutput(metrics={"auroc": 0.9}, predictions=preds, model_revisions={}, licences={})


@pytest.fixture
def cfg(paths: Paths) -> TrackBConfig:
    """Build a minimal valid Track B config scoped to the fake Vial data."""
    return TrackBConfig(
        encoder="dinov3_s",
        features=("cls",),
        pca_dim=4,
        classifier="logreg",
        k=1,
        scenarios=(Scenario.VIAL,),
        seeds=(0,),
        paths=paths,
    )


def test_run_experiment_writes_artifacts(
    cfg: TrackBConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful run writes manifest, metrics, predictions and one log line."""
    monkeypatch.setitem(RUNNERS, "B", lambda c, d: fake_output())
    res = run_experiment(cfg)
    d = res.artifact_dir
    assert res.status == "ok"
    assert res.metrics == {"auroc": 0.9}
    assert {p.name for p in d.iterdir()} >= {"manifest.json", "metrics.json", "predictions.parquet"}
    assert set(json.loads((d / "manifest.json").read_text())) == MANIFEST_KEYS
    assert len((cfg.paths.artifacts / "runs.jsonl").read_text().splitlines()) == 1


def test_failed_run_is_recorded(cfg: TrackBConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """A runner exception becomes a failed result with its traceback on disk."""

    def boom(c: TrackBConfig, d: object) -> TrackOutput:
        """Fake runner that always raises, to exercise the failure path."""
        raise ValueError("boom")

    monkeypatch.setitem(RUNNERS, "B", boom)
    res = run_experiment(cfg)
    assert res.status == "failed"
    assert res.error is not None
    assert "boom" in res.error
    manifest = read_manifest(res.artifact_dir)
    assert manifest is not None
    assert manifest["status"] == "failed"


def test_dev_run_resumes_and_failed_run_retries(
    cfg: TrackBConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ok dev run is read back from disk; a failed one runs again."""
    calls: list[int] = []
    monkeypatch.setitem(RUNNERS, "B", lambda c, d: calls.append(1) or fake_output())
    run_experiment(cfg)
    run_experiment(cfg)
    assert len(calls) == 1

    other = cfg.model_copy(update={"k": 2})
    monkeypatch.setitem(RUNNERS, "B", lambda c, d: (_ for _ in ()).throw(RuntimeError("x")))
    run_experiment(other)
    monkeypatch.setitem(RUNNERS, "B", lambda c, d: calls.append(1) or fake_output())
    assert run_experiment(other).status == "ok"
    assert len(calls) == 2


def test_lock_run_refuses_rerun(cfg: TrackBConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lock run cannot be repeated, whatever the first attempt's status."""
    monkeypatch.setitem(RUNNERS, "B", lambda c, d: fake_output())
    lock = cfg.model_copy(update={"split": "lock"})
    run_experiment(lock)
    with pytest.raises(LockAlreadyEvaluatedError):
        run_experiment(lock)


def test_lock_run_refuses_preexisting_empty_dir(
    cfg: TrackBConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing lock run directory, even empty and manifest-less, refuses the run."""
    calls: list[int] = []
    monkeypatch.setitem(RUNNERS, "B", lambda c, d: calls.append(1) or fake_output())
    lock = cfg.model_copy(update={"split": "lock"})
    (cfg.paths.artifacts / run_id(lock)).mkdir(parents=True)
    with pytest.raises(LockAlreadyEvaluatedError):
        run_experiment(lock)
    assert calls == []


def test_interrupted_lock_run_refuses_rerun(
    cfg: TrackBConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock run interrupted before writing any manifest still cannot be repeated."""

    def interrupt(c: TrackBConfig, d: object) -> TrackOutput:
        """Fake runner simulating Ctrl-C mid-run."""
        raise KeyboardInterrupt

    lock = cfg.model_copy(update={"split": "lock"})
    monkeypatch.setitem(RUNNERS, "B", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(lock)
    assert read_manifest(cfg.paths.artifacts / run_id(lock)) is None

    monkeypatch.setitem(RUNNERS, "B", lambda c, d: fake_output())
    with pytest.raises(LockAlreadyEvaluatedError):
        run_experiment(lock)


def test_renamed_lock_config_refuses_rerun(
    cfg: TrackBConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Renaming a lock config keeps its hash, so it cannot bypass the guard.

    The error names the directory of the run that already evaluated it.
    """
    monkeypatch.setitem(RUNNERS, "B", lambda c, d: fake_output())
    first = run_experiment(cfg.model_copy(update={"split": "lock"}))
    with pytest.raises(LockAlreadyEvaluatedError, match=f"as {re.escape(first.run_id)}$"):
        run_experiment(cfg.model_copy(update={"split": "lock", "name": "renamed"}))


@pytest.mark.parametrize("damage", ["corrupt_manifest", "missing_predictions"])
def test_incomplete_ok_dev_run_reruns(
    cfg: TrackBConfig, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """An ok dev run with an unreadable manifest or a missing file reruns instead of raising."""
    calls: list[int] = []
    monkeypatch.setitem(RUNNERS, "B", lambda c, d: calls.append(1) or fake_output())
    run_dir = run_experiment(cfg).artifact_dir
    if damage == "corrupt_manifest":
        (run_dir / "manifest.json").write_text('{"status": "o')
    else:
        (run_dir / "predictions.parquet").unlink()

    res = run_experiment(cfg)
    assert res.status == "ok"
    assert len(calls) == 2
    assert (run_dir / "predictions.parquet").is_file()
    manifest = read_manifest(run_dir)
    assert manifest is not None
    assert manifest["status"] == "ok"


def test_manifest_written_last(cfg: TrackBConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash while writing predictions leaves no manifest, so the next run redoes it."""

    def fail_parquet(*_args: object, **_kwargs: object) -> None:
        """Stand-in for `DataFrame.to_parquet` that fails mid-write."""
        raise OSError("disk full")

    monkeypatch.setitem(RUNNERS, "B", lambda c, d: fake_output())
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_parquet)
    with pytest.raises(OSError, match="disk full"):
        run_experiment(cfg)
    run_dir = cfg.paths.artifacts / run_id(cfg)
    assert not (run_dir / "manifest.json").exists()

    monkeypatch.undo()
    monkeypatch.setitem(RUNNERS, "B", lambda c, d: fake_output())
    assert run_experiment(cfg).status == "ok"
    assert read_manifest(run_dir) is not None


def test_manifest_provenance_captured_before_runner(
    cfg: TrackBConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Split and dataset hashes reflect the disk state when the run started, not when it ended."""

    def mutating_runner(c: TrackBConfig, d: object) -> TrackOutput:
        """Fake runner that edits the split and pins files while it runs."""
        cfg.paths.splits.mkdir(parents=True, exist_ok=True)
        (cfg.paths.splits / "vial.csv").write_text("changed mid-run")
        pins = cfg.paths.configs / "data" / "sha256sums.txt"
        pins.parent.mkdir(parents=True, exist_ok=True)
        pins.write_text(f"{'a' * 64}  vial.tar.gz\n")
        return fake_output()

    before = dataset_hash(cfg.scenarios, cfg.paths)
    monkeypatch.setitem(RUNNERS, "B", mutating_runner)
    manifest = read_manifest(run_experiment(cfg).artifact_dir)
    assert manifest is not None
    assert manifest["dataset_hash"] == before
    assert manifest["dataset_hash"] != dataset_hash(cfg.scenarios, cfg.paths)
    assert manifest["split_hash"] != split_hash(cfg.scenarios, cfg.paths)


def test_dataset_hash_without_pins_file(paths: Paths) -> None:
    """A missing pins file hashes every scenario as "missing" instead of raising."""
    assert dataset_hash([Scenario.VIAL], paths) == hashlib.sha256(b"vial:missing").hexdigest()
