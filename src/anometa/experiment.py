"""The experiment API: `run_experiment` dispatches a config to its track runner.

`RUNNERS` maps each track to its runner, as an import string resolved lazily
(so importing this module never imports a track's dependencies) or, in
tests, a callable substituted with `monkeypatch.setitem`. `run_experiment`
is the one entry point CLI, Streamlit and the search layer all call: it
guards lock runs against re-evaluation, resumes a completed dev run from
disk, and otherwise runs the track, writes its artifacts and appends to the
shared run log.
"""

import importlib
import json
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from anometa.artifacts import (
    TrackOutput,
    append_run_log,
    capture_manifest,
    read_manifest,
    write_run,
)
from anometa.config import ExperimentConfig, config_hash, run_id

RUNNERS: dict[str, str | Callable[[ExperimentConfig, Path], TrackOutput]] = {
    "A": "anometa.tracka.pipeline:run_track_a",
    "B": "anometa.trackb.pipeline:run_track_b",
}
"""Each track's runner, as an `"module:attr"` import string or a callable."""


class ExperimentResult(BaseModel):
    """The outcome of one `run_experiment` call."""

    run_id: str
    config_hash: str
    status: Literal["ok", "failed"]
    metrics: dict[str, float]
    artifact_dir: Path
    error: str | None = None


class LockAlreadyEvaluatedError(RuntimeError):
    """Raised when a lock-split config already has a run directory."""


def _resolve_runner(track: str) -> Callable[[ExperimentConfig, Path], TrackOutput]:
    """Resolve `RUNNERS[track]` to a callable, importing it if it's a string.

    Args:
        track: `"A"` or `"B"`.

    Returns:
        The track's runner function.

    Raises:
        TypeError: If the import string resolves to a non-callable.
    """
    entry = RUNNERS[track]
    if isinstance(entry, str):
        module_name, _, attr = entry.partition(":")
        runner: Callable[[ExperimentConfig, Path], TrackOutput] = getattr(
            importlib.import_module(module_name), attr
        )
        if not callable(runner):
            raise TypeError(f"runner {entry!r} is not callable")
        return runner
    return entry


def _completed_metrics(run_dir: Path) -> dict[str, float] | None:
    """Read back a completed dev run's metrics, if the run is complete on disk.

    Args:
        run_dir: The run directory to inspect.

    Returns:
        The run's metrics when its manifest is readable with `status="ok"`
        and `metrics.json` and `predictions.parquet` exist; `None` otherwise
        (no manifest, a corrupt one, a failed run or missing files), so the
        caller reruns it.
    """
    try:
        manifest = read_manifest(run_dir)
        if manifest is None or manifest.get("status") != "ok":
            return None
        if not (run_dir / "predictions.parquet").is_file():
            return None
        metrics: dict[str, float] = json.loads((run_dir / "metrics.json").read_text())
    except json.JSONDecodeError, FileNotFoundError:
        return None
    return metrics


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Run one experiment config end to end, writing its artifacts.

    Order: a lock-split config whose run directory (or any run directory
    with the same config hash, under any `name`) already exists raises
    `LockAlreadyEvaluatedError` before anything runs; a dev-split run that is
    complete on disk (`status="ok"` manifest plus its metrics and
    predictions) is read back instead of re-running. Otherwise the manifest's
    provenance (git state, dataset and split hashes, hardware, versions) is
    captured, a lock run claims its directory, the track runner runs, its
    artifacts are written (on failure too, with the traceback as `error`),
    and the manifest is appended to `runs.jsonl`. A runner exception is
    caught and recorded as a failed run; `KeyboardInterrupt` still
    propagates, and a lock run it interrupts stays claimed.

    Args:
        cfg: The experiment configuration to run.

    Returns:
        The run's outcome.

    Raises:
        LockAlreadyEvaluatedError: If `cfg.split == "lock"` and a run
            directory for its config hash already exists, whatever it holds.
    """
    rid = run_id(cfg)
    chash = config_hash(cfg)
    run_dir = cfg.paths.artifacts / rid

    if cfg.split == "lock":
        existing = next(
            (p for p in cfg.paths.artifacts.glob(f"*-{chash[:12]}") if p.is_dir()), None
        )
        if existing is not None:
            raise LockAlreadyEvaluatedError(
                f"lock run {rid} was already evaluated as {existing.name}"
            )
    else:
        done = _completed_metrics(run_dir)
        if done is not None:
            return ExperimentResult(
                run_id=rid, config_hash=chash, status="ok", metrics=done, artifact_dir=run_dir
            )

    started_at = datetime.now(UTC).isoformat()
    manifest = capture_manifest(
        cfg,
        status="failed",
        error=None,
        started_at=started_at,
        duration_s=0.0,
        model_revisions={},
        licences={},
    )
    if cfg.split == "lock":
        try:
            run_dir.mkdir(parents=True)
        except FileExistsError:
            raise LockAlreadyEvaluatedError(f"lock run {rid} was already evaluated") from None

    start = time.monotonic()
    output: TrackOutput | None = None
    error: str | None = None
    try:
        output = _resolve_runner(cfg.track)(cfg, run_dir)
        status: Literal["ok", "failed"] = "ok"
    except KeyboardInterrupt:
        raise
    except Exception:
        status = "failed"
        error = traceback.format_exc()

    metrics = output.metrics if output is not None else {}
    manifest |= {
        "status": status,
        "error": error,
        "duration_s": time.monotonic() - start,
        "model_revisions": output.model_revisions if output is not None else {},
        "licences": output.licences if output is not None else {},
    }
    write_run(
        run_dir,
        manifest,
        metrics,
        output.predictions if output is not None else None,
        output.extra_files if output is not None else {},
    )
    append_run_log(cfg.paths.artifacts, manifest)

    return ExperimentResult(
        run_id=rid,
        config_hash=chash,
        status=status,
        metrics=metrics,
        artifact_dir=run_dir,
        error=error,
    )
