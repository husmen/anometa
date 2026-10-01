"""Run artifacts: manifests, run outputs and the run log.

Defines `TrackOutput`, the value a track runner returns to `run_experiment`,
and the functions that turn it into an on-disk run directory
(`artifacts/<run-id>/manifest.json`, `metrics.json`, `predictions.parquet`
and any `extra_files`), plus `capture_manifest` for the manifest's content
and `append_run_log` for the shared `runs.jsonl` history.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from anometa.config import (
    ExperimentConfig,
    Paths,
    Scenario,
    config_hash,
    resolve_device,
    run_id,
    seed_parallelism,
)
from anometa.data.download import read_sha256sums
from anometa.data.splits import split_hash as _split_hash

LICENCES: dict[str, str] = {
    "ad2": "CC BY-NC-SA 4.0",
    "dinov3": "DINOv3 License",
    "siglip2": "Apache-2.0",
    "tabpfn": "tabpfn-3-5-license-v1.0 (non-commercial)",
    "tabpfn_api": "Prior Labs API Terms and AUP",
    "anomalib": "Apache-2.0",
}
"""Licence text for every third-party dataset and model this project uses."""

_VERSIONED_PACKAGES: tuple[str, ...] = (
    "anometa",
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "transformers",
    "tabpfn",
    "tabpfn-client",
    "tabpfn-extensions",
    "anomalib",
    "optuna",
)


@dataclass(frozen=True)
class TrackOutput:
    """One track runner's result, before it is written to a run directory.

    Attributes:
        metrics: Top-level metrics for this run; keys vary by track.
        predictions: Per-image predictions, written to `predictions.parquet`.
        model_revisions: Pinned model identifiers this run depended on (e.g.
            an encoder's HF revision), recorded in the manifest.
        licences: Licence names for the datasets and models this run used,
            recorded in the manifest.
        extra_files: Extra files written verbatim into the run directory,
            keyed by filename (e.g. `"maps.npz"`).
    """

    metrics: dict[str, float]
    predictions: pd.DataFrame
    model_revisions: dict[str, str]
    licences: dict[str, str]
    extra_files: dict[str, bytes] = field(default_factory=dict)


def dataset_hash(scenarios: Sequence[Scenario], paths: Paths) -> str:
    """Fingerprint the pinned archive checksums of a set of scenarios.

    Args:
        scenarios: Scenarios to include; hashed in canonical `Scenario`
            order regardless of the order they're given in.
        paths: Run paths; pinned checksums are read from
            `paths.configs / "data" / "sha256sums.txt"`.

    Returns:
        The hex-encoded SHA-256 digest of `"<scenario>:<digest>"` per
        scenario, joined with newlines, in `Scenario` order. A scenario
        whose archive isn't pinned yet (or a missing pins file) contributes
        `"missing"`.
    """
    sums_path = paths.configs / "data" / "sha256sums.txt"
    sums = read_sha256sums(sums_path) if sums_path.is_file() else {}
    order = list(Scenario)
    parts = [
        f"{scenario}:{sums.get(f'{scenario}.tar.gz', 'missing')}"
        for scenario in sorted(scenarios, key=order.index)
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def git_info() -> tuple[str | None, bool | None]:
    """Read the current commit and dirty state of the enclosing git repo.

    Returns:
        `(commit_sha, is_dirty)`, or `(None, None)` when run outside a git
        repository (or without git installed). Untracked files don't count
        as dirty.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError, FileNotFoundError:
        return None, None
    return commit, bool(status.strip())


def _hardware_info(cfg: ExperimentConfig) -> dict[str, object]:
    """Describe the host this run executes on.

    Args:
        cfg: The experiment config; its `device` field is resolved to the
            concrete device this run uses.

    Returns:
        `platform`, `machine`, `cpu_count`, the resolved `device`, and
        `seed_workers` and `seed_executor` from `seed_parallelism`; plus
        `cuda_device_name` and `cuda_vram_mb` when the resolved device is
        `"cuda"`.
    """
    device = resolve_device(cfg.device)
    workers, executor = seed_parallelism()
    info: dict[str, object] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "device": device,
        "seed_workers": workers,
        "seed_executor": executor,
    }
    if device == "cuda":
        import torch  # lazy: only needed to describe a CUDA device

        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_vram_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    return info


def _versions() -> dict[str, str | None]:
    """Read the installed version of every package the manifest reports.

    Returns:
        `python` (the interpreter version) plus each of
        `_VERSIONED_PACKAGES`, `None` for a package that isn't installed.
    """
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _VERSIONED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capture_manifest(
    cfg: ExperimentConfig,
    *,
    status: Literal["ok", "failed"],
    error: str | None,
    started_at: str,
    duration_s: float,
    model_revisions: dict[str, str],
    licences: dict[str, str],
) -> dict[str, object]:
    """Build a run's manifest.

    Args:
        cfg: The experiment config this run executed.
        status: Whether the run succeeded.
        error: The runner's traceback text, when `status` is `"failed"`.
        started_at: UTC ISO 8601 timestamp the run started at.
        duration_s: Wall-clock run duration in seconds.
        model_revisions: Pinned model identifiers the run depended on.
        licences: Licence names for the datasets and models the run used.

    Returns:
        A JSON-serialisable dict with exactly the manifest keys: `run_id`,
        `config`, `config_hash`, `status`, `error`, `git_commit`,
        `git_dirty`, `dataset_hash`, `split_hash`, `model_revisions`,
        `licences`, `seeds`, `hardware`, `versions`, `started_at`,
        `duration_s`.
    """
    git_commit, git_dirty = git_info()
    return {
        "run_id": run_id(cfg),
        "config": cfg.model_dump(mode="json"),
        "config_hash": config_hash(cfg),
        "status": status,
        "error": error,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "dataset_hash": dataset_hash(cfg.scenarios, cfg.paths),
        "split_hash": _split_hash(cfg.scenarios, cfg.paths),
        "model_revisions": model_revisions,
        "licences": licences,
        "seeds": list(cfg.seeds),
        "hardware": _hardware_info(cfg),
        "versions": _versions(),
        "started_at": started_at,
        "duration_s": duration_s,
    }


def write_run(
    run_dir: Path,
    manifest: dict[str, object],
    metrics: dict[str, float],
    predictions: pd.DataFrame | None,
    extra_files: dict[str, bytes],
) -> None:
    """Write one run's artifacts to disk, manifest last.

    Any stale `manifest.json` is removed first, then metrics, predictions
    and extra files are written, and `manifest.json` goes in last through a
    temporary file and `os.replace`. A run directory with a manifest is
    therefore always complete; an interrupted write leaves none.

    Args:
        run_dir: Destination directory; created if missing.
        manifest: As returned by `capture_manifest`, written to
            `manifest.json`.
        metrics: Written to `metrics.json`.
        predictions: Written to `predictions.parquet`, when given.
        extra_files: Written verbatim under `run_dir`, keyed by filename.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    if predictions is not None:
        predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    for name, data in extra_files.items():
        (run_dir / name).write_bytes(data)
    tmp_path = run_dir / "manifest.json.tmp"
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp_path, manifest_path)


def read_manifest(run_dir: Path) -> dict[str, object] | None:
    """Read a run's manifest back from disk.

    Args:
        run_dir: The run directory to read `manifest.json` from.

    Returns:
        The manifest dict, or `None` if the run has no manifest yet.
    """
    path = run_dir / "manifest.json"
    if not path.is_file():
        return None
    manifest: dict[str, object] = json.loads(path.read_text())
    return manifest


def append_run_log(root: Path, record: dict[str, object]) -> None:
    """Append one record to the shared run log.

    Args:
        root: Directory holding `runs.jsonl`; created if missing.
        record: JSON-serialisable record, written as one line.
    """
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
