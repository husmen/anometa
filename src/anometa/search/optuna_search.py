"""Optuna NSGA-II and TPE searches over the Track B unit-cube space.

`run_nsga2` runs a four-objective NSGA-II study (AUROC, calibration error,
predict latency, absolute train/test AUROC gap) that keeps every trial,
including failed runs, in the search; a failed run's all-NaN objectives make
Optuna record it as a `FAIL` trial rather than stop the study. `run_tpe` runs
a single-objective (AUROC) TPE study, warm-started from `initial_units`, the
same shared starting points the TabPFN-surrogate Bayesian optimizer begins
from, so every search method explores from identical seeds. Both studies
resume: a rerun on the same storage only runs the trials still missing from
the `n_trials` budget. `trials_frame` flattens a study into a
`pandas.DataFrame` for reporting, and `write_frame` saves one atomically.
"""

import os
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
import pandas as pd
from numpy.typing import NDArray
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler, NSGAIISampler, TPESampler
from optuna.storages import RDBStorage, RetryHeartbeatStaleTrialCallback
from optuna.trial import TrialState

from anometa.config import Paths, Scenario
from anometa.experiment import ExperimentResult, run_experiment
from anometa.search.space import PARAMS, decode_unit, scenario_suffix, suggest, to_config

_NSGA2_DIRECTIONS: list[Literal["maximize", "minimize"]] = [
    "maximize",
    "minimize",
    "minimize",
    "minimize",
]
"""Directions for `(auroc, ece_bal, predict_latency_ms, |gap_auroc|)`."""

_FINISHED: tuple[TrialState, ...] = (TrialState.COMPLETE, TrialState.FAIL)
"""Trial states that count against a study's `n_trials` budget on a rerun."""


def initial_units(n: int, seed: int) -> NDArray[np.float64]:
    """Draw `n` unit-cube points shared by every search method.

    Args:
        n: Number of points to draw.
        seed: Random seed.

    Returns:
        An `(n, len(PARAMS))` array of independent uniform draws in `[0, 1]`.
    """
    return np.random.default_rng(seed).random((n, len(PARAMS)))


def objectives(result: ExperimentResult) -> tuple[float, float, float, float]:
    """Turn one `ExperimentResult` into the four NSGA-II objective values.

    The fourth objective is `abs(gap_auroc)`: NSGA-II minimises the size
    of the train/test AUROC gap, so a config that scores test images much
    better than train ones counts as non-robust just like one that
    overfits.

    Args:
        result: The run's outcome.

    Returns:
        `(auroc, ece_bal, predict_latency_ms, abs(gap_auroc))`; all `NaN`
        when `result.status != "ok"` (Optuna then records the trial as
        `FAIL`), and NaN for any metric missing from `result.metrics`.
    """
    if result.status != "ok":
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    metrics = result.metrics
    return (
        metrics.get("auroc", float("nan")),
        metrics.get("ece_bal", float("nan")),
        metrics.get("predict_latency_ms", float("nan")),
        abs(metrics.get("gap_auroc", float("nan"))),
    )


def _run_point(
    trial: optuna.Trial, *, k: int, paths: Paths, scenarios: tuple[Scenario, ...] | None
) -> ExperimentResult:
    """Suggest one unit-cube point from `trial`, run it, and record its run id.

    The config keeps `to_config`'s default name `"search"`, shared by every
    search method, so a point two methods both try maps to one run id and
    runs once.

    Args:
        trial: The trial to suggest `PARAMS` on.
        k: Few-shot budget for the resulting config.
        paths: Filesystem layout for the run.
        scenarios: Scenarios to run on; `None` for every scenario.

    Returns:
        The run's outcome.
    """
    cfg = to_config(suggest(trial), k=k, paths=paths, scenarios=scenarios)
    result = run_experiment(cfg)
    trial.set_user_attr("run_id", result.run_id)
    return result


def _study(
    name: str,
    storage: str | None,
    paths: Paths,
    directions: list[Literal["maximize", "minimize"]],
    sampler: BaseSampler,
) -> optuna.Study:
    """Create or load a study on heartbeat-monitored RDB storage.

    A trial killed while `RUNNING` (e.g. SIGKILL) stops heartbeating; after
    the grace period the next `optimize` on this storage marks it `FAIL`
    and `RetryHeartbeatStaleTrialCallback` re-enqueues its params.

    Args:
        name: Study name.
        storage: RDB storage URL; `None` means
            `sqlite:///<paths.artifacts>/optuna.db` (the directory is
            created first).
        paths: Filesystem layout; locates the default storage.
        directions: One optimization direction per objective.
        sampler: The study's sampler.

    Returns:
        The study, loaded when it already exists in `storage`.
    """
    if storage is None:
        paths.artifacts.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{paths.artifacts / 'optuna.db'}"
    with warnings.catch_warnings():
        # Heartbeats and the retry callback are experimental in Optuna 5.
        warnings.simplefilter("ignore", ExperimentalWarning)
        rdb = RDBStorage(
            storage,
            heartbeat_interval=60,
            grace_period=120,
            heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback(),
        )
    return optuna.create_study(
        storage=rdb, sampler=sampler, study_name=name, directions=directions, load_if_exists=True
    )


def _remaining(study: optuna.Study, n_trials: int) -> int:
    """Count the trials still missing from a study's `n_trials` budget.

    Args:
        study: The (possibly resumed) study.
        n_trials: The study's total trial budget.

    Returns:
        `n_trials` minus the trials already `COMPLETE` or `FAIL`, at least 0.
    """
    return max(0, n_trials - len(study.get_trials(deepcopy=False, states=_FINISHED)))


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame to parquet atomically (temporary file, then `os.replace`).

    Args:
        frame: The frame to write.
        path: Destination parquet file; its directory must exist.
    """
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp)
    os.replace(tmp, path)


def run_nsga2(
    n_trials: int,
    *,
    k: int = 2,
    seed: int = 0,
    storage: str | None = None,
    paths: Paths = Paths(),
    population_size: int = 24,
    scenarios: tuple[Scenario, ...] | None = None,
) -> optuna.Study:
    """Run a four-objective NSGA-II search over the dev Track B space.

    Resumes: a rerun on the same storage runs only the trials still missing
    from `n_trials` (`COMPLETE` and `FAIL` trials count).

    Args:
        n_trials: Total number of trials the study should hold.
        k: Few-shot budget for every trial's config.
        seed: Sampler seed.
        storage: Optuna RDB storage URL; `None` means
            `sqlite:///<paths.artifacts>/optuna.db`.
        paths: Filesystem layout for every trial's run.
        population_size: NSGA-II population size.
        scenarios: Scenarios every trial runs on; `None` for every scenario.
            A restricted study's name ends in `scenario_suffix(scenarios)`.

    Returns:
        The study; `study.best_trials` is its dev Pareto frontier.
    """
    study = _study(
        f"nsga2-k{k}-seed{seed}{scenario_suffix(scenarios)}",
        storage,
        paths,
        _NSGA2_DIRECTIONS,
        NSGAIISampler(population_size=population_size, seed=seed),
    )
    study.optimize(
        lambda trial: objectives(_run_point(trial, k=k, paths=paths, scenarios=scenarios)),
        n_trials=_remaining(study, n_trials),
    )
    return study


def run_tpe(
    n_trials: int,
    *,
    k: int = 2,
    seed: int = 0,
    n_init: int = 8,
    storage: str | None = None,
    paths: Paths = Paths(),
    scenarios: tuple[Scenario, ...] | None = None,
) -> optuna.Study:
    """Run a single-objective (AUROC) TPE search, warm-started from shared points.

    The first `n_init` trials are `decode_unit(u)` for `u` in
    `initial_units(n_init, seed)`, enqueued before optimizing so TPE (whose
    own `n_startup_trials` is also `n_init`) starts from the identical points
    every other search method starts from. A failed run stays a `FAIL` trial
    that TPE ignores (unlike `run_tabpfn_bo`, which scores it as AUROC 0.5).

    Resumes: a rerun on the same storage skips shared points already
    enqueued and runs only the trials still missing from `n_trials`
    (`COMPLETE` and `FAIL` trials count).

    Args:
        n_trials: Total number of trials the study should hold.
        k: Few-shot budget for every trial's config.
        seed: Sampler seed; also seeds `initial_units`.
        n_init: Number of shared points to enqueue, and `TPESampler`'s
            `n_startup_trials`.
        storage: Optuna RDB storage URL; `None` means
            `sqlite:///<paths.artifacts>/optuna.db`.
        paths: Filesystem layout for every trial's run.
        scenarios: Scenarios every trial runs on; `None` for every scenario.
            A restricted study's name and parquet file end in
            `scenario_suffix(scenarios)`.

    Returns:
        The study. `trials_frame(study)` is rewritten to
        `paths.artifacts / "search" / "tpe-k{k}-seed{seed}.parquet"` after
        every trial.
    """
    name = f"tpe-k{k}-seed{seed}{scenario_suffix(scenarios)}"
    study = _study(
        name,
        storage,
        paths,
        ["maximize"],
        TPESampler(seed=seed, n_startup_trials=n_init),
    )
    for u in initial_units(n_init, seed):
        study.enqueue_trial(decode_unit(u.tolist()), skip_if_exists=True)
    out_dir = paths.artifacts / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.parquet"
    study.optimize(
        lambda trial: objectives(_run_point(trial, k=k, paths=paths, scenarios=scenarios))[0],
        n_trials=_remaining(study, n_trials),
        callbacks=[lambda st, _trial: write_frame(trials_frame(st), out_path)],
    )
    return study


def trials_frame(study: optuna.Study) -> pd.DataFrame:
    """Flatten a study into one row per trial, in trial order.

    Args:
        study: The study to flatten.

    Returns:
        Columns `trial`, `state`, `run_id`, `auroc`, `best_so_far` (the best
        AUROC seen in any `COMPLETE` trial up to and including this one,
        `NaN` before the first), plus one column per `PARAMS` name.
    """
    rows: list[dict[str, object]] = []
    best = float("nan")
    for trial in study.trials:
        auroc = trial.values[0] if trial.values is not None else float("nan")
        if trial.state == TrialState.COMPLETE and not np.isnan(auroc):
            best = auroc if np.isnan(best) else max(best, auroc)
        rows.append(
            {
                "trial": trial.number,
                "state": trial.state.name,
                "run_id": trial.user_attrs.get("run_id"),
                "auroc": auroc,
                "best_so_far": best,
                **trial.params,
            }
        )
    return pd.DataFrame(rows)
