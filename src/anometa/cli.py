"""Command-line entry point for the `anometa` package.

Registers one `argparse` subcommand per pipeline stage (download, split,
extract, run, ...). Later tasks add their own subcommand to `SUBPARSERS`
via `SUBPARSERS.add_parser(...)` and `set_defaults(func=...)`.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from pydantic import TypeAdapter

from anometa.config import (
    ExperimentConfig,
    Paths,
    Scenario,
    load_configs,
    resolve_device,
    run_id,
)
from anometa.data.ad2 import index_scenario, lighting_counts
from anometa.data.download import fetch_scenario
from anometa.data.splits import make_split, write_split
from anometa.experiment import LockAlreadyEvaluatedError, run_experiment
from anometa.features.encoders import load_encoder
from anometa.features.extract import extract_scenario
from anometa.search.grid import GridSpec, grid_configs, run_grid

parser = argparse.ArgumentParser(prog="anometa")
SUBPARSERS = parser.add_subparsers(dest="command")


def _cmd_download(args: argparse.Namespace) -> int:
    """Run the `download` subcommand: fetch, verify and extract AD2 scenarios.

    Args:
        args: Parsed arguments; `scenario` (repeatable, defaults to every
            scenario) and `keep_archive`.

    Returns:
        `0` on success.
    """
    paths = Paths()
    for scenario in args.scenario or list(Scenario):
        digest = fetch_scenario(scenario, paths, keep_archive=args.keep_archive)
        print(f"{scenario}: {digest}")
    return 0


_download_parser = SUBPARSERS.add_parser("download", help="Download AD2 scenario archives")
_download_parser.add_argument(
    "--scenario",
    action="append",
    type=Scenario,
    choices=list(Scenario),
    help="Scenario to fetch (repeatable); defaults to all scenarios",
)
_download_parser.add_argument(
    "--keep-archive",
    action="store_true",
    help="Keep the downloaded archive after extraction",
)
_download_parser.set_defaults(func=_cmd_download)


def _cmd_split(args: argparse.Namespace) -> int:
    """Run the `split` subcommand: build each downloaded scenario's dev/lock split.

    Indexes every scenario whose `train/good` folder exists under
    `paths.data`, writes `splits/<scenario>.csv` only when it doesn't already
    exist (an existing split is never overwritten), and prints its
    `lighting_counts` table.

    Args:
        args: Parsed arguments; none beyond the subcommand itself.

    Returns:
        `0` on success.
    """
    paths = Paths()
    for scenario in Scenario:
        if not (paths.data / scenario / "train/good").is_dir():
            continue
        index = index_scenario(paths.data, scenario)
        split_path = paths.splits / f"{scenario}.csv"
        if not split_path.exists():
            write_split(make_split(index), split_path)
        print(scenario)
        print(lighting_counts(index))
    return 0


_split_parser = SUBPARSERS.add_parser("split", help="Build dev/lock splits for AD2 scenarios")
_split_parser.set_defaults(func=_cmd_split)


def _cmd_extract(args: argparse.Namespace) -> int:
    """Run the `extract` subcommand: encode and cache one encoder's scenario features.

    Prints, per scenario, the backend, the cache path, the wall time and, on
    CUDA, the peak allocated VRAM.

    Args:
        args: Parsed arguments; `encoder`, `scenario` (repeatable, defaults
            to every scenario downloaded under `paths.data`), `device` and
            `backend`.

    Returns:
        `0` on success.
    """
    paths = Paths()
    device = resolve_device(args.device)
    encoder = load_encoder(args.encoder, device, args.backend)
    scenarios = args.scenario or [
        scenario for scenario in Scenario if (paths.data / scenario / "train/good").is_dir()
    ]
    for scenario in scenarios:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = extract_scenario(encoder, scenario, paths)
        timing = f"{time.perf_counter() - start:.1f} s"
        if device == "cuda":
            timing += f", peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB"
        print(f"{scenario}: backend={encoder.backend} -> {out} ({timing})")
    return 0


_extract_parser = SUBPARSERS.add_parser(
    "extract", help="Extract and cache one encoder's per-scenario features"
)
_extract_parser.add_argument(
    "--encoder",
    required=True,
    choices=["dinov3_s", "dinov3_l", "siglip2"],
    help="Encoder to run",
)
_extract_parser.add_argument(
    "--scenario",
    action="append",
    type=Scenario,
    choices=list(Scenario),
    help="Scenario to extract (repeatable); defaults to every downloaded scenario",
)
_extract_parser.add_argument(
    "--device",
    default="auto",
    choices=["auto", "cuda", "mps", "cpu"],
    help="Device to run the encoder on",
)
_extract_parser.add_argument(
    "--backend",
    default="auto",
    choices=["auto", "transformers", "timm"],
    help="Encoder loading backend",
)
_extract_parser.set_defaults(func=_cmd_extract)


def apply_overrides(data: dict[str, object], pairs: list[str]) -> dict[str, object]:
    """Apply `--set key=value` overrides to a loaded config dict.

    Each pair's value is parsed with `yaml.safe_load`, so `5`, `knn` and
    `[0, 1]` become an int, a str and a list respectively. A `null` value
    removes `key` from the result instead of setting it, so the field's own
    default applies.

    Args:
        data: The base config dict, e.g. loaded from an experiment YAML file.
        pairs: `"key=value"` strings, one per `--set` flag.

    Returns:
        A new dict with every pair applied; `data` is left unmodified.

    Raises:
        ValueError: If a pair holds no `=`.
    """
    out = dict(data)
    for pair in pairs:
        key, sep, raw_value = pair.partition("=")
        if not sep:
            raise ValueError(f"--set value must be key=value, got {pair!r}")
        value = yaml.safe_load(raw_value)
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def _cmd_run(args: argparse.Namespace) -> int:
    """Run the `run` subcommand: run one experiment config end to end.

    Refuses `split: lock` configs: only `anometa lock` builds and runs those.
    A failed run prints its traceback's last line (the exception) to stderr.

    Args:
        args: Parsed arguments; `config` (path to a YAML experiment config)
            and `set` (`key=value` overrides applied to it, from one or more
            `--set` flags that each take one or more pairs).

    Returns:
        `0` if the run's status is `"ok"`, `1` if it's `"failed"`, `2` if the
        config is a lock-split config.
    """
    data: dict[str, object] = yaml.safe_load(Path(args.config).read_text())
    data = apply_overrides(data, args.set)
    cfg = TypeAdapter(ExperimentConfig).validate_python(data)
    if cfg.split == "lock":
        print(
            f"{args.config}: split: lock configs run only through `anometa lock`",
            file=sys.stderr,
        )
        return 2
    result = run_experiment(cfg)
    print(f"{result.run_id}: {result.status}")
    if result.error:
        print(result.error.strip().splitlines()[-1], file=sys.stderr)
    for name, value in sorted(result.metrics.items()):
        print(f"{name}: {value}")
    return 0 if result.status == "ok" else 1


_run_parser = SUBPARSERS.add_parser("run", help="Run one experiment config")
_run_parser.add_argument("config", help="Path to a YAML experiment config")
_run_parser.add_argument(
    "--set",
    action="extend",
    nargs="+",
    default=[],
    metavar="key=value",
    help="Override config fields (one or more pairs, repeatable); YAML-typed, null removes it",
)
_run_parser.set_defaults(func=_cmd_run)


def _cmd_reference_check(args: argparse.Namespace) -> int:
    """Run the `reference-check` subcommand: compare `TabPFNWithImages` to Track B.

    Prints one row per seed and each side's mean AUROC.

    Args:
        args: Parsed arguments; `scenario` and `k`.

    Returns:
        `0` on success.
    """
    from anometa.trackb.reference import reference_check

    paths = Paths()
    result = reference_check(args.scenario, paths, k=args.k)
    print(result.to_string(index=False))
    print(f"reference mean AUROC: {result['reference_auroc'].mean():.4f}")
    print(f"ours mean AUROC: {result['ours_auroc'].mean():.4f}")
    return 0


_reference_check_parser = SUBPARSERS.add_parser(
    "reference-check", help="Compare TabPFNWithImages to Track B on one scenario"
)
_reference_check_parser.add_argument(
    "--scenario",
    type=Scenario,
    choices=list(Scenario),
    default=Scenario.VIAL,
    help="Scenario to check",
)
_reference_check_parser.add_argument("--k", type=int, default=5, help="Few-shot budget per seed")
_reference_check_parser.set_defaults(func=_cmd_reference_check)


def _scenarios(args: argparse.Namespace) -> tuple[Scenario, ...] | None:
    """Return a search subcommand's `--scenario` restriction.

    Args:
        args: Parsed arguments; `scenario` is `None` or a list of scenarios.

    Returns:
        The given scenarios, or `None` (every scenario) when none were given.
    """
    return tuple(args.scenario) if args.scenario else None


def _cmd_grid(args: argparse.Namespace) -> int:
    """Run the `grid` subcommand: run every config in a search grid.

    With `--dry-run`, prints the expanded config count and exits without
    running anything.

    Args:
        args: Parsed arguments; `spec` (path to a `GridSpec` YAML file),
            `scenario` (repeatable, defaults to every scenario) and `dry_run`.

    Returns:
        `0` on success.
    """
    paths = Paths()
    spec = GridSpec(**yaml.safe_load(Path(args.spec).read_text()))
    configs = grid_configs(spec, paths, _scenarios(args))
    if args.dry_run:
        print(len(configs))
        return 0
    result = run_grid(configs)
    print(result.to_string(index=False))
    return 0


_grid_parser = SUBPARSERS.add_parser("grid", help="Run a search grid of Track B configs")
_grid_parser.add_argument("spec", help="Path to a GridSpec YAML file")
_grid_parser.add_argument(
    "--scenario",
    action="append",
    type=Scenario,
    choices=list(Scenario),
    help="Scenario to search on (repeatable); defaults to all scenarios",
)
_grid_parser.add_argument(
    "--dry-run", action="store_true", help="Print the expanded config count and exit"
)
_grid_parser.set_defaults(func=_cmd_grid)


def _cmd_optuna(args: argparse.Namespace) -> int:
    """Run the `optuna` subcommand: run an NSGA-II or TPE search over dev configs.

    `nsga2` prints one line per Pareto-optimal trial (`study.best_trials`).
    `tpe` prints the single best trial (`study.best_trial`) and writes every
    trial to a parquet file (see `run_tpe`). Optuna is imported here, so
    other subcommands never load it.

    Args:
        args: Parsed arguments; `method` (`nsga2` or `tpe`), `trials`, `k`,
            `seed`, `storage` (`None` for the default under
            `paths.artifacts`) and `scenario` (repeatable, defaults to every
            scenario).

    Returns:
        `0` on success.
    """
    from anometa.search.optuna_search import run_nsga2, run_tpe

    paths = Paths()
    scenarios = _scenarios(args)
    if args.method == "nsga2":
        study = run_nsga2(
            args.trials,
            k=args.k,
            seed=args.seed,
            storage=args.storage,
            paths=paths,
            scenarios=scenarios,
        )
        for trial in study.best_trials:
            print(trial.number, trial.values, trial.params)
    else:
        study = run_tpe(
            args.trials,
            k=args.k,
            seed=args.seed,
            storage=args.storage,
            paths=paths,
            scenarios=scenarios,
        )
        print(study.best_trial.number, study.best_trial.value, study.best_trial.params)
    return 0


_optuna_parser = SUBPARSERS.add_parser("optuna", help="Run an Optuna NSGA-II or TPE search")
_optuna_parser.add_argument("method", choices=["nsga2", "tpe"], help="Search method")
_optuna_parser.add_argument("--trials", type=int, required=True, help="Number of trials to run")
_optuna_parser.add_argument("--k", type=int, default=2, help="Few-shot budget per trial")
_optuna_parser.add_argument("--seed", type=int, default=0, help="Sampler seed")
_optuna_parser.add_argument(
    "--storage",
    default=None,
    help="Optuna RDB storage URL; defaults to sqlite:///<artifacts>/optuna.db",
)
_optuna_parser.add_argument(
    "--scenario",
    action="append",
    type=Scenario,
    choices=list(Scenario),
    help="Scenario to search on (repeatable); defaults to all scenarios",
)
_optuna_parser.set_defaults(func=_cmd_optuna)


def _cmd_bo(args: argparse.Namespace) -> int:
    """Run the `bo` subcommand: run a TabPFN-surrogate Bayesian optimization loop.

    Prints the best trial found (highest `auroc`).

    Args:
        args: Parsed arguments; `trials`, `k`, `seed`, `device` (where the
            TabPFN surrogate runs) and `scenario` (repeatable, defaults to
            every scenario).

    Returns:
        `0` on success.
    """
    from anometa.search.tabpfn_bo import run_tabpfn_bo

    paths = Paths()
    df = run_tabpfn_bo(
        args.trials,
        k=args.k,
        seed=args.seed,
        device=args.device,
        paths=paths,
        scenarios=_scenarios(args),
    )
    best = df.loc[df["auroc"].idxmax()]
    print(best["trial"], best["auroc"], best["run_id"])
    return 0


_bo_parser = SUBPARSERS.add_parser("bo", help="Run a TabPFN-surrogate Bayesian optimization loop")
_bo_parser.add_argument("--trials", type=int, required=True, help="Number of trials to run")
_bo_parser.add_argument("--k", type=int, default=2, help="Few-shot budget per trial")
_bo_parser.add_argument("--seed", type=int, default=0, help="Sampler seed")
_bo_parser.add_argument(
    "--device",
    default="auto",
    choices=["auto", "cuda", "mps", "cpu"],
    help="Device to run the TabPFN surrogate on",
)
_bo_parser.add_argument(
    "--scenario",
    action="append",
    type=Scenario,
    choices=list(Scenario),
    help="Scenario to search on (repeatable); defaults to all scenarios",
)
_bo_parser.set_defaults(func=_cmd_bo)


def _cmd_report(args: argparse.Namespace) -> int:
    """Run the `report` subcommand: build the results page and figures for one split.

    Args:
        args: Parsed arguments; `split` (`dev` or `lock`).

    Returns:
        `0` on success.
    """
    from anometa.report import build_report

    print(build_report(Paths().artifacts, Path("reports"), args.split))
    return 0


_report_parser = SUBPARSERS.add_parser("report", help="Build results tables and figures")
_report_parser.add_argument(
    "--split", default="dev", choices=["dev", "lock"], help="Which split's runs to report"
)
_report_parser.set_defaults(func=_cmd_report)


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Run the `inspect` subcommand: write a Rerun recording of one run.

    Prints the `.rrd` path and the command that opens it.

    Args:
        args: Parsed arguments; `run_id` and `max_images`.

    Returns:
        `0` on success.
    """
    from anometa.rerun_log import write_rrd

    paths = Paths()
    out = write_rrd(paths.artifacts / args.run_id, paths, max_images=args.max_images)
    print(out)
    print(f"rerun {out}")
    return 0


_inspect_parser = SUBPARSERS.add_parser("inspect", help="Write a Rerun recording of one run")
_inspect_parser.add_argument("run_id", help="Run directory name under the artifacts folder")
_inspect_parser.add_argument(
    "--max-images", type=int, default=50, help="Most evaluation images to log"
)
_inspect_parser.set_defaults(func=_cmd_inspect)


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run the `demo` subcommand: launch the Streamlit few-shot dashboard.

    Args:
        args: Parsed arguments; none beyond the subcommand itself.

    Returns:
        The Streamlit subprocess's exit code.
    """
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)], check=False
    )
    return completed.returncode


_demo_parser = SUBPARSERS.add_parser("demo", help="Launch the Streamlit few-shot dashboard")
_demo_parser.set_defaults(func=_cmd_demo)


def _cmd_parity(args: argparse.Namespace) -> int:
    """Run the `parity` subcommand: compare DINOv3's transformers and timm backends.

    Args:
        args: Parsed arguments; `encoder`, `scenario`, `n` and `device`.

    Returns:
        `0` on success.
    """
    from anometa.features.parity import run_parity

    result = run_parity(args.encoder, args.scenario, Paths(), n_images=args.n, device=args.device)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


_parity_parser = SUBPARSERS.add_parser(
    "parity", help="Compare DINOv3's transformers and timm backends"
)
_parity_parser.add_argument("--encoder", required=True, choices=["dinov3_s", "dinov3_l"])
_parity_parser.add_argument(
    "--scenario", type=Scenario, choices=list(Scenario), default=Scenario.VIAL
)
_parity_parser.add_argument("--n", type=int, default=32, help="Images compared")
_parity_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
_parity_parser.set_defaults(func=_cmd_parity)


def lock_configs(frozen_dir: Path) -> list[ExperimentConfig]:
    """Load every frozen config and force it onto the lock split.

    Args:
        frozen_dir: Folder of YAML files, each holding one config or a list.

    Returns:
        Every config of every `*.yaml` file (files in name order), revalidated
        with `split="lock"`.
    """
    adapter: TypeAdapter[ExperimentConfig] = TypeAdapter(ExperimentConfig)
    return [
        adapter.validate_python(cfg.model_dump() | {"split": "lock"})
        for path in sorted(frozen_dir.glob("*.yaml"))
        for cfg in load_configs(path)
    ]


def _cmd_lock(args: argparse.Namespace) -> int:
    """Run the `lock` subcommand: evaluate every frozen config once on the lock split.

    A config whose lock run already exists is reported as already evaluated
    and never rerun. Writes one row per config (`run_id`, `status`, `error`,
    then the run's metrics) to `artifacts/lock_summary.csv`.

    Args:
        args: Parsed arguments; `frozen_dir`.

    Returns:
        `0` if no run failed in this invocation, else `1`.
    """
    import pandas as pd

    rows: list[dict[str, object]] = []
    for cfg in lock_configs(Path(args.frozen_dir)):
        try:
            result = run_experiment(cfg)
        except LockAlreadyEvaluatedError as exc:
            print(f"{run_id(cfg)}: already evaluated ({exc})")
            rows.append({"run_id": run_id(cfg), "status": "already evaluated", "error": None})
            continue
        error = result.error.strip().splitlines()[-1] if result.error else None
        print(f"{result.run_id}: {result.status}" + (f" ({error})" if error else ""))
        rows.append({"run_id": result.run_id, "status": result.status, "error": error})
        rows[-1] |= result.metrics
    summary = Paths().artifacts / "lock_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary, index=False)
    print(summary)
    return 1 if any(row["status"] == "failed" for row in rows) else 0


_lock_parser = SUBPARSERS.add_parser(
    "lock", help="Evaluate the frozen configs once on the lock split"
)
_lock_parser.add_argument("frozen_dir", help="Folder of frozen YAML configs, e.g. configs/frozen")
_lock_parser.set_defaults(func=_cmd_lock)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults
            to `sys.argv[1:]` when `None`.

    Returns:
        The subcommand's exit code, or `1` when no subcommand was given.
    """
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    result: int = func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
