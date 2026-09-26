"""Command-line entry point for the `anometa` package.

Registers one `argparse` subcommand per pipeline stage (download, split,
extract, run, ...). Later tasks add their own subcommand to `SUBPARSERS`
via `SUBPARSERS.add_parser(...)` and `set_defaults(func=...)`.
"""

import argparse
import sys
from pathlib import Path

from anometa.config import Paths, Scenario, load_config
from anometa.data.ad2 import index_scenario, lighting_counts
from anometa.data.download import fetch_scenario
from anometa.data.splits import make_split, write_split
from anometa.experiment import run_experiment

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


def _cmd_run(args: argparse.Namespace) -> int:
    """Run the `run` subcommand: run one experiment config end to end.

    Refuses `split: lock` configs: only `anometa lock` builds and runs those.

    Args:
        args: Parsed arguments; `config` (path to a YAML experiment config).

    Returns:
        `0` if the run's status is `"ok"`, `1` if it's `"failed"`, `2` if the
        config is a lock-split config.
    """
    cfg = load_config(Path(args.config))
    if cfg.split == "lock":
        print(
            f"{args.config}: split: lock configs run only through `anometa lock`",
            file=sys.stderr,
        )
        return 2
    result = run_experiment(cfg)
    print(f"{result.run_id}: {result.status}")
    for name, value in sorted(result.metrics.items()):
        print(f"{name}: {value}")
    return 0 if result.status == "ok" else 1


_run_parser = SUBPARSERS.add_parser("run", help="Run one experiment config")
_run_parser.add_argument("config", help="Path to a YAML experiment config")
_run_parser.set_defaults(func=_cmd_run)


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
