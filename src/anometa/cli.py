"""Command-line entry point for the `anometa` package.

Registers one `argparse` subcommand per pipeline stage (download, split,
extract, run, ...). Later tasks add their own subcommand to `SUBPARSERS`
via `SUBPARSERS.add_parser(...)` and `set_defaults(func=...)`.
"""

import argparse
import sys

from anometa.config import Paths, Scenario
from anometa.data.ad2 import index_scenario, lighting_counts
from anometa.data.download import fetch_scenario
from anometa.data.splits import make_split, write_split

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
