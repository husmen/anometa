"""Command-line entry point for the `anometa` package.

Registers one `argparse` subcommand per pipeline stage (download, split,
extract, run, ...). Later tasks add their own subcommand to `SUBPARSERS`
via `SUBPARSERS.add_parser(...)` and `set_defaults(func=...)`.
"""

import argparse
import sys

from anometa.config import Paths, Scenario
from anometa.data.download import fetch_scenario

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
