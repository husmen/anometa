"""Command-line entry point for the `anometa` package.

Registers one `argparse` subcommand per pipeline stage (download, split,
extract, run, ...). Later tasks add their own subcommand to `SUBPARSERS`
via `SUBPARSERS.add_parser(...)` and `set_defaults(func=...)`.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from pydantic import TypeAdapter

from anometa.config import ExperimentConfig, Paths, Scenario, resolve_device
from anometa.data.ad2 import index_scenario, lighting_counts
from anometa.data.download import fetch_scenario
from anometa.data.splits import make_split, write_split
from anometa.experiment import run_experiment
from anometa.features.encoders import load_encoder
from anometa.features.extract import extract_scenario

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
