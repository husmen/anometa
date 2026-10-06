# Getting started

This page covers setup, access to models and data, and every `anometa` command.

## Requirements

- Python 3.14 (3.14.2 or newer, below 3.15), managed with [uv](https://docs.astral.sh/uv/).
- macOS on Apple silicon (arm64) or Linux on x86_64. The lock file resolves for these two platforms only.
- For CUDA on Linux: an NVIDIA driver r580 or newer. The PyPI `torch` wheels bundle CUDA 13.0, so you don't need a separate CUDA toolkit.

All reported metrics, latencies and VRAM figures come from an RTX 3090 on Linux. Track B on cached features also runs on a laptop CPU or on Apple silicon. See [reproducibility](reproducibility.md).

## Install

```bash
uv sync                 # package and dev tools
uv sync --extra gui     # adds Streamlit and Rerun for the demo and inspection
uv run pre-commit install   # once, after cloning
```

The `gui` extra holds `streamlit` and `rerun-sdk`. The `docs` dependency group holds Sphinx and its extensions; `uv run --group docs ...` installs it on demand.

## Licence and access checklist

- **TabPFN-3.5.** The weights are non-commercial (`tabpfn-3-5-license-v1.0`). Accept the licence on [platform.priorlabs.ai](https://platform.priorlabs.ai). Then set `TABPFN_TOKEN`, `TABPFN_NO_BROWSER=1` (headless runs) and `TABPFN_DISABLE_TELEMETRY=1`. Weights download only when they are missing from the cache.
- **DINOv3.** The default source is the gated `facebook/dinov3-*` repositories on Hugging Face, under the DINOv3 License. Request access and set `HF_TOKEN`.
  - Without access, `anometa extract --backend auto` (the default) falls back to timm's ungated copies of the same weights, with a warning.
  - The backend is part of each experiment. Configs that use timm features need `encoder_backend: timm`, and timm features have their own cache tag. A run never mixes the two backends.
  - The `parity` command measures the difference between the two backends. It is within noise; see [reproducibility](reproducibility.md#dinov3-backend-parity).
- **SigLIP2.** Apache-2.0 and ungated.
- **MVTec AD 2.** CC BY-NC-SA 4.0, non-commercial. `anometa download` fetches it from the official links. The data is never redistributed.

Each run's manifest records the licences of the models and data it used.

## Download the data

```bash
uv run anometa download --scenario vial   # one scenario
uv run anometa download                   # all 8 scenarios
```

- The command downloads one `.tar.gz` archive per scenario into `data/ad2/`. The links live in `configs/data/ad2.yaml`.
- Interrupted downloads resume with HTTP range requests.
- MVTec publishes no checksums. The SHA-256 of each archive is pinned in `configs/data/sha256sums.txt`. A download that doesn't match its pinned hash fails.
- Archives are extracted with tarfile's `data` filter. `--keep-archive` keeps the archive after extraction.
- The full dataset is about 30.4 GB. Vial (0.77 GB) is the smallest scenario and a good smoke test; Fabric (10 GB) is the largest.

Then build the dev/lock splits and extract features:

```bash
uv run anometa split
uv run anometa extract --encoder dinov3_l
```

The committed split files in `splits/` are never overwritten. See [protocol](protocol.md) for how the split works.

## Command reference

| stage | command |
|---|---|
| download AD2 (hash-pinned) | `uv run anometa download --scenario vial` (omit `--scenario` for all 8) |
| dev/lock splits | `uv run anometa split` |
| extract features | `uv run anometa extract --encoder dinov3_l` |
| one experiment | `uv run anometa run configs/experiments/smoke_vial.yaml --set k=5 classifier=tabpfn` |
| Track A | `uv run anometa run configs/experiments/tracka_patchcore.yaml` |
| frozen grid | `uv run anometa grid configs/search/grid.yaml` |
| Optuna NSGA-II / TPE | `uv run anometa optuna nsga2 --trials 200`, `uv run anometa optuna tpe --trials 60 --seed 0` |
| TabPFN-surrogate BO | `uv run anometa bo --trials 60 --seed 0` |
| report | `uv run anometa report --split dev` |
| lock evaluation | `uv run anometa lock configs/frozen` |
| backend parity | `uv run anometa parity --encoder dinov3_l` |
| Rerun recording | `uv run anometa inspect <run-id>` |
| few-shot demo | `uv run anometa demo` |

`uv run anometa <command> --help` lists every flag. The main ones:

- `download`: `--scenario` (repeatable; default all 8), `--keep-archive`.
- `split`: indexes every downloaded scenario and writes `splits/<scenario>.csv` if it doesn't exist yet.
- `extract`: `--encoder dinov3_s|dinov3_l|siglip2` (required), `--scenario` (repeatable; default every downloaded scenario), `--device auto|cuda|mps|cpu`, `--backend auto|transformers|timm`.
- `run <config>`: runs one YAML config through `run_experiment`. `--set key=value` overrides config fields; see below. It refuses configs with `split: lock`.
- `grid <spec>`: runs every config of a grid spec. `--scenario` restricts it; `--dry-run` prints the config count only.
- `optuna nsga2|tpe`: `--trials` (required), `--k` (default 2), `--seed` (default 0), `--storage` (default `sqlite:///artifacts/optuna.db`), `--scenario`.
- `bo`: `--trials` (required), `--k` (default 2), `--seed` (default 0), `--device` (where the TabPFN surrogate runs), `--scenario`.
- `report`: `--split dev|lock` (default `dev`). Writes `reports/<split>/results.md` and its figures.
- `lock <frozen_dir>`: evaluates every frozen config once on the lock split and writes `artifacts/lock_summary.csv`. A config whose lock run already exists is reported as already evaluated and never rerun.
- `parity`: `--encoder dinov3_s|dinov3_l` (required), `--scenario` (default `vial`), `--n` (images compared, default 32), `--device`.
- `inspect <run-id>`: writes `artifacts/<run-id>/inspect.rrd`. `--max-images` (default 50).
- `demo`: launches the Streamlit app. Needs the `gui` extra.
- `reference-check`: compares `TabPFNWithImages` from tabpfn-extensions with Track B on one scenario (`--scenario`, default `vial`; `--k`, default 5). It is a sanity check of the feature pipeline.

The search commands only build dev-split configs. See [search](dev-search.md).

### Overriding config fields

`--set key=value` takes one or more pairs and can repeat. Each value is parsed as YAML, so `5` is an integer, `knn` a string and `[0, 1]` a list. The value `null` removes the field, so it falls back to its default. It does not set the field to `None`.

```bash
uv run anometa run configs/experiments/smoke_vial.yaml --set k=5 classifier=tabpfn seeds=[0,1,2]
```

### Running seeds in parallel

Track B fits one seed at a time by default. Two environment variables change that:

- `ANOMETA_SEED_WORKERS=<n>` fits n seeds at once (default 1).
- `ANOMETA_SEED_EXECUTOR=thread|process` picks the pool (default `thread`). MPS needs `process`.

These are execution settings. They don't change run ids, and the manifest's `hardware` block records them. Concurrent seeds inflate the per-seed latencies, so reported latencies come from serial runs.

## Development checks

```bash
uv run ruff format --check
uv run ruff check
uv run pyrefly check
uv run pytest
uv run pytest -m data          # checks against the real data in data/ad2
```

By default, pytest skips tests marked `data` (needs `data/ad2`), `models` (needs Hugging Face or TabPFN weights), `gpu` (needs a CUDA device) and `api` (calls the Prior Labs TabPFN API). Select them with `-m`.

## Build these docs

```bash
uv run --group docs sphinx-build -W docs docs/_build/html
```

`-W` turns warnings into errors. Open `docs/_build/html/index.html` to read the result.
