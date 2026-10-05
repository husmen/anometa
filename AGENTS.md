# AGENTS.md

## Status

Implemented: Milestones 1-2 (data, splits, `run_experiment`, encoders, feature cache, Track B, metrics), pixel metrics, Track A, search (grid, Optuna, TabPFN-BO, all restrictable with `--scenario`), the cached Thinking scorer, the Streamlit demo, the report builder (with paired bootstrap comparisons) and Rerun inspection (Tasks 1-24), the `lock` command (Task 25 Steps 1-4) and the DINOv3 backend `parity` check (Task 27). Cross-hardware reproduction on the M4 Pro and a CPU-only Ryzen is done (Task 29). Post-freeze studies: the lighting-adaptation study (track `L`, `LightingConfig`, protocol in PLAN_1 § Post-freeze study) and a dev-only Track C prototype (scratch script on the host, results in the run log). Encoders run in bf16 on CUDA (fp16 overflows dinov3_l). On the 3090, all [3090] dev runs, the serial latency runs and the dev report are done on all 8 scenarios (results in `docs/plans/PLAN_1_RUN_LOG.md`). The configuration is frozen (`configs/frozen/`) and the single lock evaluation is done (`reports/lock/results.md`); never rerun or change lock runs. Pending: the Hugging Face upload (Task 25 Step 7, needs the user's approval) and Task 26 (README review, fresh-clone check, submission). Task 21 Thinking was declined for the lock. Plan checkboxes in `docs/plans/PLAN_1_IMPLEMENTATION.md` are the progress record. The active plan is `docs/plans/PLAN_1.md` (TabPFN-3.5 hackathon scope, deadline 2026-10-06), with task-by-task steps in `docs/plans/PLAN_1_IMPLEMENTATION.md`; `docs/plans/PLAN_0.md` is the long-term vision. Read PLAN_1 (including its split decision), `docs/CONTEXT.md` (glossary), and `docs/plans/PLAN_1_DIAGRAMS.md` before implementing.

Dev commands:

```bash
uv sync                        # install (extra `gui` for Streamlit/Rerun: uv sync --extra gui)
uv run ruff format --check
uv run ruff check
uv run pyrefly check
uv run pytest
uv run pre-commit install      # once, after cloning
uv run anometa download --scenario vial   # AD2 archives into data/ad2 (hash-pinned)
uv run pytest -m data          # checks against the real data
uv run anometa demo            # Streamlit few-shot demo
```

Track B seeds run serially by default. `ANOMETA_SEED_WORKERS=<n>` fits n seeds at once, and `ANOMETA_SEED_EXECUTOR=thread|process` picks the pool (default `thread`; MPS needs `process`). These are execution settings: run ids don't change, the manifest's `hardware` block records them, and concurrent seeds inflate the per-seed latencies, so take reported latencies from serial runs.

Linux/CUDA host: plain `uv sync` — PyPI `torch` wheels bundle CUDA, so no separate CUDA toolkit install is needed; the lock bundles CUDA 13.0, so the host's NVIDIA driver must be r580 or newer. The RTX 3090 host is canonical for all reported metrics, latencies and VRAM.

## Project

Anometa tests whether TabPFN-3.5 on frozen vision-foundation-model features (DINOv3, SigLIP2) works for few-shot industrial anomaly detection. Primary benchmark: MVTec AD 2 (8 scenarios, defect-free train/validation, labelled public test, unlabelled private test). Submitted to the Prior Labs TabPFN-3.5 hackathon (`docs/HACKATHON.md`): Apache-2.0, non-commercial, strict reproducibility.

## Planned architecture (from PLAN_1)

- Python 3.14 managed with `uv`. Strict typing with pyrefly (`preset = "strict"`); lint and format with ruff; both run in pre-commit. Package and import name: `anometa` under `src/anometa/`. YAML configs under `configs/`, parsed into pydantic models. GUI extra `anometa[gui]`: Streamlit for control and the few-shot demo, Rerun for inspection.
- Two separate tracks. Keep them distinct in code and reporting:
  - Track A: unsupervised baselines (PatchCore, EfficientAD-S via anomalib).
  - Track B: supervised few-shot adaptation (frozen VFM, PCA, TabPFN-3.5 and controls).
- One experiment API is the core invariant. CLI, Streamlit and the search layer (grid, Optuna, TabPFN-surrogate Bayesian optimization) call `run_experiment(ExperimentConfig) -> ExperimentResult`. Metrics are a dict because tracks report different sets. Keep that interface stable.
- Artifact-first runs: each run writes `artifacts/<exp-id>/` with `manifest.json` (git commit, dataset hash, split hash, model revision, licences, config hash, seed, hardware, versions), `metrics.json`, `predictions.parquet`. Features cached separately. `.rrd` only on demand.

## Guardrails against benchmark leakage

- Never select features, tune hyperparameters, or change splits based on lock-split results.
- Optimizers and search agents must never see lock-split labels or modify evaluation code.
- Order is fixed: dev-split experiments, then freeze configuration, then a single lock-split evaluation.
- Few-shot anomalies are sampled from regular-lit dev images only.
- Record every experiment, including failed ones.
