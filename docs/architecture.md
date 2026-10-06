# Architecture

Anometa is one Python package, `anometa`, under `src/anometa/`. Every experiment goes through one function and writes one artifact directory.

## System overview

One experiment API: every entry point builds an `ExperimentConfig` and calls `run_experiment`, which dispatches on `track` and writes one artifact directory per run.

```mermaid
flowchart LR
    yaml["configs/*.yaml"] --> cfg["ExperimentConfig<br/>TrackAConfig or TrackBConfig"]
    cli["anometa CLI"] --> run
    search["search<br/>grid · Optuna · TabPFN-BO"] --> run
    runtab["Streamlit Run tab"] --> run
    cfg --> run["run_experiment"]
    run -->|"track = A"| ta["Track A<br/>PatchCore · EfficientAD-S<br/>patch distance map"]
    run -->|"track = B"| tb["Track B<br/>cached features → PCA →<br/>TabPFN-3.5 / Fast / Thinking<br/>logreg · kNN · one-class"]
    ta --> art[("artifacts/run-id/<br/>manifest.json · metrics.json<br/>predictions.parquet · maps.npz")]
    tb --> art
    run --> res["ExperimentResult<br/>metrics dict"]
    art --> report["anometa report"]
    art --> rrd["Rerun .rrd"]
    art --> view["Streamlit results tab"]
```

## The experiment API

`run_experiment(ExperimentConfig) -> ExperimentResult` is the core invariant. The CLI, the Streamlit Run tab and the search layer (grid, Optuna, TabPFN-surrogate Bayesian optimisation) all call it. Nothing else runs an experiment.

- `RUNNERS` in `anometa.experiment` maps each track to its runner: `A` to `run_track_a`, `B` to `run_track_b`, and `L` to `run_lighting` (the post-freeze lighting study). Entries are import strings, resolved on first use, so importing `anometa.experiment` doesn't import a track's dependencies.
- A runner returns a `TrackOutput`: metrics, per-image predictions, model revisions, licences and extra files.
- `ExperimentResult` holds the run id, the config hash, the status (`ok` or `failed`), the metrics, the artifact directory and the error text of a failed run.
- Metrics are a `dict[str, float]`, because the tracks report different metric sets. See [metrics](metrics.md).

Metric keys follow one scheme:

- `<metric>`: the mean over scenarios and seeds.
- `<scenario>/<metric>`: the value for one scenario.
- `gap_<metric>`: the robustness gap, regular minus shifted lighting.
- `n_nan_<metric>`: the number of NaN values (Track B).
- `fit_latency_ms`, `predict_latency_ms` (Track B medians per scenario and seed) and `peak_vram_mb` (CUDA only).

Bootstrap intervals are not part of a run. `anometa report` computes them from `predictions.parquet`, so search trials stay fast.

### `run_experiment` lifecycle

Resume, failure recording and the lock guard.

```mermaid
flowchart TD
    a["run_experiment(cfg)"] --> b["run_id = name + first 12 hex of config hash"]
    b --> c{"cfg.split is lock?"}
    c -->|"yes"| d{"artifacts/run_id exists?"}
    d -->|"yes"| e["raise LockAlreadyEvaluatedError"]
    d -->|"no"| g["resolve runner for track · run it"]
    c -->|"no"| f{"manifest status is ok?"}
    f -->|"yes"| h["return result read from disk"]
    f -->|"no"| g
    g --> i{"runner raised?"}
    i -->|"no"| j["write manifest · metrics · predictions · extra files<br/>status ok"]
    i -->|"yes"| k["write manifest with traceback<br/>status failed"]
    j --> l["append artifacts/runs.jsonl"]
    k --> l
    l --> r["return ExperimentResult"]
```

- **Lock guard.** A lock-split run refuses to start if any run directory with the same config hash exists, under any name and whatever its status. Deleting that directory is a deliberate, manual act. Only `anometa lock` builds lock configs; `anometa run` and the Streamlit Run tab refuse them, and the search layer builds dev configs only. See [protocol](protocol.md).
- **Dev-run resume.** A dev run whose manifest says `ok`, and whose `metrics.json` and `predictions.parquet` exist, is read back from disk. An interrupted grid or search therefore resumes where it stopped.
- **Track A resume inside a run.** Track A saves each scenario's results to `artifacts/<run-id>/scenarios/<scenario>/` as soon as they are computed. A rerun loads the finished scenarios and starts at the first unfinished one. A resumed run's `peak_vram_mb` covers only the scenarios computed in that process.
- **Failures.** A runner exception is caught and recorded as a failed run, with the traceback in the manifest. `KeyboardInterrupt` still stops the process.

## Configs

Experiment configs are YAML files under `configs/`, parsed into pydantic models.

- `ExperimentConfig` is a discriminated union on `track`: `TrackAConfig`, `TrackBConfig` or `LightingConfig`. All three are frozen and reject unknown fields (`extra="forbid"`).
- Shared fields: `name`, `split` (`dev` or `lock`), `scenarios` (default all 8), `seeds`, `device` (`auto`, `cuda`, `mps` or `cpu`) and `paths`.
- Validators reject combinations that can't run. For example, a one-class classifier requires `k: 0` and a single seed, and `encoder_backend: timm` requires a DINOv3 encoder.
- **Config hash.** The SHA-256 of the config's canonical JSON dump (sorted keys), without `name` and `paths`. Renaming a run or moving its files doesn't change the hash; every other field does.
- **Run id.** `<name>-<first 12 hex characters of the hash>`. Two configs that differ only in `name` share the hash but get different run ids.
- `configs/data/` holds the AD2 links and pinned archive hashes, `configs/experiments/` single experiments, `configs/search/` the grid spec and `configs/frozen/` the frozen configs for the lock evaluation.

### Core types

Configs are frozen pydantic models; `ExperimentConfig` is a discriminated union on `track`. Protocols mark the swap points (encoders, scorers).

```mermaid
classDiagram
    direction LR
    class Paths {
        +Path data
        +Path cache
        +Path artifacts
        +Path splits
        +Path configs
    }
    class ConfigBase {
        <<abstract>>
        +str name
        +str split
        +tuple~Scenario~ scenarios
        +tuple~int~ seeds
        +str device
        +Paths paths
    }
    class TrackAConfig {
        +str track
        +str model
        +str encoder
        +str encoder_backend
        +tuple~int~ image_size
        +int max_steps
    }
    class TrackBConfig {
        +str track
        +str encoder
        +tuple~str~ features
        +int pca_dim
        +str classifier
        +dict classifier_params
        +int k
        +str shot_lighting
        +str encoder_backend
    }
    class ExperimentConfig {
        <<union>>
    }
    class experiment {
        <<module>>
        +run_experiment(cfg) ExperimentResult
    }
    class ExperimentResult {
        +str run_id
        +str config_hash
        +str status
        +dict metrics
        +Path artifact_dir
        +str error
    }
    class TrackOutput {
        +dict metrics
        +DataFrame predictions
        +dict model_revisions
        +dict licences
        +dict extra_files
    }
    class Encoder {
        <<protocol>>
        +str name
        +str backend
        +int dim
        +str resolution_tag
        +dict revisions
        +encode(image) Encoded
    }
    class Encoded {
        +Tensor cls
        +Tensor patches
    }
    class Features {
        +NDArray image_id
        +NDArray cls
        +NDArray mean_patch
        +NDArray novelty
        +NDArray distmap
        +rows(ids) NDArray
    }
    class Scorer {
        <<protocol>>
        +bool probabilistic
        +fit(X, y) Scorer
        +anomaly_score(X) NDArray
    }
    ConfigBase <|-- TrackAConfig
    ConfigBase <|-- TrackBConfig
    ConfigBase *-- Paths
    ExperimentConfig <|.. TrackAConfig : track A
    ExperimentConfig <|.. TrackBConfig : track B
    experiment ..> ExperimentConfig : consumes
    experiment ..> TrackOutput : runner returns
    experiment ..> ExperimentResult : returns
    Encoder ..> Encoded : returns
    Features ..> Encoder : extracted with
    TrackBConfig ..> Scorer : classifier builds
```

## Artifact-first runs

Every run writes `artifacts/<run-id>/`:

- `manifest.json`: `run_id`, the full `config`, `config_hash`, `status`, `error`, `git_commit`, `git_dirty` (runs from before the history cleanup also carry `git_commit_original` and `git_commit_mapping`, see [provenance](provenance.md)), `dataset_hash` (from the pinned archive hashes), `split_hash` (from the split files), `model_revisions`, `licences`, `seeds`, `hardware`, `versions` (Python and key packages), `started_at` and `duration_s`.
- `metrics.json`: the metrics dict.
- `predictions.parquet`: one row per (scenario, seed, image) with `scenario, seed, image_id, scene_id, label, lighting, score`, plus `score_balanced` for Track B.
- `maps.npz` (Track A): float16 anomaly maps at model resolution, keyed `<scenario>/<image_id>`.
- `inspect.rrd`: only on demand, written by `anometa inspect`. See [dashboard and inspection](dashboard.md).

The manifest's `hardware` block records the platform, CPU count, resolved device, seed pool settings and, on CUDA, the GPU name and VRAM.

The manifest is written last, through a temporary file and an atomic rename. A run directory with a manifest is therefore always complete.

Every attempt, successful or failed, is appended to the run log `artifacts/runs.jsonl`. Data (`data/ad2/`), caches (`cache/`) and runs (`artifacts/`) are not committed. `configs/`, `splits/` and the generated `reports/` are.

## Feature cache

Feature extraction is the only step that needs a GPU. Everything downstream reads the cache.

- Path: `cache/features/<encoder>-<tag>/<scenario>-<archive sha256[:12]>.npz`.
- Tags: `r512` (DINOv3 via transformers, 512-pixel short side), `r512-timm` (DINOv3 via the timm fallback) and `p1024` (SigLIP2, up to 1,024 patches).
- Arrays: `image_id`, `cls` (N×D), `mean_patch` (N×D), `novelty` (N×2: the maximum and the top-1% mean of nearest-neighbour patch distances) and `distmap` (N×h×w, float16, a per-patch distance grid).
- Patch tokens are not stored. The novelty features and distance maps are computed from them during extraction.
- Encoders run in bfloat16 on CUDA and float32 elsewhere. Features are stored as float32.

## Package layout

```text
src/anometa/
├── config.py            Scenario, Paths, TrackAConfig, TrackBConfig, LightingConfig, config_hash, run_id
├── experiment.py        run_experiment, ExperimentResult, RUNNERS, lock guard
├── artifacts.py         TrackOutput, manifest capture, run directory writes and reads, runs.jsonl
├── data/
│   ├── ad2.py           scenario index, filename parsing, RGB loading
│   ├── download.py      resumable download, sha256 pin and verify, safe extract
│   └── splits.py        dev/lock split, split hash, few-shot sampling, evaluation rows
├── features/
│   ├── encoders.py      Encoder protocol, DINOv3 (transformers, timm fallback) and SigLIP2
│   ├── novelty.py       nearest-neighbour patch distances, novelty statistics, distance map
│   ├── extract.py       feature extraction and cache
│   └── parity.py        DINOv3 transformers versus timm comparison and speed benchmark
├── metrics/
│   ├── image.py         AUROC, AUPRC, NLL, ECE, Brier, prior correction
│   ├── pixel.py         AU-PRO, SegF1, ClassF1
│   ├── _pro_reference.py  vendored MVTec AD reference AU-PRO
│   └── aggregate.py     metrics dict from predictions, robustness gap, bootstrap intervals
├── trackb/
│   ├── classifiers.py   TabPFN and scikit-learn scorers, one-class controls
│   ├── pipeline.py      run_track_b
│   ├── lighting.py      run_lighting, the post-freeze lighting study
│   └── reference.py     TabPFNWithImages reference check
├── tracka/
│   ├── anomalib_models.py  PatchCore and EfficientAD-S fit and anomaly maps
│   └── pipeline.py      run_track_a
├── search/
│   ├── space.py         search space, unit-cube decoding, params to TrackBConfig
│   ├── grid.py          grid expansion and runner
│   ├── optuna_search.py NSGA-II and TPE studies
│   └── tabpfn_bo.py     TabPFN-surrogate Bayesian optimisation loop
├── report.py            results tables and figures from artifacts
├── cli.py               the anometa entry point
├── rerun_log.py         Rerun .rrd writer (extra anometa[dashboard])
└── dashboard/           extra anometa[dashboard]
    └── app.py           Streamlit dashboard
```

See [tracks](tracks.md) for what Track A and Track B compute, and the [API reference](api/index.md) for every module.

### Package dependencies

Arrows point from importer to imported module. `experiment` resolves the track runners lazily, so importing it stays light.

```mermaid
flowchart LR
    cli["cli"]
    report["report"]
    subgraph core["core"]
        config["config"]
        artifacts["artifacts"]
        experiment["experiment"]
    end
    subgraph datapkg["data"]
        ad2["ad2"]
        download["download"]
        splits["splits"]
    end
    subgraph featpkg["features"]
        encoders["encoders"]
        novelty["novelty"]
        extract["extract"]
        parity["parity (optional)"]
    end
    subgraph metricspkg["metrics"]
        image["image"]
        aggregate["aggregate"]
        pixel["pixel"]
        proref["_pro_reference"]
    end
    subgraph trackbpkg["trackb"]
        classifiers["classifiers"]
        tbpipe["pipeline"]
        reference["reference"]
    end
    subgraph trackapkg["tracka"]
        anomalib_models["anomalib_models"]
        tapipe["pipeline"]
    end
    subgraph searchpkg["search"]
        space["space"]
        grid["grid"]
        optuna_search["optuna_search"]
        tabpfn_bo["tabpfn_bo"]
    end
    subgraph dashpkg["dashboard (extra)"]
        app["app"]
    end
    rerun_log["rerun_log (extra)"]
    experiment --> config & artifacts
    experiment -.->|"lazy"| tbpipe & tapipe
    artifacts --> config & splits & download
    ad2 --> config
    download --> config
    splits --> ad2
    encoders --> config
    novelty --> encoders
    extract --> ad2 & encoders & novelty & download
    parity --> encoders & extract & ad2 & aggregate & artifacts
    aggregate --> image
    pixel --> image & proref
    classifiers --> config
    tbpipe --> extract & splits & classifiers & aggregate & artifacts
    reference --> tbpipe
    tapipe --> anomalib_models & extract & splits & pixel & aggregate & artifacts
    space --> config
    grid --> space & experiment
    optuna_search --> space & experiment
    tabpfn_bo --> optuna_search & experiment
    report --> aggregate & artifacts
    app --> tbpipe & extract & splits & experiment
    rerun_log --> extract & ad2 & artifacts
    cli --> experiment & download & splits & extract & grid & optuna_search & tabpfn_bo & report & reference & app & rerun_log & parity
```
