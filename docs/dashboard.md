# Dashboard and inspection

The `dashboard` extra adds two tools: a Streamlit dashboard for few-shot detection and Rerun recordings for deep inspection of one run.

```bash
uv sync --extra dashboard
```

## Few-shot dashboard

```bash
uv run anometa demo
```

The command launches the Streamlit app in `anometa.dashboard.app`. It needs cached features, so run `anometa extract` first. It reads only cached features and dev images, never lock data.

The sidebar picks a scenario (Vial by default), an encoder and a backend among the cached features, a classifier (TabPFN-3.5-Fast by default, or TabPFN-3.5), which lighting to show (`regular`, `shifted` or `all`) and the anomaly map: the cached DINOv3 patch distance map, or the maps of any finished dev Track A run (PatchCore or EfficientAD-S) found under `artifacts/`. A toggle hides the maps.

The dashboard has three tabs.

### Detect

The page puts images and anomaly maps first.

1. **Normal parts:** six defect-free training images, each next to its anomaly map.
2. **Pick defect examples:** the dev images of the scenario as an unlabelled grid, 18 per page. Tick the ones that look defective; they become the k few-shot shots. Labels and maps stay hidden here, so the picks are by eye.
3. **Results:** the classifier refits on the train normals plus your picks and scores the rest of the dev images. Four numbers show the number of picks, the dev AUROC under regular and under shifted lighting, and the fit-and-score time. Nine cards show the most (or least) anomalous images. Each card shows the image with its ground-truth defect outlined in red, its anomaly map, the defect probability and the true label.
4. **Inspect one image:** any scored image, large, next to every available anomaly map.

All maps of one source share one colour scale (the 1st to 99.5th percentile of the scenario's maps), so the same colour means the same score on every image.

Images from the scenes you picked are left out of the scoring, as in Track B. The demo fits on `cls` and `mean_patch` features with PCA to 16 dimensions; these are not exposed as controls.

The labelling loop calls the Track B building blocks directly, not `run_experiment`, because user-picked shots are not an `ExperimentConfig`. It uses the same fit-and-score helper as `run_track_b`, so the two paths can't diverge. Interactive picks are not recorded as runs.

Loaded features are cached with `st.cache_resource`, once per scenario, encoder and backend.

#### Demo labelling loop

The labelling loop reuses the Track B building blocks on cached features; the Run tab goes through `run_experiment` instead.

```mermaid
sequenceDiagram
    actor U as User
    participant St as Streamlit app
    participant C as st.cache_resource
    participant P as Track B building blocks
    participant T as TabPFN-3.5-Fast
    U->>St: pick scenario and encoder
    St->>C: load_features and load_split (dev rows only)
    C-->>St: Features and split rows
    U->>St: select k defect images in the gallery
    St->>P: fit_pca on train normals, design_matrix
    P-->>St: X_fit and X_eval
    St->>T: fit(train normals + picks)
    St->>T: anomaly_score(dev rows minus picked scenes)
    T-->>St: P(anomalous)
    St->>St: prior_correct, AUROC regular and shifted, gap, refit time
    St-->>U: ranked gallery with balanced probabilities
    U->>St: toggle lighting
    St-->>U: filtered ranking and gap
```

### Run

The Run tab launches one config from `configs/experiments/` through `run_experiment`, like `anometa run`. Overrides go in a text box, one `key=value` per line, with the same rules as `--set`. The tab refuses lock-split configs and shows the run id, status and metrics.

### Results

The Results tab shows the figures in `reports/dev/figures/`. Run `uv run anometa report --split dev` first.

## Rerun inspection

Rerun recordings (`.rrd`) are written only on demand:

```bash
uv run anometa inspect <run-id>
rerun artifacts/<run-id>/inspect.rrd
```

`anometa inspect` reads a finished run directory and writes `inspect.rrd` next to its artifacts. It prints the path and the command that opens it. The recording holds, on an `image` timeline:

- One step per evaluation image of the run's first seed, highest score first, up to `--max-images` (default 50).
- The RGB image, downscaled to at most 1,024 pixels on the longest side.
- Its ground-truth mask as a segmentation image.
- For Track A, the anomaly map as a 50% opaque overlay, with one colour range for all logged maps.
- The image's score.

Once per run, it also logs a 3-component PCA of the evaluation images' CLS features for each scenario, coloured by label and labelled with the lighting.

Open the file in the Rerun desktop viewer (`rerun`, from `rerun-sdk`).
