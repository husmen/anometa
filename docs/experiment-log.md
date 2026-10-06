# Experiment log

This log records every experiment of the hackathon phase, including failed and dropped ones, oldest first after the summary. Results are on the dev split unless a section says lock. Reported metrics, latencies and VRAM come from the RTX 3090 host unless a section names another machine.

```{note}
Commit SHAs in this log refer to the history before it was rewritten on 2026-10-06. See [provenance](provenance.md) for the old-to-new map.
```

## Summary

The single lock evaluation ran on the freeze commit `68f5301` on 2026-10-03: 28/28 runs `ok`. Full numbers are on the [results](results.md) page.

- **TabPFN-3.5 vs logistic regression, AUROC:** +0.078 [+0.009, +0.146] at k=1, significant before correction but not after a Bonferroni correction over the 3 budgets. About +0.04 at k = 2 and 5, within noise. TabPFN 0.761 / 0.778 / 0.789, logistic regression 0.683 / 0.740 / 0.745.
- **Calibration:** TabPFN wins clearly at every k (balanced NLL 0.65 vs 1.25 at k=1).
- **Label-free controls:** a tie (Mahalanobis 0.766, TabPFN outlier score 0.778). kNN loses badly.
- **TabPFN-3.5-Fast matches 3.5.** `n_normals: 32` did not replicate (−0.02 to −0.03 AUROC).
- **Track A:** the training-free DINOv3-L patch distance map is the best of our three pixel methods (AU-PRO@0.05 0.385, against PatchCore 0.222 and EfficientAD-S 0.182). PatchCore and EfficientAD-S are untuned and score below the AD2 paper's numbers for the same methods (see the 2026-10-06 correction).
- The dev-split wall plugs story reversed on lock. It was scene-sampling noise.
- **Post-freeze, 2026-10-05:** lighting adaptation improves TabPFN's calibration (balanced NLL −0.05) but not its ranking (+0.002 AUROC, primary hypothesis not supported). The Track C prototype (dev only, all 8 scenarios) beats the DINOv3 distance map by +0.056 / +0.071 AU-PRO@0.05 at k = 2 / 5.
- **Post-freeze, 2026-10-06:** TabPFN-3.5-Thinking at k=2 gives the same AUROC and calibration as local TabPFN-3.5.

## Working rules

- Work slice-first: prove every step on Vial and Fruit Jelly, then run all 8 scenarios.
- No lock-split evaluation before the configuration is frozen.
- No Thinking API spend without explicit approval.
- The RTX 3090 host only runs code; it commits nothing.

## 2026-10-01: data, first fixes and the slice

### Data

- Pinned the official archive sha256 for all 8 scenarios (`6682afa`) and committed the splits for the remaining 7 (`6f0f9b1`). Every scenario has 15 defect scenes and 4–12 good scenes, and every scenario has `regular` among its lightings.
- Checked the local AD2 copy against the official archives. All 8 scenarios are byte-identical, file for file (Vial 1129 files to Walnuts 1176).

### Fixes found on the RTX 3090

- **DINOv3-L overflows to NaN in fp16 on CUDA.** All encoders now run in bf16 (`5580c03`), and features are upcast to float32 before caching (`064ce68`). The patch cosine of bf16 against fp32 is 0.977, and the CLS cosine is at least 0.99 (tested).
- **The EfficientAD checkpoint could not be pickled.** anomalib's max-steps progress bar holds a local class, so the progress bar is now off for EfficientAD (`61c5e9a`).
- **EfficientAD validated after every epoch,** about 240 passes per scenario. That turned a ~14 min fit into more than 1 h. It now validates once, after the last step, which also uses the final weights for the map quantiles (`6e9691f`).
- **AU-PRO computed the PRO curve twice** (for 0.05 and 0.30). It now computes the curve once and integrates it at both limits, with identical values (`1497f2c`).

### Seed pool and the first benchmark

- Added an optional seed-level pool: `ANOMETA_SEED_WORKERS` and `ANOMETA_SEED_EXECUTOR=thread|process` (`a0c6875`). These are execution settings: run ids do not change, and the manifest's `hardware` block records them. Threads are refused on MPS, because torch's MPS backend aborts under concurrent threads. Scores equal serial runs exactly (tested on CPU and CUDA).
- Benchmark 1, before the checkpoint cache: the pool gave at most 1.33× (process ×4), and process ×8 was slower than serial.
- The profile showed why: 47% of the run was `torch.Tensor.uniform_`. Every TabPFN `fit` rebuilt and randomly initialised the network before loading the checkpoint over it. That was 0.76 s of a 0.88 s fit.
- Fix (`a740377`): each process loads each checkpoint once and passes it as `ClassifierModelSpecs` / `RegressorModelSpecs`, TabPFN's own supported `model_path` input. This also covers the `tabpfn_outlier` regressor, which refits for every conditional. Exact equivalence with fresh loads is tested for both versions, two `n_estimators` settings and the regressor, on CPU and CUDA.

### Slice results (Vial and Fruit Jelly)

- The Python 3.14 gate passed on the RTX 3090. Reference check: ours 0.801 against the reference 0.677, passed.
- Budget runs: 14/14 `ok`. Grid: 366/366 `ok`. Best: TabPFN-3.5-Fast, k=5, DINOv3-L, CLS + mean patch, PCA 128, AUROC 0.937.
- Track A: PatchCore AU-PRO@0.05 0.457, distance map DINOv3-S 0.543, DINOv3-L 0.561.
- Open questions: Vial saturates (TabPFN 1.000). On Fruit Jelly, AUROC drops from k=2 to k=5 for TabPFN and logistic regression. Both are answered below.

## 2026-10-01 to 02: all 8 scenarios

### EfficientAD-S data loaders: a wrong diagnosis (reverted in `aee511e`)

- With per-epoch validation gone, a full Vial fit still took about 37 min. Data loader workers looked like they were respawning, so persistent workers were added (`990fff8`).
- Persistent workers then broke anomalib's end-of-training checkpoint. A follow-up (`f6368b1`) did not fix that.
- Measured over 3000 steps on Vial: default, `fork` and persistent loaders all train at **28.6 it/s**. Respawns were not the bottleneck, so both commits were reverted. A 300-step end-to-end run then passed.
- EfficientAD-S simply trains at about 29 it/s on this host: about 41 min per scenario for 70,000 steps, about 5.4 h for all 8.

### TabPFN checkpoint cache (`a740377`)

One slice configuration (TabPFN-3.5, DINOv3-L, CLS + mean patch, PCA 128, k=5, 20 fits):

| setting | run time | fit latency | scores vs pre-cache |
|---|---|---|---|
| serial, before cache | 25.6 s | 911 ms | – |
| **serial, cache** | **8.2 s (3.1×)** | **11 ms** | identical |
| threads ×2/×4/×8 | 8.9 / 10.5 / 13.7 s | 22–106 ms | identical |
| processes ×2/×4/×8 | 12.0 / 13.1 / 17.5 s | 20–85 ms | identical |

- Decision: **seeds stay serial by default.** With a fit at 11 ms, a pool only adds overhead. The pool stays available as an opt-in.
- `tabpfn_outlier` (SigLIP2, CLS + mean patch + novelty, PCA 16): **335.5 s → 72.2 s (4.6×)**, AUROC 0.8971 in both. The scores are not bit-identical to the pre-cache run (max difference 0.29 in log-density).
- Cause: the old per-fit rebuild drew random weights from the global torch RNG. `.outliers()` takes its permutations and noise from the same RNG, so removing the rebuild changes the stream it sees. The new scores are deterministic, and every full run uses the new code, so the slice outlier numbers are superseded.

### Slice close-out

- `anometa report --split dev` on the 430 slice runs took 13 s. Integers printed as floats (`k 1.000`); fixed in `aabdb5b`.
- `anometa inspect` wrote a 21 MB `.rrd` for the slice PatchCore run in 4.3 s.
- Thinking was not run. Even the cost estimate may contact Prior Labs' service, so it waited for approval.

### Run plan: two lanes on one GPU

- A serial queue with EfficientAD-S inside would take about 11.5 h. The runs were split into two lanes. Lane 1: feature extraction (3 encoders), budget runs, grid, PatchCore, the distance maps, then the searches. Lane 2: EfficientAD-S on all 8 scenarios, started after extraction.
- Trade-off: runs that overlapped share the GPU, so their `fit_s`, per-seed latencies and `peak_vram_mb` are not canonical. Quality metrics are unaffected. Canonical latencies come from a later serial rerun (see 2026-10-02).
- Later that night, lane 1's tail was reordered by value: after the grid came PatchCore and the distance maps, then TPE vs TabPFN-BO at seed 0, then NSGA-II, then the seed 1–2 replicates.

### Why Fruit Jelly AUROC drops at k=5

Slice budget runs, PCA 32:

| k | dev defect scenes left | eval images | TabPFN AUROC | per-seed range | logreg AUROC |
|---|---|---|---|---|---|
| 1 | 7 | 40 | 0.709 ± 0.103 | 0.55–0.82 | 0.497 |
| 2 | 6 | 36 | 0.725 ± 0.066 | 0.63–0.83 | 0.508 |
| 5 | 3 | 24 | 0.647 ± 0.125 | 0.37–0.79 | 0.392 |

- This is the protocol, not the model. Fruit Jelly has 8 dev defect scenes, and every shot's scene leaves the evaluation set. So k=5 is scored on only 3 defect scenes.
- Each k is scored on a different subset of scenes, so the k columns are not a like-for-like curve. This led to the fixed-row label budget table (2026-10-02).

### Why Vial saturates at AUROC 1.000

- The label-free controls reach 0.983 (Mahalanobis) and 0.995 (`tabpfn_outlier`) on Vial with no defect labels. Vial is easy for frozen global embeddings, so label leakage cannot explain the TabPFN result. Tests cover the leakage guards.
- Vial carries little signal for comparing classifiers.

### Budget runs (all 8 scenarios, DINOv3-S, CLS + mean patch + novelty, PCA 32)

| classifier | k=1 | k=2 | k=5 |
|---|---|---|---|
| tabpfn | 0.675 | 0.677 | 0.682 |
| tabpfn_fast | 0.677 | 0.678 | 0.687 |
| logreg | 0.662 | 0.687 | 0.700 |
| knn | 0.534 | 0.573 | 0.596 |
| mahalanobis (k=0) | 0.665 | | |
| tabpfn_outlier (k=0) | 0.666 | | |

- 14/14 `ok`, no `BudgetError` in any scenario.
- At this default configuration, few-shot labels barely beat the label-free controls.
- Can (TabPFN 0.40–0.45) and Wall Plugs (TabPFN 0.37, Mahalanobis 0.36) score **below 0.5 for every method**, including the one-class controls.
- Latency is not canonical (shared GPU): TabPFN fit 6 ms + predict 281 ms per (scenario, seed), Fast 86 ms, `tabpfn_outlier` 51 s per scenario.

### Why Can and Wall Plugs score below 0.5

- Wall Plugs: under every lighting, **including regular**, good test images score higher than defective ones. TabPFN gives 0.033 vs 0.011 under regular light, Mahalanobis 59.5 vs 17.8. This is not a lighting effect.
- Median Mahalanobis distance to the train normals (PCA 16 on CLS + mean patch, regular lighting):

| scenario | encoder | train | validation | test good | test bad |
|---|---|---|---|---|---|
| wallplugs | dinov3_s | 14.4 | 14.1 | **21.9** | 15.4 |
| can | siglip2 | 15.1 | 14.3 | **16.4** | 13.3 |
| fabric | siglip2 | 14.5 | 14.2 | **17.0** | 9.6 |
| rice | dinov3_s | 15.1 | 14.6 | 12.0 | 17.0 (normal order) |

- Validation matches train, but **test-good images sit further from train than the defects do** in Can, Wall Plugs and Fabric. The patch-novelty statistic barely separates them either (Wall Plugs 3.72 good vs 3.61 bad).
- Image sizes and modes are identical across splits, and the splits are consistent. No pipeline bug was found.
- Reading: AD2's test-good images carry normal variation that train lacks, while the defects are small and local, which global embeddings miss. This sets a ceiling for global-embedding few-shot methods on AD2 and argues for patch-level maps.

### Grid (366/366 `ok`, 3.5 h)

Best mean dev AUROC per classifier and k, over encoders, feature sets and PCA dimensions:

| classifier | k=1 | k=2 | k=5 |
|---|---|---|---|
| logreg | 0.707 | **0.744** | **0.775** (dinov3_l, cls+mean_patch+novelty, pca16) |
| tabpfn | 0.710 (dinov3_l novelty) | 0.715 (novelty) | 0.761 (dinov3_l, cls+mean_patch, pca128) |
| tabpfn_fast | 0.711 (novelty) | 0.720 (novelty) | 0.760 (dinov3_l, cls+mean_patch, pca64) |
| knn | 0.585 | 0.630 | 0.674 |
| mahalanobis (k=0) | 0.723 (dinov3_l novelty) | | |
| tabpfn_outlier (k=0) | 0.717 (dinov3_l novelty) | | |

Best at k=2 per encoder (logreg / TabPFN / Mahalanobis): DINOv3-L 0.744 / 0.715 / 0.723; SigLIP2 0.731 / 0.698 / 0.676; DINOv3-S 0.712 / 0.677 / 0.679.

- Point estimates only; the paired comparison (2026-10-02) supersedes this reading. On all 8 scenarios, the best TabPFN cell scores below the best logistic regression cell at k=2 and k=5. At k≤2 Mahalanobis scores above every TabPFN cell.
- The slice flattered TabPFN (0.937), because Vial saturates.
- DINOv3-L is the best encoder for every classifier. At small k, TabPFN does best on novelty-only features.
- Best per-scenario AUROC anywhere in the grid: Vial 1.000, Sheet Metal 1.000, Walnuts 1.000, Fruit Jelly 0.883, Fabric 0.837, Rice 0.825, Wall Plugs 0.696, Can 0.685.
- The grid runs TabPFN and logistic regression at default settings. The searches also tune `n_estimators` and `C`.

### Track A

| method | AU-PRO@0.05 | AU-PRO@0.30 | SegF1 | ClassF1 | image AUROC | wall time |
|---|---|---|---|---|---|---|
| **patch distance dinov3_l** (training-free) | **0.391** | **0.611** | **0.267** | **0.777** | **0.717** | 12 min |
| patch distance dinov3_s | 0.308 | 0.508 | 0.186 | 0.761 | 0.687 | 11 min |
| PatchCore (wide_resnet50, coreset 0.01) | 0.202 | 0.453 | 0.162 | 0.686 | 0.652 | 16 min (fit_s 37, 5.3 GB) |
| EfficientAD-S (70000 steps) | 0.160 | 0.319 | 0.131 | 0.757 | 0.641 | 8.8 h (median fit_s 4289, 2.7 GB) |

AU-PRO@0.05 per scenario:

- Distance map DINOv3-L: Vial 0.864, Rice 0.691, Sheet Metal 0.556, Walnuts 0.424, Fruit Jelly 0.258, Fabric 0.207, Can 0.073, Wall Plugs 0.056.
- PatchCore: Vial 0.529, Fruit Jelly 0.442, Walnuts 0.267, Rice 0.195, Sheet Metal 0.054, Fabric 0.064, Wall Plugs 0.051, Can 0.018.
- EfficientAD-S: Vial 0.734, Fruit Jelly 0.316, Walnuts 0.139, Sheet Metal 0.047, Wall Plugs 0.036, Can 0.010, Fabric 0.000, Rice 0.000.

Findings:

- The training-free DINOv3 patch distance map beats PatchCore on every pixel metric. PatchCore and EfficientAD-S are untuned (see the 2026-10-06 correction).
- **PatchCore also scores below 0.5 image AUROC on Can (0.41) and Wall Plugs (0.36).** The test-good shift is a property of AD2, not of our features or pipeline.
- The robustness gaps (regular minus shifted) are small for all three: AU-PRO@0.05 0.02–0.05, AUROC ±0.02.
- EfficientAD-S is the weakest pixel method. It collapses on Fabric and Rice (AU-PRO@0.05 0.000, AU-PRO@0.30 0.21 and 0.12) and is near chance on image AUROC there (0.52, 0.49). It beats PatchCore only on Vial (0.734 vs 0.529). Its image AUROC on Can (0.51) and Wall Plugs (0.58) is near chance.
- EfficientAD-S's per-scenario `gap_auroc` swings from −0.25 (Rice) to +0.32 (Fabric). That is noise around a weak model, not a lighting effect.
- PatchCore's and EfficientAD-S's `fit_s` and VRAM are not canonical (shared GPU). Four EfficientAD-S fits took about 86 min each while sharing the GPU, and the other four about 47 min each, mostly alone. The 29 it/s benchmark predicts about 41 min per fit on an idle GPU.

### Searches (k=2, all 8 scenarios)

| study | trials | best AUROC | best@10 | best@30 | best@60 |
|---|---|---|---|---|---|
| TPE seed 0 | 60 | 0.747 | 0.695 | 0.746 | 0.747 |
| TPE seed 1 | 60 | 0.747 | 0.712 | 0.712 | 0.747 |
| TPE seed 2 | 60 | 0.747 | 0.740 | 0.746 | 0.747 |
| TabPFN-BO seed 0 | 60 (crashed at 19, resumed) | 0.744 | 0.695 | 0.734 | 0.744 |
| TabPFN-BO seed 1 | 60 | 0.747 | 0.712 | 0.742 | 0.747 |
| TabPFN-BO seed 2 | 60 | 0.742 | 0.740 | 0.740 | 0.742 |
| NSGA-II | 200 | 0.748 (85 Pareto points) | | | |

- **Every study converges on logistic regression, DINOv3-L, CLS + mean patch + novelty, PCA 64 (C ≈ 0.001–0.01), AUROC about 0.747.** No search puts TabPFN at the top, which matches the grid.
- TPE and TabPFN-BO reach the same plateau, and BO does not converge faster. The search space has one dominant basin, so it cannot separate the two methods.
- **TabPFN-BO seed 0 crashed at trial 19** with `CUDA error: unspecified launch failure` in the TabPFN regressor surrogate, while sharing the GPU with EfficientAD-S. EfficientAD-S was unaffected. The study resumed from its saved trials and finished 60 trials in 337 s with no further errors.
- A logging bug in the run scripts made their exit codes meaningless. Each step was verified from its artifacts instead: 366 `ok` grid lines, Track A manifests `ok`, 60-trial study files.

### The logistic regression lead is almost entirely Wall Plugs

Best grid configuration per classifier (dev AUROC):

| config | mean, 8 scenarios | mean, 7 (no wallplugs) | wallplugs | fruit_jelly | fabric | can |
|---|---|---|---|---|---|---|
| k=2 tabpfn (dinov3_l novelty) | 0.715 | **0.769** | 0.338 | 0.702 | **0.718** | **0.475** |
| k=2 logreg (dinov3_l +novelty pca64) | **0.744** | 0.761 | **0.629** | **0.811** | 0.600 | 0.428 |
| k=5 tabpfn (dinov3_l pca128) | 0.761 | **0.808** | 0.431 | **0.862** | 0.665 | **0.553** |
| k=5 logreg (dinov3_l +novelty pca16) | **0.775** | 0.798 | **0.619** | 0.712 | **0.695** | 0.524 |

- Without Wall Plugs, TabPFN slightly leads at both k. On Wall Plugs it stays below chance (0.34–0.43), while logistic regression recovers to about 0.62.
- Hypothesis (tested on 2026-10-02): with 2–5 defects among about 300 normals, TabPFN's prediction stays close to a distance-from-normal score, which inherits the test-good shift.

### Per-scenario resume for Track A (`7830589`)

- Before this change, Track A held every scenario in memory and wrote everything at the end. A crash in the last EfficientAD-S scenario would have lost about 7 h.
- Now each scenario's predictions, pixel metrics, gaps, `fit_s`, model revisions, git commit and maps are written atomically to `artifacts/<run-id>/scenarios/<scenario>/`. An interrupted dev run resumes at the first unfinished scenario. Lock runs still never rerun.
- `peak_vram_mb` of a resumed run covers only the scenarios computed in that process.
- The EfficientAD-S run already in flight could not use it. It finished at 08:49 on 2026-10-02 (manifest `ok`, `git_dirty: false`, commit `aee511e`).

## 2026-10-02: comparisons, latency and the freeze

### Paired comparison: TabPFN against each control (`4d4dcec`, `ea41c90`)

The earlier gaps were point estimates, and per-run intervals cannot test a difference. `paired_bootstrap_diff` resamples the same seeds and scenes for both runs, so its interval covers the noise in the difference. Shots depend only on seed, scenario, k and shot lighting, so two classifiers at the same k score exactly the same rows. The function refuses unpaired runs.

Best TabPFN run minus each control's best run, dev AUROC, 95% paired bootstrap interval (1000 replicates):

| k | vs logreg | without wallplugs | vs mahalanobis | vs tabpfn_outlier | vs knn |
|---|---|---|---|---|---|
| 1 | +0.003 [−0.063, +0.067] | +0.040 [−0.031, +0.110] | −0.013 [−0.042, +0.014] | −0.008 [−0.036, +0.019] | **+0.125 [+0.070, +0.180]** |
| 2 | −0.033 [−0.097, +0.034] | +0.008 [−0.058, +0.076] | −0.001 [−0.032, +0.029] | +0.003 [−0.026, +0.030] | **+0.071 [+0.014, +0.134]** |
| 5 | −0.014 [−0.095, +0.059] | +0.011 [−0.063, +0.093] | +0.041 [−0.027, +0.110] | +0.049 [−0.015, +0.117] | **+0.087 [+0.020, +0.150]** |

Matched grid cells (27 per k: same encoder, features and PCA dimension, both at default settings, so no best-run selection), share of cells where TabPFN beats logistic regression: k=1 81% (mean +0.016), k=2 22% (−0.020), k=5 41% (−0.013). Without Wall Plugs: 96%, 44%, 67%.

- **TabPFN and logistic regression are statistically indistinguishable on dev at every k**, with or without Wall Plugs.
- TabPFN clearly beats kNN at every k.
- At k ≤ 2, TabPFN ties the label-free controls. At k=5 it leads them by about 0.04–0.05, but the intervals include zero.
- TabPFN-3.5 and 3.5-Fast are interchangeable (|diff| ≤ 0.006).

### TabPFN behaves like a novelty score

Spearman correlation between each run's scores and the best Mahalanobis scores on the same rows (DINOv3-L, default settings, matched configurations):

| k | features, PCA | TabPFN: wallplugs / other 7 | logreg: wallplugs / other 7 | wallplugs AUROC TabPFN / logreg |
|---|---|---|---|---|
| 2 | cls+mean_patch+novelty, 16 | 0.91 / 0.82 | −0.59 / 0.58 | 0.37 / 0.65 |
| 2 | cls+mean_patch+novelty, 64 | 0.86 / 0.81 | −0.47 / 0.53 | 0.35 / 0.63 |
| 2 | cls+mean_patch, 128 | 0.81 / 0.64 | −0.16 / 0.39 | 0.38 / 0.53 |
| 5 | cls+mean_patch+novelty, 16 | 0.86 / 0.79 | −0.82 / 0.55 | 0.38 / 0.62 |
| 5 | cls+mean_patch+novelty, 64 | 0.80 / 0.76 | −0.72 / 0.46 | 0.40 / 0.69 |
| 5 | cls+mean_patch, 128 | 0.77 / 0.57 | −0.33 / 0.29 | 0.43 / 0.61 |

- With 1–5 defects among about 300 normals, TabPFN's score stays close to a distance-from-normal score everywhere (ρ 0.57–0.91). Logistic regression fits a direction toward the few defects, which on Wall Plugs runs against novelty (ρ down to −0.82).
- Where novelty is informative (7 scenarios), that helps TabPFN slightly. On Wall Plugs, where test-good images are more novel than the defects, it hurts.
- The AD2 paper lists allowed variation that train may cover only partly ("Wall plugs can vary in their quantity and positions and may overlap"; the can "is randomly rotated and contains a glitter foil"; the fabric "pattern is variable and varies in orientation"). It does not say test-good images differ from train, so these lines are a plausible cause only.

### Tried and dropped: TabPFN + logistic regression rank ensemble

Averaging per-seed score ranks of TabPFN and logistic regression on the 27 matched cells per k (existing predictions, no new runs): mean AUROC k=1 0.660 (TabPFN 0.657, logreg 0.641), k=2 0.688 (0.670, 0.690), k=5 0.712 (0.698, 0.711). It beats both members in only 30–52% of cells. Its best cell (0.761 at k=5) is below the best logistic regression (0.775). No measurable gain, so no code was added.

### Label budget on fixed rows (`605d6e1`)

A secondary report section. Each classifier's best configuration at its largest k, with smaller k rescored on that run's evaluation rows (shots are nested):

| classifier (config) | k=0 | k=1 | k=2 | k=5 |
|---|---|---|---|---|
| tabpfn (dinov3_l, cls+mean_patch, pca128) | | 0.675 | 0.700 | 0.761 |
| tabpfn_fast (dinov3_l, cls+mean_patch, pca64) | | 0.700 | 0.722 | 0.760 |
| logreg (dinov3_l, cls+mean_patch+novelty, pca16) | | 0.703 | 0.750 | 0.775 |
| knn (dinov3_l, cls+mean_patch+novelty, pca64) | | 0.585 | 0.637 | 0.674 |
| mahalanobis (dinov3_l, novelty) | 0.721 | | | |
| tabpfn_outlier (dinov3_l, novelty) | 0.712 | | | |

- On the same scenes, every classifier gains steadily from k=1 to k=5. The non-monotone curves in the main table came from scoring each k on different scenes.
- On these rows, the label-free controls (0.71–0.72) match TabPFN at k=2 and are beaten only at k=5.

### Dev report

- `anometa report --split dev` took 192 s with 30 paired bootstraps.
- `load_runs` now keeps one run per config hash, so the latency reruns of the best configurations do not count twice in means.
- A rerun on the latest code: 191 s, 781 completed dev runs, paired numbers identical.

### Canonical latency (RTX 3090, serial seeds, nothing else on the GPU)

Best dev configurations, all 8 scenarios. Fit and predict are medians per (scenario, seed); predict covers one scenario's whole dev evaluation set.

| config | AUROC | fit | predict | peak VRAM | run wall time |
|---|---|---|---|---|---|
| tabpfn k=5, dinov3_l cls+mean_patch pca128 | 0.761 | 11.2 ms | 276 ms | 1106 MB | 27.0 s |
| tabpfn_fast k=5, dinov3_l cls+mean_patch pca64 | 0.760 | 5.7 ms | 70.6 ms | 470 MB | 9.2 s |
| logreg k=5, dinov3_l cls+mean_patch+novelty pca16 | 0.775 | 19.6 ms | 0.22 ms | – (CPU) | 4.5 s |
| mahalanobis k=0, dinov3_l novelty | 0.723 | 0.60 ms | 0.26 ms | – (CPU) | 0.6 s |
| tabpfn_outlier k=0, dinov3_l novelty | 0.717 | 0.16 ms | 776 ms | 876 MB | 8.8 s |

- Every latency rerun reproduces its grid AUROC exactly.
- TabPFN's fit is cheap because it is in-context: "fit" only stores the context, and the work happens at predict time.
- 3.5-Fast is 3.9× faster at predict time than 3.5, with the same AUROC.
- `tabpfn_outlier` on the novelty features takes 8.8 s for all 8 scenarios, against 72 s for the SigLIP2 PCA 16 configuration. Its cost grows with the number of features, because it refits a regressor for every conditional.
- Failed runs: passing `pca_dim=null` through `--set` deletes the key instead of setting `None`. Both one-class latency runs failed validation before starting and were rerun from a YAML file. The README now documents that `null` removes a field.

### Lock command (`1c86368`)

`anometa lock <frozen_dir>` runs every frozen configuration once. A configuration whose lock run exists is reported as "already evaluated" and never rerun. A failed run makes the command exit 1 after the others ran. It was committed ahead of the freeze, so the freeze commit holds only `configs/frozen/`. Nothing ran on the lock split at this point.

### Calibration: TabPFN is calibrated without tuning (`3826948`)

On the 27 matched grid cells per k (default settings for both):

| k | balanced NLL, TabPFN vs logreg | cells TabPFN better | balanced ECE, TabPFN vs logreg | cells TabPFN better |
|---|---|---|---|---|
| 1 | 0.728 vs 1.251 | 100% | 0.216 vs 0.394 | 100% |
| 2 | 0.702 vs 1.132 | 93% | 0.216 vs 0.341 | 96% |
| 5 | 0.705 vs 0.924 | 89% | 0.259 vs 0.268 | 56% |

TabPFN beats kNN on every cell and metric. 3.5 and 3.5-Fast are indistinguishable.

- A tuned logistic regression closes the gap. All 85 NSGA-II Pareto points (k=2, AUROC against balanced ECE) are logistic regression, mostly with strong regularisation (`C` ≈ 0.005–0.007). Example: AUROC 0.734 at ECE 0.178, against TabPFN's best 0.715 at 0.216.
- The sampler gave logistic regression 137 of its 200 trials and TabPFN 49. TabPFN's `n_estimators` barely changes its scores (0.715 tuned and untuned).
- Framing: TabPFN-3.5 is well calibrated out of the box. Logistic regression matches it only after tuning `C` with dev labels from all 8 scenarios, far more labels than a few-shot user has. Logistic regression's AUROC-best `C` (16.6) is badly calibrated (balanced ECE 0.377).

### Cross-hardware: CPU only (Ryzen 7 7700)

The 5 latency configurations, `device=cpu` with `CUDA_VISIBLE_DEVICES=""`, serial seeds. Paired CPU minus RTX 3090 AUROC:

| config | CPU | 3090 | diff [95% paired CI] | run time CPU / 3090 | predict CPU / 3090 |
|---|---|---|---|---|---|
| tabpfn k=5 | 0.7612 | 0.7612 | −0.0000 [−0.0034, +0.0034] | 363 s / 27 s | 4.43 s / 276 ms |
| tabpfn_fast k=5 | 0.7593 | 0.7602 | −0.0009 [−0.0039, +0.0017] | 82 s / 9.2 s | 957 ms / 71 ms |
| logreg k=5 | 0.7752 | 0.7752 | 0 | 3.4 s / 4.5 s | 0.20 / 0.22 ms |
| mahalanobis | 0.7231 | 0.7231 | 0 | 0.5 s / 0.6 s | 0.25 / 0.26 ms |
| tabpfn_outlier | 0.7168 | 0.7168 | +0.0000 [−0.0015, +0.0014] | 40.5 s / 8.8 s | 4.68 s / 776 ms |

- Track B reproduces on a CPU-only machine from the cached features. TabPFN scores move by at most 0.001 AUROC. The scikit-learn controls do not move.
- On the CPU, TabPFN-3.5 takes about 4.4 s per (scenario, seed) prediction and 3.5-Fast about 1 s. The full Track B grid is CPU-feasible only with 3.5-Fast.

### DINOv3 transformers versus timm parity (`7d79ba8`)

- Token agreement on 32 Vial images, bf16 on the RTX 3090: mean CLS cosine 0.99985 (DINOv3-S) and 0.99990 (DINOv3-L); mean patch cosine 0.99987 and 0.99981; worst single patch 0.995 and 0.953.
- Throughput and VRAM are the same on both backends (DINOv3-L: 11.7 images/s on a Sheet Metal-sized input and 49 on a Fabric-sized one, 1.2–1.3 GB).
- Downstream, transformers minus timm AUROC (DINOv3-S, PCA 32, k=2, 10 seeds, paired interval): Vial logreg +0.003 [−0.001, +0.014] and TabPFN 0.000 (both saturate at 1.000); Fruit Jelly logreg +0.001 [−0.023, +0.032] and TabPFN +0.012 [−0.036, +0.070].
- The ungated timm fallback is equivalent within noise.

### Cross-hardware: M4 Pro (MPS)

The same 5 configurations on the M4 Pro, from a copy of the 31 MB DINOv3-L feature cache. Run ids match the RTX 3090 latency runs. Paired M4 Pro minus RTX 3090 AUROC:

| config | M4 Pro | 3090 | diff [95% paired CI] | run time M4 / 3090 | predict M4 / 3090 |
|---|---|---|---|---|---|
| tabpfn k=5 | 0.7595 | 0.7612 | −0.0017 [−0.0061, +0.0018] | 115 s / 27 s | 1.35 s / 276 ms |
| tabpfn_fast k=5 | 0.7563 | 0.7602 | −0.0039 [−0.0153, +0.0042] | 31 s / 9.2 s | 329 ms / 71 ms |
| logreg k=5 | 0.7752 | 0.7752 | 0 | 1.5 s / 4.5 s | 0.11 / 0.22 ms |
| mahalanobis | 0.7231 | 0.7231 | 0 | 1.6 s / 0.6 s | – |
| tabpfn_outlier | 0.7172 | 0.7168 | +0.0004 [−0.0020, +0.0028] | 32 s / 8.8 s | 3.27 s / 776 ms |

- MPS moves TabPFN by up to 0.004 AUROC, all inside the paired intervals. The scikit-learn controls are bit-identical.
- Three of these manifests say `git_dirty: true`, because a documentation file had uncommitted edits while they ran. The code was `3c53bc1` for all five.

### Free-threaded Python 3.14t: not started

Since the checkpoint cache, a serial TabPFN fit takes 11 ms, and the seed pool lost to serial runs at every width. The GIL does not limit a run, so a free-threaded interpreter has no measurable gain to offer here.

### Probe: fewer train normals in TabPFN's context

Hypothesis: about 300 normals against 2–5 defects push TabPFN toward novelty scoring. The probe keeps the shots, evaluation rows and PCA of the grid run, and subsamples only the label-0 context to m normals per (scenario, seed). With m = all it reproduces the grid exactly (TabPFN 0.708 / 0.730, logreg 0.736 / 0.775 at k = 2 / 5). Configuration: DINOv3-L, CLS + mean patch + novelty, PCA 16, 10 seeds. Paired difference against m = all:

| classifier, k | m=64 AUROC | m=8 AUROC | balanced NLL, m = 64 / 32 / 16 / 8 |
|---|---|---|---|
| tabpfn, 2 | −0.006 [−0.026, +0.014] | +0.014 [−0.028, +0.054] | **−0.061 / −0.062 / −0.066 / −0.078**, every CI below 0 |
| tabpfn, 5 | −0.001 [−0.029, +0.028] | +0.022 [−0.026, +0.072] | **−0.075 / −0.080 / −0.083 / −0.068**, every CI below 0 |
| logreg, 2 | −0.011 [−0.032, +0.008] | −0.015 [−0.052, +0.024] | −0.045 / −0.037 / −0.105 / −0.063, mixed |
| logreg, 5 | −0.032 [−0.063, +0.001] | −0.032 [−0.088, +0.021] | +0.063 / +0.083 / +0.168 / +0.158, worse |

- Ranking: no significant change for TabPFN at any m. The hypothesis is not supported for AUROC.
- Calibration: a smaller context makes TabPFN measurably better calibrated (balanced NLL about 0.66 to 0.59 at k=2, 0.66 to 0.57 at k=5), at no AUROC cost. Logistic regression gets no such benefit. Predict cost also scales with context size.

### `n_normals` (`0bf2998`)

- `classifier_params: {n_normals: m}` fits any few-shot scorer on a seeded subsample of m train normals, drawn from a random stream separate from shot sampling. PCA still fits on all train normals. The balanced-prior correction uses the context the scorer saw. One-class classifiers and invalid values are rejected.
- m = 32, fixed before these runs: TabPFN and Fast over every grid cell (162 dev runs, all `ok`, about 40 min).

Matched cells (27 per row), all normals → 32 normals:

| classifier, k | AUROC | n32 better | balanced NLL | n32 better | balanced ECE | n32 better |
|---|---|---|---|---|---|---|
| tabpfn, 1 | 0.657 → 0.649 | 33% | 0.728 → 0.653 | 93% | 0.216 → 0.196 | 93% |
| tabpfn, 2 | 0.670 → 0.659 | 22% | 0.702 → 0.652 | 85% | 0.216 → 0.200 | 89% |
| tabpfn, 5 | 0.698 → 0.689 | 26% | 0.705 → 0.648 | 85% | 0.259 → 0.238 | 89% |
| tabpfn_fast, 1 | 0.659 → 0.650 | 37% | 0.735 → 0.663 | 100% | 0.217 → 0.201 | 96% |
| tabpfn_fast, 2 | 0.673 → 0.662 | 22% | 0.708 → 0.653 | 89% | 0.217 → 0.199 | 96% |
| tabpfn_fast, 5 | 0.699 → 0.691 | 26% | 0.705 → 0.646 | 81% | 0.258 → 0.237 | 89% |

At the frozen configuration (DINOv3-L, CLS + mean patch + novelty, PCA 16), paired n32 minus all: AUROC −0.015 / −0.007 / +0.002 (k = 1 / 2 / 5, all intervals include zero); balanced NLL −0.091 / −0.062 / −0.080 (all intervals below zero); balanced ECE −0.021 / −0.032 / −0.018.

- A trade-off, not a free win: across the grid, 32 normals cost about 0.01 AUROC and buy about 0.06–0.08 balanced NLL.
- Both variants were frozen: all normals as the main row, `n_normals: 32` as an extra row.

### Freeze

Selection rule, dev data only: the TabPFN-3.5 configuration with the highest mean dev AUROC over k ∈ {1, 2, 5} at default settings, so that one configuration serves every budget.

| dinov3_l configuration | TabPFN k=1 / 2 / 5 | mean | logreg k=1 / 2 / 5 | mean |
|---|---|---|---|---|
| **cls+mean_patch+novelty, pca16** | 0.708 / 0.708 / 0.730 | **0.715** | 0.707 / 0.736 / 0.775 | **0.740** |
| cls+mean_patch+novelty, pca64 | 0.692 / 0.702 / 0.740 | 0.711 | 0.690 / 0.744 / 0.758 | 0.731 |
| cls+mean_patch, pca64 | 0.677 / 0.692 / 0.757 | 0.708 | 0.667 / 0.731 / 0.735 | 0.711 |
| cls+mean_patch, pca128 (best TabPFN at k=5) | 0.666 / 0.689 / 0.761 | 0.705 | | |
| novelty only (best TabPFN at k ≤ 2) | 0.710 / 0.715 / 0.684 | 0.703 | | |

The same configuration is also logistic regression's best by this rule. Freezing it for every classifier gives each side its own best and tunes neither against the other.

`configs/frozen/` holds 28 configurations, committed in `68f5301` before any lock run:

- `trackb_final.yaml`: tabpfn, tabpfn_fast, logreg, knn × k ∈ {1, 2, 5} on DINOv3-L, CLS + mean patch + novelty, PCA 16, default settings, seeds 0–9 (12).
- `trackb_n_normals.yaml`: tabpfn and tabpfn_fast with `n_normals: 32`, same configuration, k ∈ {1, 2, 5} (6).
- `trackb_one_class.yaml`: mahalanobis and tabpfn_outlier on their dev-best features (DINOv3-L novelty, no PCA), k=0, seed 0 (2). Using TabPFN's features instead would flatter TabPFN.
- `trackb_ablation.yaml`: tabpfn and logreg with `shot_lighting: all`, k ∈ {10, 20} (4).
- `tracka.yaml`: PatchCore, EfficientAD-S, distance maps DINOv3-S and DINOv3-L (4).
- No Thinking configuration.

Not frozen: the search winners (logistic regression, PCA 64, tuned `C` ≈ 0.003, 0.747 at k=2). The searches never ranked a TabPFN trial first, so freezing a tuned logistic regression next to a default TabPFN would compare a tuned control with an untuned model.

## 2026-10-03: the single lock evaluation

`anometa lock configs/frozen` on the freeze commit `68f5301`: **28/28 `ok`**, 6.9 h (EfficientAD-S took about 5.7 h of it). Each configuration ran once. Nothing was rerun or changed. The committed report is `reports/lock/results.md` (`4fa3308`). Two display fixes to the report came after the runs and before that commit, and neither changes a number: the `n_normals: 32` runs became separate rows instead of being pooled into "best per classifier" (`74b7bd4`), and figure names became file-safe, with no dev search figures in the lock report (`7155d40`).

### Track B, lock AUROC (95% bootstrap interval), DINOv3-L, CLS + mean patch + novelty, PCA 16, seeds 0–9

| classifier | k=1 | k=2 | k=5 |
|---|---|---|---|
| **tabpfn** | **0.761** [0.702, 0.814] | **0.778** [0.722, 0.828] | **0.789** [0.734, 0.840] |
| tabpfn_fast | 0.758 [0.700, 0.811] | 0.775 [0.722, 0.825] | 0.790 [0.735, 0.840] |
| logreg | 0.683 [0.633, 0.732] | 0.740 [0.688, 0.785] | 0.745 [0.681, 0.802] |
| knn | 0.567 [0.538, 0.598] | 0.608 [0.572, 0.646] | 0.662 [0.610, 0.708] |
| tabpfn, `n_normals: 32` | 0.729 [0.666, 0.785] | 0.755 [0.694, 0.810] | 0.769 [0.710, 0.824] |
| tabpfn_fast, `n_normals: 32` | 0.724 [0.662, 0.782] | 0.751 [0.690, 0.805] | 0.768 [0.706, 0.823] |
| mahalanobis (k=0, label-free) | 0.766 [0.702, 0.822] | | |
| tabpfn_outlier (k=0, label-free) | 0.778 [0.723, 0.827] | | |

Paired, TabPFN minus control (95% interval):

| k | vs logreg AUROC | vs logreg, without wallplugs | vs logreg balanced NLL | vs logreg balanced ECE | vs mahalanobis | vs tabpfn_outlier | vs knn |
|---|---|---|---|---|---|---|---|
| 1 | **+0.078 [+0.009, +0.146]** | +0.063 [−0.010, +0.131] | **−0.595 [−0.823, −0.410]** | **−0.139 [−0.171, −0.101]** | −0.005 [−0.035, +0.026] | −0.017 [−0.051, +0.015] | **+0.194** |
| 2 | +0.038 [−0.020, +0.095] | +0.023 [−0.030, +0.073] | **−0.542 [−0.766, −0.350]** | **−0.110 [−0.138, −0.073]** | +0.012 [−0.015, +0.042] | −0.001 [−0.029, +0.027] | **+0.169** |
| 5 | +0.044 [−0.011, +0.097] | +0.021 [−0.026, +0.068] | **−0.429 [−0.605, −0.269]** | **−0.094 [−0.121, −0.058]** | +0.023 [−0.010, +0.057] | +0.011 [−0.026, +0.042] | **+0.127** |

- Ablation, shots from every lighting: TabPFN minus logistic regression +0.003 [−0.032, +0.036] at k=10 and +0.000 [−0.034, +0.036] at k=20 (both about 0.78–0.79).
- `n_normals: 32` minus all normals: AUROC −0.032 [−0.061, −0.008] / −0.023 [−0.048, +0.001] / −0.020 [−0.054, +0.012]; balanced NLL −0.060 [−0.087, −0.031] / −0.000 / +0.010 (k = 1 / 2 / 5).
- Robustness gap (regular minus shifted AUROC): TabPFN +0.021, logistic regression −0.012, Mahalanobis −0.044. All small.

### Track A, lock

| method | AU-PRO@0.05 | AU-PRO@0.30 | SegF1 | ClassF1 | image AUROC |
|---|---|---|---|---|---|
| **patch distance dinov3_l** (training-free) | **0.385** | **0.583** | **0.375** | 0.800 | **0.780** |
| patch distance dinov3_s | 0.315 | 0.533 | 0.260 | 0.787 | 0.715 |
| PatchCore | 0.222 | 0.460 | 0.198 | 0.702 | 0.720 |
| EfficientAD-S | 0.182 | 0.371 | 0.150 | 0.806 | 0.653 |

### Reading the lock results

1. **TabPFN-3.5 beats logistic regression at k=1** (+0.078 AUROC, the interval excludes zero) and leads it by about 0.04 at k=2 and k=5, but those intervals include zero. With 3 pre-declared budgets, the k=1 result would not survive a Bonferroni correction on its own: its share of replicates at or below zero is about 1%, and the corrected threshold is about 0.8% one-sided. It is "significant at k=1 before correction".
2. **Calibration is the clear win.** TabPFN's balanced NLL is less than half of logistic regression's at k=1 (0.65 vs 1.25), and its balanced ECE is lower at every k. All intervals exclude zero. This is against logistic regression at default settings, as frozen.
3. **TabPFN ties the label-free controls.** With 1–5 labelled defects it does not significantly beat Mahalanobis or TabPFN's own outlier score on the same features. On AD2, a handful of labels does not buy much over a good novelty score. TabPFN is the only supervised model here that matches one. Logistic regression and kNN score below both label-free controls.
4. **TabPFN-3.5-Fast matches 3.5** within 0.003 at every k, at about a third of the predict time (frozen configuration, k = 2: 59 ms vs 0.19 s on the RTX 3090).
5. **`n_normals: 32` did not replicate.** On lock it costs 0.02–0.03 AUROC and improves calibration only at k=1. The dev trade-off was mostly noise or dev-specific. It is a negative ablation.
6. **Dev to lock:** the dev picture (TabPFN ≈ logistic regression, Wall Plugs favouring logistic regression) did not hold. On lock, logistic regression is weaker (0.68–0.75 vs dev 0.71–0.78) and TabPFN stronger (0.76–0.79 vs dev 0.71–0.73). **Wall Plugs now favours TabPFN** (0.61–0.66 vs logistic regression 0.43–0.48). Dev evaluates only the scenes left after shot sampling, lock all of its own. With 7–8 defect scenes per half, per-scenario differences of 0.1–0.2 are within scene-sampling noise. The Wall Plugs story from dev was a small-sample effect, and the lock split did its job.
7. **Track A confirms dev.** The training-free DINOv3-L patch distance map is the best of our three pixel methods (AU-PRO@0.05 0.385 vs PatchCore 0.222 and EfficientAD-S 0.182; baselines untuned, see the 2026-10-06 correction). Its image AUROC (0.780) matches TabPFN's k=5 (0.789) without any labels.

### The frozen configuration on three machines (dev split)

All 20 frozen Track B configurations (`trackb_final`, `trackb_n_normals`, `trackb_one_class`) were rerun on the dev split, one seed at a time, from the same cached features: on the RTX 3090, on the M4 Pro (MPS) and on the Ryzen 7 7700 (CPU only). Nothing touched the lock split.

- **Same scores everywhere.** Logistic regression, kNN and Mahalanobis are bit-identical. TabPFN, Fast and the outlier score move by at most 0.005 AUROC on the M4 Pro and 0.001 on the CPU. All 40 paired intervals against the RTX 3090 include zero.
- **TabPFN-3.5 at k=2, one scenario's dev images:** fit 4.4 / 5.1 / 4.8 ms and predict 0.19 / 0.82 / 1.54 s (3090 / M4 Pro / CPU). TabPFN-3.5-Fast predicts in 59 / 202 / 470 ms. The outlier score takes 0.76 / 3.3 / 4.8 s.
- **Encoders, one Fabric-sized image:** DINOv3-L 20 ms / 234 ms / 1.12 s; DINOv3-S 12 / 33 / 110 ms. MPS runs the encoders in float32, CUDA in bfloat16. The encoder dominates Track B's cost on every machine.
- The M4 Pro runs overlapped with a data copy, which may have slowed them slightly.

Track A on other hardware (dev split, speed first):

- PatchCore, all 8 scenarios on the M4 Pro (MPS, `PYTORCH_ENABLE_MPS_FALLBACK=1`): 32 min in total, median 169 s per scenario (fit + maps). Pixel metrics match the RTX 3090 (AU-PRO@0.05 0.207 vs 0.202). Image AUROC is 0.03 lower (0.619 vs 0.652, mostly Fabric 0.53 vs 0.72), likely because the random coreset draw differs on MPS. PatchCore on Vial with the CPU only: 152 s.
- EfficientAD-S, timed only. On the M4 Pro, 1,000 / 2,000 steps took 343 / 551 s on Vial (0.21 s per step), so 70,000 steps take about 4.1 h. On the CPU, 500 / 1,000 steps took 317 / 577 s (0.52 s per step), about 10.1 h. The RTX 3090 trained Vial's 70,000 steps in 44 min (lock run, GPU alone).
- Vial end to end (every image), 3090 / M4 Pro / CPU: TabPFN pipeline (DINOv3-L encode, fit, predict) 14 s / 2.1 min / 9.8 min; PatchCore 18 s / 2.6 min / 2.5 min; EfficientAD-S 44 min / about 4.1 h / about 10.1 h.
- The TabPFN pipeline is slightly faster than PatchCore on the RTX 3090 and the M4 Pro, and far faster than EfficientAD-S everywhere. On the CPU, PatchCore is about 4 times faster, because DINOv3-L encoding at a 512-pixel short side is expensive there. TabPFN itself never costs more than 1.5 s per scenario. DINOv3-S encodes 10 times faster on the CPU.

### Visual report

The visual HTML report was built from the run artifacts: question, dataset and protocol, model and pipeline diagrams, testing framework, all lock results with charts, example anomaly maps, dev findings, speed and cross-hardware agreement, limitations and references. Nothing was uploaded to Hugging Face and no Prior Labs API was called. It is now the [full report](https://husmen.github.io/anometa/report/).

## 2026-10-05: post-freeze studies

Both studies were approved before they ran. The lighting-adaptation protocol was committed before any run (`e8c2498`). Neither touches the frozen configuration or the lock benchmark. Summaries are on the [results](results.md) page.

### Track C prototype, first 3 scenarios (superseded)

- An uncommitted prototype script. One row per DINOv3-L patch (16×16 px at a 512-px short side, about 1,200–1,400 per image). Features: PCA-32 of the patch token (PCA fitted on 40 train images), the DINOv3 nearest-patch distance, the PatchCore map averaged to the patch grid, and the 3×3 maxima of both maps (36 columns).
- Context: 15,000 patches from half of the validation images (label 0) plus every patch of the k shot images, labelled from the mask (coverage 0 → 0, at least 0.3 → 1, in between left out). About 17,000–20,000 rows with 100–600 defect rows. The other half of validation sets the SegF1/ClassF1 threshold for every method.
- Evaluation: the dev images left after the shots, every lighting, scored at full resolution with the project's pixel metrics.
- Baselines on the same images: the DINOv3 distance map, the PatchCore map, their z-scored average (`fused_z`, no labels), and a per-patch logistic regression on the same rows.
- Cost on the RTX 3090: TabPFN-3.5-Fast 14–19 s and TabPFN-3.5 53–71 s per (scenario, k, seed) for about 100,000 scored patches; patch-token extraction and a PatchCore refit about 2–5 min per scenario.

AU-PRO@0.05, mean over 3 seeds (no intervals; exploratory):

| scenario, k | DINOv3 distance | PatchCore | fused (no labels) | logreg rows | TabPFN-3.5 rows | TabPFN-Fast rows |
|---|---|---|---|---|---|---|
| Vial, 2 | **0.857** | 0.514 | 0.844 | 0.812 | 0.819 | 0.827 |
| Vial, 5 | **0.867** | 0.503 | 0.845 | 0.743 | 0.804 | 0.834 |
| Fruit Jelly, 2 | 0.226 | 0.351 | 0.351 | **0.384** | 0.371 | 0.376 |
| Fruit Jelly, 5 | 0.139 | 0.161 | 0.164 | 0.162 | 0.168 | **0.170** |
| Walnuts, 2 | 0.418 | 0.264 | 0.413 | 0.390 | 0.413 | **0.444** |
| Walnuts, 5 | 0.504 | 0.419 | 0.540 | 0.521 | 0.602 | **0.616** |

- Where it helps: on Walnuts at k=5, TabPFN-Fast reaches 0.616, against 0.540 for the best label-free map and 0.521 for logistic regression on the same rows. On Fruit Jelly at k=2 the learned maps (0.37–0.38) beat the DINOv3 distance (0.23), but only match PatchCore and the fusion (0.35).
- Where it does not: Vial is saturated for the DINOv3 distance (0.86), and every learned combination loses a little to it. Fruit Jelly at k=5 is scored on only 3 defect scenes, and everything collapses (about 0.16).
- SegF1 is often lower for the learned maps, because the threshold rule (validation mean + 3 std) suits a distance better than a probability.

### Lighting-adaptation study (protocol `e8c2498`, code `bdcfade`, report section `aae9c30`)

Track `L`: scene-level folds over every public test scene, 10 folds, frozen Track B features. 30 of the 36 pre-declared runs finished first. The 6 `tabpfn_outlier` runs take much longer: with 18 features the outlier score refits 18 regressors per fit. Pre-declared comparisons, shifted lighting, paired 95% interval:

| comparison | k = 2 | k = 1 |
|---|---|---|
| **TabPFN, m = 2 minus m = 0, AUROC (primary)** | +0.002 [−0.007, +0.012] | +0.005 [−0.006, +0.015] |
| TabPFN, m = 1 minus m = 0, AUROC | +0.001 [−0.007, +0.009] | +0.004 [−0.004, +0.013] |
| TabPFN, m = 2 minus m = 0, balanced NLL | **−0.049 [−0.057, −0.041]** | **−0.065 [−0.073, −0.058]** |
| TabPFN minus logreg at m = 2, AUROC | +0.025 [−0.001, +0.051] | +0.015 [−0.011, +0.041] |
| TabPFN minus Mahalanobis at m = 2, AUROC | **+0.046 [+0.024, +0.067]** | **+0.044 [+0.026, +0.063]** |
| TabPFN minus logreg at m = 2, balanced NLL | **−0.41 [−0.50, −0.34]** | **−0.43 [−0.51, −0.36]** |
| logreg, m = 2 minus m = 0, AUROC | −0.001 [−0.004, +0.002] | −0.000 [−0.002, +0.002] |
| Mahalanobis, m = 2 minus m = 0, AUROC | +0.001 [−0.002, +0.005] | +0.001 [−0.002, +0.005] |

Mean AUROC at k = 2, m = 0 (regular / shifted / gap): TabPFN 0.735 / 0.727 / 0.009; TabPFN-Fast 0.744 / 0.731 / 0.013; logreg 0.703 / 0.705 / −0.002; Mahalanobis 0.687 / 0.682 / 0.006; kNN 0.607 / 0.616 / −0.009. Hardest lightings for TabPFN: `shift_3` (0.701) and `shift_1` (0.710), against 0.735 regular.

- **The primary hypothesis is not supported.** A few good images under the new light do not improve TabPFN's ranking there (+0.002, the interval includes zero). With DINOv3-L features the lighting gap is only about 0.01 AUROC for TabPFN and zero for logistic regression.
- **They do improve calibration.** Balanced NLL on shifted lighting drops by 0.05–0.065 with two adaptation scenes. The other models' ranking does not move either.
- **Under shifted light, TabPFN beats Mahalanobis** (+0.045, the interval excludes zero) and leads logistic regression by 0.015–0.025 (intervals touch zero), with far better calibration (balanced NLL about 0.6 vs 1.0).
- The frozen DINOv3-L features already make every model nearly lighting-invariant on AD2. TabPFN's in-context adaptation buys calibration, not ranking.

### Track C prototype, all 8 scenarios

- The first attempt stopped on Can. Its defects are so small that no shot patch reaches 30% mask coverage, so the context had no defect rows and TabPFN refused to fit (one class only). This was a gap in the labelling rule, not in the results.
- New rule, applied to every scenario: a shot patch is a defect row if its mask coverage is at least min(0.3, half of that image's largest patch coverage). Clean patches stay normal rows; the patches in between are left out. All 8 scenarios were rerun with it (3 seeds, k ∈ {2, 5}, the same models), which supersedes the 3-scenario table above.

Mean AU-PRO@0.05, 48 units of (scenario, k, seed):

| k | DINOv3 distance | PatchCore | fused (no labels) | logreg rows | TabPFN-3.5 rows | TabPFN-Fast rows |
|---|---|---|---|---|---|---|
| 2 | 0.393 | 0.188 | 0.363 | 0.428 | 0.432 | **0.449** |
| 5 | 0.397 | 0.189 | 0.353 | 0.402 | 0.460 | **0.468** |

Paired, TabPFN-Fast map minus each baseline, bootstrap over (scenario, seed) units, 95%:

| baseline | k = 2 | k = 5 |
|---|---|---|
| DINOv3 distance map | **+0.056 [+0.022, +0.091]** | **+0.071 [+0.034, +0.109]** |
| fused map (no labels) | **+0.086 [+0.038, +0.137]** | **+0.115 [+0.059, +0.178]** |
| logreg on the same rows | **+0.021 [+0.006, +0.034]** | **+0.065 [+0.025, +0.106]** |
| PatchCore | **+0.261 [+0.186, +0.342]** | **+0.278 [+0.188, +0.373]** |

- Per scenario at k = 5 (DINOv3 distance → TabPFN-Fast): Wall Plugs 0.006 → 0.202, Sheet Metal 0.554 → 0.723, Walnuts 0.504 → 0.638, Rice 0.691 → 0.736, Fabric 0.286 → 0.289, Can 0.126 → 0.140 (logistic regression 0.253 is best there), Fruit Jelly 0.139 → 0.178, Vial 0.867 → 0.834 (the only clear loss).
- Image AUROC from the maps' maxima: TabPFN-Fast 0.740 / 0.717 vs DINOv3 distance 0.710 / 0.672 (k = 2 / 5).
- SegF1 is lower for every learned map (TabPFN-Fast 0.18 vs distance 0.26), because the distance-style threshold rule does not suit probabilities.
- Reading: TabPFN on one row per patch learns a better defect map than either input map, their fusion, or logistic regression on the same rows. This is the strongest TabPFN-specific result of the project.
- Caveats: dev split only, 3 seeds, intervals over (scenario, seed) units rather than scenes, and the labelling rule changed once (after a crash on Can, not after looking at scores). A clean confirmation needs its own frozen protocol and a held-out set.
- Cost on the RTX 3090, all 8 scenarios: 3.5 h for everything, sharing the GPU with the lighting study. TabPFN-Fast takes 15–30 s per (scenario, k, seed), for about 100,000–250,000 scored patches each.

### Lighting study: first `tabpfn_outlier` run

The first `tabpfn_outlier` run (k = 2, m = 0) finished after 5.8 h while sharing the GPU and CPU with Track C: regular 0.702, shifted 0.710 AUROC. No lighting gap either. These runs feed one secondary comparison only.

## 2026-10-06: Thinking ablation and follow-ups

### TabPFN-3.5-Thinking ablation (protocol `2ab89a9`, dev split, post-freeze)

- Runs (M4 Pro, Prior Labs API, medium effort): frozen configuration, k = 2, all 8 scenarios. Seeds 0–4 are complete (40 Thinking fits, run `thinking-3a67cd0b6af6`, `ok`). Seeds 5–9 have 10 of 40 predictions cached and wait for budget. Every prediction is cached, so nothing is paid twice.
- Cost: the API's estimate is about 49k tokens per (scenario, seed). The real cost was about 200k, about 4 times the estimate. The 5M daily token limit covered 25 fits on 5 October and 25 on 6 October. Each fit takes about 10–20 s, and the API caps Thinking fits at 10 calls per minute.
- Code fixes: the cost check sent the estimate in a shape the server rejects (`7fb8c4a`). Thinking calls now wait and retry after a per-minute rate-limit reply (`8e746d4`), but stop at once on the daily token limit (`ff30021`). On the M4 Pro, Thinking runs need `device=cpu`, because the seed thread pool refuses MPS. The cache key ignores the device.

Pre-declared comparison, k = 2, seeds 0–4, paired 95% interval over seeds and scenes:

| Thinking minus | AUROC | balanced NLL |
|---|---|---|
| local TabPFN-3.5 (`grid-59bd9f739153`, same seeds) | +0.0004 [−0.011, +0.012] | +0.011 [−0.011, +0.031] |
| logistic regression (`grid-155da266f380`) | −0.034 [−0.091, +0.019] | **−0.31 [−0.48, −0.14]** |

Mean dev AUROC / balanced NLL: Thinking 0.708 / 0.688, TabPFN-3.5 0.707 / 0.677, logistic regression 0.742 / 1.000. Per scenario, Thinking tracks TabPFN-3.5 within 0.02 everywhere (largest: Fabric 0.560 vs 0.539).

- Reading: with about 300 normal rows, 2 defect rows and 18 features, Thinking at medium effort gives the same ranking and calibration as local TabPFN-3.5, at about 200k tokens per fit. The extra test-time compute has nothing to work with when there are only two positives. Local TabPFN-3.5 is the better choice here: free, about 0.2 s per scenario, identical results.

### Lighting study: `tabpfn_outlier` at k = 2

- The three k = 2 `tabpfn_outlier` runs finished (4.6–5.8 h each). Shifted AUROC 0.710 / 0.710 / 0.713 at m = 0 / 1 / 2 (regular 0.702): no lighting gap and no adaptation effect.
- Secondary comparison, TabPFN-3.5 minus `tabpfn_outlier` at m = 2, k = 2, shifted lighting: +0.016 [−0.005, +0.039]. TabPFN leads, but the interval includes zero.
- The three k = 1 outlier runs are still running (about 15 h). They only add the k = 1 version of this comparison.

### Track C example images, chosen by a fixed rule

- The first example (Walnuts, first dev defect image by id) showed a miss. The new examples, one per scenario, follow a fixed rule: among dev defect images under regular lighting, keep the larger half by defect area, then take the image where TabPFN-Fast's map has the highest pixel AUROC (k = 5, seed 0). They are good cases for TabPFN by construction, and are labelled that way.
- Median per-image pixel AUROC over the same candidates (DINOv3 distance / PatchCore / TabPFN-Fast): Can 0.98 / 0.67 / 0.73, Fabric 0.92 / 0.82 / 0.93, Fruit Jelly 0.85 / 0.83 / 0.75, Rice 0.99 / 0.79 / 1.00, Sheet Metal 0.96 / 0.76 / 0.99, Vial 0.93 / 0.87 / 0.95, Wall Plugs 0.83 / 0.52 / 0.99, Walnuts 1.00 / 1.00 / 1.00.
- TabPFN's maps suppress the background much better. On Can and Fruit Jelly the plain distance map is better per image.
- The video's map scene uses the Wall Plugs example, where TabPFN finds a defect the other maps miss (pixel AUROC 1.00 vs 0.71 and 0.07).

### 2026-10-06 correction: comparison with the AD2 paper

This correction compares our lock numbers with Table VII of the AD2 paper (private test set, regular lighting; table on the [results](results.md) page).

- The distance map's lock mean AU-PRO@0.05 (38.5%) is above the paper's best mean (30.8%, EfficientAD). But it beats the paper's best method on only 4 of 8 scenarios (Rice, Sheet Metal, Vial, Wall Plugs).
- Our PatchCore (22.2 vs 28.8) and EfficientAD-S (18.2 vs 30.8) score below the paper on 6 and 7 of 8 scenarios. Both run untuned with anomalib defaults, and EfficientAD-S uses Imagenette instead of ImageNet as its penalty set.
- The test sets differ (public lock half, pooled lighting), so this shows promise, not a ranking. A conclusive result needs tuned baselines, a tuned pipeline and a server submission.
- Earlier wording in this log ("best pixel method", "not comparable with published numbers") is qualified accordingly.
