# Results

The lock split was evaluated once, on 2026-10-03, after the configuration was frozen on the dev split. All 28 frozen configurations finished with status `ok`. The [protocol](protocol.md) page explains the split and the freeze, the [tracks](tracks.md) page the methods, and the [metrics](metrics.md) page the metrics. Every experiment on the way to the lock run is in the [experiment log](experiment-log.md).

## Lock results

### Track B, image level

Track B uses DINOv3-L CLS, mean-patch and patch-novelty features, PCA to 16 dimensions, default settings and 10 seeds. Brackets are 95% bootstrap intervals over seeds and scenes.

| image AUROC, all 8 scenarios | k=1 | k=2 | k=5 |
|---|---|---|---|
| **TabPFN-3.5** | **0.761** [0.702, 0.814] | **0.778** [0.722, 0.828] | **0.789** [0.734, 0.840] |
| TabPFN-3.5-Fast | 0.758 | 0.775 | 0.790 |
| Logistic regression | 0.683 [0.633, 0.732] | 0.740 [0.688, 0.785] | 0.745 [0.681, 0.802] |
| kNN | 0.567 | 0.608 | 0.662 |
| Mahalanobis, no labels | 0.766 | | |
| TabPFN unsupervised outlier score, no labels | 0.778 | | |

- With one labelled defect, TabPFN-3.5 beats logistic regression by 0.078 AUROC (paired 95% interval [+0.009, +0.146]). This result is significant before correction. It does not survive a Bonferroni correction over the 3 pre-declared budgets.
- At k = 2 and 5, TabPFN-3.5 leads logistic regression by about 0.04. Both intervals include zero.
- TabPFN is far better calibrated. Its balanced NLL is about half of logistic regression's (0.65 vs 1.25 at k=1). Its balanced ECE is lower at every k. All these intervals exclude zero. Logistic regression runs at its default `C` here, as frozen.
- TabPFN ties the label-free controls. On AD2, 1–5 labels add little over a good novelty score. TabPFN is the only supervised model here that matches one. Logistic regression and kNN score below both label-free controls.
- TabPFN-3.5-Fast matches TabPFN-3.5 within 0.003 at about a third of the predict time (59 ms vs 0.19 s for one scenario at k = 2 on an RTX 3090).
- A context of 32 train normals (`n_normals: 32`) did not replicate its dev result. On lock it costs 0.02–0.03 AUROC and improves calibration only at k=1. It is a negative ablation.
- With shots from every lighting, TabPFN and logistic regression tie at k=10 (+0.003 [−0.032, +0.036]) and k=20 (+0.000 [−0.034, +0.036]). Both score about 0.78–0.79.
- The robustness gap (regular minus shifted AUROC) is small for every model: TabPFN +0.021, logistic regression −0.012, Mahalanobis −0.044.
- On the dev split, wall plugs favoured logistic regression. On lock, wall plugs favours TabPFN. With 7–8 defect scenes per half, per-scenario differences of 0.1–0.2 are within scene-sampling noise.

### Track A, pixel level

| pixel-level (Track A, unsupervised) | AU-PRO@0.05 | AU-PRO@0.30 | SegF1 | image AUROC |
|---|---|---|---|---|
| **DINOv3-L patch distance (training-free)** | **0.385** | 0.583 | **0.375** | **0.780** |
| PatchCore | 0.222 | 0.460 | 0.198 | 0.720 |
| PatchCore, 512×512, tuned on dev (post-freeze) | **0.385** | **0.602** | 0.233 | 0.739 |
| EfficientAD-S | 0.182 | 0.371 | 0.150 | 0.653 |

The training-free DINOv3-L patch distance map ties PatchCore tuned on dev to 512×512 on AU-PRO@0.05 and has the best SegF1 and image AUROC; its image AUROC (0.780) matches TabPFN at k=5 (0.789) without any labels. The tuned PatchCore row is a post-freeze study, evaluated once on lock (see [protocol](protocol.md)).

### Comparison with the MVTec AD 2 paper

AU-PRO@0.05 in %. Ours: lock split of the public test set, all lighting conditions pooled. Paper: Table VII of [Heckler-Kram et al.](https://arxiv.org/abs/2503.21622), private test set, regular lighting (`TESTpriv`), evaluated by MVTec's server. Best of 7 methods: PatchCore, RD, RD++, EfficientAD, MSFlow, SimpleNet, DSR.

| scenario | DINOv3-L distance (ours) | PatchCore 512 px, tuned (ours) | PatchCore 256 px, ours / paper | EfficientAD-S, ours / paper | best in paper |
|---|---|---|---|---|---|
| Can | 6.3 | 12.9 | 0.1 / 4.7 | 4.1 / 9.6 | **13.9** (DSR) |
| Fabric | 19.6 | **37.7** | 4.5 / 11.0 | 15.7 / 22.2 | 22.2 (EfficientAD) |
| Fruit Jelly | 42.9 | 48.0 | 35.4 / 46.7 | 34.1 / 50.5 | **54.4** (RD++) |
| Rice | **39.5** | 15.7 | 11.7 / 25.6 | 2.9 / 27.6 | 27.6 (EfficientAD) |
| Sheet Metal | **47.6** | 15.9 | 9.4 / 15.2 | 8.7 / 11.8 | 18.0 (DSR) |
| Vial | **84.6** | 73.4 | 47.0 / 62.2 | 62.2 / 55.6 | 63.0 (RD++) |
| Wall Plugs | 26.8 | **32.0** | 15.4 / 12.8 | 2.4 / 20.3 | 20.3 (EfficientAD) |
| Walnuts | 41.1 | **72.1** | 54.3 / 51.8 | 15.7 / 48.8 | 51.8 (PatchCore) |
| **Mean** | **38.5** | **38.5** | 22.2 / 28.8 | 18.2 / 30.8 | 30.8 (EfficientAD) |

- Our two best maps tie on the mean: the training-free DINOv3-L distance map and PatchCore at 512×512, tuned on the dev split after the freeze (38.5 each). Between them, they beat the paper's best method on 6 of 8 scenarios (Fabric, Rice, Sheet Metal, Vial, Wall Plugs, Walnuts) and lose on Can and Fruit Jelly. They are complementary: the distance map wins on Rice, Sheet Metal and Vial, PatchCore on the other five.
- At the frozen 256×256 default, our PatchCore scores below the paper's on 6 of 8 scenarios. A declared dev-split sweep of input size and coreset ratio showed that resolution drives most of that gap: at 512×512 its lock mean rises from 22.2 to 38.5. The paper ran PatchCore at 256×256, so this shows the effect of resolution, not a better PatchCore. EfficientAD-S was not tuned (about 44 min per scenario) and stays below the paper on 7 of 8 scenarios; it also uses Imagenette instead of ImageNet as its penalty set.
- The test sets differ (public vs private, pooled lighting vs regular lighting) and the lock sets are small, so this is not a ranking. A conclusive comparison needs the private test set, scored through MVTec's server.

### Reproduction

The best dev configurations give the same AUROC within 0.001 on a CPU-only Ryzen 7 7700 and within 0.004 on an M4 Pro. The scikit-learn controls are bit-identical across machines. The ungated timm DINOv3 weights match the gated ones within noise. See the [experiment log](experiment-log.md) for timings.

## Post-freeze studies

These studies ran after the single lock evaluation. They are not part of the lock benchmark. Nothing in the frozen configuration changed because of them. Each protocol was fixed before its first run.

### Lighting adaptation

Question: when the lighting changes, does TabPFN-3.5 recover detection if a few good parts photographed under the new light are added to its context, without retraining?

- Model: the frozen Track B configuration.
- Scenes: every public test scene, in 10 scene-level folds per scenario. The study tunes nothing, so it may reuse lock scenes.
- Context for target lighting L: all train normals, k regular-lit shots, and the L-lit images of m good adaptation scenes. Evaluation: the L-lit images of every other scene.
- Grid: m ∈ {0, 1, 2}, k ∈ {1, 2}, six classifiers, 36 runs.

Pre-declared comparisons on shifted lighting, paired 95% intervals:

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

- The primary hypothesis is not supported. A few good images under the new light do not improve TabPFN's ranking there (+0.002, the interval includes zero).
- There is little to recover. With DINOv3-L features, the lighting gap at m = 0 is only about 0.01 AUROC for TabPFN and zero for logistic regression.
- The adaptation images do improve calibration. Balanced NLL on shifted lighting drops by 0.05–0.065 with two adaptation scenes.
- Under shifted light, TabPFN beats Mahalanobis by about 0.045 and leads logistic regression by 0.015–0.025 (intervals touch zero).
- The TabPFN outlier score shows no lighting gap either. At m = 2, TabPFN-3.5 leads it by +0.016 [−0.005, +0.039] at k = 2 and by +0.012 [−0.009, +0.032] at k = 1; both intervals include zero. All 36 runs of the study are complete.

The frozen DINOv3-L features already make every model nearly lighting-invariant on AD2. TabPFN's in-context adaptation buys calibration, not ranking.

### TabPFN-3.5-Thinking ablation

TabPFN-3.5-Thinking (Prior Labs API, medium effort) ran on the frozen configuration at k = 2 on the dev split, all 8 scenarios, seeds 0–4. Only dev feature rows (18 numbers per image) and their labels were sent to the API, never images.

| Thinking minus | AUROC | balanced NLL |
|---|---|---|
| local TabPFN-3.5, same seeds | +0.0004 [−0.011, +0.012] | +0.011 [−0.011, +0.031] |
| logistic regression | −0.034 [−0.091, +0.019] | **−0.31 [−0.48, −0.14]** |

- Mean dev AUROC / balanced NLL: Thinking 0.708 / 0.688, TabPFN-3.5 0.707 / 0.677, logistic regression 0.742 / 1.000.
- Per scenario, Thinking tracks TabPFN-3.5 within 0.02 everywhere.
- Each fit cost about 200k tokens, about 4 times the API's estimate.
- With about 300 normal rows, 2 defect rows and 18 features, the extra test-time compute has nothing to work with. Local TabPFN-3.5 gives identical results for free, in about 0.2 s per scenario.

Seeds 5–9 and the budgets k = 1 and k = 5 wait for API budget (see the [roadmap](roadmap.md)).

### Track C prototype: TabPFN anomaly maps

Track C gives TabPFN one row per DINOv3-L patch and learns a defect map from a few labelled shot images. The prototype ran on the dev split only, as an exploratory study.

- Rows: one per patch (16×16 px at a 512-px short side). 36 columns: PCA-32 of the patch token, the DINOv3 nearest-patch distance, the PatchCore map averaged to the patch grid, and the 3×3 maxima of both maps.
- Context: 15,000 patches from half of the validation images (label 0), plus the patches of the k shot images labelled from their masks.
- Baselines on the same images: the DINOv3 distance map, PatchCore, their z-scored fusion (no labels), and logistic regression on the same rows.
- 3 seeds, k ∈ {2, 5}, all 8 scenarios.

Mean AU-PRO@0.05:

| k | DINOv3 distance | PatchCore | fused (no labels) | logreg rows | TabPFN-3.5 rows | TabPFN-Fast rows |
|---|---|---|---|---|---|---|
| 2 | 0.393 | 0.188 | 0.363 | 0.428 | 0.432 | **0.449** |
| 5 | 0.397 | 0.189 | 0.353 | 0.402 | 0.460 | **0.468** |

TabPFN-Fast map minus each baseline, bootstrap over (scenario, seed) units, 95%:

| baseline | k = 2 | k = 5 |
|---|---|---|
| DINOv3 distance map | **+0.056 [+0.022, +0.091]** | **+0.071 [+0.034, +0.109]** |
| fused map (no labels) | **+0.086 [+0.038, +0.137]** | **+0.115 [+0.059, +0.178]** |
| logreg on the same rows | **+0.021 [+0.006, +0.034]** | **+0.065 [+0.025, +0.106]** |
| PatchCore | **+0.261 [+0.186, +0.342]** | **+0.278 [+0.188, +0.373]** |

- This is the strongest TabPFN-specific result of the project.

With PatchCore tuned to 512×512 (post-freeze rerun, same images, seeds and context), the PatchCore columns of the rows and the PatchCore baseline change; everything else stays the same:

| k | DINOv3 distance | PatchCore 512 | fused (no labels) | logreg rows | TabPFN-3.5 rows | TabPFN-Fast rows |
|---|---|---|---|---|---|---|
| 2 | 0.393 | 0.331 | 0.435 | 0.427 | 0.433 | **0.452** |
| 5 | 0.397 | 0.296 | 0.413 | 0.413 | 0.469 | **0.470** |

| TabPFN-Fast map minus | k = 2 | k = 5 |
|---|---|---|
| DINOv3 distance map | **+0.059 [+0.021, +0.103]** | **+0.073 [+0.039, +0.109]** |
| fused map (no labels) | +0.018 [−0.016, +0.058] | **+0.057 [+0.006, +0.109]** |
| logreg on the same rows | **+0.025 [+0.008, +0.043]** | **+0.056 [+0.024, +0.090]** |
| PatchCore 512 | **+0.121 [+0.051, +0.199]** | **+0.174 [+0.086, +0.267]** |

- TabPFN's maps barely change with the better PatchCore columns (+0.003 and +0.002, intervals around zero): they rely mostly on the DINOv3 columns.
- The lead over PatchCore halves but stays clear. The label-free fusion gets much stronger and ties TabPFN-Fast at k = 2; at k = 5 TabPFN still leads.
- The biggest gains at k = 5 are on Wall Plugs (0.006 → 0.202), Sheet Metal (0.554 → 0.723) and Walnuts (0.504 → 0.638). Vial is the only clear loss (0.867 → 0.834).
- SegF1 is lower for every learned map (TabPFN-Fast 0.18 vs distance 0.26). The threshold rule (validation mean + 3 std) suits a distance better than a probability.
- TabPFN-Fast takes 15–30 s per (scenario, k, seed) on an RTX 3090, for about 100,000–250,000 scored patches.
- Caveats: dev split only, 3 seeds, intervals over (scenario, seed) units rather than scenes, and the labelling rule changed once (after a crash on Can, not after looking at scores). Track C learns from labelled test defects, so it breaks the benchmark's unsupervised protocol. A clean confirmation needs its own frozen protocol and a held-out set.

## Full lock tables

The generated lock report follows unchanged.

```{include} ../reports/lock/results.md
:heading-offset: 2
:relative-images:
```
