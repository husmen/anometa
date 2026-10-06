# Metrics and statistics

This page defines the metrics each track reports, the robustness gap, and the bootstrap behind every confidence interval. The code lives in `anometa.metrics`. Tracks report different metric sets, so a run's metrics are a dict keyed by metric name (see [architecture](architecture.md)).

## Image metrics

Track B reports image-level metrics on the classifier's anomaly score. `anometa.metrics.image` implements them, and `anometa.metrics.image.image_metrics` computes the set for one scenario and seed.

| Key | Metric | Notes |
|---|---|---|
| `auroc` | area under the ROC curve | Computed from ranks (Mann-Whitney U, ties averaged). |
| `auprc` | area under the precision-recall curve | Average precision. |
| `nll` | mean binary negative log-likelihood | Probabilities clipped to [1e-7, 1 − 1e-7]. |
| `ece` | expected calibration error | 10 equal-width bins on the anomaly probability. |
| `brier` | Brier score | Mean squared error between probability and label. |

Anomalous is the positive class.

### Raw and balanced calibration

Calibration metrics are reported twice:

- Raw (`nll`, `ece`, `brier`): on the classifier's probabilities.
- Balanced (`nll_bal`, `ece_bal`, `brier_bal`): after the post-hoc prior correction to a 50/50 prior, `anometa.metrics.image.prior_correct`.

The correction uses π, the share of anomalous rows in the fit set, computed from training class counts only (see [tracks](tracks.md#prior-correction)):

```text
p' = (p/π) / (p/π + (1 − p)/(1 − π))
```

The training prior is a few percent, while the evaluation sets hold many more anomalies. Raw values therefore mostly measure this prior shift. The balanced values are the ones to compare.

One-class controls output unbounded scores, not probabilities. They report AUROC and AUPRC only.

### Track A image score

For Track A, the image score is the maximum of the full-resolution anomaly map. Track A reports image AUROC on this score, next to its pixel metrics.

### Single-class subsets

A metric on a subset with only one class is NaN. Small lighting subsets can be single-class. NaN values are recorded, ignored by the means, and counted in `n_nan_<metric>` next to them.

## Pixel metrics

Track A reports the MVTec AD 2 pixel metrics. `anometa.metrics.pixel.pixel_metrics` computes them per scenario. The implementation follows the MVTec AD 2 paper and the MVTec evaluation code as closely as the public material allows.

### AU-PRO

AU-PRO is the area under the per-region overlap (PRO) curve, integrated up to a limit on the global pixel false positive rate and divided by that limit. Anometa reports two limits:

- `au_pro_005`: AU-PRO@0.05, the evaluation server's metric.
- `au_pro_030`: AU-PRO@0.30, the paper's other limit.

The code is vendored from the MVTec AD reference evaluation code (`compute_pro` and `trapezoid`, BSD-style licence, notice kept) in `anometa.metrics._pro_reference`. The only change from upstream replaces a SciPy call that current SciPy removed. The PRO curve is computed once per scenario and integrated at both limits.

- PRO is the mean overlap over every ground-truth connected component (8-connectivity), with all images of the scenario pooled.
- The false positive rate counts every normal pixel, including every pixel of good images.

### SegF1 and ClassF1

Both use one threshold per scenario, from defect-free validation images only.

- **Threshold:** the mean plus 3 standard deviations of every pixel of every validation map, after upsampling to full resolution (`anometa.metrics.pixel.validation_threshold`). This is the baseline rule named by the MVTec AD 2 checker.
- **SegF1** (`seg_f1`): pixel-level F1 with true positives, false positives and false negatives pooled over all images of the scenario. Good images contribute false positives only. The paper does not say whether SegF1 is pooled or averaged per image; pooled is our reading.
- **ClassF1** (`class_f1`): image-level F1. An image is predicted anomalous when any pixel of its map is above the threshold. Anomalous is the positive class.

### Resolution and float16

The evaluation server upsamples submitted maps to the original image size with bilinear interpolation. Anometa does the same:

- Each map is bilinearly upsampled to the original image size (`anometa.metrics.pixel.upsample`, `align_corners=False`).
- The upsampled map is cast to float16 before any metric, which mimics the server's float16 input format.
- Thresholding and every pixel metric run at full resolution.
- Masks are binarised as any value above 0. Good images get an all-zero mask.

### Pooling rules

- Every pixel metric is computed per scenario, with all evaluation images of that scenario pooled, good images included.
- The run reports the per-scenario values (`<scenario>/<metric>`) and their unweighted mean over scenarios (`<metric>`).
- Metrics are also computed on the regular-lit and the shifted-lit images separately, which gives the Track A robustness gap.

Lock-split pixel metrics approximate the server protocol, but they are not comparable one-to-one with leaderboard numbers, which use different images.

## Robustness gap

The **robustness gap** is a metric on regular-lit images minus the same metric on shifted-lit images of the same evaluation set. Every scene appears under every lighting condition, so the gap is a paired comparison.

- Track B (`anometa.metrics.aggregate.group_metrics`): `gap_<metric>` per (scenario, seed), then averaged like any other metric.
- Track A: the gap per scenario, then the mean over scenarios.
- A positive AUROC or AUPRC gap means regular lighting scores better. For NLL, ECE and Brier, lower is better, so a positive gap means regular lighting scores worse.
- The gap is reported on the lock split. The dev AUROC gap is one of the search objectives (see [search](dev-search.md)).

## Aggregation

`anometa.metrics.aggregate.group_metrics` reduces per-image predictions to one row per (scenario, seed). `anometa.metrics.aggregate.summarize` then reports:

- `<metric>`: the NaN-ignoring mean over all (scenario, seed) rows.
- `<scenario>/<metric>`: the mean over that scenario's rows.
- `n_nan_<metric>`: the NaN count.

Track B runs also record `fit_latency_ms` and `predict_latency_ms` (medians over the (scenario, seed) fits of a run, so the first fit's weight loading does not count) and, on CUDA, `peak_vram_mb`.

## Bootstrap

Confidence intervals come from `anometa report`, which reads `predictions.parquet`. Runs do not compute them, which keeps search trials fast.

### Confidence intervals

`anometa.metrics.aggregate.bootstrap_ci` gives a 95% interval from 1,000 replicates. Each replicate, per scenario:

1. Resamples the scenario's seeds with replacement.
2. Per label, draws one multiset of evaluation scenes with replacement.
3. Scores every drawn seed on that same scene multiset. A drawn scene brings all its lighting variants. Scenes a seed never evaluated (its dev shot scenes) are skipped.

The replicate's value is the NaN-ignoring mean of the metric over every drawn (scenario, seed). The interval is the 2.5th to 97.5th percentile of the replicates. Sharing the scene draw across seeds keeps seed copies from averaging away the scene-sampling noise. A `gap_` metric recomputes the gap inside each replicate.

Resampling scenes matters because lock sets are small (see [protocol](protocol.md#why-split-the-public-test-set)).

### Paired differences

`anometa.metrics.aggregate.paired_bootstrap_diff` compares two runs that score the same rows: the same scenarios, seeds and images. Two classifiers at the same k and shot lighting do, because shots depend only on seed, scenario, k and lighting.

- Each replicate draws seeds and scenes once and scores both runs on that draw. The interval therefore reflects the noise in the difference, not in each run alone.
- It returns the point difference (A minus B), the 95% interval, and the **share of positive replicates**: the share of replicates where A minus B is above zero.
- The function refuses runs that do not score the same rows with the same labels.

The lighting study uses the same function with every (scenario, lighting) pair as its own stratum, because each pair has its own fitted context (see [protocol](protocol.md#lighting-adaptation)). Results with paired intervals are on the [results](results.md) page.
