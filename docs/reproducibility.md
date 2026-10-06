# Reproducibility and hardware

Every run records what it depended on, so another machine can rerun it and compare. This page covers what makes runs deterministic, which machine is canonical, and how close other machines get.

## Determinism

- **Splits.** `anometa split` splits each scenario's labelled public test set by scene, with seed 0, stratified by label. The split files `splits/<scenario>.csv` are committed and never overwritten. See [protocol](protocol.md).
- **Few-shot shots.** Shots come from a generator keyed by (seed, scenario). A smaller k always gets a prefix of a larger k's shots for the same seed. Track B runs seeds 0–9 by default.
- **Hashes.** Each manifest records a `dataset_hash` (from the pinned archive SHA-256s in `configs/data/sha256sums.txt`), a `split_hash` (from the split files), the `config_hash`, the git commit and whether the tree was dirty. See [architecture](architecture.md#artifact-first-runs).
- **Pinned models.** Encoders load fixed Hugging Face revisions:
  - `facebook/dinov3-vits16-pretrain-lvd1689m` @ `114c1379950215c8b35dfcd4e90a5c251dde0d32`
  - `facebook/dinov3-vitl16-pretrain-lvd1689m` @ `ea8dc2863c51be0a264bab82070e3e8836b02d51`
  - `timm/vit_small_patch16_dinov3.lvd1689m` @ `3bf4720a82ec2066db88137180ff1f83a675cef0` (fallback)
  - `timm/vit_large_patch16_dinov3.lvd1689m` @ `30c1109559f65dea34316b0d4842d35c5771fe11` (fallback)
  - `google/siglip2-so400m-patch16-naflex` @ `cc24074f717b612951c2dead130904ab9b65a81e`
- **Pinned packages.** `uv.lock` fixes every dependency version. The manifest records the Python version and the versions of the key packages.
- **Run ids.** A run id is the config name plus the first 12 hex characters of the config hash. The same config gives the same run id on every machine, so runs on different hardware can be paired.

## Canonical hardware

The RTX 3090 host (Linux) is canonical for every reported metric, latency and VRAM figure. It needs an NVIDIA driver r580 or newer, because the locked PyPI `torch` wheels bundle CUDA 13.0.

### bf16 on CUDA

Encoders run in bfloat16 on CUDA and float32 on MPS and CPU. DINOv3-L overflows to NaN in float16 on CUDA, so float16 is not used. Features are upcast to float32 before caching, and extraction rejects non-finite features. Against float32, bfloat16 gives a CLS cosine of at least 0.99 and a patch cosine of 0.977.

## Latency and VRAM

Canonical figures: RTX 3090, one seed at a time, nothing else on the GPU. Best dev configurations, all 8 scenarios. Fit and predict are medians per (scenario, seed); predict covers one scenario's whole dev evaluation set.

| configuration | dev AUROC | fit | predict | peak VRAM | run wall time |
|---|---|---|---|---|---|
| TabPFN-3.5, k=5, DINOv3-L cls+mean_patch, PCA 128 | 0.761 | 11.2 ms | 276 ms | 1106 MB | 27.0 s |
| TabPFN-3.5-Fast, k=5, DINOv3-L cls+mean_patch, PCA 64 | 0.760 | 5.7 ms | 70.6 ms | 470 MB | 9.2 s |
| logistic regression, k=5, DINOv3-L cls+mean_patch+novelty, PCA 16 | 0.775 | 19.6 ms | 0.22 ms | – (CPU) | 4.5 s |
| Mahalanobis, k=0, DINOv3-L novelty | 0.723 | 0.60 ms | 0.26 ms | – (CPU) | 0.6 s |
| TabPFN outlier score, k=0, DINOv3-L novelty | 0.717 | 0.16 ms | 776 ms | 876 MB | 8.8 s |

- TabPFN's fit is cheap because it learns in context. "Fit" stores the context; the work happens at predict time.
- TabPFN-3.5-Fast predicts 3.9 times faster than TabPFN-3.5, with the same AUROC.
- Each process loads a TabPFN checkpoint once and reuses it. Before that, every fit rebuilt and randomly initialised the network, and a fit took 911 ms instead of 11 ms.
- The TabPFN outlier score refits a regressor for every conditional, so its cost grows with the number of features.

Track A: EfficientAD-S trains at about 29 steps per second on the 3090, so its 70,000 steps take about 41 minutes per scenario on an idle GPU. On the lock split, Vial took 44 minutes.

## Seed execution and latencies

Track B fits seeds one at a time by default. `ANOMETA_SEED_WORKERS=<n>` fits n seeds at once, and `ANOMETA_SEED_EXECUTOR=thread|process` picks the pool (default `thread`). Threads are refused on MPS, because torch's MPS backend aborts under concurrent threads, so MPS needs `process`.

These are execution settings, not experiment settings. Run ids don't change, and the manifest's `hardware` block records them. Scores are exactly equal to serial runs.

Concurrent seeds inflate the per-seed latencies, so reported latencies come from serial runs. With a cached checkpoint, a pool only adds overhead. One benchmark config (TabPFN, DINOv3-L, cls+mean_patch, PCA 128, k=5, 20 fits):

| setting | run time | fit latency |
|---|---|---|
| serial | 8.2 s | 11 ms |
| threads ×2 / ×4 / ×8 | 8.9 / 10.5 / 13.7 s | 22–106 ms |
| processes ×2 / ×4 / ×8 | 12.0 / 13.1 / 17.5 s | 20–85 ms |

## Cross-hardware reproduction

Track B reruns from the cached features on two more machines: an M4 Pro (MPS) and a CPU-only Ryzen 7 7700 (run with `CUDA_VISIBLE_DEVICES=""`).

All 20 frozen Track B configurations, rerun on the dev split on all three machines:

- Logistic regression, kNN and Mahalanobis are bit-identical.
- TabPFN-3.5, TabPFN-3.5-Fast and the TabPFN outlier score move by at most 0.005 AUROC on the M4 Pro and 0.001 on the CPU.
- All 40 paired 95% intervals against the 3090 include zero.

The differences come from float kernels: MPS kernels differ more from CUDA than CPU kernels do.

Best dev configurations, AUROC relative to the 3090 (paired 95% interval):

| configuration | 3090 | Ryzen CPU | M4 Pro |
|---|---|---|---|
| TabPFN-3.5, k=5 | 0.7612 | 0.7612 (−0.0000 [−0.0034, +0.0034]) | 0.7595 (−0.0017 [−0.0061, +0.0018]) |
| TabPFN-3.5-Fast, k=5 | 0.7602 | 0.7593 (−0.0009 [−0.0039, +0.0017]) | 0.7563 (−0.0039 [−0.0153, +0.0042]) |
| logistic regression, k=5 | 0.7752 | 0.7752 | 0.7752 |
| Mahalanobis | 0.7231 | 0.7231 | 0.7231 |
| TabPFN outlier score | 0.7168 | 0.7168 (+0.0000 [−0.0015, +0.0014]) | 0.7172 (+0.0004 [−0.0020, +0.0028]) |

### Speed on each machine

Frozen configuration, TabPFN-3.5 at k=2, one scenario's dev images:

| | RTX 3090 | M4 Pro | Ryzen CPU |
|---|---|---|---|
| TabPFN-3.5 fit | 4.4 ms | 5.1 ms | 4.8 ms |
| TabPFN-3.5 predict | 0.19 s | 0.82 s | 1.54 s |
| TabPFN-3.5-Fast predict | 59 ms | 202 ms | 470 ms |
| TabPFN outlier score | 0.76 s | 3.3 s | 4.8 s |
| DINOv3-L encode, one Fabric-sized image | 20 ms | 234 ms | 1.12 s |
| DINOv3-S encode, one Fabric-sized image | 12 ms | 33 ms | 110 ms |

Vial end to end, every image:

| pipeline | RTX 3090 | M4 Pro | Ryzen CPU |
|---|---|---|---|
| Track B: DINOv3-L encode, TabPFN fit and predict | 14 s | 2.1 min | 9.8 min |
| PatchCore | 18 s | 2.6 min | 2.5 min |
| EfficientAD-S (70,000 steps) | 44 min | about 4.1 h | about 10.1 h |

- The encoder dominates Track B's cost on every machine. TabPFN itself never takes more than 1.5 s per scenario.
- On the CPU, PatchCore is about 4 times faster than the Track B pipeline, because DINOv3-L encoding at a 512-pixel short side is expensive there. DINOv3-S encodes 10 times faster on the CPU.
- EfficientAD-S times on the M4 Pro and the CPU are extrapolated from short runs (1,000–2,000 and 500–1,000 steps).
- PatchCore on all 8 scenarios on the M4 Pro (with `PYTORCH_ENABLE_MPS_FALLBACK=1`) matches the 3090's pixel metrics (AU-PRO@0.05 0.207 against 0.202). Its image AUROC is 0.03 lower (0.619 against 0.652), likely because the random coreset draw differs on MPS.

## DINOv3 backend parity

The gated `facebook/dinov3-*` weights load through transformers. The ungated timm copies are the fallback. The two implementations differ slightly: transformers loads explicit zero QKV biases and keeps RoPE periods as bfloat16 buffers, while timm drops the zero biases and computes RoPE periods in float32.

```bash
uv run anometa parity --encoder dinov3_l
```

`anometa parity` encodes the first `--n` images (default 32) of a scenario (default `vial`) with both backends. It reports the CLS cosine, the mean and minimum patch cosine and the maximum absolute difference, plus throughput and peak VRAM on a Sheet Metal-sized and a Fabric-sized input. It writes the result to `artifacts/parity_<encoder>.json`.

Results on 32 Vial images, bfloat16 on the 3090:

| | DINOv3-S | DINOv3-L |
|---|---|---|
| mean CLS cosine | 0.99985 | 0.99990 |
| mean patch cosine | 0.99987 | 0.99981 |
| worst single patch cosine | 0.995 | 0.953 |

- Throughput and VRAM are the same on both backends. DINOv3-L encodes 11.7 images per second on a Sheet Metal-sized input and 49 on a Fabric-sized one, with 1.2–1.3 GB of VRAM.
- Downstream (DINOv3-S, PCA 32, k=2, 10 seeds), the transformers-minus-timm AUROC difference stays inside its paired interval: on Vial, logistic regression +0.003 [−0.001, +0.014] and TabPFN 0.000; on Fruit Jelly, logistic regression +0.001 [−0.023, +0.032] and TabPFN +0.012 [−0.036, +0.070].

Without DINOv3 access, the timm fallback gives results equivalent within noise.
