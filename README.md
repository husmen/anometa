<p align="center">
    <img src="/logo.png" height="100">
    <h3 align="center">Anometa</h3>
</p>

<p align="center">
  Bringing TabPFN 3.5 to industrial vision
</p>

<p align="center">
  <a href="https://husmen.github.io/anometa/">Project page</a> ·
  <a href="https://husmen.github.io/anometa/docs/">Docs</a> ·
  <a href="https://husmen.github.io/anometa/report/">Full report</a> ·
  <a href="https://husmen.github.io/anometa/#video">Video</a>
</p>

## What is Anometa?

**Anometa** = **ano**maly + **meta**. "Meta" has two meanings:

- **Meta-learning, today.** TabPFN is a meta-learned model: it was pre-trained on millions of synthetic tasks and learns a new task in context, from a table of examples, with no training per task. Anometa applies it to anomaly detection on frozen vision-foundation-model features.
- **A meta-framework, tomorrow.** The experiment API, artifacts and protocol are not tied to one model or dataset, so other foundation models, detectors and benchmarks can plug into the same pipeline.

The initial focus is TabPFN-3.5 on MVTec AD 2, built for the Prior Labs TabPFN-3.5 hackathon. Anometa investigates whether TabPFN 3.5 can leverage frozen vision-foundation-model representations for few-shot industrial anomaly detection: how visual representations, compact feature projections and TabPFN interact under limited labelled data, with attention to label efficiency, calibration and robustness to lighting and domain shifts. More datasets, models and tracks can follow (see the [roadmap](https://husmen.github.io/anometa/docs/roadmap.html)).

## How it works

- **Track A, unsupervised baselines:** PatchCore and EfficientAD-S (anomalib) and a training-free DINOv3 patch distance map, all fitted on defect-free images only. They report image AUROC and pixel metrics (AU-PRO@0.05/0.30, SegF1).
- **Track B, few-shot adaptation:** frozen DINOv3 or SigLIP2 features (CLS, mean patch and a patch-novelty statistic), PCA, then TabPFN-3.5 (or 3.5-Fast) trained on the defect-free images plus k = 1, 2 or 5 labelled defects. Logistic regression and kNN are the supervised controls; Mahalanobis distance and TabPFN's unsupervised outlier score are label-free controls.
- **Protocol:** each scenario's labelled public test set is split by scene into a dev half for every experiment and a lock half that is evaluated once, after the configuration is frozen. Few-shot defects come only from regular-lit dev images. Every run writes `artifacts/<run-id>/` with a manifest (git commit, dataset and split hashes, model revisions, licences, seeds, hardware), metrics and per-image predictions.

## Results

Lock split, evaluated once after the configuration was frozen on the dev split (full tables: [`reports/lock/results.md`](reports/lock/results.md); every experiment on the way there: [experiment log](https://husmen.github.io/anometa/docs/experiment-log.html)). Track B uses DINOv3-L CLS, mean-patch and patch-novelty features, PCA to 16 dimensions, 10 seeds; brackets are 95% bootstrap intervals over seeds and scenes.

| image AUROC, all 8 scenarios | k=1 | k=2 | k=5 |
|---|---|---|---|
| **TabPFN-3.5** | **0.761** [0.702, 0.814] | **0.778** [0.722, 0.828] | **0.789** [0.734, 0.840] |
| TabPFN-3.5-Fast | 0.758 | 0.775 | 0.790 |
| Logistic regression | 0.683 [0.633, 0.732] | 0.740 [0.688, 0.785] | 0.745 [0.681, 0.802] |
| kNN | 0.567 | 0.608 | 0.662 |
| Mahalanobis, no labels | 0.766 | | |
| TabPFN unsupervised outlier score, no labels | 0.778 | | |

- With one labelled defect, TabPFN-3.5 beats logistic regression by 0.078 AUROC (paired 95% interval [+0.009, +0.146]); at k = 2 and 5 it leads by about 0.04, within noise.
- TabPFN is far better calibrated: its balanced NLL is about half of logistic regression's at every k (0.65 vs 1.25 at k=1), with intervals well clear of zero.
- It ties the label-free controls: on AD2, 1–5 labels add little over a good novelty score, and TabPFN is the only supervised model here that matches one (logistic regression and kNN score below both label-free controls).
- TabPFN-3.5-Fast matches TabPFN-3.5 within 0.003 at about a third of the predict time (59 ms vs 0.19 s for one scenario at k = 2 on an RTX 3090).

| pixel-level (Track A, unsupervised) | AU-PRO@0.05 | AU-PRO@0.30 | SegF1 | image AUROC |
|---|---|---|---|---|
| **DINOv3-L patch distance (training-free)** | **0.385** | **0.583** | **0.375** | **0.780** |
| PatchCore | 0.222 | 0.460 | 0.198 | 0.720 |
| EfficientAD-S | 0.182 | 0.371 | 0.150 | 0.653 |

### Comparison with the MVTec AD 2 paper

AU-PRO@0.05 in %. Ours: lock split of the public test set, all lighting conditions pooled. Paper: Table VII of [Heckler-Kram et al.](https://arxiv.org/abs/2503.21622), private test set, regular lighting (`TESTpriv`), evaluated by MVTec's server. Best of 7 methods: PatchCore, RD, RD++, EfficientAD, MSFlow, SimpleNet, DSR.

| scenario | DINOv3-L distance (ours) | PatchCore, ours / paper | EfficientAD-S, ours / paper | best in paper |
|---|---|---|---|---|
| Can | 6.3 | 0.1 / 4.7 | 4.1 / 9.6 | **13.9** (DSR) |
| Fabric | 19.6 | 4.5 / 11.0 | 15.7 / 22.2 | **22.2** (EfficientAD) |
| Fruit Jelly | 42.9 | 35.4 / 46.7 | 34.1 / 50.5 | **54.4** (RD++) |
| Rice | **39.5** | 11.7 / 25.6 | 2.9 / 27.6 | 27.6 (EfficientAD) |
| Sheet Metal | **47.6** | 9.4 / 15.2 | 8.7 / 11.8 | 18.0 (DSR) |
| Vial | **84.6** | 47.0 / 62.2 | 62.2 / 55.6 | 63.0 (RD++) |
| Wall Plugs | **26.8** | 15.4 / 12.8 | 2.4 / 20.3 | 20.3 (EfficientAD) |
| Walnuts | 41.1 | 54.3 / 51.8 | 15.7 / 48.8 | **51.8** (PatchCore) |
| **Mean** | **38.5** | 22.2 / 28.8 | 18.2 / 30.8 | 30.8 (EfficientAD) |

- The training-free DINOv3-L distance map has the highest mean. It beats the paper's best method on 4 of 8 scenarios (Rice, Sheet Metal, Vial, Wall Plugs) and loses on the other 4 (Can, Fabric, Fruit Jelly, Walnuts). The higher mean comes mainly from Sheet Metal, Rice and Vial.
- Our PatchCore and EfficientAD-S score below the paper's on 6 and 7 of 8 scenarios. Neither was tuned: both use the paper's stated settings with anomalib defaults (256×256 input; PatchCore wide_resnet50_2, coreset ratio 0.01; EfficientAD-S 70,000 steps), one setting for all scenarios, and Imagenette instead of ImageNet as EfficientAD's penalty set. Our pixel-level DINOv3 methods were not tuned either.
- The test sets differ (public vs private, pooled lighting vs regular lighting) and the lock sets are small, so this is not a ranking. The results show promise for frozen DINOv3 patch features. A conclusive comparison needs tuned baselines and a tuned pipeline of our own, scored on the private test set through MVTec's server.

Reproduction: the best dev configurations give the same AUROC within 0.001 on a CPU-only Ryzen 7 7700 and within 0.004 on an M4 Pro, and the ungated timm DINOv3 weights match the gated ones within noise.

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). Reported numbers come from an RTX 3090 (Linux, NVIDIA driver r580 or newer, since the PyPI torch wheels bundle CUDA 13.0). Track B on cached features also runs on a laptop CPU or Apple silicon.

```bash
uv sync --extra gui
uv run pytest
```

## Licence and access checklist

- **TabPFN-3.5** weights are non-commercial (`tabpfn-3-5-license-v1.0`). Accept the licence on [platform.priorlabs.ai](https://platform.priorlabs.ai), then set `TABPFN_TOKEN`, `TABPFN_NO_BROWSER=1` and `TABPFN_DISABLE_TELEMETRY=1`.
- **DINOv3** weights come from the gated `facebook/dinov3-*` repositories: request access on Hugging Face and set `HF_TOKEN`. Without access, `anometa extract --backend auto` falls back to timm's ungated copies of the same weights (configs then need `encoder_backend: timm`, since the backend is part of each experiment).
- **SigLIP2** is Apache-2.0 and ungated.
- **MVTec AD 2** is CC BY-NC-SA 4.0 and non-commercial. It is downloaded from the official links and never redistributed; the archive hashes are pinned in `configs/data/sha256sums.txt`.

## Quickstart

```bash
uv run anometa download --scenario vial   # AD2 archive, hash-pinned (omit --scenario for all 8)
uv run anometa split                      # dev/lock splits
uv run anometa extract --encoder dinov3_l # frozen DINOv3-L features
uv run anometa run configs/experiments/smoke_vial.yaml --set k=5 classifier=tabpfn
uv run anometa demo                       # Streamlit few-shot demo
```

Every command (Track A, grid, Optuna, TabPFN-BO, report, lock, parity, Rerun), the `--set` overrides and the seed-pool settings are in the [getting started guide](https://husmen.github.io/anometa/docs/getting-started.html). Build the docs locally with `uv run --group docs sphinx-build -W docs docs/_build/html`.

## Limitations

- Lock-split numbers use half of AD2's public test set (split by scene), so they are not directly comparable with published AD2 numbers, which use the private test set (see [Comparison with the MVTec AD 2 paper](#comparison-with-the-mvtec-ad-2-paper)).
- PatchCore and EfficientAD-S run with fixed default settings and no tuning, and they score below the paper's numbers for the same methods.
- SegF1 is pooled over all images of a scenario, which is our reading of the metric.
- Lock sets are small (7 defect scenes and 2–6 good scenes per scenario, each under 4–7 lighting conditions), so the bootstrap intervals are wide.
- Track A runs at 256×256 (as in the AD2 paper); Track B encodes DINOv3 at a 512-pixel short side and SigLIP2 at up to 1,024 patches.

## Acknowledgements

- DINOv3 by Meta AI: this project uses DINO Materials under the DINOv3 License.
- TabPFN-3.5 by Prior Labs.
- MVTec AD 2 by MVTec Software GmbH (Heckler-Kram et al., 2025), CC BY-NC-SA 4.0.

## Citation

```bibtex
@misc{anometa2026,
  title  = {Anometa: Few-Shot Industrial Anomaly Detection with TabPFN-3.5
            on Frozen Vision Foundation Model Features},
  author = {Menhour, Houssem},
  year   = {2026},
  url    = {https://github.com/husmen/anometa}
}
```

## Licence

Apache-2.0 (see `LICENSE`). Model weights and the dataset keep their own licences.

## References

- [TabPFN 3.5: Technical Report](https://arxiv.org/abs/2609.17895)
- [TabPFN beyond tabular Data: Calibration and Accuracy on Multimodal Embeddings](https://arxiv.org/abs/2607.11007)
- [The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection](https://arxiv.org/abs/2503.21622)
- [The MVTec AD 2 Dataset: Downloads](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2/downloads)
- [Anomalib: A deep learning library for anomaly detection](https://github.com/openvinotoolkit/anomalib)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)
