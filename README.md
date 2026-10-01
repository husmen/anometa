<p align="center">
    <img src="/logo.png" height="100">
    <h3 align="center">Anometa</h3>
</p>

<p align="center">
  Bringing TabPFN 3.5 to industrial vision
</p>

## What is Anometa?

Anometa investigates whether TabPFN 3.5 can effectively leverage frozen vision-foundation-model representations for few-shot industrial anomaly detection. Using MVTec AD 2 as the primary benchmark, the project explores how visual representations, compact feature projections, and TabPFN interact under limited labeled data, with particular attention to label efficiency, calibration, and robustness to illumination and domain shifts. The initial focus is on establishing a reproducible TabPFN 3.5 pipeline and understanding when and why it can complement powerful pretrained vision representations.

## How it works

- **Track A, unsupervised baselines:** PatchCore and EfficientAD-S (anomalib) and a training-free DINOv3 patch distance map, all fitted on defect-free images only. They report image AUROC and pixel metrics (AU-PRO@0.05/0.30, SegF1).
- **Track B, few-shot adaptation:** frozen DINOv3 or SigLIP2 features (CLS, mean patch and a patch-novelty statistic), PCA, then TabPFN-3.5 (or 3.5-Fast) trained on the defect-free images plus k = 1, 2 or 5 labelled defects. Logistic regression and kNN are the supervised controls; Mahalanobis distance and TabPFN's unsupervised outlier score are label-free controls.
- **Protocol:** each scenario's labelled public test set is split by scene into a dev half for every experiment and a lock half that is evaluated once, after the configuration is frozen. Few-shot defects come only from regular-lit dev images. Every run writes `artifacts/<run-id>/` with a manifest (git commit, dataset and split hashes, model revisions, licences, seeds, hardware), metrics and per-image predictions.

## Results

Lock-split results are added after the single lock evaluation. Dev-split progress is recorded in [`docs/plans/PLAN_1_RUN_LOG.md`](docs/plans/PLAN_1_RUN_LOG.md).

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

## Commands

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
| Rerun recording | `uv run anometa inspect <run-id>` |
| few-shot demo | `uv run anometa demo` |

`--set key=value` overrides config fields (YAML-typed); `null` removes a field, so it falls back to its default. Track B runs one seed at a time by default; see `AGENTS.md` for the optional seed pool.

## Limitations

- Lock-split numbers use half of AD2's public test set (split by scene), so they are not comparable with published AD2 numbers, which use the private test set.
- SegF1 is pooled over all images of a scenario, which is our reading of the metric.
- Lock sets are small (7 defect scenes and 2–6 good scenes per scenario, each under 4–7 lighting conditions), so the bootstrap intervals are wide.
- Track A runs at 256×256 (as in the AD2 paper); Track B encodes DINOv3 at a 512-pixel short side and SigLIP2 at up to 1,024 patches.

## Acknowledgements

- DINOv3 by Meta AI: this project uses DINO Materials under the DINOv3 License.
- TabPFN-3.5 by Prior Labs.
- MVTec AD 2 by MVTec Software GmbH (Heckler-Kram et al., 2025), CC BY-NC-SA 4.0.

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
