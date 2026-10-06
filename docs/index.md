# Anometa

**Few-shot industrial anomaly detection with TabPFN-3.5 on frozen vision foundation model features.**

[Project page](https://husmen.github.io/anometa/) · [Full report](https://husmen.github.io/anometa/report/) · [Code](https://github.com/husmen/anometa) · [Explainer video](https://husmen.github.io/anometa/#video)

Factories need to catch defects, but defects are rare: often there are a handful of labelled examples, sometimes one. Anometa asks whether TabPFN-3.5, a foundation model for tables, can turn a few labelled defect images into a useful detector. It never trains a vision model. A frozen DINOv3 encoder turns each image into one short table row, and TabPFN classifies new rows in context, in one forward pass.

The name is **ano**maly + **meta**. Today "meta" means meta-learning: TabPFN was pre-trained on millions of synthetic tasks and learns a new one from its context. Tomorrow it means a meta-framework: the experiment API, artifacts and protocol are not tied to one model or dataset. TabPFN-3.5 and MVTec AD 2 are the starting point.

```{figure} report/data/trackc_gallery.webp
:alt: One defect image per MVTec AD 2 scenario with the DINOv3-L distance, PatchCore and TabPFN-Fast anomaly maps

TabPFN as a defect-map generator (dev-split prototype): one row per image patch, scored by TabPFN-3.5-Fast. Rows: image, DINOv3-L distance, PatchCore (tuned, 512 px), TabPFN-Fast. Best case per scenario, chosen on purpose (Fabric has no image where TabPFN's peak sits on the defect); typical results are in [results](results.md).
```

## At a glance

Lock split of MVTec AD 2, evaluated once after the configuration was frozen on the dev split. Image AUROC over 8 scenarios and 10 seeds, with 95% bootstrap intervals over seeds and scenes.

| | k = 1 | k = 2 | k = 5 |
|---|---|---|---|
| **TabPFN-3.5** | **0.761** [0.702, 0.814] | **0.778** [0.722, 0.828] | **0.789** [0.734, 0.840] |
| Logistic regression | 0.683 [0.633, 0.732] | 0.740 [0.688, 0.785] | 0.745 [0.681, 0.802] |
| Mahalanobis, no labels | 0.766 | | |

- **One labelled defect is enough to beat logistic regression:** +0.078 AUROC at k = 1 (paired interval [+0.009, +0.146]). At k = 2 and 5 the lead of about 0.04 is within noise.
- **Calibrated without tuning:** TabPFN's balanced log loss is about half of logistic regression's (0.65 vs 1.25 at k = 1).
- **The features do most of the work:** label-free novelty scores on the same DINOv3 features tie TabPFN. The training-free DINOv3-L distance map ties the best pixel method (AU-PRO@0.05 0.385, the same as PatchCore tuned on dev to 512×512; PatchCore 0.222 at its 256×256 default, EfficientAD-S 0.182).
- **Fast and reproducible:** a TabPFN fit takes about 5 ms; one scenario runs end to end in about 14 s on an RTX 3090. Results match within 0.005 AUROC on an Apple M4 Pro and within 0.001 on a CPU.

The [results](results.md) page has every table, the per-scenario comparison with the MVTec AD 2 paper and the post-freeze studies.

## How it works

1. **Encode.** A frozen DINOv3-L vision transformer reads each image. Its global token and mean patch, compressed by PCA, plus two patch-novelty numbers form one row of 18 values ([tracks](tracks.md)).
2. **Classify in context.** TabPFN-3.5 reads about 300 normal rows and k labelled defect rows as its context and predicts the defect probability of every test row. There are no gradient updates.
3. **Evaluate honestly.** The labelled public test set is split by scene into a dev half for every experiment and a lock half that is scored once, after a commit freezes the configuration ([protocol](protocol.md), [metrics](metrics.md)).
4. **Record everything.** One function, `run_experiment`, runs every experiment and writes a manifest, metrics and per-image predictions ([architecture](architecture.md), [experiment log](experiment-log.md)).

## Quickstart

```bash
uv sync --extra gui
uv run anometa download --scenario vial   # MVTec AD 2 archive, hash-pinned
uv run anometa split                      # dev/lock splits
uv run anometa extract --encoder dinov3_l # frozen DINOv3-L features
uv run anometa run configs/experiments/smoke_vial.yaml --set k=5 classifier=tabpfn
uv run anometa demo                       # Streamlit few-shot demo
```

TabPFN-3.5 weights need a licence acceptance and a token; DINOv3 needs gated Hugging Face access or the timm fallback. See [getting started](getting-started.md).

## Where to go next

| page | what you find there |
|---|---|
| [Getting started](getting-started.md) | install, licences and access, every command |
| [Data and protocol](protocol.md) | MVTec AD 2, the dev/lock split, guardrails, post-freeze studies |
| [Tracks and models](tracks.md) | Track A baselines, Track B features and classifiers, the patch-map prototype |
| [Results](results.md) | lock tables, the paper comparison, lighting, Thinking and patch maps |
| [Roadmap](roadmap.md) | TabPFN anomaly maps, unsupervised TabPFN, server submission, broader scope |
| [API reference](api/index.md) | every module of the `anometa` package |

```{toctree}
:caption: Guide
:maxdepth: 2
:hidden:

getting-started
protocol
architecture
tracks
metrics
dev-search
gui
reproducibility
```

```{toctree}
:caption: Results
:maxdepth: 2
:hidden:

results
experiment-log
```

```{toctree}
:caption: Project
:maxdepth: 2
:hidden:

roadmap
provenance
glossary
api/index
```
