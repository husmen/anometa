"""Collect every number and figure the HTML report needs into docs/report/data/ (run from the repo root where artifacts/ and data/ exist)."""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from anometa.config import Scenario
from anometa.metrics.aggregate import bootstrap_ci, group_metrics, paired_bootstrap_diff, summarize
from anometa.report import load_runs, one_class_on_fewshot_rows, with_variants

OUT = Path("docs/report/data")
OUT.mkdir(parents=True, exist_ok=True)
SC = [s.value for s in Scenario]
data: dict = {}


def run_dir(name):
    (d,) = glob.glob(f"artifacts/{name}-*")
    return Path(d)


def pred(name):
    return pd.read_parquet(run_dir(name) / "predictions.parquet")


# Lock Track B: every frozen run with CI, calibration and per-scenario AUROC.
rows = []
for d in sorted(glob.glob("artifacts/lock-*")):
    name = Path(d).name.rsplit("-", 1)[0]
    if "tracka" in name:
        continue
    cfg = json.loads(Path(d, "manifest.json").read_text())["config"]
    p = pd.read_parquet(Path(d) / "predictions.parquet")
    prob = "score_balanced" in p and p.score_balanced.notna().any()
    a, lo, hi = bootstrap_ci(p, "auroc", probabilistic=False)
    s = summarize(group_metrics(p, probabilistic=bool(prob)))
    rows.append(
        dict(
            name=name,
            classifier=cfg["classifier"],
            k=cfg["k"],
            shot_lighting=cfg["shot_lighting"],
            n_normals=(cfg.get("classifier_params") or {}).get("n_normals"),
            auroc=a,
            lo=lo,
            hi=hi,
            nll_bal=s.get("nll_bal"),
            ece_bal=s.get("ece_bal"),
            gap_auroc=s.get("gap_auroc"),
            per_scenario={sc: s[f"{sc}/auroc"] for sc in SC},
        )
    )
    print("lock", name, round(a, 3), flush=True)
data["lock_trackb"] = rows

# Lock paired comparisons, TabPFN minus control.
paired = []
for k in (1, 2, 5):
    t = pred(f"lock-tabpfn-k{k}")
    ctls = {c: pred(f"lock-{c}-k{k}") for c in ("logreg", "knn", "tabpfn-fast", "tabpfn-n32")}
    ctls["mahalanobis"] = one_class_on_fewshot_rows(pred("lock-mahalanobis"), t)
    ctls["tabpfn-outlier"] = one_class_on_fewshot_rows(pred("lock-tabpfn-outlier"), t)
    for c, cp in ctls.items():
        for metric in ("auroc", "nll_bal", "ece_bal"):
            prob = metric != "auroc"
            if prob and c in ("mahalanobis", "tabpfn-outlier"):
                continue
            d, lo, hi, share = paired_bootstrap_diff(t, cp, metric, probabilistic=prob)
            paired.append(dict(k=k, control=c, metric=metric, diff=d, lo=lo, hi=hi, share=share))
    print("paired k", k, flush=True)
for k in (10, 20):
    d, lo, hi, share = paired_bootstrap_diff(
        pred(f"lock-ablation-tabpfn-k{k}"),
        pred(f"lock-ablation-logreg-k{k}"),
        "auroc",
        probabilistic=False,
    )
    paired.append(
        dict(
            k=k, control="logreg (all lightings)", metric="auroc", diff=d, lo=lo, hi=hi, share=share
        )
    )
data["lock_paired"] = paired

# Lock Track A.
ta = []
for d in sorted(glob.glob("artifacts/lock-tracka-*")):
    m = json.loads(Path(d, "metrics.json").read_text())
    ta.append(dict(name=Path(d).name.rsplit("-", 1)[0], metrics={k: v for k, v in m.items()}))
data["lock_tracka"] = ta

# Reliability bins (balanced probabilities), lock k=2.
rel = {}
for n in ("lock-tabpfn-k2", "lock-logreg-k2", "lock-knn-k2"):
    p = pred(n)
    y = p.label.to_numpy(float)
    q = p.score_balanced.to_numpy(float)
    edges = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(q, edges) - 1, 0, 9)
    rel[n] = [
        dict(
            bin=i, conf=float(q[b == i].mean()), acc=float(y[b == i].mean()), n=int((b == i).sum())
        )
        for i in range(10)
        if (b == i).any()
    ]
data["reliability"] = rel

# Dev: searches, budget curves, grid best per classifier and k.
curves = {}
for f in sorted(glob.glob("artifacts/search/*.parquet")):
    df = pd.read_parquet(f)
    curves[Path(f).stem] = df.sort_values("trial")["best_so_far"].tolist()
data["dev_search"] = curves
r = load_runs(Path("artifacts"), "dev")
b = with_variants(r[r.track == "B"])
data["dev_budget"] = [
    dict(classifier=c, k=int(k), best=float(g.auroc.max()), mean=float(g.auroc.mean()), n=len(g))
    for (c, k), g in b.groupby(["classifier", "k"])
]
data["dev_counts"] = dict(
    runs=len(r), trackb=int((r.track == "B").sum()), tracka=int((r.track == "A").sum())
)

# Latency, RTX 3090 (serial, idle GPU).
lat = []
for d in sorted(glob.glob("artifacts/latency-*")):
    m = json.loads(Path(d, "metrics.json").read_text())
    c = json.loads(Path(d, "manifest.json").read_text())
    lat.append(
        dict(
            classifier=c["config"]["classifier"],
            auroc=m["auroc"],
            fit_ms=m["fit_latency_ms"],
            predict_ms=m["predict_latency_ms"],
            duration_s=c["duration_s"],
        )
    )
data["latency_3090"] = lat
data["tracka_fit_3090"] = {
    Path(d).name: json.loads(Path(d, "metrics.json").read_text()).get("fit_s")
    for d in glob.glob("artifacts/tracka-*")
}
data["parity"] = {
    n: json.loads(Path(f"artifacts/parity_{n}.json").read_text()) for n in ("dinov3_s", "dinov3_l")
}

# Dataset: images per scenario, split and lighting.
ds = []
for sc in SC:
    split = pd.read_csv(f"splits/{sc}.csv")
    z = np.load(glob.glob(f"cache/features/dinov3_l-r512/{sc}-*.npz")[0])
    ids = z["image_id"].astype(str)
    ds.append(
        dict(
            scenario=sc,
            train=int(sum(i.startswith("train/") for i in ids)),
            validation=int(sum(i.startswith("validation/") for i in ids)),
            dev_good=int(((split.split == "dev") & (split.label == 0)).sum()),
            dev_bad=int(((split.split == "dev") & (split.label == 1)).sum()),
            lock_good=int(((split.split == "lock") & (split.label == 0)).sum()),
            lock_bad=int(((split.split == "lock") & (split.label == 1)).sum()),
            lightings=sorted(split.lighting.unique().tolist()),
            image_size=list(Image.open(next(Path(f"data/ad2/{sc}/train/good").glob("*.png"))).size),
        )
    )
data["dataset"] = ds

# Example anomaly maps: one lock defect per scenario, regular lighting.
dist = np.load(run_dir("lock-tracka-distance-dinov3-l") / "maps.npz")
pc = np.load(run_dir("lock-tracka-patchcore") / "maps.npz")
ead = np.load(run_dir("lock-tracka-efficientad-s") / "maps.npz")
examples = []
for sc in ("vial", "walnuts", "sheet_metal", "fruit_jelly", "can", "fabric"):
    split = pd.read_csv(f"splits/{sc}.csv")
    cand = (
        split[(split.split == "lock") & (split.label == 1) & (split.lighting == "regular")]
        .image_id.sort_values()
        .tolist()
    )
    iid = cand[0]
    img = Image.open(f"data/ad2/{sc}/{iid}.png").convert("RGB")
    mask = Image.open(
        f"data/ad2/{sc}/"
        + iid.replace("test_public/bad/", "test_public/ground_truth/bad/")
        + "_mask.png"
    ).convert("L")
    w, h = img.size
    scale = 260 / h
    size = (max(1, int(w * scale)), 260)
    img_s = np.asarray(img.resize(size))
    mask_s = np.asarray(mask.resize(size, Image.NEAREST)) > 127
    panels = [("image", img_s, None), ("ground truth", img_s, mask_s)]
    for label, maps in (("DINOv3-L distance", dist), ("PatchCore", pc), ("EfficientAD-S", ead)):
        m = maps[f"{sc}/{iid}"].astype(np.float32)
        m = np.asarray(Image.fromarray(m).resize(size, Image.BILINEAR))
        panels.append((label, img_s, m))
    fig = Figure(figsize=(len(panels) * size[0] / 100, 2.9), dpi=100)
    axes = fig.subplots(1, len(panels))
    for ax, (label, im, ov) in zip(axes, panels, strict=False):
        ax.imshow(im)
        if ov is not None and ov.dtype == bool:
            ax.contour(ov, levels=[0.5], colors="#ff3b30", linewidths=1.2)
        elif ov is not None:
            lo_, hi_ = np.percentile(ov, [1, 99.5])
            ax.imshow(np.clip((ov - lo_) / (hi_ - lo_ + 1e-9), 0, 1), cmap="inferno", alpha=0.55)
            ax.contour(mask_s, levels=[0.5], colors="#36d1ff", linewidths=0.8)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.01, wspace=0.03)
    fig.savefig(OUT / f"example_{sc}.png", dpi=100)
    examples.append(dict(scenario=sc, image_id=iid))
data["examples"] = examples
(OUT / "data.json").write_text(json.dumps(data, indent=1, default=float))
print("REPORT DATA DONE")
