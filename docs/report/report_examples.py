"""Render the example anomaly maps (one lock defect per scenario) into docs/report/data/ (run from the repo root)."""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from anometa.config import Scenario

OUT = Path("docs/report/data")
OUT.mkdir(parents=True, exist_ok=True)
SC = [s.value for s in Scenario]
data: dict = {}


def run_dir(name):
    (d,) = glob.glob(f"artifacts/{name}-*")
    return Path(d)


def pred(name):
    return pd.read_parquet(run_dir(name) / "predictions.parquet")


data = json.loads((OUT / "data.json").read_text())
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
    panels = [("image, defect outlined", img_s, mask_s)]
    for label, maps in (("DINOv3-L distance", dist), ("PatchCore", pc), ("EfficientAD-S", ead)):
        m = maps[f"{sc}/{iid}"].astype(np.float32)
        m = np.asarray(Image.fromarray(m).resize(size, Image.BILINEAR))
        panels.append((label, img_s, m))
    wide = w / h > 1.6
    if wide:
        pw = 650 / 100
        ph = pw * size[1] / size[0]
        fig = Figure(figsize=(2 * pw, 2 * (ph + 0.3)), dpi=100)
        axes = fig.subplots(2, 2).ravel()
    else:
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
    if wide:
        fig.subplots_adjust(
            left=0.005, right=0.995, top=0.93, bottom=0.005, wspace=0.02, hspace=0.16
        )
    else:
        fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.01, wspace=0.03)
    fig.savefig(OUT / f"example_{sc}.png", dpi=100)
    examples.append(dict(scenario=sc, image_id=iid))
data["examples"] = examples
(OUT / "data.json").write_text(json.dumps(data, indent=1, default=float))
print("REPORT DATA DONE")
