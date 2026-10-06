"""Assemble hw.json for the report from the per-machine timing files in hwdata/."""

import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "hwdata"
MACHINES = [
    ("3090", "RTX 3090 (Ryzen 7 7700 host)", "s-tab"),
    ("m4", "M4 Pro, MPS", "s-log"),
    ("ryzen", "Ryzen 7 7700, CPU only", "s-knn"),
]


def load(name):
    p = SRC / name
    return json.loads(p.read_text()) if p.exists() else None


def frozen(tag, classifier, k, field):
    rows = load(f"frozen_dev_{tag}.json") or []
    for r in rows:
        if r["classifier"] == classifier and r["k"] == k and not r.get("n_normals"):
            return r[field]
    return None


def enc_ms(tag, name, sc="fabric"):
    d = load(f"encoder_bench_{tag}.json")
    if not d or f"{name}/{sc}" not in d:
        return None
    return 1000 / d[f"{name}/{sc}"]["images_per_s"]


TRACKA = load("tracka_fit.json") or {}
VIAL_IMAGES = 472  # Vial: train, validation and public test images


def stage(label, sub, fn):
    return {"label": label, "sub": sub, "values": {m: fn(m) for m, _, _ in MACHINES}}


def setup_b(m):
    e = enc_ms(m, "dinov3_l", "vial")
    f = frozen(m, "tabpfn", 2, "fit_ms")
    q = frozen(m, "tabpfn", 2, "predict_ms")
    return None if e is None or f is None else e * VIAL_IMAGES + f + q


stages = [
    stage("TabPFN-3.5 fit", "one scenario, k = 2", lambda m: frozen(m, "tabpfn", 2, "fit_ms")),
    stage(
        "TabPFN-3.5 predict",
        "one scenario's dev images",
        lambda m: frozen(m, "tabpfn", 2, "predict_ms"),
    ),
    stage(
        "TabPFN-3.5-Fast predict",
        "one scenario's dev images",
        lambda m: frozen(m, "tabpfn_fast", 2, "predict_ms"),
    ),
    stage(
        "Logistic regression fit", "one scenario, k = 2", lambda m: frozen(m, "logreg", 2, "fit_ms")
    ),
    stage(
        "TabPFN outlier score",
        "one scenario, no labels",
        lambda m: frozen(m, "tabpfn_outlier", 0, "predict_ms"),
    ),
    stage("DINOv3-L encode", "one 2448×2048 image", lambda m: enc_ms(m, "dinov3_l")),
    stage("DINOv3-S encode", "one 2448×2048 image", lambda m: enc_ms(m, "dinov3_s")),
    stage(
        "TabPFN pipeline, Vial end to end", f"encode {VIAL_IMAGES} images, fit, predict", setup_b
    ),
    stage(
        "PatchCore, Vial end to end",
        "fit + maps for every image",
        lambda m: TRACKA.get(f"patchcore/{m}"),
    ),
    stage(
        "EfficientAD-S, Vial end to end",
        "70,000 steps; M4 and CPU extrapolated",
        lambda m: TRACKA.get(f"efficientad/{m}"),
    ),
]


def fmt(v):
    if v is None:
        return "not run"
    if v >= 3600000:
        return f"{v / 3600000:.1f} h"
    if v >= 60000:
        return f"{v / 60000:.1f} min"
    if v >= 1000:
        return f"{v / 1000:.2f} s"
    return f"{v:.1f} ms" if v >= 1 else f"{v:.2f} ms"


rows = [
    [st["label"] + f' <span class="ci">{st["sub"]}</span>']
    + [fmt(st["values"][m]) for m, _, _ in MACHINES]
    for st in stages
]
auc = []
for m, _lab, _ in MACHINES:
    a = frozen(m, "tabpfn", 2, "auroc")
    auc.append(f"{a:.4f}" if a is not None else "not run")
rows.append(['TabPFN-3.5 dev AUROC, k = 2 <span class="ci">same config everywhere</span>', *auc])
out = {
    "machines": [list(x) for x in MACHINES],
    "stages": [s for s in stages if any(v is not None for v in s["values"].values())],
    "table_headers": ["stage"] + [lab for _, lab, _ in MACHINES],
    "table_rows": rows,
    "xhw": load("xhw_paired.json") or [],
    "text": (HERE / "hw_text.html").read_text() if (HERE / "hw_text.html").exists() else "",
}
(HERE / "hw.json").write_text(json.dumps(out, indent=1))
print(json.dumps(rows, indent=0)[:2000])
