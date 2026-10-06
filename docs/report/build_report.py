"""Build the static HTML report (`index.html`) from `data/`, `hw.json` and the templates here.

Rebuild: `uv run python docs/report/build_report.py`. The inputs are committed, so this needs no
artifacts. Refreshing the inputs needs the run artifacts: `report_data.py` and `report_examples.py`
(run from the repo root on the machine with `artifacts/` and `data/ad2`) rewrite `data/`, and
`assemble_hw.py` rewrites `hw.json` from `hwdata/`.
"""

import base64
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
D = json.loads((HERE / "data" / "data.json").read_text())
HW = json.loads((HERE / "hw.json").read_text()) if (HERE / "hw.json").exists() else {}
SC = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
SC_LABEL = {s: s.replace("_", " ").title() for s in SC}
SC_LABEL["fruit_jelly"] = "Fruit Jelly"
SC_LABEL["sheet_metal"] = "Sheet Metal"
SC_LABEL["wallplugs"] = "Wall Plugs"

esc = html.escape


def f3(x):
    return f"{x:.3f}"


def lock(name):
    return next(r for r in D["lock_trackb"] if r["name"] == name)


def tracka(model):
    return next(r["metrics"] for r in D["lock_tracka"] if r["name"] == f"lock-tracka-{model}")


def paired(k, control, metric="auroc"):
    return next(
        p
        for p in D["lock_paired"]
        if p["k"] == k and p["control"] == control and p["metric"] == metric
    )


# ---------- SVG helpers ----------
def svg(w, h, body, label, cls="chart"):
    return (
        f'<svg class="{cls}" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" '
        f'preserveAspectRatio="xMidYMid meet">{body}</svg>'
    )


def text(x, y, s, cls="ax", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{esc(str(s))}</text>'


def line(x1, y1, x2, y2, cls="grid", extra=""):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" {extra}/>'


def rect(x, y, w, h, cls, extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" class="{cls}" {extra}/>'


def box(x, y, w, h, title, sub="", cls="node"):
    out = rect(x, y, w, h, cls, 'rx="6"')
    lines_ = [title] + ([sub] if isinstance(sub, str) and sub else list(sub) if sub else [])
    n = len(lines_)
    for i, s in enumerate(lines_):
        ty = y + h / 2 + (i - (n - 1) / 2) * 15 + 4
        out += text(x + w / 2, ty, s, "nt" if i == 0 else "ns", "middle")
    return out


def arrow(x1, y1, x2, y2, cls="edge"):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" marker-end="url(#ah)"/>'


def path_arrow(d, cls="edge"):
    return f'<path d="{d}" class="{cls}" fill="none" marker-end="url(#ah)"/>'


DEFS = (
    '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahead"/></marker></defs>'
)

SERIES = {
    "tabpfn": ("TabPFN-3.5", "s-tab"),
    "tabpfn_fast": ("TabPFN-3.5-Fast", "s-fast"),
    "logreg": ("Logistic regression", "s-log"),
    "knn": ("kNN", "s-knn"),
    "tabpfn-n32": ("TabPFN-3.5, 32 normals", "s-n32"),
}


# ---------- Charts ----------
def chart_budget():
    W, H, L, R, T, B = 720, 380, 56, 200, 20, 44
    ymin, ymax = 0.5, 0.86
    xs = {1: 0, 2: 1, 5: 2}

    def X(k, off=0.0):
        return L + (xs[k] + 0.5) * (W - L - R) / 3 + off

    def Y(v):
        return T + (ymax - v) / (ymax - ymin) * (H - T - B)

    body = ""
    for v in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        body += line(L, Y(v), W - R, Y(v)) + text(L - 8, Y(v) + 4, f"{v:.2f}", "ax", "end")
    for k in xs:
        body += text(X(k), H - B + 22, f"k = {k}", "ax", "middle")
    body += text(
        L + (W - L - R) / 2, H - 6, "labelled defect images per scenario (shots)", "axl", "middle"
    )
    # Label-free references.
    refs = [
        (lock("lock-tabpfn-outlier")["auroc"], "TabPFN outlier score, no labels", "ref-a"),
        (lock("lock-mahalanobis")["auroc"], "Mahalanobis, no labels", "ref-b"),
        (tracka("distance-dinov3-l")["auroc"], "DINOv3-L patch distance, no labels", "ref-c"),
        (tracka("patchcore")["auroc"], "PatchCore, no labels", "ref-d"),
        (tracka("efficientad-s")["auroc"], "EfficientAD-S, no labels", "ref-e"),
    ]
    placed = []
    for v, lab, c in sorted(refs, key=lambda r: -r[0]):
        body += line(L, Y(v), W - R, Y(v), c)
        ty = Y(v) + 4
        while any(abs(ty - p) < 13 for p in placed):
            ty += 13
        placed.append(ty)
        body += text(W - R + 8, ty, f"{lab} {v:.3f}", "lab " + c + "-t")
    offs = {"tabpfn": -14, "tabpfn_fast": -5, "logreg": 5, "knn": 14, "tabpfn-n32": 0}
    for key in ("knn", "logreg", "tabpfn-n32", "tabpfn_fast", "tabpfn"):
        label, cls = SERIES[key]
        name = {"tabpfn_fast": "tabpfn-fast"}.get(key, key)
        pts = []
        for k in (1, 2, 5):
            r = lock(f"lock-{name}-k{k}")
            x = X(k, offs[key])
            pts.append((x, Y(r["auroc"])))
            body += line(x, Y(r["lo"]), x, Y(r["hi"]), cls + " ci")
        body += '<polyline class="{} ln" points="{}"/>'.format(
            cls, " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        )
        for x, y in pts:
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" class="{cls} pt"/>'
    # Legend.
    ly = T + 4
    for key in ("tabpfn", "tabpfn_fast", "tabpfn-n32", "logreg", "knn"):
        label, cls = SERIES[key]
        body += f'<circle cx="{L + 14}" cy="{ly}" r="4" class="{cls} pt"/>' + text(
            L + 24, ly + 4, label, "lab"
        )
        ly += 15
    return svg(W, H, body, "Lock AUROC against the number of shots, with 95% intervals")


def chart_forest():
    rows = []
    names = {
        "logreg": "Logistic regression",
        "knn": "kNN",
        "mahalanobis": "Mahalanobis (no labels)",
        "tabpfn-outlier": "TabPFN outlier score (no labels)",
        "tabpfn-fast": "TabPFN-3.5-Fast",
        "tabpfn-n32": "TabPFN-3.5, 32 normals",
    }
    for k in (1, 2, 5):
        rows.append(("head", f"k = {k}", None))
        for c in names:
            rows.append(("row", names[c], paired(k, c)))
    rows.append(("head", "Shots from every lighting (ablation)", None))
    for k in (10, 20):
        rows.append(("row", f"Logistic regression, k = {k}", paired(k, "logreg (all lightings)")))
    W, L, R, T = 720, 250, 70, 26
    rh = 19
    H = T + rh * len(rows) + 40
    xmin, xmax = -0.15, 0.30

    def X(v):
        return L + (v - xmin) / (xmax - xmin) * (W - L - R)

    body = ""
    for v in (-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        body += line(X(v), T - 6, X(v), H - 34, "grid zero" if v == 0 else "grid")
        body += text(X(v), H - 18, f"{v:+.2f}" if v else "0", "ax", "middle")
    body += text(
        X(0.075),
        14,
        "TabPFN-3.5 AUROC minus the other method (paired, 95% interval)",
        "axl",
        "middle",
    )
    y = T
    for kind, label, p in rows:
        if kind == "head":
            body += text(8, y + 14, label, "hd")
        else:
            sig = p["lo"] > 0 or p["hi"] < 0
            cls = "fp sig" if sig else "fp"
            body += text(18, y + 13, label, "lab")
            body += line(X(max(p["lo"], xmin)), y + 9, X(min(p["hi"], xmax)), y + 9, cls + " ci")
            body += f'<circle cx="{X(p["diff"]):.1f}" cy="{y + 9:.1f}" r="4" class="{cls} pt"/>'
            body += text(
                W - 4, y + 13, f"{p['diff']:+.3f}", "num " + ("sigt" if sig else ""), "end"
            )
        y += rh
    return svg(
        W, H, body, "Paired differences between TabPFN-3.5 and every other method on the lock split"
    )


def chart_calibration():
    W, H, L, R, T, B = 720, 314, 56, 16, 46, 44
    groups = [1, 2, 5]
    keys = [
        ("tabpfn", "s-tab"),
        ("tabpfn-fast", "s-fast"),
        ("tabpfn-n32", "s-n32"),
        ("logreg", "s-log"),
        ("knn", "s-knn"),
    ]
    vmax = 1.4

    def Y(v):
        return T + (vmax - min(v, vmax)) / vmax * (H - T - B)

    body = ""
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        body += line(L, Y(v), W - R, Y(v)) + text(L - 8, Y(v) + 4, f"{v:.1f}", "ax", "end")
    gw = (W - L - R) / 3
    bw = gw / (len(keys) + 1.5)
    for gi, k in enumerate(groups):
        gx = L + gi * gw + bw * 0.75
        for i, (name, cls) in enumerate(keys):
            v = lock(f"lock-{name}-k{k}")["nll_bal"]
            x = gx + i * bw
            body += rect(x, Y(v), bw - 3, Y(0) - Y(v), cls + " bar")
            body += text(
                x + (bw - 3) / 2, Y(v) - 4, f"{v:.2f}" if v < vmax else f"{v:.1f}↑", "num", "middle"
            )
        body += text(L + gi * gw + gw / 2, H - B + 20, f"k = {k}", "ax", "middle")
    body += text(
        L, 16, "Balanced negative log-likelihood on the lock split (lower is better)", "axl"
    )
    lx = L + 8
    for name, cls in keys:
        lab = {
            "tabpfn": "TabPFN-3.5",
            "tabpfn-fast": "Fast",
            "tabpfn-n32": "32 normals",
            "logreg": "Logistic regression",
            "knn": "kNN (off scale)",
        }[name]
        body += rect(lx, H - 16, 10, 10, cls + " bar") + text(lx + 14, H - 7, lab, "lab")
        lx += 30 + len(lab) * 6.2
    return svg(W, H, body, "Balanced NLL per classifier and k on the lock split")


def chart_reliability():
    W, H, L, R, T, B = 340, 350, 50, 14, 30, 66
    s = min(W - L - R, H - T - B)

    def X(v):
        return L + v * s

    def Y(v):
        return T + (1 - v) * s

    body = ""
    for v in (0, 0.25, 0.5, 0.75, 1):
        body += line(X(v), Y(0), X(v), Y(1)) + line(X(0), Y(v), X(1), Y(v))
        body += text(X(v), Y(0) + 16, f"{v:g}", "ax", "middle") + text(
            X(0) - 6, Y(v) + 4, f"{v:g}", "ax", "end"
        )
    body += line(X(0), Y(0), X(1), Y(1), "diag")
    for name, cls in (
        ("lock-knn-k2", "s-knn"),
        ("lock-logreg-k2", "s-log"),
        ("lock-tabpfn-k2", "s-tab"),
    ):
        pts = [(X(b["conf"]), Y(b["acc"])) for b in D["reliability"][name]]
        body += '<polyline class="{} ln" points="{}"/>'.format(
            cls, " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        )
        for (x, y), b in zip(pts, D["reliability"][name], strict=False):
            r = 2 + 3.5 * math.sqrt(b["n"] / 2500)
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" class="{cls} pt"/>'
    body += text(L, 16, "Reliability, k = 2 (balanced prior)", "axl")
    body += text(X(0.5), Y(0) + 32, "predicted P(anomalous)", "ax", "middle")
    lx = L
    for lab, cls in (("TabPFN-3.5", "s-tab"), ("Logistic regression", "s-log"), ("kNN", "s-knn")):
        body += f'<circle cx="{lx + 5}" cy="{H - 10}" r="4" class="{cls} pt"/>' + text(
            lx + 13, H - 6, lab, "lab"
        )
        lx += 24 + len(lab) * 6.4
    body += text(
        12,
        Y(0.5),
        "observed share anomalous",
        "ax",
        "middle",
        f'transform="rotate(-90 12 {Y(0.5):.1f})"',
    )
    return svg(W, H, body, "Reliability diagram for TabPFN, logistic regression and kNN at k = 2")


def heat(rows, title, vmin, vmax, fmt=f3, note=""):
    W = 760
    L, T, cw, rh = 230, 46, (760 - 230 - 8) / 8, 24
    H = T + rh * len(rows) + 12
    body = text(L, 16, title, "axl")
    for j, sc in enumerate(SC):
        body += text(L + cw * (j + 0.5), T - 10, SC_LABEL[sc], "ax", "middle")
    for i, (label, vals, cls) in enumerate(rows):
        y = T + i * rh
        body += text(L - 10, y + rh / 2 + 4, label, "lab " + (cls or ""), "end")
        for j, sc in enumerate(SC):
            v = vals.get(sc)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            a = max(0.04, min(1, (v - vmin) / (vmax - vmin)))
            body += rect(
                L + cw * j + 1,
                y + 1,
                cw - 2,
                rh - 2,
                "hc",
                f'style="fill-opacity:{0.06 + 0.62 * a:.2f}"',
            )
            body += text(L + cw * (j + 0.5), y + rh / 2 + 4, fmt(v), "num", "middle")
    if note:
        body += text(L, H - 2, note, "ax")
    return svg(W, H, body, title, "chart wide")


def chart_scenarios():
    rows = []
    for name, label, cls in (
        ("lock-tabpfn-k2", "TabPFN-3.5, k = 2", "s-tab-t"),
        ("lock-tabpfn-k5", "TabPFN-3.5, k = 5", "s-tab-t"),
        ("lock-logreg-k2", "Logistic regression, k = 2", ""),
        ("lock-logreg-k5", "Logistic regression, k = 5", ""),
        ("lock-knn-k5", "kNN, k = 5", ""),
        ("lock-mahalanobis", "Mahalanobis, no labels", ""),
        ("lock-tabpfn-outlier", "TabPFN outlier, no labels", ""),
    ):
        rows.append((label, lock(name)["per_scenario"], cls))
    for model, label in (
        ("distance-dinov3-l", "DINOv3-L distance, no labels"),
        ("patchcore", "PatchCore, no labels"),
        ("efficientad-s", "EfficientAD-S, no labels"),
    ):
        m = tracka(model)
        rows.append((label, {sc: m[f"{sc}/auroc"] for sc in SC}, ""))
    return heat(rows, "Image AUROC per scenario, lock split (darker is better)", 0.3, 1.0)


def chart_tracka():
    W, H, L, R, T, B = 720, 290, 56, 16, 30, 50
    methods = [
        ("distance-dinov3-l", "DINOv3-L distance", "s-dl"),
        ("distance-dinov3-s", "DINOv3-S distance", "s-ds"),
        ("patchcore", "PatchCore", "s-pc"),
        ("efficientad-s", "EfficientAD-S", "s-ead"),
    ]
    metrics = [
        ("au_pro_005", "AU-PRO@0.05"),
        ("au_pro_030", "AU-PRO@0.30"),
        ("seg_f1", "SegF1"),
        ("auroc", "image AUROC"),
    ]
    vmax = 0.85

    def Y(v):
        return T + (vmax - v) / vmax * (H - T - B)

    body = ""
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        body += line(L, Y(v), W - R, Y(v)) + text(L - 8, Y(v) + 4, f"{v:.1f}", "ax", "end")
    gw = (W - L - R) / len(metrics)
    bw = gw / (len(methods) + 1.2)
    for gi, (mk, ml) in enumerate(metrics):
        for i, (model, _lab, cls) in enumerate(methods):
            v = tracka(model)[mk]
            x = L + gi * gw + bw * 0.6 + i * bw
            body += rect(x, Y(v), bw - 3, Y(0) - Y(v), cls + " bar")
            body += text(x + (bw - 3) / 2, Y(v) - 4, f"{v:.2f}", "num", "middle")
        body += text(L + gi * gw + gw / 2, H - B + 18, ml, "ax", "middle")
    lx = L + 8
    for _model, lab, cls in methods:
        body += rect(lx, H - 16, 10, 10, cls + " bar") + text(lx + 14, H - 7, lab, "lab")
        lx += 34 + len(lab) * 6.3
    body += text(
        L, 16, "Track A on the lock split: pixel metrics and image AUROC (higher is better)", "axl"
    )
    return svg(W, H, body, "Track A lock metrics")


def chart_tracka_scen():
    rows = []
    for model, label in (
        ("distance-dinov3-l", "DINOv3-L distance"),
        ("distance-dinov3-s", "DINOv3-S distance"),
        ("patchcore", "PatchCore"),
        ("efficientad-s", "EfficientAD-S"),
    ):
        m = tracka(model)
        rows.append((label, {sc: m[f"{sc}/au_pro_005"] for sc in SC}, ""))
    return heat(rows, "AU-PRO@0.05 per scenario, lock split", 0.0, 0.9)


def chart_search():
    W, H, L, R, T, B = 720, 280, 56, 140, 30, 44
    ymin, ymax = 0.68, 0.755
    n = 60

    def X(i):
        return L + (i - 1) / (n - 1) * (W - L - R)

    def Y(v):
        return T + (ymax - max(v, ymin)) / (ymax - ymin) * (H - T - B)

    body = ""
    for v in (0.68, 0.70, 0.72, 0.74):
        body += line(L, Y(v), W - R, Y(v)) + text(L - 8, Y(v) + 4, f"{v:.2f}", "ax", "end")
    for i in (1, 10, 20, 30, 40, 50, 60):
        body += text(X(i), H - B + 18, str(i), "ax", "middle")
    body += text(L + (W - L - R) / 2, H - 8, "trial", "axl", "middle")
    for meth, lab, cls in (("tpe", "Optuna TPE", "s-log"), ("bo", "TabPFN-surrogate BO", "s-tab")):
        curves = [v for k, v in D["dev_search"].items() if k.startswith(meth)]
        for c in curves:
            body += '<polyline class="{} ln thin" points="{}"/>'.format(
                cls, " ".join(f"{X(i + 1):.1f},{Y(v):.1f}" for i, v in enumerate(c))
            )
        mean = [sum(c[i] for c in curves) / len(curves) for i in range(n)]
        body += '<polyline class="{} ln" points="{}"/>'.format(
            cls, " ".join(f"{X(i + 1):.1f},{Y(v):.1f}" for i, v in enumerate(mean))
        )
        body += text(
            W - R + 8,
            Y(mean[-1]) + (4 if meth == "bo" else -6),
            f"{lab} {mean[-1]:.3f}",
            "lab " + cls + "-t",
        )
    body += text(
        L, 16, "Best dev AUROC found so far, k = 2 (3 seeds each; thick line is the mean)", "axl"
    )
    return svg(W, H, body, "Search convergence on the dev split")


def chart_devlock():
    W, H, L, R, T, B = 420, 300, 50, 150, 30, 30
    ymin, ymax = 0.66, 0.80
    dev = {"tabpfn": [0.708, 0.708, 0.730], "logreg": [0.707, 0.736, 0.775]}
    lck = {
        "tabpfn": [lock(f"lock-tabpfn-k{k}")["auroc"] for k in (1, 2, 5)],
        "logreg": [lock(f"lock-logreg-k{k}")["auroc"] for k in (1, 2, 5)],
    }

    def Y(v):
        return T + (ymax - v) / (ymax - ymin) * (H - T - B)

    x0, x1 = L + 20, W - R - 10
    body = line(x0, T, x0, H - B, "grid") + line(x1, T, x1, H - B, "grid")
    body += text(x0, H - 10, "dev", "ax", "middle") + text(x1, H - 10, "lock", "ax", "middle")
    for v in (0.68, 0.72, 0.76, 0.80):
        body += text(L - 6, Y(v) + 4, f"{v:.2f}", "ax", "end")
    placed = []
    for key, cls, lab in (("tabpfn", "s-tab", "TabPFN"), ("logreg", "s-log", "Logreg")):
        for i, k in enumerate((1, 2, 5)):
            body += line(x0, Y(dev[key][i]), x1, Y(lck[key][i]), cls + " ln")
            body += f'<circle cx="{x0}" cy="{Y(dev[key][i]):.1f}" r="3.5" class="{cls} pt"/>'
            body += f'<circle cx="{x1}" cy="{Y(lck[key][i]):.1f}" r="3.5" class="{cls} pt"/>'
            ty = Y(lck[key][i]) + 4
            while any(abs(ty - p) < 12 for p in placed):
                ty += 12
            placed.append(ty)
            body += text(x1 + 8, ty, f"{lab} k={k}  {lck[key][i]:.3f}", "lab " + cls + "-t")
    body += text(L, 16, "Frozen configuration, dev against lock AUROC", "axl")
    return svg(W, H, body, "Dev and lock AUROC for TabPFN and logistic regression")


def chart_dataset():
    W = 760
    L, T, rh = 110, 34, 24
    H = T + rh * len(SC) + 40
    rows = {r["scenario"]: r for r in D["dataset"]}
    parts = [
        ("train", "train (normal)", "d-train"),
        ("validation", "validation (normal)", "d-val"),
        ("dev_good", "dev good", "d-devg"),
        ("dev_bad", "dev defect", "d-devb"),
        ("lock_good", "lock good", "d-lockg"),
        ("lock_bad", "lock defect", "d-lockb"),
    ]
    vmax = max(sum(rows[s][p] for p, _, _ in parts) for s in SC)
    scale = (W - L - 70) / vmax
    body = text(
        L,
        16,
        "Images per scenario and role (public test images include every lighting condition)",
        "axl",
    )
    for i, sc in enumerate(SC):
        y = T + i * rh
        body += text(L - 10, y + 16, SC_LABEL[sc], "lab", "end")
        x = L
        for p, _, cls in parts:
            w = rows[sc][p] * scale
            body += rect(x, y + 3, w, rh - 6, cls)
            x += w
        body += text(x + 6, y + 16, str(sum(rows[sc][p] for p, _, _ in parts)), "num")
    lx = L
    for _p, lab, cls in parts:
        body += rect(lx, H - 18, 10, 10, cls) + text(lx + 14, H - 9, lab, "lab")
        lx += 30 + len(lab) * 6.2
    return svg(W, H, body, "Dataset composition", "chart wide")


def chart_hardware():
    """Log-scale bars of measured times per stage and machine."""
    if not HW:
        return '<p class="note">Hardware timings were not collected.</p>'
    stages = HW["stages"]
    machines = HW["machines"]
    W = 760
    L, T, rh = 250, 40, 15
    gh = rh * len(machines) + 10
    H = T + gh * len(stages) + 50
    vals = [v for st in stages for v in st["values"].values() if v]
    lo, hi = math.floor(math.log10(min(vals))), math.ceil(math.log10(max(vals)))

    def X(ms):
        return L + (math.log10(max(ms, 10**lo)) - lo) / (hi - lo) * (W - L - 70)

    body = text(L, 16, "Measured time, log scale (shorter is faster)", "axl")
    for e in range(lo, hi + 1):
        x = X(10**e)
        lab = f"{10**e:g} ms" if e < 3 else f"{10 ** (e - 3):,.0f} s"
        body += line(x, T - 6, x, H - 40) + text(x, H - 26, lab, "ax", "middle")
    y = T
    for st in stages:
        body += text(L - 10, y + gh / 2 + 2, st["label"], "lab", "end")
        if st.get("sub"):
            body += text(L - 10, y + gh / 2 + 15, st["sub"], "ax", "end")
        for j, (mk, _mlab, cls) in enumerate(machines):
            v = st["values"].get(mk)
            yy = y + j * rh + 4
            if v is None:
                body += text(L + 4, yy + 10, "not run", "ax")
                continue
            body += rect(L, yy, X(v) - L, rh - 3, cls + " bar")
            body += text(X(v) + 5, yy + 10, fmt_ms(v), "num")
        y += gh
    lx = L
    for _mk, mlab, cls in machines:
        body += rect(lx, H - 14, 10, 10, cls + " bar") + text(lx + 14, H - 5, mlab, "lab")
        lx += 40 + len(mlab) * 6.2
    return svg(W, H, body, "Measured times per stage and machine", "chart wide")


def chart_xhw():
    """Paired AUROC difference to the RTX 3090 for every frozen config, per machine."""
    rows = [
        r
        for r in HW.get("xhw", [])
        if r["classifier"] in ("tabpfn", "tabpfn_fast", "tabpfn_outlier")
    ]
    if not rows:
        return ""
    names = {
        "tabpfn": "TabPFN-3.5",
        "tabpfn_fast": "TabPFN-3.5-Fast",
        "tabpfn_outlier": "TabPFN outlier",
    }
    order = sorted(rows, key=lambda r: (r["machine"], r["classifier"], r["k"], r["params"]))
    W, L, R, T, rh = 720, 300, 60, 30, 16
    H = T + rh * len(order) + 40
    xmin, xmax = -0.02, 0.02

    def X(v):
        return L + (v - xmin) / (xmax - xmin) * (W - L - R)

    body = text(
        L,
        16,
        "AUROC difference to the RTX 3090 run of the same config (paired, 95% interval)",
        "axl",
    )
    for v in (-0.02, -0.01, 0, 0.01, 0.02):
        body += line(X(v), T - 4, X(v), H - 34, "grid zero" if v == 0 else "grid") + text(
            X(v), H - 18, f"{v:+.2f}" if v else "0", "ax", "middle"
        )
    y = T
    for r in order:
        mach = {"m4": "M4 Pro", "ryzen": "Ryzen CPU"}[r["machine"]]
        n32 = ", 32 normals" if "n_normals" in r["params"] else ""
        k = f", k = {r['k']}" if r["k"] else ""
        cls = "s-log" if r["machine"] == "m4" else "s-knn"
        body += text(L - 10, y + 11, f"{mach}: {names[r['classifier']]}{k}{n32}", "lab", "end")
        body += line(X(r["lo"]), y + 7, X(r["hi"]), y + 7, cls + " ci")
        body += f'<circle cx="{X(r["diff"]):.1f}" cy="{y + 7:.1f}" r="3.5" class="{cls} pt"/>'
        y += rh
    return svg(W, H, body, "Cross-hardware agreement of TabPFN scores")


def fmt_ms(v):
    if v >= 3600000:
        return f"{v / 3600000:.1f} h"
    if v >= 60000:
        return f"{v / 60000:.1f} min"
    if v >= 1000:
        return f"{v / 1000:.2f} s"
    if v >= 1:
        return f"{v:.1f} ms"
    return f"{v:.2f} ms"


LIGHT = json.loads((HERE / "data" / "lighting_analysis.json").read_text())
TRACKC = [
    json.loads(row)
    for row in (HERE / "data" / "results.jsonl").read_text().splitlines()
    if row.strip()
]


def chart_lighting():
    """Shifted-lighting AUROC and balanced NLL against the number of adaptation scenes, k = 2."""
    W, H = 760, 300
    panels = [
        ("shifted/auroc", "AUROC on shifted lighting", 0.56, 0.76),
        ("shifted/nll_bal", "balanced NLL on shifted lighting", 0.4, 1.2),
    ]
    series = [
        ("tabpfn", "TabPFN-3.5", "s-tab"),
        ("tabpfn_fast", "TabPFN-3.5-Fast", "s-fast"),
        ("logreg", "Logistic regression", "s-log"),
        ("mahalanobis", "Mahalanobis (no labels)", "s-ref"),
        ("knn", "kNN", "s-knn"),
    ]
    body = ""
    pw = 300
    for pi, (metric, title, lo, hi) in enumerate(panels):
        L = 60 + pi * (pw + 80)
        T, B = 30, 50

        def X(m, L=L):
            return L + m / 2 * pw

        def Y(v, T=T, B=B, lo=lo, hi=hi):
            return T + (hi - min(max(v, lo), hi)) / (hi - lo) * (H - T - B)

        body += text(L, 16, title + " (k = 2)", "axl")
        for v in [lo + i * (hi - lo) / 4 for i in range(5)]:
            body += line(L, Y(v), L + pw, Y(v)) + text(L - 8, Y(v) + 4, f"{v:.2f}", "ax", "end")
        for m in (0, 1, 2):
            body += text(X(m), H - B + 18, str(m), "ax", "middle")
        body += text(
            L + pw / 2,
            H - 14,
            "good scenes under the new light added to the context",
            "ax",
            "middle",
        )
        for clf, _lab, cls in series:
            vals = [LIGHT["metrics"].get(f"{clf}|2|{m}", {}).get(metric) for m in (0, 1, 2)]
            if any(v is None or v != v or v > hi for v in vals):
                continue  # kNN's balanced NLL (about 8) is off this scale
            body += '<polyline class="{} ln" points="{}"/>'.format(
                cls, " ".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in enumerate(vals))
            )
            for m, v in enumerate(vals):
                body += f'<circle cx="{X(m):.1f}" cy="{Y(v):.1f}" r="3.8" class="{cls} pt"/>'
    lx = 60
    for _clf, lab, cls in series:
        body += f'<circle cx="{lx + 5}" cy="{H + 12}" r="4" class="{cls} pt"/>' + text(
            lx + 13, H + 16, lab, "lab"
        )
        lx += 30 + len(lab) * 6.3
    return svg(
        W,
        H + 24,
        body,
        "Lighting adaptation: shifted-lighting AUROC and calibration against the number of adaptation scenes",
    )


def table_lighting():
    names = {
        "tabpfn m=2 - m=0, k=2": "TabPFN, 2 adaptation scenes vs none (primary), AUROC",
        "tabpfn m=2 - m=0 nll_bal, k=2": "TabPFN, 2 adaptation scenes vs none, balanced NLL",
        "tabpfn - mahalanobis, m=2, k=2": "TabPFN vs Mahalanobis, AUROC",
        "tabpfn - logreg, m=2, k=2": "TabPFN vs logistic regression, AUROC",
        "tabpfn - logreg nll_bal, m=2, k=2": "TabPFN vs logistic regression, balanced NLL",
        "logreg m=2 - m=0, k=2": "Logistic regression, 2 adaptation scenes vs none, AUROC",
    }
    comp = {c["name"]: c for c in LIGHT["comparisons"]}
    rows = []
    for key, lab in names.items():
        c1 = comp.get(key)
        c2 = comp.get(key.replace("k=2", "k=1"))
        rows.append(
            [lab]
            + [
                pdiff({"diff": c["diff"], "lo": c["lo"], "hi": c["hi"]}) if c else "–"
                for c in (c2, c1)
            ]
        )
    return table(["comparison on shifted lighting", "k = 1", "k = 2"], rows)


TC_METHODS = [
    ("dinov3_distance", "DINOv3 distance", "s-dl"),
    ("patchcore", "PatchCore", "s-pc"),
    ("fused_z", "fused (no labels)", "s-ref"),
    ("logreg", "logreg rows", "s-log"),
    ("tabpfn", "TabPFN-3.5 rows", "s-n32"),
    ("tabpfn_fast", "TabPFN-Fast rows", "s-tab"),
]


def _tc_mean(sc, k, m, metric="au_pro_005"):
    import statistics

    vals = [
        r[metric]
        for r in TRACKC
        if (sc is None or r["scenario"] == sc) and r["k"] == k and r["method"] == m
    ]
    return statistics.mean(vals) if vals else None


def chart_trackc(k=5):
    groups = [*SC, None]
    W, H, L, R, T, B = 760, 300, 46, 10, 30, 62
    vmax = 0.9

    def Y(v):
        return T + (vmax - v) / vmax * (H - T - B)

    body = text(
        L, 16, f"AU-PRO@0.05 on dev images, k = {k}, mean of 3 seeds (higher is better)", "axl"
    )
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        body += line(L, Y(v), W - R, Y(v)) + text(
            L - 8, Y(v) + 4, f"{v:.1f}", "ax", "middle" if False else "end"
        )
    gw = (W - L - R) / len(groups)
    bw = gw / (len(TC_METHODS) + 1.3)
    for gi, sc in enumerate(groups):
        for i, (m, _lab, cls) in enumerate(TC_METHODS):
            v = _tc_mean(sc, k, m)
            if v is None:
                continue
            x = L + gi * gw + bw * 0.65 + i * bw
            body += rect(x, Y(v), bw - 1.5, Y(0) - Y(v), cls + " bar")
        lab = "mean" if sc is None else SC_LABEL[sc]
        body += text(L + gi * gw + gw / 2, H - B + 16, lab, "ax" if sc else "axl", "middle")
    lx = L
    for _m, lab, cls in TC_METHODS:
        body += rect(lx, H - 22, 10, 10, cls + " bar") + text(lx + 14, H - 13, lab, "lab")
        lx += 26 + len(lab) * 6.0
    return svg(W, H, body, f"Track C prototype, AU-PRO@0.05 per scenario, k = {k}", "chart wide")


def table_trackc():
    import numpy as np

    by = {}
    for r in TRACKC:
        by.setdefault((r["scenario"], r["k"], r["seed"]), {})[r["method"]] = r["au_pro_005"]
    rng = np.random.default_rng(0)
    rows = []
    for b, lab in (
        ("dinov3_distance", "DINOv3 distance map"),
        ("fused_z", "fused map, no labels"),
        ("logreg", "logistic regression on the same rows"),
        ("patchcore", "PatchCore map"),
    ):
        cells = []
        for k in (2, 5):
            d = np.array(
                [
                    v["tabpfn_fast"] - v[b]
                    for u, v in by.items()
                    if u[1] == k and "tabpfn_fast" in v and b in v
                ]
            )
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(pdiff({"diff": float(d.mean()), "lo": float(lo), "hi": float(hi)}))
        rows.append([lab, *cells])
    return table(["TabPFN-Fast map minus", "k = 2", "k = 5"], rows)


# ---------- Diagrams ----------
def diag_split():
    W, H = 760, 300
    b = DEFS
    b += box(10, 20, 150, 56, "MVTec AD 2 scenario", "one of 8")
    b += box(10, 120, 150, 56, "train + validation", "normal images only")
    b += box(10, 210, 150, 70, "test_public", ("labelled good + defect", "every lighting"))
    b += arrow(85, 76, 85, 118) + path_arrow("M 160 40 L 190 40 L 190 226 L 162 226", "edge faint")
    b += box(220, 200, 150, 40, "split by scene", "", "node alt")
    b += arrow(160, 258, 218, 228)
    b += box(430, 150, 150, 56, "dev half", ("all experiments", "and model choice"), "node dev")
    b += box(430, 240, 150, 56, "lock half", ("evaluated once,", "after the freeze"), "node lockn")
    b += arrow(370, 215, 428, 182) + arrow(370, 225, 428, 262)
    b += box(620, 120, 130, 50, "k shots", ("regular-lit defects", "from dev only"), "node shot")
    b += box(620, 192, 130, 48, "dev evaluation", ("dev minus the", "shots' scenes"))
    b += box(620, 252, 130, 44, "lock evaluation", "all lock images")
    b += arrow(580, 170, 618, 148) + arrow(580, 182, 618, 214) + arrow(580, 268, 618, 274)
    b += box(
        220,
        40,
        360,
        70,
        "fit set for Track B",
        ("all train normals (label 0)", "+ k shots (label 1)"),
        "node fit",
    )
    b += arrow(160, 148, 218, 86) + path_arrow("M 685 120 L 685 75 L 582 75")
    b += text(12, 196, "Track A fits on train images only", "ax")
    return svg(W, H, b, "How each scenario is split into training, dev and lock data", "diagram")


def diag_timeline():
    W, H = 760, 120
    b = DEFS
    steps = [
        ("Dev experiments", "grid, searches, ablations"),
        ("Freeze", "configs committed (68f5301)"),
        ("Single lock run", "28 configs, once"),
        ("Report", "reports/lock"),
    ]
    w = 160
    for i, (t, s) in enumerate(steps):
        x = 10 + i * (w + 36)
        b += box(x, 30, w, 60, t, s, "node " + ("lockn" if i == 2 else "dev" if i == 0 else ""))
        if i:
            b += arrow(x - 34, 60, x - 2, 60)
    b += text(10, 16, "Order is fixed: no lock result can feed back into any choice.", "axl")
    return svg(W, H, b, "Protocol timeline", "diagram")


def diag_trackb():
    W, H = 780, 330
    b = DEFS
    b += box(10, 120, 110, 60, "image", ("full resolution,", "any lighting"))
    b += box(160, 110, 140, 80, "frozen encoder", ("DINOv3-S / L", "or SigLIP2"), "node enc")
    b += arrow(120, 150, 158, 150)
    b += box(340, 20, 150, 46, "CLS token", "global summary")
    b += box(340, 86, 150, 46, "mean of patches", "average appearance")
    b += box(340, 170, 150, 60, "patch tokens", ("one per 16×16 px", "(not stored)"))
    b += arrow(300, 130, 338, 43) + arrow(300, 140, 338, 109) + arrow(300, 160, 338, 200)
    b += box(
        340, 255, 150, 60, "train patch bank", ("every patch of", "every train normal"), "node alt"
    )
    b += box(
        530, 205, 140, 70, "nearest-neighbour", ("distance per patch", "→ distance map"), "node"
    )
    b += arrow(490, 200, 528, 228) + arrow(490, 285, 528, 255)
    b += box(
        530,
        20,
        140,
        112,
        "PCA",
        ("fit on train normals", "CLS + mean patch", "→ 16 to 128 dims"),
        "node",
    )
    b += arrow(490, 43, 528, 60) + arrow(490, 109, 528, 95)
    b += box(530, 150, 140, 42, "novelty", "max, top-1% mean", "node")
    b += arrow(600, 205, 600, 194)
    b += box(696, 60, 80, 150, "table row", ("one per", "image"), "node row")
    b += arrow(670, 76, 694, 100) + arrow(670, 171, 694, 160)
    b += path_arrow("M 670 255 L 736 255 L 736 300 L 520 300", "edge faint")
    b += text(560, 318, "Track A uses the map directly", "ax")
    return svg(W, H, b, "Track B feature pipeline: from an image to one table row", "diagram")


def diag_tabpfn():
    W, H = 780, 290
    b = DEFS
    b += text(
        10,
        16,
        "Few-shot detection as one table: TabPFN reads the labelled rows and predicts the rest in one forward pass",
        "axl",
    )
    cols = ["f1", "f2", "…", "f18", "label"]
    x0, y0, cw, rh = 20, 40, 44, 20
    rows_ = (
        [("normal", "0", "ctx")] * 5
        + [("…", "", "ctx")]
        + [("defect", "1", "shot")] * 2
        + [("test", "?", "test")] * 3
    )
    for j, c in enumerate(cols):
        b += text(x0 + cw * (j + 0.5), y0 - 6, c, "ax", "middle")
    for i, (lab, y, cls) in enumerate(rows_):
        yy = y0 + i * rh
        for j in range(len(cols)):
            b += rect(x0 + cw * j, yy, cw - 2, rh - 2, "cell " + cls)
        b += text(x0 + cw * (len(cols) - 0.5), yy + 14, y, "num", "middle")
        b += text(x0 + cw * len(cols) + 8, yy + 14, lab, "ax")
    b += text(
        x0,
        y0 + rh * len(rows_) + 16,
        "about 300 normal rows + k defect rows (context), then test rows",
        "ax",
    )
    # Model.
    mx = 360
    b += box(mx, 50, 230, 210, "TabPFN-3.5", "", "node enc")
    b += box(mx + 20, 90, 190, 40, "attention across features", "within each row", "node inner")
    b += box(
        mx + 20, 145, 190, 40, "attention across rows", "test rows read context rows", "node inner"
    )
    b += text(mx + 115, 205, "× many layers", "ns", "middle")
    b += text(mx + 115, 225, "pre-trained on synthetic tables;", "ns", "middle")
    b += text(mx + 115, 240, "no gradient updates at use time", "ns", "middle")
    b += arrow(270, 150, mx - 2, 150)
    b += box(
        625,
        110,
        145,
        80,
        "P(defect)",
        ("per test row,", "then prior-corrected", "to a 50/50 prior"),
        "node row",
    )
    b += arrow(mx + 230, 150, 623, 150)
    return svg(W, H, b, "How TabPFN scores test images in context", "diagram")


def diag_encoder():
    W, H = 780, 180
    b = DEFS
    b += box(10, 30, 120, 80, "image", ("resized: short side", "512 px (DINOv3)"))
    b += box(160, 30, 120, 80, "16×16 patches", ("linear embedding", "+ RoPE positions"))
    b += arrow(130, 70, 158, 70)
    toks = ["CLS", "R1", "R2", "R3", "R4", "p1", "p2", "…", "pN"]
    for i, t in enumerate(toks):
        cls = "cell shot" if t == "CLS" else "cell ctx" if t.startswith("R") else "cell test"
        b += rect(310 + i * 30, 55, 27, 30, cls) + text(310 + i * 30 + 13.5, 74, t, "ns", "middle")
    b += arrow(280, 70, 308, 70)
    b += text(310, 45, "tokens: CLS, 4 registers, patches", "ax")
    b += box(
        590,
        20,
        180,
        100,
        "ViT blocks",
        ("self-attention + MLP", "ViT-S: 12 layers, 384-d", "ViT-L: 24 layers, 1024-d"),
        "node enc",
    )
    b += arrow(580, 70, 588, 70)
    b += text(
        10,
        150,
        "DINOv3 is self-supervised on 1.7 B images (LVD-1689M) and distilled from a 7 B model; SigLIP2 is trained on image-text pairs",
        "ax",
    )
    b += text(
        10,
        166,
        "with a sigmoid loss, and its NaFlex variant keeps the aspect ratio (up to 1,024 patches) and pools with an attention head.",
        "ax",
    )
    return svg(W, H, b, "Vision transformer encoder", "diagram")


def diag_tracka():
    W, H = 780, 330
    b = DEFS
    y = 20
    rows_ = [
        (
            "PatchCore",
            [
                ("WideResNet-50", "layers 2 + 3"),
                ("local 3×3 pooling", "patch features"),
                ("memory bank", "1% coreset of train patches"),
                ("distance to nearest", "bank patch → map"),
            ],
            "s-pc",
        ),
        (
            "EfficientAD-S",
            [
                ("teacher PDN", "pre-trained, frozen"),
                ("student PDN", "mimics teacher on normals"),
                ("autoencoder", "for logical anomalies"),
                ("student–teacher +", "AE gaps → map"),
            ],
            "s-ead",
        ),
        (
            "DINOv3 distance",
            [
                ("DINOv3 patch tokens", "frozen, no training"),
                ("train patch bank", "all train patches"),
                ("Euclidean distance", "to nearest patch"),
                ("patch grid", "→ map"),
            ],
            "s-dl",
        ),
    ]
    for name, steps, cls in rows_:
        b += text(10, y + 30, name, "hd " + cls + "-t")
        for i, (t, s) in enumerate(steps):
            x = 140 + i * 160
            b += box(x, y, 140, 52, t, s)
            if i:
                b += arrow(x - 18, y + 26, x - 2, y + 26)
        y += 76
    b += text(
        10,
        y + 18,
        "Every map is upsampled to the image size; the image score is the map's maximum.",
        "ax",
    )
    b += text(
        10,
        y + 34,
        "PatchCore and EfficientAD-S run at 256×256 as in the AD2 paper; the DINOv3 map comes from the 512-px patch tokens.",
        "ax",
    )
    return svg(W, H, b, "Track A unsupervised baselines", "diagram")


def diag_stats():
    W, H = 780, 210
    b = DEFS
    b += box(10, 60, 150, 70, "per-image scores", ("8 scenarios ×", "10 seeds"))
    b += box(200, 20, 180, 60, "resample seeds", "with replacement")
    b += box(200, 110, 180, 60, "resample scenes", ("per label, shared", "by all drawn seeds"))
    b += arrow(160, 85, 198, 55) + arrow(160, 105, 198, 135)
    b += box(420, 60, 160, 70, "metric per draw", ("both methods on the", "same draw (paired)"))
    b += arrow(380, 50, 418, 85) + arrow(380, 140, 418, 110)
    b += box(
        620, 60, 150, 70, "1,000 draws", ("2.5% and 97.5%", "percentiles = 95% CI"), "node row"
    )
    b += arrow(580, 95, 618, 95)
    b += text(
        10,
        195,
        "Scenes, not images, are resampled, so every lighting variant of a scene moves together.",
        "ax",
    )
    return svg(W, H, b, "Bootstrap over seeds and scenes", "diagram")


# AU-PRO@0.05 (%) on TESTpriv, MVTec AD 2 paper (Heckler-Kram et al., arXiv:2503.21622), Table VII.
PAPER_PRO = {
    "patchcore": dict(zip(SC, (4.7, 11.0, 46.7, 25.6, 15.2, 62.2, 12.8, 51.8), strict=True)),
    "efficientad": dict(zip(SC, (9.6, 22.2, 50.5, 27.6, 11.8, 55.6, 20.3, 48.8), strict=True)),
}
PAPER_MEAN = {"patchcore": 28.8, "efficientad": 30.8}
PAPER_BEST = {
    "can": ("DSR", 13.9),
    "fabric": ("EfficientAD", 22.2),
    "fruit_jelly": ("RD++", 54.4),
    "rice": ("EfficientAD", 27.6),
    "sheet_metal": ("DSR", 18.0),
    "vial": ("RD++", 63.0),
    "wallplugs": ("EfficientAD", 20.3),
    "walnuts": ("PatchCore", 51.8),
    None: ("EfficientAD", 30.8),
}


def table_paper():
    """Per-scenario lock AU-PRO@0.05 next to the AD2 paper's private-test numbers."""
    dl, pc, ead = tracka("distance-dinov3-l"), tracka("patchcore"), tracka("efficientad-s")

    def pro(m, sc):
        return 100 * m["au_pro_005" if sc is None else f"{sc}/au_pro_005"]

    rows = []
    for sc in [*SC, None]:
        ours, (best_name, best) = pro(dl, sc), PAPER_BEST[sc]
        pc_paper = PAPER_MEAN["patchcore"] if sc is None else PAPER_PRO["patchcore"][sc]
        ead_paper = PAPER_MEAN["efficientad"] if sc is None else PAPER_PRO["efficientad"][sc]
        a, b = (
            (f"<b>{ours:.1f}</b>", f"{best:.1f}")
            if ours > best
            else (f"{ours:.1f}", f"<b>{best:.1f}</b>")
        )
        rows.append(
            [
                "<b>Mean</b>" if sc is None else SC_LABEL[sc],
                a,
                f"{pro(pc, sc):.1f} / {pc_paper:.1f}",
                f"{pro(ead, sc):.1f} / {ead_paper:.1f}",
                f"{b} ({best_name})",
            ]
        )
    return table(
        [
            "scenario",
            "DINOv3-L distance (ours)",
            "PatchCore, ours / paper",
            "EfficientAD-S, ours / paper",
            "best in paper",
        ],
        rows,
    )


# ---------- Images ----------
def img_uri(p):
    return "data:image/webp;base64," + base64.b64encode(p.read_bytes()).decode()


def examples_html():
    out = []
    for ex in D["examples"]:
        sc = ex["scenario"]
        p = HERE / "data" / f"example_{sc}.webp"
        out.append(
            f'<figure class="ex"><img src="{img_uri(p)}" alt="{esc(SC_LABEL[sc])}: image with the defect outlined, and three anomaly maps">'
            f"<figcaption>{esc(SC_LABEL[sc])}, lock image <code>{esc(ex['image_id'])}</code>. Red outline: ground-truth defect. Heat: anomaly map (yellow = most anomalous), with the defect outlined in blue.</figcaption></figure>"
        )
    return "\n".join(out)


def table(headers, rows, cls=""):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def ci(r):
    return f'{r["auroc"]:.3f} <span class="ci">[{r["lo"]:.3f}, {r["hi"]:.3f}]</span>'


def pdiff(p):
    sig = p["lo"] > 0 or p["hi"] < 0
    s = f'{p["diff"]:+.3f} <span class="ci">[{p["lo"]:+.3f}, {p["hi"]:+.3f}]</span>'
    return f"<strong>{s}</strong>" if sig else s


# ---------- Page ----------
def build():
    tab = {k: lock(f"lock-tabpfn-k{k}") for k in (1, 2, 5)}
    log = {k: lock(f"lock-logreg-k{k}") for k in (1, 2, 5)}
    dl = tracka("distance-dinov3-l")
    pc = tracka("patchcore")
    ead = tracka("efficientad-s")
    lock_rows = []
    for key, lab in (
        ("tabpfn", "TabPFN-3.5"),
        ("tabpfn-fast", "TabPFN-3.5-Fast"),
        ("logreg", "Logistic regression"),
        ("knn", "kNN"),
        ("tabpfn-n32", "TabPFN-3.5, 32 normals"),
        ("tabpfn-fast-n32", "TabPFN-3.5-Fast, 32 normals"),
    ):
        lock_rows.append(
            [lab]
            + [ci(lock(f"lock-{key}-k{k}")) for k in (1, 2, 5)]
            + [f"{lock(f'lock-{key}-k2')['nll_bal']:.2f}"]
        )
    lock_rows.append(["Mahalanobis (no labels)", ci(lock("lock-mahalanobis")), "", "", "–"])
    lock_rows.append(
        ["TabPFN outlier score (no labels)", ci(lock("lock-tabpfn-outlier")), "", "", "–"]
    )
    paired_rows = []
    for c, lab in (
        ("logreg", "Logistic regression"),
        ("knn", "kNN"),
        ("mahalanobis", "Mahalanobis (no labels)"),
        ("tabpfn-outlier", "TabPFN outlier score (no labels)"),
        ("tabpfn-fast", "TabPFN-3.5-Fast"),
        ("tabpfn-n32", "TabPFN-3.5, 32 normals"),
    ):
        paired_rows.append([lab] + [pdiff(paired(k, c)) for k in (1, 2, 5)])
    calib_rows = []
    for c, lab in (("logreg", "Logistic regression"), ("knn", "kNN")):
        for m, ml in (("nll_bal", "balanced NLL"), ("ece_bal", "balanced ECE")):
            calib_rows.append([f"{lab}, {ml}"] + [pdiff(paired(k, c, m)) for k in (1, 2, 5)])
    ta_rows = []
    for model, lab in (
        ("distance-dinov3-l", "DINOv3-L patch distance"),
        ("distance-dinov3-s", "DINOv3-S patch distance"),
        ("patchcore", "PatchCore"),
        ("efficientad-s", "EfficientAD-S"),
    ):
        m = tracka(model)
        ta_rows.append(
            [lab]
            + [f"{m[x]:.3f}" for x in ("au_pro_005", "au_pro_030", "seg_f1", "class_f1", "auroc")]
        )
    ds_rows = []
    for r in D["dataset"]:
        light = ", ".join(x for x in r["lightings"] if x != "regular")
        ds_rows.append(
            [
                SC_LABEL[r["scenario"]],
                f"{r['image_size'][0]}×{r['image_size'][1]}",
                r["train"],
                r["validation"],
                f"{r['dev_good']} / {r['dev_bad']}",
                f"{r['lock_good']} / {r['lock_bad']}",
                esc(light),
            ]
        )
    hw_rows = HW.get("table_rows", [])
    refs = REFS
    css = CSS
    sections = SECTIONS.format(
        tab1=f3(tab[1]["auroc"]),
        tab2=f3(tab[2]["auroc"]),
        tab5=f3(tab[5]["auroc"]),
        log1=f3(log[1]["auroc"]),
        log2=f3(log[2]["auroc"]),
        log5=f3(log[5]["auroc"]),
        d1=pdiff(paired(1, "logreg")),
        nll1=f"{tab[1]['nll_bal']:.2f}",
        lnll1=f"{log[1]['nll_bal']:.2f}",
        mah=f3(lock("lock-mahalanobis")["auroc"]),
        tout=f3(lock("lock-tabpfn-outlier")["auroc"]),
        dl_pro=f3(dl["au_pro_005"]),
        pc_pro=f3(pc["au_pro_005"]),
        ead_pro=f3(ead["au_pro_005"]),
        dl_auc=f3(dl["auroc"]),
        pc_auc=f3(pc["auroc"]),
        ead_auc=f3(ead["auroc"]),
        ead_fit=f"{ead.get('fit_s', float('nan')) / 60:.0f}",
        pc_fit=f"{pc.get('fit_s', float('nan')):.0f}",
        dev_runs=D["dev_counts"]["runs"],
        chart_budget=chart_budget(),
        chart_forest=chart_forest(),
        chart_calibration=chart_calibration(),
        chart_reliability=chart_reliability(),
        chart_scenarios=chart_scenarios(),
        chart_tracka=chart_tracka(),
        chart_tracka_scen=chart_tracka_scen(),
        chart_search=chart_search(),
        chart_devlock=chart_devlock(),
        chart_dataset=chart_dataset(),
        chart_hardware=chart_hardware(),
        chart_xhw=chart_xhw(),
        chart_lighting=chart_lighting(),
        table_lighting=table_lighting(),
        chart_trackc=chart_trackc(5),
        chart_trackc2=chart_trackc(2),
        table_trackc=table_trackc(),
        trackc_example=img_uri(HERE / "data" / "best_grid.webp"),
        diag_split=diag_split(),
        diag_timeline=diag_timeline(),
        diag_trackb=diag_trackb(),
        diag_tabpfn=diag_tabpfn(),
        diag_encoder=diag_encoder(),
        diag_tracka=diag_tracka(),
        diag_stats=diag_stats(),
        examples=examples_html(),
        table_paper=table_paper(),
        table_lock=table(["method", "k = 1", "k = 2", "k = 5", "balanced NLL, k = 2"], lock_rows),
        table_paired=table(["TabPFN-3.5 minus", "k = 1", "k = 2", "k = 5"], paired_rows),
        table_calib=table(["TabPFN-3.5 minus", "k = 1", "k = 2", "k = 5"], calib_rows),
        table_tracka=table(
            ["method", "AU-PRO@0.05", "AU-PRO@0.30", "SegF1", "ClassF1", "image AUROC"], ta_rows
        ),
        table_dataset=table(
            [
                "scenario",
                "image size (w×h)",
                "train",
                "validation",
                "dev good / defect",
                "lock good / defect",
                "shifted lightings",
            ],
            ds_rows,
        ),
        table_hw=table(HW.get("table_headers", ["stage"]), hw_rows) if hw_rows else "",
        hw_text=HW.get("text", ""),
        refs=refs,
    )
    fonts = (
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700'
        '&amp;family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&amp;family=JetBrains+Mono:wght@400;600&amp;display=swap">'
    )
    head = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    page = f"{head}<title>Anometa on MVTec AD 2</title>\n{fonts}\n<style>{css}</style>\n</head>\n<body>\n{sections}\n</body>\n</html>\n"
    (HERE / "index.html").write_text(page)
    print("written", len(page) // 1024, "KB")


SITE = HERE.parent / "site"


def build_site():
    """Render the landing page `docs/site/index.html` from `docs/site/template.html`."""
    from string import Template

    def plain(p):
        return f"{p['diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]"

    def tc(k):
        return f"{_tc_mean(None, k, 'tabpfn_fast') - _tc_mean(None, k, 'dinov3_distance'):+.3f}"

    rows = [
        [lab] + [ci(lock(f"lock-{key}-k{k}")) for k in (1, 2, 5)]
        for key, lab in (
            ("tabpfn", "<b>TabPFN-3.5</b>"),
            ("tabpfn-fast", "TabPFN-3.5-Fast"),
            ("logreg", "Logistic regression"),
        )
    ]
    rows.append(["Mahalanobis (no labels)", ci(lock("lock-mahalanobis")), "", ""])
    rows.append(["TabPFN outlier score (no labels)", ci(lock("lock-tabpfn-outlier")), "", ""])
    tab1, log1 = lock("lock-tabpfn-k1"), lock("lock-logreg-k1")
    body = Template((SITE / "template.html").read_text()).substitute(
        tab1=f3(tab1["auroc"]),
        log1=f3(log1["auroc"]),
        d1_plain=plain(paired(1, "logreg")),
        nll1=f"{tab1['nll_bal']:.2f}",
        lnll1=f"{log1['nll_bal']:.2f}",
        tc2=tc(2),
        tc5=tc(5),
        diag_trackb=diag_trackb(),
        diag_tabpfn=diag_tabpfn(),
        chart_budget=f'<figure class="plot">{chart_budget()}</figure>',
        table_lock=table(["method", "k = 1", "k = 2", "k = 5"], rows),
        table_paper=table_paper(),
        trackc_example=img_uri(HERE.parent / "video" / "assets" / "trackc_example.webp"),
    )
    desc = "Few-shot industrial anomaly detection with TabPFN-3.5 on frozen DINOv3 features, tested on MVTec AD 2."
    head = (
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="description" content="{desc}">\n'
        '<meta property="og:title" content="Anometa">\n'
        f'<meta property="og:description" content="{desc}">\n'
        '<meta property="og:type" content="website">\n'
        '<meta property="og:image" content="https://husmen.github.io/anometa/assets/poster.webp">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
        '<link rel="icon" href="assets/logo.webp">'
    )
    fonts = (
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700'
        '&amp;family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&amp;family=JetBrains+Mono:wght@400;600&amp;display=swap">'
    )
    css = CSS + (SITE / "site.css").read_text()
    page = (
        f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>Anometa</title>\n{head}\n{fonts}\n'
        f"<style>{css}</style>\n</head>\n<body>\n{body}</body>\n</html>\n"
    )
    (SITE / "index.html").write_text(page)
    print("written", len(page) // 1024, "KB (site)")


CSS = (HERE / "report.css").read_text()
SECTIONS = (HERE / "report_body.html").read_text()
REFS = (HERE / "report_refs.html").read_text()

if __name__ == "__main__":
    build()
    build_site()
