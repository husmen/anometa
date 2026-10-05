"""Results tables and figures, built from run artifacts.

`load_runs` flattens every `ok` run of one split into a frame;
`paired_comparisons` and `matched_cells` compare TabPFN with each control on
the same evaluation rows, and `fixed_set_budget` compares label budgets on the
same rows; `build_report` turns that frame (plus each run's
`predictions.parquet`, the search trial frames and the NSGA-II study) into
`reports/<split>/results.md`
and `reports/<split>/figures/*.png`, one section per PLAN_1 question. Figures
use matplotlib's object-oriented `Figure` API, so no interactive backend is
ever loaded.
"""

import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from anometa.artifacts import read_manifest
from anometa.config import ONE_CLASS, Scenario
from anometa.metrics.aggregate import (
    bootstrap_ci,
    group_metrics,
    paired_bootstrap_diff,
    summarize,
)

_CONFIG_COLUMNS: list[str] = [
    "run_id",
    "track",
    "encoder",
    "features",
    "pca_dim",
    "classifier",
    "k",
    "shot_lighting",
    "model",
    "params",
    "adapt_normals",
    "config_hash",
]
"""Flattened config columns of `load_runs`, before the metrics."""

_PIXEL_COLUMNS: list[str] = ["au_pro_005", "au_pro_030", "seg_f1", "class_f1"]
_CALIBRATION_COLUMNS: list[str] = ["ece", "ece_bal", "nll_bal", "brier_bal"]
_SEARCH_FILE = re.compile(r"(tpe|bo)-k\d+-seed\d+\.parquet")
"""Trial frames of all-scenario searches; scenario-restricted ones carry a suffix."""
_NSGA2_STUDY = re.compile(r"nsga2-k\d+-seed\d+")
_REFERENCE = "tabpfn"
"""The classifier every paired comparison puts first (positive = it scores higher)."""
_SENSITIVITY_SCENARIO = "wallplugs"
"""Scenario the paired sensitivity rows leave out: every dev method ranks its
test-good images above its defects, so it can dominate a mean difference."""


def load_runs(artifacts: Path, split: Literal["dev", "lock"]) -> pd.DataFrame:
    """Collect every `ok` run of one split into one frame.

    Args:
        artifacts: Directory holding the run directories.
        split: Which split's runs to keep.

    Returns:
        One row per run config, sorted by `run_id`: `_CONFIG_COLUMNS`
        (`features` joined with `"+"`, `params` the JSON of
        `classifier_params`; fields a track lacks are `None`), every metric
        in its `metrics.json`, and `run_dir`. Runs that differ only in name
        or paths share a `config_hash` and repeat the same experiment (e.g.
        a latency rerun); only the first by `run_id` is kept.
    """
    rows: list[dict[str, object]] = []
    for run_dir in sorted(p for p in artifacts.glob("*") if p.is_dir()):
        manifest = read_manifest(run_dir)
        if manifest is None or manifest.get("status") != "ok":
            continue
        config = manifest["config"]
        assert isinstance(config, dict)
        if config.get("split") != split:
            continue
        features = config.get("features")
        row: dict[str, object] = {
            "run_id": manifest["run_id"],
            "track": config.get("track"),
            "encoder": config.get("encoder"),
            "features": "+".join(features) if isinstance(features, list) else None,
            "pca_dim": config.get("pca_dim"),
            "classifier": config.get("classifier"),
            "k": config.get("k"),
            "shot_lighting": config.get("shot_lighting"),
            "model": config.get("model"),
            "params": json.dumps(config.get("classifier_params") or {}, sort_keys=True)
            if config.get("track") == "B"
            else None,
            "adapt_normals": config.get("adapt_normals"),
            "config_hash": manifest["config_hash"],
        }
        row |= json.loads((run_dir / "metrics.json").read_text())
        row["run_dir"] = run_dir
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[*_CONFIG_COLUMNS, "run_dir"])
    return pd.DataFrame(rows).drop_duplicates("config_hash", ignore_index=True)


_INT_COLUMNS: frozenset[str] = frozenset({"k", "pca_dim", "seed", "n_estimators"})
"""Columns rendered as integers: pandas turns them into floats once one value is missing."""


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Render a frame as a GitHub-flavoured Markdown table.

    Args:
        df: The frame; its index is not rendered.
        floatfmt: Format spec for float cells, except in `_INT_COLUMNS`,
            whose whole-number values render as integers.

    Returns:
        The table, one line per row, without a trailing newline.
    """

    def cell(column: object, value: object) -> str:
        """Format one cell of `column`."""
        if isinstance(value, float):
            if column in _INT_COLUMNS and value.is_integer():
                return str(int(value))
            return format(value, floatfmt)
        return str(value)

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += [
        "| " + " | ".join(cell(c, v) for c, v in zip(columns, row, strict=True)) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join(lines)


def _predictions(run_dir: object) -> pd.DataFrame:
    """Read one run's `predictions.parquet`.

    Args:
        run_dir: A `load_runs` `run_dir` value.

    Returns:
        The predictions frame.
    """
    return pd.read_parquet(Path(str(run_dir)) / "predictions.parquet")


def _probabilistic(pred: pd.DataFrame) -> bool:
    """Tell whether a predictions frame carries calibrated probabilities.

    Args:
        pred: A predictions frame.

    Returns:
        Whether `score_balanced` exists and holds a value.
    """
    return "score_balanced" in pred.columns and bool(pred["score_balanced"].notna().any())


def with_ci(runs: pd.DataFrame, metric: str, n_boot: int = 1000) -> pd.DataFrame:
    """Add a 95% bootstrap confidence interval for one metric to each run.

    Args:
        runs: `load_runs` rows.
        metric: A `group_metrics` column, e.g. `"auroc"`.
        n_boot: Bootstrap replicates per run.

    Returns:
        `runs` with `<metric>_lo` and `<metric>_hi` from `bootstrap_ci` on
        each run's predictions; NaN where the metric does not apply (e.g.
        a calibration metric of a one-class run).
    """
    lo: list[float] = []
    hi: list[float] = []
    for run_dir in runs["run_dir"]:
        pred = _predictions(run_dir)
        try:
            _, low, high = bootstrap_ci(
                pred, metric, probabilistic=_probabilistic(pred), n_boot=n_boot
            )
        except KeyError:
            low = high = float("nan")
        lo.append(low)
        hi.append(high)
    return runs.assign(**{f"{metric}_lo": lo, f"{metric}_hi": hi})


def one_class_on_fewshot_rows(
    one_class_pred: pd.DataFrame, fewshot_pred: pd.DataFrame
) -> pd.DataFrame:
    """Restrict one-class predictions to a few-shot run's evaluation rows.

    A one-class run scores every dev image once (seed 0); a few-shot run
    drops its shot scenes per seed. Matching rows lets both be scored on the
    same images.

    Args:
        one_class_pred: The one-class run's predictions.
        fewshot_pred: The few-shot run's predictions.

    Returns:
        One row per few-shot (scenario, seed, image_id), in its order, with
        the few-shot run's `seed` and the one-class run's other columns.
    """
    keys = fewshot_pred[["scenario", "seed", "image_id"]]
    return keys.merge(one_class_pred.drop(columns="seed"), on=["scenario", "image_id"], how="inner")


def _scenario_means(pred: pd.DataFrame) -> dict[str, float]:
    """Mean AUROC per scenario of one predictions frame.

    Args:
        pred: A predictions frame.

    Returns:
        Scenario name to its mean AUROC over seeds.
    """
    summary = summarize(group_metrics(pred, probabilistic=False))
    return {sc.value: summary[f"{sc}/auroc"] for sc in Scenario if f"{sc}/auroc" in summary}


def paired_comparisons(
    best: pd.DataFrame, one_class: pd.DataFrame, n_boot: int = 1000
) -> pd.DataFrame:
    """Compare the best TabPFN run with each control's best run on the same rows.

    For every `k` with a `_REFERENCE` run in `best`, pairs that run with the
    best run of every other few-shot classifier at the same `k` and shot
    lighting, and with the best run of every one-class classifier,
    restricted to the TabPFN run's evaluation rows. Each pair gets a paired
    bootstrap of the AUROC difference over all scenarios and again without
    `_SENSITIVITY_SCENARIO`, plus per-scenario point differences. Picking the
    best run per classifier on the same dev data favours both sides, so the
    differences are optimistic for neither and noisy for both.

    Args:
        best: Best few-shot `load_runs` row per (classifier, k).
        one_class: One-class `load_runs` rows.
        n_boot: Bootstrap replicates per difference.

    Returns:
        One row per (k, control, scope): `k`, `control`, its `encoder`,
        `features` and `pca_dim`, `scope` (`"all"` or
        `"without <scenario>"`), `diff` (TabPFN minus control AUROC),
        `diff_lo`, `diff_hi` (95% CI) and `p_tabpfn_better` (share of
        replicates above zero), then one point difference per scenario
        (NaN outside the scope).
    """
    rows: list[dict[str, object]] = []
    oc_best = (
        one_class.loc[one_class.groupby("classifier")["auroc"].idxmax()]
        if not one_class.empty
        else one_class
    )
    for ref in best[best["classifier"] == _REFERENCE].itertuples(index=False):
        ref_pred = _predictions(ref.run_dir)
        controls = best[
            (best["k"] == ref.k)
            & (best["shot_lighting"] == ref.shot_lighting)
            & (best["classifier"] != _REFERENCE)
        ]
        pairs = [(c, _predictions(c.run_dir)) for c in controls.itertuples(index=False)]
        pairs += [
            (c, one_class_on_fewshot_rows(_predictions(c.run_dir), ref_pred))
            for c in oc_best.itertuples(index=False)
        ]
        for control, control_pred in pairs:
            per_ref = _scenario_means(ref_pred)
            per_control = _scenario_means(control_pred)
            for scope, keep in [
                ("all", None),
                (f"without {_SENSITIVITY_SCENARIO}", _SENSITIVITY_SCENARIO),
            ]:
                a, b = ref_pred, control_pred
                if keep is not None:
                    a, b = a[a["scenario"] != keep], b[b["scenario"] != keep]
                diff, lo, hi, share = paired_bootstrap_diff(
                    a, b, "auroc", probabilistic=False, n_boot=n_boot
                )
                row: dict[str, object] = {
                    "k": ref.k,
                    "control": control.classifier,
                    "encoder": control.encoder,
                    "features": control.features,
                    "pca_dim": control.pca_dim,
                    "scope": scope,
                    "diff": diff,
                    "diff_lo": lo,
                    "diff_hi": hi,
                    "p_tabpfn_better": share,
                }
                row |= {
                    sc: per_ref[sc] - per_control[sc] if sc != keep else float("nan")
                    for sc in per_ref
                }
                rows.append(row)
    return pd.DataFrame(rows)


def matched_cells(fewshot: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """Compare TabPFN with each few-shot control cell by cell, at default settings.

    A cell is one (encoder, features, PCA dimension, k, shot lighting); only
    runs with default `classifier_params` (the frozen grid) count, so no
    side is tuned. This avoids picking a best run per classifier.

    Args:
        fewshot: Few-shot `load_runs` rows.
        metric: A `group_metrics` column with per-scenario summaries, e.g.
            `"auroc"`, or a calibration metric such as `"nll_bal"`, for which
            lower is better.

    Returns:
        One row per (control, k, scope): `metric`, `control`, `k`, `scope`
        (`"all"` or `"without <scenario>"`), `cells` (matched cells),
        `mean_diff` and `median_diff` (TabPFN minus control, each cell's value
        the mean of its per-scenario means) and `share_tabpfn_better` (above
        zero for a ranking metric, below zero otherwise).
    """
    keys = ["encoder", "features", "pca_dim", "k", "shot_lighting"]
    grid = fewshot[fewshot["params"] == "{}"].assign(pca_dim=lambda d: d["pca_dim"].fillna(-1))
    scenario_cols = [f"{sc}/{metric}" for sc in Scenario if f"{sc}/{metric}" in grid.columns]
    kept_cols = [c for c in scenario_cols if c != f"{_SENSITIVITY_SCENARIO}/{metric}"]
    higher_is_better = metric.removeprefix("gap_") in ("auroc", "auprc")
    ref = grid[grid["classifier"] == _REFERENCE]
    rows: list[dict[str, object]] = []
    for control, g in grid[grid["classifier"] != _REFERENCE].groupby("classifier", sort=True):
        cells = ref.merge(g, on=keys, suffixes=("_ref", "_ctl"))
        for k, c in cells.groupby("k", sort=True):
            for scope, cols in [
                ("all", scenario_cols),
                (f"without {_SENSITIVITY_SCENARIO}", kept_cols),
            ]:
                diff = (
                    c[[f"{col}_ref" for col in cols]].mean(axis=1).to_numpy()
                    - c[[f"{col}_ctl" for col in cols]].mean(axis=1).to_numpy()
                )
                rows.append(
                    {
                        "metric": metric,
                        "control": control,
                        "k": k,
                        "scope": scope,
                        "cells": len(c),
                        "mean_diff": float(np.mean(diff)),
                        "median_diff": float(np.median(diff)),
                        "share_tabpfn_better": float(
                            np.mean(diff > 0 if higher_is_better else diff < 0)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def lighting_paired(lighting: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Run the lighting study's pre-declared paired comparisons (PLAN_1 § Post-freeze study).

    Compares, on shifted-lighting images, TabPFN-3.5 with m adaptation
    scenes against m = 0, and TabPFN-3.5 against each control at the
    largest m, per k. Each (scenario, lighting) pair is its own bootstrap
    stratum, because every pair has its own fitted context.

    Args:
        lighting: Track L `load_runs` rows.
        n_boot: Bootstrap replicates per comparison.

    Returns:
        One row per comparison: `k`, `comparison`, `metric`, `diff` (first
        minus second), `diff_lo`, `diff_hi` and `p_first_better`.
    """
    runs = {
        (str(c), k, m): d
        for c, k, m, d in zip(
            lighting["classifier"],
            lighting["k"].astype(int).tolist(),
            lighting["adapt_normals"].astype(int).tolist(),
            lighting["run_dir"],
            strict=True,
        )
    }

    def shifted(key: tuple[str, int, int]) -> pd.DataFrame:
        """Shifted-lighting predictions of one run, grouped by (scenario, lighting)."""
        p = _predictions(runs[key])
        p = p[p["lighting"] != "regular"]
        return p.assign(scenario=p["scenario"] + "|" + p["lighting"])

    rows: list[dict[str, object]] = []
    m_max = max(m for _, _, m in runs)
    for k in sorted({k for _, k, _ in runs}):
        plan = [
            (f"tabpfn m={m} vs m=0", ("tabpfn", k, m), ("tabpfn", k, 0))
            for m in range(1, m_max + 1)
        ]
        plan += [
            (f"tabpfn vs {c}, m={m_max}", ("tabpfn", k, m_max), (c, k, m_max))
            for c in ("logreg", "mahalanobis", "tabpfn_outlier", "tabpfn_fast", "knn")
        ]
        for name, a, b in plan:
            if a not in runs or b not in runs:
                continue
            for metric in ("auroc", "nll_bal"):
                if metric == "nll_bal" and b[0] in ONE_CLASS:
                    continue
                d, lo, hi, share = paired_bootstrap_diff(
                    shifted(a), shifted(b), metric, probabilistic=metric != "auroc", n_boot=n_boot
                )
                rows.append(
                    {
                        "k": k,
                        "comparison": name,
                        "metric": metric,
                        "diff": d,
                        "diff_lo": lo,
                        "diff_hi": hi,
                        "p_first_better": share if metric == "auroc" else 1 - share,
                    }
                )
    return pd.DataFrame(rows)


def _slug(name: str) -> str:
    """Turn a classifier name (e.g. `tabpfn n_normals=32`) into a file-name stem.

    Args:
        name: The name.

    Returns:
        `name` with every run of characters other than letters, digits and
        `_` replaced by `-`.
    """
    return re.sub(r"[^A-Za-z0-9_]+", "-", name)


def _save(fig: Figure, out: Path) -> Path:
    """Save a figure as PNG, creating its directory.

    Args:
        fig: The figure.
        out: Destination file.

    Returns:
        `out`.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    return out


def plot_budget_curves(runs: pd.DataFrame, out: Path) -> Path:
    """Plot mean dev AUROC against the label budget k, one line per classifier.

    One-class runs appear as markers at k = 0.

    Args:
        runs: `load_runs` rows of Track B runs.
        out: Destination PNG.

    Returns:
        `out`.
    """
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    means = runs.groupby(["classifier", "k"])["auroc"].mean().reset_index()
    for classifier, g in means.groupby("classifier"):
        style = "o" if classifier in ONE_CLASS else "o-"
        ax.plot(g["k"], g["auroc"], style, label=str(classifier))
    ax.set_xlabel("label budget k")
    ax.set_ylabel("AUROC (mean over configs)")
    ax.legend()
    return _save(fig, out)


def plot_reliability(pred: pd.DataFrame, out: Path) -> Path:
    """Plot reliability diagrams of raw and balanced probabilities.

    Args:
        pred: A probabilistic run's predictions.
        out: Destination PNG.

    Returns:
        `out`.
    """
    fig = Figure(figsize=(8, 4))
    axes = fig.subplots(1, 2)
    edges = np.linspace(0.0, 1.0, 11)
    labels = pred["label"].to_numpy(dtype=np.float64)
    for ax, column, title in zip(
        axes, ["score", "score_balanced"], ["raw", "balanced"], strict=True
    ):
        p = pred[column].to_numpy(dtype=np.float64)
        bins = np.clip(np.digitize(p, edges) - 1, 0, 9)
        used = [b for b in range(10) if np.any(bins == b)]
        ax.plot([0, 1], [0, 1], ":", color="grey")
        ax.plot([p[bins == b].mean() for b in used], [labels[bins == b].mean() for b in used], "o-")
        ax.set_title(title)
        ax.set_xlabel("predicted anomaly probability")
        ax.set_ylabel("fraction anomalous")
    return _save(fig, out)


def _search_files(search_dir: Path) -> list[Path]:
    """List the all-scenario TPE and BO trial frames in a directory.

    Args:
        search_dir: Directory holding the search parquet files.

    Returns:
        Matching files, sorted.
    """
    return sorted(p for p in search_dir.glob("*.parquet") if _SEARCH_FILE.fullmatch(p.name))


def plot_search(search_dir: Path, out: Path) -> Path:
    """Plot mean best-so-far AUROC per trial, TPE against TabPFN-BO.

    Args:
        search_dir: Directory with `tpe-k<k>-seed<s>.parquet` and
            `bo-k<k>-seed<s>.parquet` frames.
        out: Destination PNG.

    Returns:
        `out`.
    """
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    for method, label in [("tpe", "Optuna TPE"), ("bo", "TabPFN-BO")]:
        frames = [
            pd.read_parquet(p) for p in _search_files(search_dir) if p.name.startswith(method)
        ]
        if frames:
            curve = pd.concat(frames).groupby("trial")["best_so_far"].mean()
            ax.plot(curve.index, curve.to_numpy(), label=f"{label} ({len(frames)} seeds)")
    ax.set_xlabel("trial")
    ax.set_ylabel("best dev AUROC so far")
    ax.legend()
    return _save(fig, out)


def plot_pareto(storage: str, study_name: str, out: Path) -> Path:
    """Plot an NSGA-II study's AUROC against balanced ECE, Pareto trials highlighted.

    Args:
        storage: Optuna storage URL.
        study_name: The study to plot.
        out: Destination PNG.

    Returns:
        `out`.
    """
    import optuna

    study = optuna.load_study(study_name=study_name, storage=storage)
    done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    best = {t.number for t in study.best_trials}
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    for on_front, style in [(False, "."), (True, "o")]:
        points = [t.values for t in done if t.values is not None and (t.number in best) == on_front]
        if points:
            ax.plot(
                [v[1] for v in points],
                [v[0] for v in points],
                style,
                label="Pareto front" if on_front else "trial",
            )
    ax.set_xlabel("balanced ECE")
    ax.set_ylabel("dev AUROC")
    ax.set_title(study_name)
    ax.legend()
    return _save(fig, out)


def _section(title: str, body: list[str]) -> list[str]:
    """Format one report section.

    Args:
        title: Section heading.
        body: Paragraphs and tables.

    Returns:
        Markdown lines.
    """
    return [f"## {title}", "", *(line for part in body for line in (part, ""))]


def with_variants(track_b: pd.DataFrame) -> pd.DataFrame:
    """Rename runs with a non-default context size as their own classifier.

    `n_normals` changes what the scorer sees rather than tuning it, so a
    `tabpfn` run with `n_normals: 32` is reported as `tabpfn n_normals=32`
    and is never pooled with (or picked over) the default `tabpfn` runs.

    Args:
        track_b: Track B `load_runs` rows.

    Returns:
        `track_b` with `classifier` renamed where `params` sets `n_normals`.
    """
    names = [
        f"{classifier} n_normals={n}" if (n := json.loads(params).get("n_normals")) else classifier
        for classifier, params in zip(track_b["classifier"], track_b["params"], strict=True)
    ]
    return track_b.assign(classifier=names)


def fixed_set_budget(fewshot: pd.DataFrame, one_class: pd.DataFrame) -> pd.DataFrame:
    """Score every label budget of one configuration on the same evaluation rows.

    Each k drops its shot scenes from the dev evaluation set, so the label
    budget table scores each k on different (and, at larger k, fewer)
    scenes. Shots are nested (a smaller k's shots are a prefix of a larger
    k's for the same seed), so every smaller-k run of a configuration also
    scored the largest k's evaluation rows; this table rescores them there.
    The configuration per classifier is its best run at its largest k; the
    best run of each one-class classifier is scored on the same rows. Only
    regular-lit shots count: shots from every lighting are a separate
    ablation and not nested with them.

    Args:
        fewshot: Few-shot `load_runs` rows.
        one_class: One-class `load_runs` rows.

    Returns:
        One row per classifier: `classifier`, `encoder`, `features`,
        `pca_dim`, then `k=<k>` (AUROC on the fixed rows) for every k the
        configuration ran; one-class rows fill `k=0` only.
    """
    keys = ["encoder", "features", "pca_dim", "params", "shot_lighting"]
    fewshot = fewshot[fewshot["shot_lighting"] == "regular"]
    rows: list[dict[str, object]] = []
    ref_pred: pd.DataFrame | None = None
    for classifier, g in fewshot.groupby("classifier", sort=True):
        top = g[g["k"] == g["k"].max()]
        ref = top.loc[top["auroc"].idxmax()]
        fixed = _predictions(ref["run_dir"])[["scenario", "seed", "image_id"]]
        if classifier == _REFERENCE or ref_pred is None:
            ref_pred = fixed
        same = g[(g[keys].fillna(-1) == ref[keys].fillna(-1)).all(axis=1)]
        row: dict[str, object] = {"classifier": classifier} | {c: ref[c] for c in keys[:3]}
        for run in same.sort_values("k").itertuples(index=False):
            pred = _predictions(run.run_dir).merge(fixed, on=["scenario", "seed", "image_id"])
            row[f"k={int(run.k)}"] = summarize(group_metrics(pred, probabilistic=False))["auroc"]
        rows.append(row)
    if ref_pred is not None and not one_class.empty:
        for oc in one_class.loc[one_class.groupby("classifier")["auroc"].idxmax()].itertuples(
            index=False
        ):
            pred = one_class_on_fewshot_rows(_predictions(oc.run_dir), ref_pred)
            rows.append(
                {
                    "classifier": oc.classifier,
                    "encoder": oc.encoder,
                    "features": oc.features,
                    "pca_dim": oc.pca_dim,
                    "k=0": summarize(group_metrics(pred, probabilistic=False))["auroc"],
                }
            )
    out = pd.DataFrame(rows)
    budgets = sorted((c for c in out.columns if c.startswith("k=")), key=lambda c: int(c[2:]))
    return out[["classifier", "encoder", "features", "pca_dim", *budgets]]


def _matched_one_class(best: pd.DataFrame, one_class: pd.DataFrame) -> pd.DataFrame:
    """Score each one-class run on the rows of its matching few-shot run.

    Args:
        best: Few-shot `load_runs` rows.
        one_class: One-class `load_runs` rows.

    Returns:
        `best` plus `<one-class classifier>_auroc`: that classifier's AUROC
        (same encoder, features and PCA dimension) on the few-shot run's
        evaluation rows; NaN without a match.
    """
    out = best.copy()
    for oc_name, oc_runs in one_class.groupby("classifier"):
        values: list[float] = []
        for row in best.itertuples(index=False):
            match = oc_runs[
                (oc_runs["encoder"] == row.encoder)
                & (oc_runs["features"] == row.features)
                & (oc_runs["pca_dim"].fillna(-1) == (-1 if pd.isna(row.pca_dim) else row.pca_dim))
            ]
            if match.empty:
                values.append(float("nan"))
                continue
            rows = one_class_on_fewshot_rows(
                _predictions(match["run_dir"].iloc[0]), _predictions(row.run_dir)
            )
            values.append(summarize(group_metrics(rows, probabilistic=False))["auroc"])
        out[f"{oc_name}_auroc"] = values
    return out


def build_report(
    artifacts: Path, out: Path = Path("reports"), split: Literal["dev", "lock"] = "dev"
) -> Path:
    """Write the results page and figures for one split.

    Sections follow PLAN_1's questions: label-budget curves (with the
    one-class controls scored on the same rows), classifiers per PCA
    dimension, calibration, robustness gap, Track A against Track B, and the
    search comparison. A section whose inputs are missing says so in one line.

    Args:
        artifacts: Directory holding the run directories, `search/` and
            `optuna.db`.
        out: Report root; the page goes to `out / split / "results.md"`.
        split: Which split's runs to report.

    Returns:
        The results page path.
    """
    page_dir = out / split
    figures = page_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(artifacts, split)
    track_b = with_variants(runs[runs["track"] == "B"])
    fewshot = track_b[~track_b["classifier"].isin(ONE_CLASS)]
    one_class = track_b[track_b["classifier"].isin(ONE_CLASS)]
    track_a = runs[runs["track"] == "A"]
    lines = [f"# Results: {split} split", "", f"{len(runs)} completed runs.", ""]

    if fewshot.empty:
        lines += _section("Label budget", ["No few-shot Track B runs yet."])
    else:
        best = fewshot.loc[fewshot.groupby(["classifier", "k"])["auroc"].idxmax()]
        best = _matched_one_class(with_ci(best, "auroc"), one_class)
        cols = ["classifier", "k", "shot_lighting", "encoder", "features", "pca_dim", "auroc"]
        cols += ["auroc_lo"]
        cols += ["auroc_hi", *(f"{c}_auroc" for c in sorted(ONE_CLASS) if f"{c}_auroc" in best)]
        plot_budget_curves(track_b, figures / "budget_curves.png")
        lines += _section(
            "Label budget",
            [
                "Best configuration per classifier and k, with a 95% bootstrap CI; "
                "`<one-class>_auroc` scores the matching one-class control on the same rows.",
                markdown_table(best[cols]),
                "![Label budget curves](figures/budget_curves.png)",
            ],
        )
        lines += _section(
            "Label budget on fixed rows",
            [
                "Each classifier's best configuration at its largest k, with every k "
                "scored on that run's evaluation rows (shots are nested), so the columns "
                "compare budgets on the same scenes. The one-class controls are scored "
                "on the same rows.",
                markdown_table(fixed_set_budget(fewshot, one_class)),
            ],
        )
        paired = paired_comparisons(best, one_class)
        if not paired.empty:
            lines += _section(
                "TabPFN against each control (paired)",
                [
                    "Best TabPFN run against each control's best run, both scored on the "
                    "same evaluation rows; `diff` is TabPFN minus control AUROC with a 95% "
                    "paired bootstrap CI, and `p_tabpfn_better` is the share of bootstrap "
                    f"replicates above zero. The `without {_SENSITIVITY_SCENARIO}` rows are "
                    "a sensitivity check: every method ranks its test-good images above "
                    "its defects. Scenario columns are point differences.",
                    markdown_table(paired),
                ],
            )
        cells = matched_cells(fewshot)
        if not cells.empty:
            calibration = pd.concat(
                [cells.iloc[:0]]
                + [matched_cells(fewshot, m) for m in ("nll_bal", "ece_bal") if m in fewshot]
            )
            lines += _section(
                "TabPFN against each control (matched grid cells)",
                [
                    "Every grid cell (encoder, features, PCA dimension, k) both classifiers "
                    "ran at default settings, so no best-run selection is involved.",
                    markdown_table(cells),
                    "Calibration after the prior correction to 50/50 (lower is better), "
                    "on the same cells. Default settings only: logreg's `C` and TabPFN's "
                    "`n_estimators` are untuned here (see the search section for tuned runs).",
                    markdown_table(calibration[calibration["scope"] == "all"]),
                ],
            )
        per_pca = fewshot.pivot_table(index=["k", "pca_dim"], columns="classifier", values="auroc")
        lines += _section(
            "Classifiers per PCA dimension",
            ["Mean AUROC over encoders and feature sets.", markdown_table(per_pca.reset_index())],
        )
        calib = [c for c in _CALIBRATION_COLUMNS if c in fewshot.columns]
        for classifier, g in fewshot.groupby("classifier"):
            plot_reliability(
                _predictions(g.loc[g["auroc"].idxmax(), "run_dir"]),
                figures / f"reliability_{_slug(str(classifier))}.png",
            )
        lines += _section(
            "Calibration",
            [
                "Mean over runs; `_bal` after the prior correction to 50/50.",
                markdown_table(fewshot.groupby("classifier")[calib].mean().reset_index()),
                *(
                    f"![Reliability {c}](figures/reliability_{_slug(c)}.png)"
                    for c in sorted(fewshot["classifier"].unique())
                ),
            ],
        )
    if one_class.empty:
        lines += _section("One-class controls", ["No one-class runs yet."])
    else:
        lines += _section(
            "One-class controls",
            [
                markdown_table(
                    one_class.sort_values(["classifier", "auroc"], ascending=[True, False])[
                        ["classifier", "encoder", "features", "pca_dim", "auroc"]
                    ]
                )
            ],
        )

    gap_parts: list[str] = []
    if not track_b.empty:
        gap_parts.append(
            markdown_table(
                track_b.pivot_table(
                    index="encoder", columns="classifier", values="gap_auroc"
                ).reset_index()
            )
        )
    if not track_a.empty:
        gap_parts.append(markdown_table(track_a[["model", "encoder", "gap_auroc"]]))
    lines += _section(
        "Robustness gap",
        ["AUROC on regular-lit minus shifted-lit images.", *gap_parts]
        if gap_parts
        else ["No runs yet."],
    )

    if track_a.empty:
        lines += _section("Track A against Track B", ["No Track A runs yet."])
    else:
        pixel = [c for c in _PIXEL_COLUMNS if c in track_a.columns]
        body = [markdown_table(track_a[["model", "encoder", "auroc", *pixel]])]
        if not track_b.empty:
            body.append(markdown_table(track_b.groupby("classifier")["auroc"].max().reset_index()))
        lines += _section(
            "Track A against Track B", ["Image AUROC; Track B best per classifier.", *body]
        )

    lighting = runs[runs["track"] == "L"]
    if not lighting.empty:
        cols = ["classifier", "k", "adapt_normals", "regular/auroc", "shifted/auroc", "gap_auroc"]
        cols += [c for c in ("shifted/nll_bal",) if c in lighting.columns]
        table = lighting.sort_values(["k", "classifier", "adapt_normals"])[cols]
        lines += _section(
            "Lighting adaptation (post-freeze study)",
            [
                "Separate study (PLAN_1 § Post-freeze study), not part of the lock benchmark. "
                "Scene-level folds over every public test scene; m = good scenes whose "
                "target-lit images join the context. `shifted` averages every shifted "
                "(scenario, lighting) pair; `gap_auroc` is regular minus shifted.",
                markdown_table(table),
                "Pre-declared paired comparisons on shifted lighting (95% bootstrap CI).",
                markdown_table(lighting_paired(lighting)),
            ],
        )

    search_dir = artifacts / "search"
    search_body: list[str] = []
    if split == "lock":
        search_body.append("Searches run on the dev split only; see the dev report.")
    elif search_dir.is_dir() and _search_files(search_dir):
        plot_search(search_dir, figures / "search.png")
        search_body.append("![TPE against TabPFN-BO](figures/search.png)")
    db = artifacts / "optuna.db"
    if split == "dev" and db.is_file():
        import optuna

        storage = f"sqlite:///{db}"
        for name in sorted(optuna.get_all_study_names(storage)):
            if _NSGA2_STUDY.fullmatch(name):
                plot_pareto(storage, name, figures / f"pareto_{name}.png")
                search_body.append(f"![{name}](figures/pareto_{name}.png)")
    lines += _section("Search", search_body or ["No search results yet."])

    page = page_dir / "results.md"
    page.write_text("\n".join(lines).rstrip() + "\n")
    return page
