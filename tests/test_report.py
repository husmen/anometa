"""Tests for `anometa.report`: tables, one-class row matching and the results page."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import cfg_b

from anometa.config import Scenario, TrackAConfig
from anometa.experiment import run_experiment
from anometa.report import (
    build_report,
    fixed_set_budget,
    load_runs,
    markdown_table,
    matched_cells,
    one_class_on_fewshot_rows,
    paired_comparisons,
    with_variants,
)


def test_markdown_table():
    """Floats are formatted and the header separator is present."""
    md = markdown_table(pd.DataFrame({"a": [0.12345], "b": ["x"]}))
    assert md.splitlines() == ["| a | b |", "| --- | --- |", "| 0.123 | x |"]


def test_markdown_table_keeps_integer_columns_integral():
    """Columns k and pca_dim render as integers after pandas made them floats; NaN stays."""
    md = markdown_table(
        pd.DataFrame({"k": [1.0, 5.0], "pca_dim": [16.0, float("nan")], "a": [1.0, 0.5]})
    )
    assert md.splitlines()[2:] == ["| 1 | 16 | 1.000 |", "| 5 | nan | 0.500 |"]


def test_one_class_on_fewshot_rows():
    """One-class scores are restricted to the few-shot run's evaluation rows."""
    oc = pd.DataFrame(
        {"scenario": ["vial"] * 3, "seed": 0, "image_id": ["a", "b", "c"], "score": [1, 2, 3]}
    )
    fs = pd.DataFrame({"scenario": ["vial"] * 2, "seed": [3, 3], "image_id": ["a", "c"]})
    out = one_class_on_fewshot_rows(oc, fs)
    assert list(out.image_id) == ["a", "c"]
    assert set(out.seed) == {3}


def test_build_report_on_fake_artifacts(prepared):
    """Real (fake-encoder) Track A and Track B runs produce a results page and figures.

    `load_runs` keeps one row per ok dev run, with flattened config fields.
    """
    for clf in ["logreg", "knn"]:
        run_experiment(cfg_b(prepared, classifier=clf))
    run_experiment(
        TrackAConfig(
            model="patch_distance", encoder="dinov3_s", scenarios=(Scenario.VIAL,), paths=prepared
        )
    )
    runs = load_runs(prepared.artifacts, "dev")
    assert sorted(runs.classifier.dropna()) == ["knn", "logreg"]
    assert set(runs.track) == {"A", "B"}
    assert set(runs.features.dropna()) == {"cls+mean_patch+novelty"}
    page = build_report(prepared.artifacts, prepared.artifacts.parent / "reports", "dev")
    assert page.exists()
    assert "auroc" in page.read_text().lower()
    assert any((page.parent / "figures").glob("*.png"))


def test_load_runs_keeps_one_row_per_config(prepared):
    """A rerun under another name (same config hash) is loaded once, as the first by run_id."""
    first = run_experiment(cfg_b(prepared, classifier="logreg"))
    second = run_experiment(cfg_b(prepared, classifier="logreg", name="latency"))
    assert first.run_id != second.run_id
    runs = load_runs(prepared.artifacts, "dev")
    assert len(runs) == 1
    assert runs.params.iloc[0] == "{}"
    assert runs.run_id.iloc[0] == min(first.run_id, second.run_id)


def _write_pred(run_dir: Path, scores: dict[str, np.ndarray], seeds: int = 2) -> Path:
    """Write a predictions.parquet: two scenarios, `seeds` seeds, 10 good and 10 bad scenes.

    `scores[scenario]` gives the score per row of one seed (good rows first).
    """
    rows = [
        dict(
            scenario=sc,
            seed=s,
            scene_id=f"{kind}/{i:03d}",
            image_id=f"{kind}/{i:03d}",
            label=label,
            lighting="regular",
            score=float(scores[sc][label * 10 + i]),
        )
        for sc in scores
        for s in range(seeds)
        for label, kind in [(0, "good"), (1, "bad")]
        for i in range(10)
    ]
    run_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(run_dir / "predictions.parquet")
    return run_dir


def test_paired_comparisons_scopes_and_one_class(tmp_path):
    """TabPFN perfect on vial and inverted on wallplugs, logreg the reverse.

    Over all scenarios the AUROC difference is 0. Without wallplugs TabPFN
    leads by 1 with p_tabpfn_better 1. The one-class control (a single seed)
    is scored on TabPFN's rows, and per-scenario columns hold point differences.
    """
    label = np.r_[np.zeros(10), np.ones(10)]
    tab = _write_pred(tmp_path / "tab", {"vial": label, "wallplugs": 1 - label})
    log = _write_pred(tmp_path / "log", {"vial": 1 - label, "wallplugs": label})
    oc = _write_pred(tmp_path / "oc", {"vial": label, "wallplugs": label}, seeds=1)
    common = dict(k=2, encoder="dinov3_s", features="cls", pca_dim=16.0, shot_lighting="regular")
    best = pd.DataFrame(
        [
            dict(classifier="tabpfn", run_dir=tab, auroc=0.5, **common),
            dict(classifier="logreg", run_dir=log, auroc=0.5, **common),
        ]
    )
    one_class = pd.DataFrame([dict(classifier="mahalanobis", run_dir=oc, auroc=1.0, **common)])
    out = paired_comparisons(best, one_class, n_boot=50)
    by = out.set_index(["control", "scope"])
    assert by.loc[("logreg", "all"), "diff"] == pytest.approx(0.0)
    assert by.loc[("logreg", "without wallplugs"), "diff"] == pytest.approx(1.0)
    assert by.loc[("logreg", "without wallplugs"), "p_tabpfn_better"] == 1.0
    assert by.loc[("logreg", "all"), "vial"] == pytest.approx(1.0)
    assert pd.isna(by.loc[("logreg", "without wallplugs"), "wallplugs"])
    assert by.loc[("mahalanobis", "all"), "diff"] == pytest.approx(-0.5)


def test_matched_cells_uses_default_params_only():
    """Only default-parameter runs in the same cell are compared; tuned runs are ignored.

    Two cells: TabPFN wins one by 0.2 and loses the other by 0.1 overall;
    without wallplugs (both tied there) the per-cell differences change.
    """

    def row(classifier, pca, vial, wallplugs, params="{}"):
        return dict(
            classifier=classifier,
            encoder="dinov3_s",
            features="cls",
            pca_dim=pca,
            k=2,
            shot_lighting="regular",
            params=params,
            **{"vial/auroc": vial, "wallplugs/auroc": wallplugs},
        )

    fewshot = pd.DataFrame(
        [
            row("tabpfn", 16.0, 0.9, 0.5),
            row("logreg", 16.0, 0.7, 0.3),
            row("tabpfn", 32.0, 0.6, 0.5),
            row("logreg", 32.0, 0.6, 0.7),
            row("logreg", 32.0, 0.0, 0.0, params='{"C": 0.01}'),
        ]
    )
    out = matched_cells(fewshot).set_index("scope")
    assert set(out["metric"]) == {"auroc"}
    assert out.loc["all", "cells"] == 2
    assert out.loc["all", "mean_diff"] == pytest.approx(0.05)
    assert out.loc["all", "share_tabpfn_better"] == 0.5
    assert out.loc["without wallplugs", "mean_diff"] == pytest.approx(0.1)


def test_fixed_set_budget_scores_every_k_on_the_largest_k_rows(tmp_path):
    """k=1 is rescored on the k=2 run's rows; the one-class control on the same rows.

    The k=1 run scores scene bad/000 wrong; the k=2 run dropped that scene
    as a shot, so on the fixed rows k=1 is perfect too. A tuned run of the
    same classifier is not mistaken for the configuration.
    """
    label = np.r_[np.zeros(10), np.ones(10)]
    k1_scores = label.copy()
    k1_scores[10] = -1.0  # bad/000 ranked below every good image
    k1 = _write_pred(tmp_path / "k1", {"vial": k1_scores})
    k2 = _write_pred(tmp_path / "k2", {"vial": label})
    k2_pred = pd.read_parquet(k2 / "predictions.parquet")
    k2_pred[k2_pred.scene_id != "bad/000"].to_parquet(k2 / "predictions.parquet")
    oc = _write_pred(tmp_path / "oc", {"vial": k1_scores}, seeds=1)
    common = dict(encoder="dinov3_s", features="cls", pca_dim=16.0, shot_lighting="regular")
    fewshot = pd.DataFrame(
        [
            dict(classifier="tabpfn", k=1, run_dir=k1, auroc=0.95, params="{}", **common),
            dict(classifier="tabpfn", k=2, run_dir=k2, auroc=1.0, params="{}", **common),
            dict(classifier="tabpfn", k=1, run_dir=k2, auroc=0.0, params='{"n": 1}', **common),
        ]
    )
    one_class = pd.DataFrame(
        [dict(classifier="mahalanobis", k=0, run_dir=oc, auroc=0.9, params="{}", **common)]
    )
    out = fixed_set_budget(fewshot, one_class).set_index("classifier")
    assert list(out.columns) == ["encoder", "features", "pca_dim", "k=0", "k=1", "k=2"]
    assert out.loc["tabpfn", "k=1"] == pytest.approx(1.0)
    assert out.loc["tabpfn", "k=2"] == pytest.approx(1.0)
    assert out.loc["mahalanobis", "k=0"] == pytest.approx(1.0)


def test_matched_cells_lower_is_better_for_calibration():
    """For nll_bal, a lower TabPFN value counts as TabPFN better."""
    fewshot = pd.DataFrame(
        [
            dict(
                classifier=c,
                encoder="dinov3_s",
                features="cls",
                pca_dim=16.0,
                k=2,
                shot_lighting="regular",
                params="{}",
                **{"vial/nll_bal": v},
            )
            for c, v in [("tabpfn", 0.6), ("logreg", 1.2)]
        ]
    )
    out = matched_cells(fewshot, "nll_bal").set_index("scope")
    assert out.loc["all", "mean_diff"] == pytest.approx(-0.6)
    assert out.loc["all", "share_tabpfn_better"] == 1.0


def test_with_variants_names_context_size_runs_apart():
    """A tabpfn run with n_normals becomes its own classifier; tuned params do not."""
    runs = pd.DataFrame(
        {
            "classifier": ["tabpfn", "tabpfn", "logreg"],
            "params": ["{}", '{"n_normals": 32}', '{"C": 0.01}'],
        }
    )
    assert list(with_variants(runs)["classifier"]) == ["tabpfn", "tabpfn n_normals=32", "logreg"]
