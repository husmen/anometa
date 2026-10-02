"""Tests for `anometa.trackb.pipeline`: PCA, design matrix and the Track B runner."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import cfg_b
from numpy.typing import NDArray

import anometa.trackb.pipeline as pipeline
from anometa.artifacts import LICENCES
from anometa.cli import apply_overrides
from anometa.config import ClassifierName, Paths, Scenario
from anometa.data.splits import BudgetError, eval_rows, load_split, sample_few_shot, write_split
from anometa.features.extract import cache_path
from anometa.metrics.image import prior_correct
from anometa.trackb.classifiers import Scorer
from anometa.trackb.pipeline import context_normals, run_track_b


def record_fits(monkeypatch: pytest.MonkeyPatch, build_as: ClassifierName) -> list[tuple[int, int]]:
    """Patch `pipeline.make_scorer` to build `build_as` scorers that log every fit.

    Args:
        monkeypatch: Pytest's monkeypatch fixture.
        build_as: Classifier actually built, whatever name the pipeline asks for.

    Returns:
        A list that gains `(fit rows, positive labels)` per scorer fit.
    """
    fits: list[tuple[int, int]] = []
    real_make_scorer = pipeline.make_scorer

    def make_scorer(
        name: ClassifierName,
        params: Mapping[str, int | float | str],
        *,
        seed: int,
        device: str,
        cache_key: str | None = None,
        cache_dir: Path | None = None,
    ) -> Scorer:
        """Build the `build_as` scorer and wrap its `fit` to log its inputs.

        `cache_key`/`cache_dir` are accepted (and ignored) so this also
        stands in for `tabpfn_thinking`'s wider call signature.
        """
        scorer = real_make_scorer(build_as, params, seed=seed, device=device)
        real_fit = scorer.fit

        def fit(X: NDArray[np.float32], y: NDArray[np.int64]) -> Scorer:
            """Log `(len(X), y.sum())`, then fit."""
            fits.append((len(X), int(y.sum())))
            return real_fit(X, y)

        monkeypatch.setattr(scorer, "fit", fit)
        return scorer

    monkeypatch.setattr(pipeline, "make_scorer", make_scorer)
    return fits


@pytest.mark.parametrize(
    ("classifier", "build_as", "version", "overrides", "n_fits"),
    [
        ("tabpfn", "logreg", "v3.5", {}, 2),
        ("tabpfn_fast", "logreg", "v3.5-fast", {}, 2),
        ("tabpfn_outlier", "mahalanobis", "v3.5", {"k": 0, "seeds": (0,)}, 1),
    ],
)
def test_tabpfn_revision_read_after_fits(
    prepared: Paths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    classifier: ClassifierName,
    build_as: ClassifierName,
    version: str,
    overrides: dict[str, object],
    n_fits: int,
) -> None:
    """Every TabPFN classifier records its checkpoint, hashed only after every fit.

    On a fresh host the first fit downloads the checkpoint, so reading it
    earlier would fail; `tabpfn_outlier` records the v3.5 checkpoint too.
    """
    fits = record_fits(monkeypatch, build_as)
    fits_before_revision: list[int] = []

    def fake_revision(v: str) -> dict[str, str]:
        """Record how many fits ran before the checkpoint was read."""
        assert v == version
        fits_before_revision.append(len(fits))
        return {"tabpfn_version": "9.9", "checkpoint": "c.ckpt", "sha256": "ab" * 32}

    monkeypatch.setattr(pipeline, "tabpfn_revision", fake_revision)
    out = run_track_b(cfg_b(prepared, classifier=classifier, **overrides), tmp_path)
    assert fits_before_revision == [n_fits]
    assert out.model_revisions["tabpfn"] == "9.9@c.ckpt#abababababab"


def test_tabpfn_thinking_records_api_terms_and_no_latency(
    prepared: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `tabpfn_thinking` run records the Prior Labs API terms and NaN latencies.

    It runs on the API, not on local weights, so its manifest carries the API
    terms instead of the local weights licence, an `api:` model revision, and
    no fit/predict latency (network time isn't comparable).
    """
    record_fits(monkeypatch, "logreg")
    out = run_track_b(cfg_b(prepared, classifier="tabpfn_thinking", seeds=(0,)), tmp_path)
    assert out.licences["tabpfn_api"] == LICENCES["tabpfn_api"]
    assert "tabpfn" not in out.licences
    assert out.model_revisions["tabpfn_thinking"] == "api:v3.5 effort=medium metric=roc_auc"
    assert np.isnan(out.metrics["fit_latency_ms"])
    assert np.isnan(out.metrics["predict_latency_ms"])


def test_run_track_b_end_to_end(prepared: Paths, tmp_path: Path) -> None:
    """A logreg run predicts every dev eval row per seed and reports raw and balanced metrics.

    `score_balanced` is `score` prior-corrected from the fit prior k / (6 train + k).
    """
    out = run_track_b(cfg_b(prepared), tmp_path)
    split = load_split(Scenario.VIAL, prepared)
    pred = out.predictions
    for seed in (0, 1):
        shots = sample_few_shot(split, scenario=Scenario.VIAL, k=2, seed=seed, lighting="regular")
        eval_ids = set(eval_rows(split, split="dev", shots=shots).image_id)
        assert set(pred[pred.seed == seed].image_id) == eval_ids
    np.testing.assert_allclose(
        pred.score_balanced, prior_correct(pred.score.to_numpy(dtype=np.float64), 2 / (6 + 2))
    )
    assert {"auroc", "vial/auroc", "ece", "ece_bal", "gap_auroc", "fit_latency_ms"} <= set(
        out.metrics
    )


def test_pca_and_fit_never_see_lock_rows(
    prepared: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PCA fits on the 6 train normals; each seed's scorer fits on them plus k=2 dev shots."""
    seen: dict[str, int] = {}
    real_fit_pca = pipeline.fit_pca
    monkeypatch.setattr(
        pipeline,
        "fit_pca",
        lambda e, d: seen.setdefault("pca", e.shape[0]) and real_fit_pca(e, d),
    )
    fits = record_fits(monkeypatch, "logreg")
    out = run_track_b(cfg_b(prepared), tmp_path)
    lock = set(load_split(Scenario.VIAL, prepared).query("split == 'lock'").image_id)
    assert seen["pca"] == 6
    assert fits == [(6 + 2, 2), (6 + 2, 2)]
    assert not lock & set(out.predictions.image_id)


def test_n_normals_shrinks_only_the_scorer_context(
    prepared: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With n_normals=3, each scorer fits on 3 of the 6 train normals plus k=2 shots.

    PCA still fits on all 6 normals, and `score_balanced` is prior-corrected
    from the fit prior the scorer actually saw, k / (3 + k).
    """
    seen: dict[str, int] = {}
    real_fit_pca = pipeline.fit_pca
    monkeypatch.setattr(
        pipeline,
        "fit_pca",
        lambda e, d: seen.setdefault("pca", e.shape[0]) and real_fit_pca(e, d),
    )
    fits = record_fits(monkeypatch, "logreg")
    out = run_track_b(cfg_b(prepared, classifier_params={"n_normals": 3}), tmp_path)
    assert seen["pca"] == 6
    assert fits == [(3 + 2, 2), (3 + 2, 2)]
    pred = out.predictions
    np.testing.assert_allclose(
        pred.score_balanced, prior_correct(pred.score.to_numpy(dtype=np.float64), 2 / (3 + 2))
    )


def test_context_normals_is_seeded_and_keeps_all_when_asked_for_more() -> None:
    """Draws are sorted, reproducible and differ by seed; n >= the pool or None keeps every row."""
    rows = np.arange(10, 40, dtype=np.int64)
    a = context_normals(rows, 8, seed=0, scenario=Scenario.VIAL)
    assert len(a) == 8
    assert np.all(np.diff(a) > 0)
    assert set(a) <= set(rows)
    np.testing.assert_array_equal(a, context_normals(rows, 8, seed=0, scenario=Scenario.VIAL))
    assert not np.array_equal(a, context_normals(rows, 8, seed=1, scenario=Scenario.VIAL))
    np.testing.assert_array_equal(context_normals(rows, 30, seed=0, scenario=Scenario.VIAL), rows)
    np.testing.assert_array_equal(context_normals(rows, None, seed=0, scenario=Scenario.VIAL), rows)


def test_one_class_run_scores_all_dev(prepared: Paths, tmp_path: Path) -> None:
    """Mahalanobis at k=0 scores every dev row and reports ranking metrics only."""
    out = run_track_b(cfg_b(prepared, classifier="mahalanobis", k=0, seeds=(0,)), tmp_path)
    dev = load_split(Scenario.VIAL, prepared).query("split == 'dev'")
    assert set(out.predictions.image_id) == set(dev.image_id)
    assert "auroc" in out.metrics
    assert "ece" not in out.metrics


def test_exhaustive_shots_budget_error_fires_before_any_fit(
    prepared: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later scenario whose shots cover every dev defect scene fails before anything fits.

    Vial keeps one of its 3 dev defect scenes to evaluate at k=2; wallplugs, a
    copy of Vial's split with one dev defect scene moved to lock, has only 2.
    The BudgetError names wallplugs, k=2 and its 2 dev defect scenes, and no
    PCA or scorer was fitted for Vial first.
    """
    split = load_split(Scenario.VIAL, prepared)
    dev_defect_scenes = sorted(split.query("split == 'dev' and label == 1").scene_id.unique())
    assert len(dev_defect_scenes) == 3
    split.loc[split.scene_id == dev_defect_scenes[0], "split"] = "lock"
    write_split(split, prepared.splits / "wallplugs.csv")

    def never(*args: object, **kwargs: object) -> None:
        """Fail the test: nothing may be fitted before every budget check passed."""
        raise AssertionError("fitted before the budget check")

    monkeypatch.setattr(pipeline, "fit_pca", never)
    monkeypatch.setattr(pipeline, "make_scorer", never)
    cfg = cfg_b(prepared, scenarios=(Scenario.VIAL, Scenario.WALLPLUGS))
    with pytest.raises(BudgetError, match=r"^wallplugs: k=2 .* cover all 2 dev defect scenes"):
        run_track_b(cfg, tmp_path)


def test_mixed_feature_provenance_raises(prepared: Paths, tmp_path: Path) -> None:
    """Scenarios whose features differ in provenance raise ValueError naming both setups.

    Wallplugs reuses Vial's split and features, but its cache says it was
    extracted on MPS while Vial's says CPU.
    """
    write_split(load_split(Scenario.VIAL, prepared), prepared.splits / "wallplugs.csv")
    with np.load(cache_path("dinov3_s", Scenario.VIAL, prepared)) as data:
        arrays = {k: data[k] for k in data.files} | {"provenance_device": np.str_("mps")}
    np.savez_compressed(cache_path("dinov3_s", Scenario.WALLPLUGS, prepared), **arrays)
    cfg = cfg_b(prepared, scenarios=(Scenario.VIAL, Scenario.WALLPLUGS))
    with pytest.raises(ValueError, match=r"^wallplugs: .*device=mps.*device=cpu"):
        run_track_b(cfg, tmp_path)


def test_apply_overrides() -> None:
    """`--set` values are YAML-typed; null restores the default."""
    assert apply_overrides({"k": 1}, ["k=5", "classifier=knn", "seeds=[0, 1]"]) == {
        "k": 5,
        "classifier": "knn",
        "seeds": [0, 1],
    }
    assert apply_overrides({"k": 1, "seeds": [0]}, ["seeds=null"]) == {"k": 1}


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_parallel_seeds_match_serial(
    prepared: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, executor: str
) -> None:
    """Seeds fitted concurrently, in threads or processes, give exactly the serial predictions."""
    cfg = cfg_b(prepared, seeds=(0, 1, 2), device="cpu")
    serial = run_track_b(cfg, tmp_path).predictions
    monkeypatch.setenv("ANOMETA_SEED_WORKERS", "3")
    monkeypatch.setenv("ANOMETA_SEED_EXECUTOR", executor)
    pd.testing.assert_frame_equal(run_track_b(cfg, tmp_path).predictions, serial)


@pytest.mark.models
def test_parallel_tabpfn_seeds_match_serial(
    prepared: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TabPFN-3.5-Fast seeds fitted in threads give the serial scores, on CUDA or else CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = cfg_b(prepared, classifier="tabpfn_fast", seeds=(0, 1, 2), device=device)
    serial = run_track_b(cfg, tmp_path).predictions
    monkeypatch.setenv("ANOMETA_SEED_WORKERS", "3")
    pd.testing.assert_frame_equal(run_track_b(cfg, tmp_path).predictions, serial)


def test_threaded_seeds_refuse_mps(
    prepared: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Threads on MPS raise a ValueError pointing at the process executor, before any fit."""
    monkeypatch.setattr(pipeline, "resolve_device", lambda device: "mps")
    monkeypatch.setenv("ANOMETA_SEED_WORKERS", "2")
    with pytest.raises(ValueError, match="ANOMETA_SEED_EXECUTOR=process"):
        run_track_b(cfg_b(prepared), tmp_path)
