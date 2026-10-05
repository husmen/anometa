"""Tests for the lighting-adaptation study: `LightingConfig` and `run_lighting`."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import anometa.trackb.lighting as lighting
from anometa.config import ClassifierName, LightingConfig, Paths, Scenario
from anometa.data.splits import lighting_fold, load_split
from anometa.experiment import run_experiment
from anometa.trackb.classifiers import Scorer


def cfg_l(paths: Paths, **kw: object) -> LightingConfig:
    """Build a lighting-study config over the fake Vial scenario, overriding any field."""
    base: dict[str, object] = dict(
        encoder="dinov3_s",
        pca_dim=4,
        classifier="logreg",
        k=1,
        scenarios=(Scenario.VIAL,),
        seeds=(0, 1),
        paths=paths,
    )
    return LightingConfig.model_validate(base | kw)


def record_fits(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Patch `lighting.make_scorer` so every scorer logs `(fit rows, positive labels)`."""
    fits: list[tuple[int, int]] = []
    real = lighting.make_scorer

    def make_scorer(
        name: ClassifierName, params: Mapping[str, int | float | str], *, seed: int, device: str
    ) -> Scorer:
        """Build the real scorer and wrap its `fit` to log its inputs."""
        scorer = real(name, params, seed=seed, device=device)
        fit = scorer.fit

        def logged(x: np.ndarray, y: np.ndarray) -> Scorer:
            """Record the fit set's size and positives, then fit."""
            fits.append((len(y), int(y.sum())))
            return fit(x, y)

        object.__setattr__(scorer, "fit", logged)
        return scorer

    monkeypatch.setattr(lighting, "make_scorer", make_scorer)
    return fits


def test_lighting_config_rules(paths: Paths) -> None:
    """Defaults are the frozen config; too many adaptation scenes, Thinking or lock are rejected."""
    cfg = LightingConfig(classifier="tabpfn", paths=paths)
    assert (cfg.encoder, cfg.features, cfg.pca_dim, cfg.k) == (
        "dinov3_l",
        ("cls", "mean_patch", "novelty"),
        16,
        2,
    )
    for bad in (
        dict(adapt_normals=3, adapt_max=2),
        dict(classifier="tabpfn_thinking"),
        dict(split="lock"),
        dict(features=("novelty",), pca_dim=16),
    ):
        with pytest.raises(ValidationError):
            cfg_l(paths, **bad)


def test_context_adds_adaptation_images_of_the_target_lighting(
    prepared: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With m = 2, each fit holds 6 train normals, 2 target-lit adaptation images and 1 shot.

    One fit per (seed, target lighting): 2 seeds x 3 lightings. With m = 0
    the context is the train normals and the shot only.
    """
    fits = record_fits(monkeypatch)
    lighting.run_lighting(cfg_l(prepared, adapt_normals=2), tmp_path)
    assert fits == [(6 + 2 + 1, 1)] * 6
    fits.clear()
    lighting.run_lighting(cfg_l(prepared, adapt_normals=0), tmp_path)
    assert fits == [(6 + 1, 1)] * 6


def test_evaluation_rows_exclude_context_scenes_and_ignore_m(
    prepared: Paths, tmp_path: Path
) -> None:
    """Every target lighting scores the fold's held-out scenes; m never changes those rows."""
    split = load_split(Scenario.VIAL, prepared)
    out0 = lighting.run_lighting(cfg_l(prepared, adapt_normals=0), tmp_path)
    out2 = lighting.run_lighting(cfg_l(prepared, adapt_normals=2), tmp_path)
    for seed in (0, 1):
        fold = lighting_fold(split, scenario=Scenario.VIAL, seed=seed, k=1, adapt_max=2)
        p0 = out0.predictions[out0.predictions.seed == seed]
        assert set(p0.image_id) == set(fold.eval_rows.image_id)
        assert not set(p0.scene_id) & set(fold.adapt_scenes)
        p2 = out2.predictions[out2.predictions.seed == seed]
        assert set(p2.image_id) == set(p0.image_id)


def test_run_experiment_writes_lighting_metrics(prepared: Paths) -> None:
    """A track-L run reports regular, shifted, gap and per-lighting metrics.

    One-class classifiers run too: they fit on normals (plus adaptation
    images) and take no shots.
    """
    result = run_experiment(cfg_l(prepared, adapt_normals=1))
    assert result.status == "ok", result.error
    for key in (
        "regular/auroc",
        "shifted/auroc",
        "gap_auroc",
        "overexposed/auroc",
        "shifted/nll_bal",
    ):
        assert key in result.metrics
    assert result.run_id.startswith("run-")
    one_class = run_experiment(cfg_l(prepared, classifier="mahalanobis", adapt_normals=1))
    assert one_class.status == "ok", one_class.error
    assert "shifted/nll_bal" not in one_class.metrics
