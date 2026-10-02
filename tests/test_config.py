"""Tests for `anometa.config`: experiment schema, hashing and YAML loading."""

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from anometa.config import (
    EncoderName,
    Paths,
    Scenario,
    TrackAConfig,
    TrackBConfig,
    config_hash,
    load_config,
    run_id,
    seed_parallelism,
)


def test_load_trackb_yaml_canonicalises_features(tmp_path: Path) -> None:
    """Features given in any order load in canonical order."""
    p = tmp_path / "c.yaml"
    p.write_text(
        "track: B\nencoder: dinov3_s\nfeatures: [novelty, cls]\npca_dim: 16\n"
        "classifier: tabpfn\nk: 2\n"
    )
    cfg = load_config(p)
    assert isinstance(cfg, TrackBConfig)
    assert cfg.features == ("cls", "novelty")


def test_load_smoke_vial_config() -> None:
    """The shipped `smoke_vial.yaml` loads as a Track B logreg run on Vial."""
    cfg = load_config(Path(__file__).parents[1] / "configs/experiments/smoke_vial.yaml")
    assert isinstance(cfg, TrackBConfig)
    assert cfg.name == "smoke_vial"
    assert cfg.scenarios == (Scenario.VIAL,)
    assert cfg.seeds == (0, 1)
    assert cfg.features == ("cls", "mean_patch", "novelty")


def test_load_config_rejects_unknown_key(tmp_path: Path) -> None:
    """A YAML key that isn't a config field raises ValidationError."""
    p = tmp_path / "c.yaml"
    p.write_text("track: A\nmodel: patchcore\nmax_step: 10\n")
    with pytest.raises(ValidationError, match="max_step"):
        load_config(p)


def test_config_hash_ignores_name_and_paths() -> None:
    """Renaming a run or moving its paths keeps the hash; changing k does not."""
    a = TrackBConfig(encoder="dinov3_s", features=("cls",), pca_dim=16, classifier="logreg", k=1)
    assert config_hash(a) == config_hash(
        a.model_copy(update={"name": "x", "paths": Paths(data=Path("/d"))})
    )
    assert config_hash(a) != config_hash(a.model_copy(update={"k": 2}))
    assert re.fullmatch(r"run-[0-9a-f]{12}", run_id(a))


def test_config_hash_ignores_classifier_params_order() -> None:
    """Configs whose classifier_params differ only in key order hash equal."""
    base = dict(encoder="dinov3_s", features=("cls",), pca_dim=16, classifier="tabpfn", k=1)
    a = TrackBConfig.model_validate(base | {"classifier_params": {"n": 4, "t": 0.5}})
    b = TrackBConfig.model_validate(base | {"classifier_params": {"t": 0.5, "n": 4}})
    assert config_hash(a) == config_hash(b)


def test_name_rejects_path_traversal() -> None:
    """A run name that could escape `artifacts/` through `run_id` raises ValidationError."""
    with pytest.raises(ValidationError, match="name"):
        TrackAConfig(model="patchcore", name="../x")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(features=("cls",), pca_dim=None, classifier="logreg", k=1),  # embedding needs PCA
        dict(
            features=("novelty",), pca_dim=16, classifier="logreg", k=1
        ),  # novelty alone has no PCA
        dict(features=("cls",), pca_dim=16, classifier="mahalanobis", k=2),  # one-class means k=0
        dict(features=("cls",), pca_dim=16, classifier="mahalanobis", k=0, seeds=(0, 1)),
        dict(features=("cls",), pca_dim=16, classifier="logreg", k=0),  # few-shot means k>0
        dict(features=("cls",), pca_dim=16, classifier="logreg", k=10, shot_lighting="all"),  # dev
        dict(
            features=("cls",),
            pca_dim=16,
            classifier="tabpfn_thinking",
            k=1,
            classifier_params={"n_estimators": 4},
        ),  # tabpfn_thinking takes no params
        dict(
            encoder="siglip2",
            features=("cls",),
            pca_dim=16,
            classifier="logreg",
            k=1,
            encoder_backend="timm",
        ),
        dict(
            features=("cls",),
            pca_dim=16,
            classifier="mahalanobis",
            k=0,
            seeds=(0,),
            classifier_params={"n_normals": 8},
        ),  # n_normals is for few-shot classifiers
        dict(
            features=("cls",),
            pca_dim=16,
            classifier="tabpfn",
            k=1,
            classifier_params={"n_normals": 0},
        ),
        dict(
            features=("cls",),
            pca_dim=16,
            classifier="tabpfn",
            k=1,
            classifier_params={"n_normals": 2.5},
        ),
    ],
)
def test_trackb_rejects_inconsistent_configs(kwargs: dict[str, object]) -> None:
    """Each inconsistent Track B combination raises ValidationError."""
    with pytest.raises(ValidationError):
        TrackBConfig.model_validate({"encoder": "dinov3_s"} | kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(features=(), pca_dim=None),
        dict(scenarios=()),
        dict(seeds=()),
        dict(pca_dim=0),
        dict(k=-1),
    ],
)
def test_trackb_rejects_empty_or_out_of_range_fields(kwargs: dict[str, object]) -> None:
    """Empty features, scenarios or seeds, a non-positive pca_dim or a negative k raise."""
    base = dict(encoder="dinov3_s", features=("cls",), pca_dim=16, classifier="logreg", k=1)
    with pytest.raises(ValidationError):
        TrackBConfig.model_validate(base | kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scenarios=()),
        dict(seeds=()),
        dict(max_steps=0),
        dict(image_size=(0, 256)),
        dict(image_size=(256, 0)),
        dict(encoder="dinov3_s"),  # only patch_distance takes an encoder
    ],
)
def test_tracka_rejects_invalid_fields(kwargs: dict[str, object]) -> None:
    """Empty scenarios or seeds, non-positive sizes or steps, or a stray encoder raise."""
    with pytest.raises(ValidationError):
        TrackAConfig.model_validate({"model": "patchcore"} | kwargs)


@pytest.mark.parametrize("encoder", [None, "siglip2"])
def test_patch_distance_requires_dinov3(encoder: EncoderName | None) -> None:
    """The patch distance map is defined on DINOv3 features only."""
    with pytest.raises(ValidationError):
        TrackAConfig(model="patch_distance", encoder=encoder)


@pytest.mark.parametrize("encoder", [None, "siglip2"])
def test_track_a_timm_backend_requires_dinov3(encoder: EncoderName | None) -> None:
    """The timm encoder backend is only wired up for DINOv3 checkpoints."""
    with pytest.raises(ValidationError):
        TrackAConfig(model="patchcore", encoder=encoder, encoder_backend="timm")


@pytest.mark.parametrize(
    ("workers", "executor"), [("0", "thread"), ("two", "thread"), ("2", "fork")]
)
def test_seed_parallelism_rejects_bad_env(
    monkeypatch: pytest.MonkeyPatch, workers: str, executor: str
) -> None:
    """A non-positive or non-integer worker count, or an unknown executor, raises ValueError."""
    monkeypatch.setenv("ANOMETA_SEED_WORKERS", workers)
    monkeypatch.setenv("ANOMETA_SEED_EXECUTOR", executor)
    with pytest.raises(ValueError, match="ANOMETA_SEED_"):
        seed_parallelism()


def test_seed_parallelism_defaults_to_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the environment variables, seeds run serially in the calling thread."""
    monkeypatch.delenv("ANOMETA_SEED_WORKERS", raising=False)
    monkeypatch.delenv("ANOMETA_SEED_EXECUTOR", raising=False)
    assert seed_parallelism() == (1, "thread")
