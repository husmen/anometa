"""Experiment configuration schema, validation and content hashing.

Defines the pydantic models that describe one experiment run for either
track, plus `load_config`, `config_hash` and `run_id` to load, hash and
identify a config. `TrackAConfig` and `TrackBConfig` share the common fields
in `_CommonConfig` and are combined into `ExperimentConfig`, a discriminated
union keyed on `track`.
"""

import hashlib
import json
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    TypeAdapter,
    field_validator,
    model_validator,
)


class Scenario(StrEnum):
    """MVTec AD 2 scenario folder names, in the canonical reporting order."""

    CAN = "can"
    FABRIC = "fabric"
    FRUIT_JELLY = "fruit_jelly"
    RICE = "rice"
    SHEET_METAL = "sheet_metal"
    VIAL = "vial"
    WALLPLUGS = "wallplugs"
    WALNUTS = "walnuts"


EncoderName = Literal["dinov3_s", "dinov3_l", "siglip2"]
"""Frozen vision-foundation-model encoders available for feature extraction."""

FeatureBlock = Literal["cls", "mean_patch", "novelty"]
"""Feature blocks a Track B run can concatenate into its tabular input."""

ClassifierName = Literal[
    "tabpfn", "tabpfn_fast", "tabpfn_thinking", "logreg", "knn", "mahalanobis", "tabpfn_outlier"
]
"""Classifiers a Track B run can fit on the extracted features."""

ONE_CLASS: frozenset[str] = frozenset({"mahalanobis", "tabpfn_outlier"})
"""Classifiers that fit on normal samples only and take no labelled shots."""

_DINOV3_ENCODERS: tuple[EncoderName, ...] = ("dinov3_s", "dinov3_l")
_FEATURE_ORDER: dict[str, int] = {"cls": 0, "mean_patch": 1, "novelty": 2}


def _check_timm_requires_dinov3(
    encoder: EncoderName | None, encoder_backend: Literal["transformers", "timm"]
) -> None:
    """Raise unless the `timm` backend is paired with a DINOv3 encoder.

    Args:
        encoder: The configured encoder, or None when a track allows it.
        encoder_backend: The backend used to load `encoder`'s weights.

    Raises:
        ValueError: If `encoder_backend` is `"timm"` and `encoder` isn't DINOv3.
    """
    if encoder_backend == "timm" and encoder not in _DINOV3_ENCODERS:
        raise ValueError("encoder_backend='timm' requires a DINOv3 encoder")


class Paths(BaseModel):
    """Filesystem layout for a run's inputs and outputs.

    All fields are relative by default; callers may pass absolute paths to
    relocate a run without changing any other config field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    data: Path = Path("data/ad2")
    cache: Path = Path("cache")
    artifacts: Path = Path("artifacts")
    splits: Path = Path("splits")
    configs: Path = Path("configs")


class _CommonConfig(BaseModel):
    """Fields shared by every experiment track.

    Frozen so a validated config can't change under a running experiment;
    its content hash is `config_hash`. Build a modified copy with
    `type(cfg).model_validate(cfg.model_dump() | changes)`, which re-runs
    validation, instead of mutating.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")] = "run"
    split: Literal["dev", "lock"] = "dev"
    scenarios: Annotated[tuple[Scenario, ...], Field(min_length=1)] = tuple(Scenario)
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    paths: Paths = Paths()


class TrackBConfig(_CommonConfig):
    """Supervised few-shot adaptation: frozen VFM features, PCA, a classifier."""

    track: Literal["B"] = "B"
    seeds: Annotated[tuple[int, ...], Field(min_length=1)] = tuple(range(10))
    encoder: EncoderName
    features: Annotated[tuple[FeatureBlock, ...], Field(min_length=1)]
    pca_dim: PositiveInt | None
    classifier: ClassifierName
    classifier_params: dict[str, int | float | str] = {}
    k: NonNegativeInt
    shot_lighting: Literal["regular", "all"] = "regular"
    encoder_backend: Literal["transformers", "timm"] = "transformers"

    @field_validator("features")
    @classmethod
    def _canonicalise_features(cls, value: tuple[FeatureBlock, ...]) -> tuple[FeatureBlock, ...]:
        """Deduplicate feature blocks and sort into `cls, mean_patch, novelty` order."""
        return tuple(sorted(set(value), key=lambda block: _FEATURE_ORDER[block]))

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Reject combinations that don't correspond to a runnable experiment."""
        has_embedding = bool(set(self.features) & {"cls", "mean_patch"})
        if has_embedding and self.pca_dim is None:
            raise ValueError("pca_dim is required when features include cls or mean_patch")
        if not has_embedding and self.pca_dim is not None:
            raise ValueError("pca_dim must be None when features is novelty only")
        if self.classifier in ONE_CLASS:
            if self.k != 0:
                raise ValueError(f"{self.classifier} is one-class and requires k=0")
            if len(self.seeds) != 1:
                raise ValueError(f"{self.classifier} is one-class and takes a single seed")
        elif self.k <= 0:
            raise ValueError("k must be > 0 for a few-shot classifier")
        if self.classifier == "tabpfn_thinking" and self.classifier_params:
            raise ValueError("tabpfn_thinking takes no classifier_params")
        if self.shot_lighting == "all" and self.split != "lock":
            raise ValueError("shot_lighting='all' requires split='lock'")
        _check_timm_requires_dinov3(self.encoder, self.encoder_backend)
        return self


class TrackAConfig(_CommonConfig):
    """Unsupervised baseline: PatchCore, EfficientAD-S or a patch-distance map."""

    track: Literal["A"] = "A"
    seeds: Annotated[tuple[int, ...], Field(min_length=1)] = (0,)
    model: Literal["patchcore", "efficientad_s", "patch_distance"]
    encoder: EncoderName | None = None
    encoder_backend: Literal["transformers", "timm"] = "transformers"
    image_size: tuple[PositiveInt, PositiveInt] = (256, 256)
    max_steps: PositiveInt = 70000

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Reject combinations that don't correspond to a runnable experiment."""
        if self.model == "patch_distance" and self.encoder not in _DINOV3_ENCODERS:
            raise ValueError("patch_distance requires a DINOv3 encoder")
        if self.model != "patch_distance" and self.encoder is not None:
            raise ValueError(f"encoder is only used by patch_distance, not {self.model}")
        _check_timm_requires_dinov3(self.encoder, self.encoder_backend)
        return self


ExperimentConfig = Annotated[TrackAConfig | TrackBConfig, Field(discriminator="track")]
"""A validated Track A or Track B experiment config, selected by `track`."""

_experiment_config_adapter: TypeAdapter[TrackAConfig | TrackBConfig] = TypeAdapter(ExperimentConfig)


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Args:
        path: Path to a YAML file with a top-level `track: "A"` or `track: "B"` key.

    Returns:
        The validated `TrackAConfig` or `TrackBConfig`.
    """
    data: object = yaml.safe_load(path.read_text())
    return _experiment_config_adapter.validate_python(data)


def config_hash(cfg: ExperimentConfig, *, exclude: Iterable[str] = ()) -> str:
    """Compute a stable content hash of an experiment config.

    Excludes `name` and `paths`: renaming a run or relocating its files
    leaves the hash unchanged, while any field that affects the experiment's
    behaviour changes it. Keys are sorted, so the order of
    `classifier_params` entries doesn't matter.

    Args:
        cfg: The experiment configuration to hash.
        exclude: Further top-level fields to leave out, e.g. for a cache key
            that must not depend on them.

    Returns:
        The hex-encoded SHA-256 digest of the config's canonical JSON dump.
    """
    payload = json.dumps(
        cfg.model_dump(mode="json", exclude={"name", "paths", *exclude}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_device(device: str) -> Literal["cuda", "mps", "cpu"]:
    """Resolve a configured device string to a concrete, available device.

    Imports `torch` lazily so loading this module stays cheap.

    Args:
        device: `"auto"`, `"cuda"`, `"mps"` or `"cpu"`. `"auto"` picks CUDA,
            else MPS, else CPU.

    Returns:
        The resolved device kind.

    Raises:
        ValueError: If an explicit `"cuda"` or `"mps"` device isn't
            available on this host, or `device` isn't a known value.
    """
    import torch

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if device == "auto":
        return "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if device == "cuda":
        if not cuda_available:
            raise ValueError("device='cuda' requested but no CUDA device is available")
        return "cuda"
    if device == "mps":
        if not mps_available:
            raise ValueError("device='mps' requested but MPS is not available")
        return "mps"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"unknown device: {device!r}")


def run_id(cfg: ExperimentConfig) -> str:
    """Build a short, content-addressed identifier for an experiment run.

    Args:
        cfg: The experiment configuration to identify.

    Returns:
        `f"{cfg.name}-{config_hash(cfg)[:12]}"`.
    """
    return f"{cfg.name}-{config_hash(cfg)[:12]}"
