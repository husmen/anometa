"""Tests for `anometa.features.extract`: feature extraction and cache."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from conftest import FakeEncoder
from numpy.typing import NDArray

import anometa.cli as cli
import anometa.features.extract as extract_module
from anometa.artifacts import git_info
from anometa.cli import main
from anometa.config import Paths, Scenario
from anometa.features.encoders import Encoded
from anometa.features.extract import (
    PROVENANCE_KEYS,
    cache_path,
    extract_scenario,
    load_features,
)


@dataclass
class OverflowEncoder(FakeEncoder):
    """`FakeEncoder` whose CLS overflows to infinity, as fp16 can on CUDA."""

    @override
    def encode(self, image: NDArray[np.uint8]) -> Encoded:
        """Encode like `FakeEncoder`, then replace `cls` with infinities.

        Args:
            image: HxWx3 uint8 array.

        Returns:
            `Encoded` with an all-infinite `cls`.
        """
        enc = super().encode(image)
        return Encoded(cls=torch.full_like(enc.cls, float("inf")), patches=enc.patches)


@dataclass
class Bf16Encoder(FakeEncoder):
    """`FakeEncoder` that returns bf16 tensors, as encoders do on CUDA."""

    @override
    def encode(self, image: NDArray[np.uint8]) -> Encoded:
        """Encode like `FakeEncoder`, then cast to bf16.

        Args:
            image: HxWx3 uint8 array.

        Returns:
            `Encoded` with bf16 `cls` and `patches`.
        """
        enc = super().encode(image)
        return Encoded(cls=enc.cls.bfloat16(), patches=enc.patches.bfloat16())


def test_extract_scenario_writes_cache(ad2_root: Path, paths: Paths) -> None:
    """Every indexed image gets features; train novelty is positive because it is leave-one-out."""
    out = extract_scenario(FakeEncoder(), Scenario.VIAL, paths)
    f = load_features("dinov3_s", Scenario.VIAL, paths)
    assert out == cache_path("dinov3_s", Scenario.VIAL, paths)
    assert len(f.image_id) == 38
    assert f.cls.shape == (38, 8)
    assert f.novelty.shape == (38, 2)
    assert f.distmap.shape == (38, 4, 6)
    assert f.distmap.dtype == np.float16
    train = np.char.startswith(f.image_id.astype(str), "train/")
    assert (f.novelty[train, 0] > 0).all()


def test_extract_scenario_accepts_bf16(ad2_root: Path, paths: Paths) -> None:
    """bf16 encodings are stored as float32 features, and provenance records bfloat16."""
    extract_scenario(Bf16Encoder(), Scenario.VIAL, paths)
    f = load_features("dinov3_s", Scenario.VIAL, paths)
    assert f.cls.dtype == np.float32
    assert np.isfinite(f.mean_patch).all()
    assert f.provenance["dtype"] == "bfloat16"


def test_extract_skips_existing(ad2_root: Path, paths: Paths) -> None:
    """A second extraction reuses the cache and encodes nothing."""
    enc = FakeEncoder()
    extract_scenario(enc, Scenario.VIAL, paths)
    n = enc.calls
    extract_scenario(enc, Scenario.VIAL, paths)
    assert enc.calls == n


def test_features_rows(ad2_root: Path, paths: Paths) -> None:
    """Row lookup by image id; unknown ids raise KeyError."""
    extract_scenario(FakeEncoder(), Scenario.VIAL, paths)
    f = load_features("dinov3_s", Scenario.VIAL, paths)
    assert f.image_id[f.rows(["train/good/000_regular"])[0]] == "train/good/000_regular"
    with pytest.raises(KeyError):
        f.rows(["nope"])


def test_extract_scenario_crash_leaves_no_cache(
    ad2_root: Path, paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-write leaves no cache file, so a rerun re-extracts instead of trusting it."""
    cache = cache_path("dinov3_s", Scenario.VIAL, paths)

    def _boom(*_args: object, **_kwargs: object) -> None:
        """Stand-in for `np.savez_compressed` that crashes mid-write."""
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(extract_module.np, "savez_compressed", _boom)
    with pytest.raises(OSError, match="simulated crash"):
        extract_scenario(FakeEncoder(), Scenario.VIAL, paths)

    assert not cache.exists()
    assert list(cache.parent.iterdir()) == []

    monkeypatch.undo()
    extract_scenario(FakeEncoder(), Scenario.VIAL, paths)
    assert cache.is_file()


def test_load_features_missing_raises(paths: Paths) -> None:
    """A cache that was never extracted raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="anometa extract"):
        load_features("dinov3_s", Scenario.VIAL, paths)


def test_load_features_other_backend_hint(ad2_root: Path, paths: Paths) -> None:
    """A cache written under the other backend names it as the fix."""
    extract_scenario(FakeEncoder(backend="timm"), Scenario.VIAL, paths)
    with pytest.raises(FileNotFoundError, match="encoder_backend='timm'"):
        load_features("dinov3_s", Scenario.VIAL, paths, backend="transformers")


def test_cli_extract_writes_cache_and_prints_backend(
    ad2_root: Path,
    paths: Paths,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`anometa extract` runs the given encoder over the given scenario and reports its backend.

    The device is pinned to CPU so the timing text never carries a CUDA VRAM suffix.
    """
    monkeypatch.setattr(cli, "Paths", lambda: paths)
    monkeypatch.setattr(cli, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(cli, "load_encoder", lambda name, device, backend: FakeEncoder())

    assert main(["extract", "--encoder", "dinov3_s", "--scenario", "vial"]) == 0

    out = capsys.readouterr().out
    assert "backend=transformers" in out
    assert re.search(r"\(\d+\.\d s\)", out)
    assert cache_path("dinov3_s", Scenario.VIAL, paths).is_file()


def test_features_provenance_round_trips(ad2_root: Path, paths: Paths) -> None:
    """Provenance scalars written at extraction load back as the same strings."""
    extract_scenario(FakeEncoder(revisions={"org/repo": "abc123"}), Scenario.VIAL, paths)
    assert load_features("dinov3_s", Scenario.VIAL, paths).provenance == {
        "encoder": "dinov3_s",
        "backend": "transformers",
        "revision": "org/repo@abc123",
        "device": "cpu",
        "dtype": "float32",
        "git_commit": git_info()[0] or "unknown",
    }


def test_cache_without_provenance_loads_unknown(ad2_root: Path, paths: Paths) -> None:
    """A cache written before provenance existed loads with every value "unknown"."""
    path = extract_scenario(FakeEncoder(), Scenario.VIAL, paths)
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files if not k.startswith("provenance_")}
    np.savez_compressed(path, **arrays)
    assert load_features("dinov3_s", Scenario.VIAL, paths).provenance == dict.fromkeys(
        PROVENANCE_KEYS, "unknown"
    )


def test_extract_rejects_non_finite_features(ad2_root: Path, paths: Paths) -> None:
    """A non-finite encoding raises ValueError naming the image and writes no cache."""
    with pytest.raises(ValueError, match="train/good/000_regular"):
        extract_scenario(OverflowEncoder(), Scenario.VIAL, paths)
    assert not cache_path("dinov3_s", Scenario.VIAL, paths).exists()
