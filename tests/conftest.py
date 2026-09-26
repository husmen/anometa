"""Shared pytest fixtures: a fake AD2 tree, `Paths` and `FakeEncoder`."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from PIL import Image

from anometa.config import Paths
from anometa.features.encoders import Encoded

_LIGHTINGS: tuple[str, ...] = ("regular", "overexposed", "shift_1")
_SIZE: tuple[int, int] = (16, 24)  # (height, width)
_MASK_ROWS = slice(6, 10)
_MASK_COLS = slice(10, 14)
_FAKE_PATCH = 4
_FAKE_PROJECTION: NDArray[np.float32] = (
    np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
)
"""Fixed seeded 3->8 projection `FakeEncoder` applies to its block means."""


def _random_grey(rng: np.random.Generator) -> NDArray[np.uint8]:
    """Draw one seeded random grey-scale HxW array.

    Args:
        rng: Seeded random generator shared across a fixture's images.

    Returns:
        A `_SIZE`-shaped `uint8` array of pixel values.
    """
    return rng.integers(0, 256, size=_SIZE, dtype=np.uint8)


def _save_grey(path: Path, array: NDArray[np.uint8]) -> None:
    """Write a grey-scale array to `path` as a mode-`L` PNG.

    Args:
        path: Destination file; parent directories are created as needed.
        array: HxW `uint8` array to encode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


@pytest.fixture
def ad2_root(tmp_path: Path) -> Path:
    """Build a fake AD2 tree for scenario `vial` under `tmp_path`.

    Layout: `train/good` (6 images), `validation/good` (2 images),
    `test_public/good` (4 scenes) and `test_public/bad` (6 scenes), each
    scene shot under `_LIGHTINGS`, plus a 0/255 mask per bad image in
    `test_public/ground_truth/bad`. Every image is a 16x24 px grey PNG with
    seeded random pixels; bad images carry a bright 4x4 square where the
    matching mask is 255.

    Args:
        tmp_path: Pytest's per-test temporary directory.

    Returns:
        The AD2 data root, i.e. the parent of the `vial` scenario folder.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "data" / "ad2"
    scenario_dir = root / "vial"

    for i in range(6):
        _save_grey(scenario_dir / "train/good" / f"{i:03d}_regular.png", _random_grey(rng))
    for i in range(2):
        _save_grey(scenario_dir / "validation/good" / f"{i:03d}_regular.png", _random_grey(rng))
    for scene in range(4):
        for lighting in _LIGHTINGS:
            _save_grey(
                scenario_dir / "test_public/good" / f"{scene:03d}_{lighting}.png",
                _random_grey(rng),
            )
    for scene in range(6):
        for lighting in _LIGHTINGS:
            stem = f"{scene:03d}_{lighting}"
            image = _random_grey(rng)
            mask = np.zeros_like(image)
            image[_MASK_ROWS, _MASK_COLS] = 250
            mask[_MASK_ROWS, _MASK_COLS] = 255
            _save_grey(scenario_dir / "test_public/bad" / f"{stem}.png", image)
            _save_grey(scenario_dir / "test_public/ground_truth/bad" / f"{stem}_mask.png", mask)

    return root


@pytest.fixture
def paths(tmp_path: Path, ad2_root: Path) -> Paths:
    """Build `Paths` with `data` at `ad2_root` and the rest under `tmp_path`.

    `configs` points at an empty `tmp_path / "configs"`, so tests never read
    the repository's pinned archive checksums.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        ad2_root: The fake AD2 data root.

    Returns:
        A `Paths` scoped entirely to `tmp_path`.
    """
    return Paths(
        data=ad2_root,
        cache=tmp_path / "cache",
        artifacts=tmp_path / "artifacts",
        splits=tmp_path / "splits",
        configs=tmp_path / "configs",
    )


@dataclass
class FakeEncoder:
    """Deterministic, CPU-only stand-in for `Encoder` (`anometa.features.encoders`).

    `encode` projects 4x4 block means of the input image through a fixed
    seeded 3->8 matrix, so results are reproducible without loading any
    model weights. `calls` counts `encode` invocations.
    """

    name: str = "dinov3_s"
    backend: Literal["transformers", "timm"] = "transformers"
    dim: int = 8
    resolution_tag: str = "r512"
    revisions: dict[str, str] = field(default_factory=dict)
    calls: int = 0

    def encode(self, image: NDArray[np.uint8]) -> Encoded:
        """Encode one HxWx3 uint8 image into 4x4-patch fake features.

        Args:
            image: HxWx3 uint8 array; each side is cropped down to a multiple
                of the 4px patch size before pooling.

        Returns:
            `Encoded` with `patches` from block means projected to `dim` and
            `cls` as their mean.
        """
        self.calls += 1
        h, w, c = image.shape
        ph, pw = h // _FAKE_PATCH, w // _FAKE_PATCH
        cropped = image[: ph * _FAKE_PATCH, : pw * _FAKE_PATCH]
        blocks = cropped.reshape(ph, _FAKE_PATCH, pw, _FAKE_PATCH, c).astype(np.float32)
        means = blocks.mean(axis=(1, 3))
        patches = means @ _FAKE_PROJECTION
        cls = patches.mean(axis=(0, 1))
        return Encoded(cls=torch.from_numpy(cls), patches=torch.from_numpy(patches))
