"""Shared pytest fixtures: a fake AD2 tree and the `Paths` pointing at it."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from anometa.config import Paths

_LIGHTINGS: tuple[str, ...] = ("regular", "overexposed", "shift_1")
_SIZE: tuple[int, int] = (16, 24)  # (height, width)
_MASK_ROWS = slice(6, 10)
_MASK_COLS = slice(10, 14)


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
