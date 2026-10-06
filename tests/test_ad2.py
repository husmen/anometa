"""Tests for `anometa.data.ad2`: scenario indexing, lighting counts and RGB loads."""

from pathlib import Path

import numpy as np
import pytest

from anometa.config import Paths, Scenario
from anometa.data.ad2 import INDEX_COLUMNS, index_scenario, lighting_counts, load_rgb, parse_stem


def test_parse_stem() -> None:
    """Index and lighting token split at the first underscore; bad stems raise."""
    assert parse_stem("007_shift_2") == ("007", "shift_2")
    assert parse_stem("000_regular") == ("000", "regular")
    with pytest.raises(ValueError, match="stem does not match"):
        parse_stem("regular_000")


def test_index_scenario_rows(ad2_root: Path) -> None:
    """The fake tree indexes into train, validation and scene-tagged test rows."""
    idx = index_scenario(ad2_root, Scenario.VIAL)
    assert list(idx.columns) == INDEX_COLUMNS
    train = idx[idx.source == "train"]
    assert len(train) == 6
    assert train.scene_id.isna().all()
    assert (idx.source == "validation").sum() == 2
    bad = idx[(idx.source == "test_public") & (idx.label == 1)]
    assert len(bad) == 18
    assert bad.mask_path.map(lambda p: Path(p).exists()).all()
    assert set(bad.scene_id) == {f"bad/{i:03d}" for i in range(6)}
    assert set(bad.lighting) == {"regular", "overexposed", "shift_1"}
    one = idx[idx.image_id == "test_public/bad/000_regular"]
    assert one.path.tolist() == [str(ad2_root / "vial/test_public/bad/000_regular.png")]


def test_index_scenario_missing_scenario_raises(tmp_path: Path) -> None:
    """A never-extracted scenario raises FileNotFoundError naming the missing folder."""
    with pytest.raises(FileNotFoundError, match="can/train/good"):
        index_scenario(tmp_path, Scenario.CAN)


def test_load_rgb_converts_grey(ad2_root: Path) -> None:
    """Grey PNGs load as three-channel uint8 arrays."""
    img = load_rgb(ad2_root / "vial/train/good/000_regular.png")
    assert img.shape == (16, 24, 3)
    assert img.dtype == np.uint8


@pytest.mark.data
def test_vial_matches_published_counts() -> None:
    """Real Vial: 291 train, 41 validation, 35 good and 105 bad test images, 7 lightings."""
    idx = index_scenario(Paths().data, Scenario.VIAL)
    counts = {"train": 291, "validation": 41, "test_public": 140}
    assert idx.source.value_counts().to_dict() == counts
    assert lighting_counts(idx).shape == (7, 2)
