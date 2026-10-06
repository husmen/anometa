"""Tests for `anometa.dashboard.app`: the Streamlit few-shot dashboard's pure functions."""

from pathlib import Path

import pytest

from anometa.config import Paths, Scenario
from anometa.data.splits import eval_rows, load_split
from anometa.features.extract import load_features

pytest.importorskip("streamlit")

from anometa.dashboard.app import fit_and_score, thumbnail_data_url


def test_thumbnail_data_url(ad2_root: Path) -> None:
    """Thumbnails are inline PNG data URLs."""
    assert thumbnail_data_url(ad2_root / "vial/train/good/000_regular.png").startswith(
        "data:image/png;base64,"
    )


def test_fit_and_score_excludes_picked_scenes(prepared: Paths) -> None:
    """Scoring covers dev rows except every image of a picked scene."""
    split = load_split(Scenario.VIAL, prepared)
    picked = list(
        split.query("split == 'dev' and label == 1 and lighting == 'regular'").image_id[:2]
    )
    rows, ms = fit_and_score(
        load_features("dinov3_s", Scenario.VIAL, prepared),
        split,
        picked,
        features=("cls", "mean_patch"),
        pca_dim=4,
        classifier="logreg",
        device="cpu",
    )
    assert set(rows.image_id) == set(eval_rows(split, split="dev", shots=picked).image_id)
    assert ms > 0
