"""Tests for `anometa.trackb.reference`: fit/eval frames matching Track B's rows."""

from pathlib import Path

from anometa.config import Paths, Scenario
from anometa.data.ad2 import index_scenario
from anometa.data.splits import eval_rows, make_split, sample_few_shot
from anometa.trackb.reference import reference_frames


def test_reference_frames_match_track_b_rows(ad2_root: Path, paths: Paths) -> None:
    """The reference fits on train normals plus shots and evaluates Track B's dev rows."""
    idx = index_scenario(ad2_root, Scenario.VIAL)
    split = make_split(idx)
    shots = sample_few_shot(split, scenario=Scenario.VIAL, k=2, seed=0, lighting="regular")
    x_fit, y_fit, x_eval, _y_eval = reference_frames(idx, split, shots, "dev")
    assert list(x_fit.columns) == ["image"]
    assert y_fit.sum() == 2
    assert len(x_fit) == 6 + 2
    assert len(x_eval) == len(eval_rows(split, split="dev", shots=shots))
    assert Path(x_eval.image.iloc[0]).exists()
