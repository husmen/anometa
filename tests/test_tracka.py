"""Tests for `anometa.tracka`: anomalib model settings and the Track A runner."""

import pandas as pd
import pytest

import anometa.tracka.pipeline as pipeline
from anometa.config import Scenario, TrackAConfig
from anometa.data.splits import load_split
from anometa.tracka.anomalib_models import build_model
from anometa.tracka.pipeline import run_track_a


def test_run_track_a_patch_distance(prepared, tmp_path):
    """The patch distance map yields pixel metrics, a gap and max-of-map image scores."""
    cfg = TrackAConfig(
        model="patch_distance", encoder="dinov3_s", scenarios=(Scenario.VIAL,), paths=prepared
    )
    out = run_track_a(cfg, tmp_path)
    dev = load_split(Scenario.VIAL, prepared).query("split == 'dev'")
    assert set(out.predictions.image_id) == set(dev.image_id)
    assert {
        "au_pro_005",
        "au_pro_030",
        "seg_f1",
        "class_f1",
        "auroc",
        "gap_au_pro_005",
        "vial/au_pro_005",
    } <= set(out.metrics)
    assert "maps.npz" in out.extra_files


def test_run_track_a_resumes_saved_scenarios(prepared, tmp_path, monkeypatch):
    """A rerun loads each saved scenario instead of recomputing it, with identical output."""
    cfg = TrackAConfig(
        model="patch_distance", encoder="dinov3_s", scenarios=(Scenario.VIAL,), paths=prepared
    )
    first = run_track_a(cfg, tmp_path)
    assert (tmp_path / "scenarios" / "vial" / "result.json").is_file()

    def fail(*args, **kwargs):
        raise AssertionError("a saved scenario was recomputed")

    monkeypatch.setattr(pipeline, "_scenario_result", fail)
    second = run_track_a(cfg, tmp_path)
    pd.testing.assert_frame_equal(second.predictions, first.predictions)
    assert second.metrics == pytest.approx(first.metrics, nan_ok=True)
    assert second.extra_files.keys() == first.extra_files.keys()


def test_run_track_a_ignores_half_written_scenario(prepared, tmp_path):
    """A leftover temporary directory from an interrupted save is not taken as a result."""
    cfg = TrackAConfig(
        model="patch_distance", encoder="dinov3_s", scenarios=(Scenario.VIAL,), paths=prepared
    )
    stale = tmp_path / "scenarios" / ".vial.tmp"
    stale.mkdir(parents=True)
    (stale / "predictions.parquet").write_bytes(b"truncated")
    out = run_track_a(cfg, tmp_path)
    assert not stale.exists()
    assert (tmp_path / "scenarios" / "vial" / "maps.npz").is_file()
    assert len(out.predictions) > 0


@pytest.mark.models
def test_build_model_settings():
    """PatchCore uses a 0.01 coreset and no center crop; EfficientAD is the small model."""
    pc = build_model(TrackAConfig(model="patchcore"))
    assert pc.coreset_sampling_ratio == 0.01
    assert "CenterCrop" not in repr(pc.pre_processor)
    assert "small" in repr(build_model(TrackAConfig(model="efficientad_s")).model_size).lower()
