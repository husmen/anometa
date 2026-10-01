"""Tests for `anometa.gui.rerun_log`: Rerun inspection recordings of runs."""

import pytest

from anometa.config import Scenario, TrackAConfig
from anometa.experiment import run_experiment

pytest.importorskip("rerun")

from anometa.gui.rerun_log import write_rrd


def test_write_rrd_for_track_a_run(prepared):
    """A patch-distance run produces a non-empty recording."""
    res = run_experiment(
        TrackAConfig(
            model="patch_distance", encoder="dinov3_s", scenarios=(Scenario.VIAL,), paths=prepared
        )
    )
    out = write_rrd(res.artifact_dir, prepared, max_images=5)
    assert out.name == "inspect.rrd"
    assert out.stat().st_size > 0
