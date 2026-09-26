"""Tests for `anometa.data.splits`: dev/lock split and few-shot sampling."""

import pandas as pd
import pytest

import anometa.cli as cli
from anometa.cli import main
from anometa.config import Paths, Scenario
from anometa.data.ad2 import index_scenario
from anometa.data.splits import (
    BudgetError,
    eval_rows,
    load_split,
    make_split,
    sample_few_shot,
    split_hash,
    write_split,
)


@pytest.fixture
def split_df(ad2_root):
    """Build the dev/lock split of the fake Vial `test_public` index at seed 0."""
    return make_split(index_scenario(ad2_root, Scenario.VIAL), seed=0)


def test_split_keeps_scenes_whole(split_df):
    """No scene has images in both dev and lock."""
    assert split_df.groupby("scene_id").split.nunique().max() == 1


def test_split_halves_scenes_by_label(split_df):
    """6 bad scenes split 3/3, 4 good scenes 2/2; an odd count gives dev the extra scene."""
    scenes = split_df.drop_duplicates("scene_id")
    assert scenes.groupby(["label", "split"]).size().to_dict() == {
        (0, "dev"): 2,
        (0, "lock"): 2,
        (1, "dev"): 3,
        (1, "lock"): 3,
    }
    odd = split_df[split_df.scene_id != "bad/005"]
    assert (
        (
            make_split(odd.assign(source="test_public"), 0)
            .drop_duplicates("scene_id")
            .query("label == 1")
            .split.value_counts()["dev"]
        )
        == 3
    )


def test_split_is_deterministic(split_df, ad2_root, paths):
    """Same seed, same split, same hash."""
    again = make_split(index_scenario(ad2_root, Scenario.VIAL), seed=0)
    pd.testing.assert_frame_equal(split_df, again)
    write_split(split_df, paths.splits / "vial.csv")
    assert split_hash([Scenario.VIAL], paths) == split_hash([Scenario.VIAL], paths)


def test_sample_few_shot_nested_and_regular(split_df):
    """k=1 is a prefix of k=3; shots are regular-lit dev defects."""
    one = sample_few_shot(split_df, scenario=Scenario.VIAL, k=1, seed=4, lighting="regular")
    three = sample_few_shot(split_df, scenario=Scenario.VIAL, k=3, seed=4, lighting="regular")
    rows = split_df.set_index("image_id").loc[three]
    assert three[:1] == one
    assert (rows.lighting == "regular").all()
    assert (rows.split == "dev").all()
    assert (rows.label == 1).all()


def test_budget_error_names_scenario_and_pool(split_df):
    """Asking for more regular-lit shots than dev holds raises a precise BudgetError."""
    with pytest.raises(BudgetError, match=r"vial.*k=4.*3"):
        sample_few_shot(split_df, scenario=Scenario.VIAL, k=4, seed=0, lighting="regular")


def test_eval_rows_drop_whole_sampled_scenes(split_df):
    """Dev evaluation drops every lighting variant of a sampled scene and keeps other dev rows."""
    shots = sample_few_shot(split_df, scenario=Scenario.VIAL, k=2, seed=0, lighting="regular")
    ev = eval_rows(split_df, split="dev", shots=shots)
    sampled = set(split_df.set_index("image_id").loc[shots, "scene_id"])
    assert not sampled & set(ev.scene_id)
    assert len(ev) == (split_df.split == "dev").sum() - 3 * len(sampled)
    assert (eval_rows(split_df, split="lock", shots=shots).split == "lock").all()


def test_load_split_missing_raises_naming_command(paths: Paths) -> None:
    """A split never generated raises FileNotFoundError naming `anometa split`."""
    with pytest.raises(FileNotFoundError, match="anometa split"):
        load_split(Scenario.VIAL, paths)


def test_load_split_reads_back_written_split(split_df: pd.DataFrame, paths: Paths) -> None:
    """`load_split` reads back exactly what `write_split` wrote."""
    write_split(split_df, paths.splits / "vial.csv")
    pd.testing.assert_frame_equal(load_split(Scenario.VIAL, paths), split_df)


def test_cli_split_writes_only_downloaded_scenarios_and_never_overwrites(
    paths: Paths, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`anometa split` writes vial's split once, skips undownloaded scenarios, prints counts."""
    monkeypatch.setattr(cli, "Paths", lambda: paths)

    assert main(["split"]) == 0
    written = (paths.splits / "vial.csv").read_bytes()
    assert not (paths.splits / "can.csv").exists()
    assert "vial" in capsys.readouterr().out

    assert main(["split"]) == 0
    assert (paths.splits / "vial.csv").read_bytes() == written
