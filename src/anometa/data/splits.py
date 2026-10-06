"""Dev/lock split and nested few-shot sampling over an AD2 `test_public` index.

`make_split` partitions `test_public` scenes into a dev half and a lock half,
keeping every scene (all its lighting variants) on one side. `sample_few_shot`
draws a deterministic, nested few-shot defect set from the dev half, and
`eval_rows` builds the matching evaluation rows for either split, dropping
whole scenes that were used as shots so a defect never appears in both roles.
`write_split`/`load_split` persist a split to `splits/<scenario>.csv`, and
`split_hash` fingerprints a set of scenario splits for a run's manifest.
`lighting_fold` draws one fold of the post-freeze lighting-adaptation study
(docs: protocol, post-freeze studies), which rotates whole scenes between context and
evaluation.
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from anometa.config import Paths, Scenario

SPLIT_COLUMNS: list[str] = ["image_id", "scene_id", "label", "lighting", "split"]
"""Columns of the `DataFrame` returned by `make_split` and read by `load_split`."""


class BudgetError(ValueError):
    """Raised when a few-shot request exceeds the available shot pool."""


def make_split(index: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Split a scenario's `test_public` scenes into a dev half and a lock half.

    One `np.random.default_rng(seed)` draws a permutation of each label's
    sorted unique `scene_id`s in turn (label 0, then label 1); the first
    `ceil(n/2)` scenes of each permutation are assigned to `dev`, the rest to
    `lock`. Every image of a scene shares its scene's split, so no scene is
    split across dev and lock. Any pre-existing `split` column in `index` is
    ignored; only `source == "test_public"` rows are used.

    Args:
        index: A scenario index as returned by `index_scenario`.
        seed: Seed for the scene permutation.

    Returns:
        One row per `test_public` image, columns `SPLIT_COLUMNS`, sorted by
        `image_id`.
    """
    test_public = index[index["source"] == "test_public"]
    rng = np.random.default_rng(seed)
    scene_split: dict[str, str] = {}
    for label in (0, 1):
        scenes = sorted(test_public.loc[test_public["label"] == label, "scene_id"].unique())
        n_dev = math.ceil(len(scenes) / 2)
        for rank, position in enumerate(rng.permutation(len(scenes))):
            scene_split[scenes[position]] = "dev" if rank < n_dev else "lock"
    out = test_public[["image_id", "scene_id", "label", "lighting"]].copy()
    out["split"] = out["scene_id"].map(scene_split)
    return out.sort_values("image_id", ignore_index=True)[SPLIT_COLUMNS]


def write_split(df: pd.DataFrame, path: Path) -> None:
    """Write a split `DataFrame` to a CSV file, creating its parent directory.

    Args:
        df: A split as returned by `make_split`.
        path: Destination CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_split(scenario: Scenario, paths: Paths) -> pd.DataFrame:
    """Load a scenario's previously generated dev/lock split.

    Args:
        scenario: The scenario whose split to load.
        paths: Run paths; the split is read from `paths.splits / f"{scenario}.csv"`.

    Returns:
        The split `DataFrame`, columns `SPLIT_COLUMNS`.

    Raises:
        FileNotFoundError: If the split was never generated for `scenario`.
    """
    path = paths.splits / f"{scenario}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `anometa split` first")
    return pd.read_csv(path)


def split_hash(scenarios: Sequence[Scenario], paths: Paths) -> str:
    r"""Fingerprint the on-disk splits of a set of scenarios.

    Args:
        scenarios: Scenarios to include; hashed in canonical `Scenario` order
            regardless of the order they're given in.
        paths: Run paths; each scenario's split is read from
            `paths.splits / f"{scenario}.csv"`.

    Returns:
        The hex-encoded SHA-256 digest of `"<scenario>\\n"` plus each
        scenario's split CSV bytes, in `Scenario` order. A scenario whose
        split hasn't been generated yet contributes `b"missing"`.
    """
    order = list(Scenario)
    hasher = hashlib.sha256()
    for scenario in sorted(scenarios, key=order.index):
        hasher.update(f"{scenario}\n".encode())
        path = paths.splits / f"{scenario}.csv"
        hasher.update(path.read_bytes() if path.is_file() else b"missing")
    return hasher.hexdigest()


def sample_few_shot(
    split_df: pd.DataFrame,
    *,
    scenario: Scenario,
    k: int,
    seed: int,
    lighting: Literal["regular", "all"],
) -> list[str]:
    """Draw a deterministic, nested few-shot set of dev defect images.

    The pool is dev rows with `label == 1` (and `lighting == "regular"` when
    `lighting="regular"`), sorted by `image_id` and then permuted by a
    `(seed, scenario)`-keyed generator; the returned list is a prefix of that
    permutation, so a smaller `k` always returns a prefix of a larger `k`'s
    result for the same `seed` and `scenario`.

    Args:
        split_df: A scenario's split as returned by `make_split`.
        scenario: The scenario `split_df` belongs to, used to key the RNG so
            different scenarios draw independent permutations for the same
            seed.
        k: Number of shots to draw.
        seed: Seed for the shot permutation.
        lighting: `"regular"` restricts the pool to regular-lit images;
            `"all"` allows every lighting condition.

    Returns:
        The sampled shots' `image_id`s, in draw order.

    Raises:
        BudgetError: If `k` exceeds the pool size.
    """
    pool = split_df[(split_df["split"] == "dev") & (split_df["label"] == 1)]
    if lighting == "regular":
        pool = pool[pool["lighting"] == "regular"]
    pool = pool.sort_values("image_id")
    if k > len(pool):
        raise BudgetError(
            f"{scenario}: requested k={k} {lighting}-lit shots but only {len(pool)} available"
        )
    rng = np.random.default_rng([seed, list(Scenario).index(scenario)])
    order = pool.iloc[rng.permutation(len(pool))]
    return order["image_id"].tolist()[:k]


def eval_rows(
    split_df: pd.DataFrame, *, split: Literal["dev", "lock"], shots: Sequence[str]
) -> pd.DataFrame:
    """Build the evaluation rows for one split, excluding sampled shot scenes.

    Args:
        split_df: A scenario's split as returned by `make_split`.
        split: Which half to evaluate.
        shots: `image_id`s sampled as few-shot shots (e.g. by
            `sample_few_shot`); ignored when `split="lock"`.

    Returns:
        `split_df` rows for `split`. For `split="dev"`, every row whose
        `scene_id` matches a shot's scene is dropped (all its lighting
        variants), so a sampled defect never also appears in the evaluation
        set. For `split="lock"`, all lock rows are returned unchanged.
    """
    subset = split_df[split_df["split"] == split]
    if split == "dev":
        sampled_scenes = set(split_df.set_index("image_id").loc[list(shots), "scene_id"])
        subset = subset[~subset["scene_id"].isin(sampled_scenes)]
    return subset


@dataclass(frozen=True)
class LightingFold:
    """One fold of the lighting-adaptation study for one scenario.

    Attributes:
        adapt_scenes: Good `scene_id`s whose images may join the context, in
            draw order; the first m serve a run with m adaptation scenes.
        shots: Regular-lit `image_id`s of the defect shot scenes, one per
            scene, in draw order.
        eval_rows: Split rows of every other scene in the pool, under every
            lighting.
    """

    adapt_scenes: list[str]
    shots: list[str]
    eval_rows: pd.DataFrame


def lighting_fold(
    split_df: pd.DataFrame,
    *,
    scenario: Scenario,
    seed: int,
    k: int,
    adapt_max: int,
    pool: Literal["all", "dev"] = "all",
) -> LightingFold:
    """Draw one scene-level fold of the lighting-adaptation study.

    One generator keyed `(seed, scenario index, 11)` permutes the pool's
    sorted good scenes and then its sorted defect scenes. The first
    `adapt_max` good scenes form the adaptation pool, the first `k` defect
    scenes give the shots (their regular-lit image), and every other scene
    is evaluated, so no scene is both in a context and evaluated, under any
    lighting.

    Args:
        split_df: A scenario's split as returned by `make_split`.
        scenario: The scenario, used to key the generator.
        seed: The fold seed.
        k: Number of defect shot scenes.
        adapt_max: Size of the adaptation pool; runs with fewer adaptation
            scenes use a prefix of it, so their evaluation rows match.
        pool: `"all"` uses every `test_public` scene (dev and lock halves),
            `"dev"` the dev half only.

    Returns:
        The fold.

    Raises:
        BudgetError: If the pool leaves no good or no defect scene to
            evaluate.
    """
    rows = split_df if pool == "all" else split_df[split_df["split"] == "dev"]
    rng = np.random.default_rng([seed, list(Scenario).index(scenario), 11])
    good = sorted(rows.loc[rows["label"] == 0, "scene_id"].unique())
    bad = sorted(rows.loc[rows["label"] == 1, "scene_id"].unique())
    if adapt_max >= len(good):
        raise BudgetError(
            f"{scenario}: adapt_max={adapt_max} leaves none of {len(good)} good scenes to evaluate"
        )
    if k >= len(bad):
        raise BudgetError(f"{scenario}: k={k} leaves none of {len(bad)} defect scenes to evaluate")
    adapt = [good[i] for i in rng.permutation(len(good))[:adapt_max]]
    shot_scenes = [bad[i] for i in rng.permutation(len(bad))[:k]]
    regular = rows[rows["lighting"] == "regular"].set_index("scene_id")["image_id"]
    held = set(adapt) | set(shot_scenes)
    return LightingFold(
        adapt_scenes=adapt,
        shots=[str(regular[s]) for s in shot_scenes],
        eval_rows=rows[~rows["scene_id"].isin(held)].reset_index(drop=True),
    )
