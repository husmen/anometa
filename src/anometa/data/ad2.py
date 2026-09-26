"""AD2 scenario indexing: enumerate images into one tidy `pandas.DataFrame`.

`index_scenario` walks an extracted MVTec AD 2 scenario folder (`train`,
`validation`, `test_public`) and returns one row per image with its split,
label, lighting condition and, for `test_public`, the scene it belongs to and
its ground-truth mask path. `lighting_counts` summarises the `test_public`
rows for a quick sanity check, and `load_rgb` loads an image (single-channel
or not) as an HxWx3 `uint8` array.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from anometa.config import Scenario

INDEX_COLUMNS: list[str] = [
    "image_id",
    "path",
    "source",
    "label",
    "lighting",
    "scene_id",
    "mask_path",
]
"""Columns of the `DataFrame` returned by `index_scenario`, in order."""

_STEM_RE = re.compile(r"^(\d{3})_([a-z0-9_]+)$")

_SPLITS: tuple[tuple[str, str, int], ...] = (
    ("train/good", "train", 0),
    ("validation/good", "validation", 0),
    ("test_public/good", "test_public", 0),
    ("test_public/bad", "test_public", 1),
)
"""(subfolder, `source` value, `label` value) for every folder indexed."""


def parse_stem(stem: str) -> tuple[str, str]:
    r"""Split an AD2 filename stem into its scene index and lighting token.

    Args:
        stem: A file stem, e.g. `"007_shift_2"`.

    Returns:
        `(index, lighting)`, e.g. `("007", "shift_2")`.

    Raises:
        ValueError: If `stem` doesn't match `^\d{3}_[a-z0-9_]+$`.
    """
    match = _STEM_RE.match(stem)
    if match is None:
        raise ValueError(f"stem does not match ^\\d{{3}}_[a-z0-9_]+$: {stem!r}")
    return match.group(1), match.group(2)


def index_scenario(root: Path, scenario: Scenario) -> pd.DataFrame:
    """Index every image of one extracted AD2 scenario.

    Args:
        root: Directory holding the scenario folder (i.e. `root / scenario`
            is the extracted archive's top-level scenario directory).
        scenario: The scenario to index.

    Returns:
        One row per image in `train/good`, `validation/good` and
        `test_public/{good,bad}`, sorted by `image_id`, with columns
        `INDEX_COLUMNS`. `scene_id` is `"<good|bad>/<NNN>"` for
        `test_public` rows and `None` otherwise; `mask_path` is set only for
        `test_public/bad` rows.

    Raises:
        FileNotFoundError: If `root / scenario / "train/good"` doesn't
            exist, i.e. the scenario was never extracted under `root`.
    """
    scenario_dir = root / scenario
    train_dir = scenario_dir / "train/good"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"{train_dir} not found; run `anometa download` first")
    rows: list[dict[str, str | int | None]] = []
    for subdir, source, label in _SPLITS:
        for image_path in sorted((scenario_dir / subdir).glob("*.png")):
            index_id, lighting = parse_stem(image_path.stem)
            image_id = image_path.relative_to(scenario_dir).with_suffix("").as_posix()
            scene_id: str | None = None
            mask_path: str | None = None
            if source == "test_public":
                folder = "bad" if label else "good"
                scene_id = f"{folder}/{index_id}"
                if label:
                    mask_dir = scenario_dir / "test_public/ground_truth/bad"
                    mask_path = str(mask_dir / f"{image_path.stem}_mask.png")
            rows.append(
                {
                    "image_id": image_id,
                    "path": str(image_path),
                    "source": source,
                    "label": label,
                    "lighting": lighting,
                    "scene_id": scene_id,
                    "mask_path": mask_path,
                }
            )
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    return index.sort_values("image_id", ignore_index=True)


def lighting_counts(index: pd.DataFrame) -> pd.DataFrame:
    """Crosstab `test_public` rows by lighting condition and label.

    Args:
        index: A `DataFrame` as returned by `index_scenario`.

    Returns:
        A `lighting` x `label` count table restricted to `source ==
        "test_public"` rows.
    """
    test_public = index[index["source"] == "test_public"]
    return pd.crosstab(test_public["lighting"], test_public["label"])


def load_rgb(path: Path) -> NDArray[np.uint8]:
    """Load an image as an HxWx3 `uint8` array, converting grey to RGB.

    Args:
        path: Image file to load.

    Returns:
        The image as an HxWx3 `uint8` array.
    """
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)
