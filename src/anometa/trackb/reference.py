"""TabPFNWithImages reference check: sanity-check Track B's extraction pipeline.

`reference_frames` builds the fit/eval rows Track B itself scores (train
normals plus few-shot shots, dev evaluation rows) as one-column file-path
frames. `reference_check` fits `tabpfn_extensions.image.TabPFNWithImages`
(which encodes images with its own DINOv3 ViT-S/16, at the processor's
default resolution) directly on those paths and compares its AUROC, seed by
seed, to Track B's own cached-feature pipeline. A material shortfall points
at an extraction bug in our own pipeline rather than a modelling difference.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from anometa.config import Paths, Scenario, TrackBConfig, resolve_device
from anometa.data.ad2 import index_scenario
from anometa.data.splits import eval_rows, load_split, sample_few_shot
from anometa.metrics.image import auroc
from anometa.trackb.pipeline import run_track_b


def reference_frames(
    index: pd.DataFrame,
    split_df: pd.DataFrame,
    shots: Sequence[str],
    split: Literal["dev"],
) -> tuple[pd.DataFrame, NDArray[np.int64], pd.DataFrame, NDArray[np.int64]]:
    """Build `TabPFNWithImages`-ready fit/eval frames matching Track B's rows.

    Args:
        index: A scenario index as returned by `index_scenario`; supplies
            every fit/eval image's file path through its `path` column.
        split_df: The scenario's dev/lock split, as returned by `make_split`.
        shots: Few-shot `image_id`s, as returned by `sample_few_shot`.
        split: Which half to evaluate; only `"dev"` is supported.

    Returns:
        `(X_fit, y_fit, X_eval, y_eval)`. `X_fit`/`X_eval` are one-column
        `DataFrame`s named `"image"` holding file paths. `X_fit` covers train
        normals (`y_fit=0`) plus `shots` (`y_fit=1`), in that order; `X_eval`
        covers `eval_rows(split_df, split=split, shots=shots)`.
    """
    path_by_id = index.set_index("image_id")["path"]
    train_ids = index.loc[index["source"] == "train", "image_id"].tolist()
    fit_ids = [*train_ids, *shots]
    y_fit: NDArray[np.int64] = np.array([0] * len(train_ids) + [1] * len(shots), dtype=np.int64)
    x_fit = pd.DataFrame({"image": path_by_id.loc[fit_ids].to_numpy()})

    eval_df = eval_rows(split_df, split=split, shots=shots)
    x_eval = pd.DataFrame({"image": path_by_id.loc[eval_df["image_id"]].to_numpy()})
    y_eval: NDArray[np.int64] = eval_df["label"].to_numpy(dtype=np.int64)
    return x_fit, y_fit, x_eval, y_eval


def reference_check(
    scenario: Scenario,
    paths: Paths,
    *,
    k: int = 5,
    seeds: Sequence[int] = range(10),
    device: str = "auto",
) -> pd.DataFrame:
    """Compare `TabPFNWithImages` to Track B's own pipeline on one scenario.

    Per seed, fits a fresh `TabPFNWithImages` (wrapping
    `TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, ...)`)
    directly on `reference_frames`' file-path rows, and reads off the
    matching seed's AUROC from one `run_track_b` call at `features=("cls",)`,
    `pca_dim=30`, `classifier="tabpfn"` (Track B's cached dinov3_s features).
    Both use the same few-shot shots per seed, so they score the same rows.

    Args:
        scenario: Scenario to check; its dinov3_s features must already be
            cached (`anometa extract --encoder dinov3_s`) and its dev/lock
            split generated (`anometa split`).
        paths: Run paths.
        k: Few-shot budget per seed.
        seeds: Seeds to check.
        device: Device for both TabPFN models; see `config.resolve_device`.

    Returns:
        One row per seed, columns `seed`, `reference_auroc`, `ours_auroc`;
        also written to `artifacts/reference_check_<scenario>.csv`.
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.model_loading import ModelVersion
    from tabpfn_extensions.image import TabPFNWithImages

    resolved_device = resolve_device(device)
    index = index_scenario(paths.data, scenario)
    split_df = load_split(scenario, paths)
    seeds = tuple(seeds)

    track_b = run_track_b(
        TrackBConfig(
            scenarios=(scenario,),
            seeds=seeds,
            encoder="dinov3_s",
            features=("cls",),
            pca_dim=30,
            classifier="tabpfn",
            k=k,
            split="dev",
            device=resolved_device,
            paths=paths,
        ),
        paths.artifacts / f"reference_check_{scenario}",
    )

    rows: list[dict[str, float | int]] = []
    for seed in seeds:
        shots = sample_few_shot(split_df, scenario=scenario, k=k, seed=seed, lighting="regular")
        x_fit, y_fit, x_eval, y_eval = reference_frames(index, split_df, shots, "dev")

        estimator = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, device=resolved_device, random_state=seed
        )
        reference = TabPFNWithImages(estimator, image_features_indices=[0], n_components=30)
        reference.fit(x_fit, y_fit)
        positive_col = list(reference.classes_).index(1)
        reference_score = np.asarray(reference.predict_proba(x_eval))[:, positive_col]

        seed_pred = track_b.predictions[track_b.predictions["seed"] == seed]
        rows.append(
            {
                "seed": seed,
                "reference_auroc": auroc(y_eval, reference_score),
                "ours_auroc": auroc(
                    seed_pred["label"].to_numpy(), seed_pred["score"].to_numpy(dtype=np.float64)
                ),
            }
        )

    result = pd.DataFrame(rows)
    out_path = paths.artifacts / f"reference_check_{scenario}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    return result
