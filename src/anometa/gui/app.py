"""Streamlit few-shot labelling demo: pick anomalies by eye, refit, rescore.

The gallery tab shows an unlabelled dev-image gallery for one cached
(scenario, encoder, backend); picking rows there refits a Track B classifier
on train normals plus the picks and shows a ranked table of the rest of dev,
with the live regular-vs-shifted AUROC gap. The run tab launches one
`configs/experiments/*.yaml` config, with `key=value` overrides, through
`run_experiment`. The results tab renders `reports/dev/figures/*.png`.

`thumbnail_data_url`, `gallery_frame` and `fit_and_score` are pure and unit
tested (`tests/test_app.py`); `main` is the Streamlit page itself, exercised
by hand and by a headless health check.
"""

import base64
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image
from pydantic import TypeAdapter

from anometa.cli import apply_overrides
from anometa.config import (
    ClassifierName,
    EncoderName,
    ExperimentConfig,
    FeatureBlock,
    Paths,
    Scenario,
    resolve_device,
)
from anometa.data.ad2 import index_scenario
from anometa.data.splits import load_split
from anometa.experiment import run_experiment
from anometa.features.extract import Features, cache_path, load_features
from anometa.metrics.image import auroc
from anometa.trackb.classifiers import make_scorer
from anometa.trackb.pipeline import embedding, fit_and_score_shots, fit_pca

_ENCODER_NAMES: tuple[EncoderName, ...] = ("dinov3_s", "dinov3_l", "siglip2")
_BACKENDS: tuple[Literal["transformers", "timm"], ...] = ("transformers", "timm")
_CLASSIFIER_OPTIONS: tuple[ClassifierName, ClassifierName] = ("tabpfn_fast", "tabpfn")
_LIGHTING_OPTIONS: tuple[Literal["regular", "shifted", "all"], ...] = (
    "regular",
    "shifted",
    "all",
)
_FEATURES: tuple[FeatureBlock, ...] = ("cls", "mean_patch")
_PCA_DIM = 16
"""Feature blocks and PCA width the demo fits with; not exposed as controls."""


def thumbnail_data_url(path: Path, size: int = 160) -> str:
    """Render an image as an inline base64 PNG data URL.

    `st.column_config.ImageColumn` doesn't accept local file paths, so the
    gallery embeds each thumbnail directly in the dataframe instead.

    Args:
        path: Image file to thumbnail.
        size: Maximum thumbnail side length in pixels.

    Returns:
        A `"data:image/png;base64,..."` URL for a PNG thumbnail of `path`,
        at most `size` px on its longest side.
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    rgb.thumbnail((size, size))
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def gallery_frame(
    split_df: pd.DataFrame, index: pd.DataFrame, lighting: Literal["regular", "shifted", "all"]
) -> pd.DataFrame:
    """Build the unlabelled dev-image gallery for one lighting filter.

    Args:
        split_df: The scenario's dev/lock split, as returned by `load_split`.
        index: The scenario's full image index, as returned by
            `index_scenario`, used to look up each dev image's file path.
        lighting: `"regular"` or `"shifted"` (anything but regular)
            restricts the gallery to that lighting condition; `"all"` shows
            every dev image.

    Returns:
        One row per matching dev image with `thumb` (an inline PNG data
        URL), `image_id` and `lighting`. No `label` column: the demo asks
        the user to pick anomalies by eye before any label is shown.
    """
    dev = split_df[split_df["split"] == "dev"]
    if lighting == "regular":
        dev = dev[dev["lighting"] == "regular"]
    elif lighting == "shifted":
        dev = dev[dev["lighting"] != "regular"]
    image_path = index.set_index("image_id")["path"]
    thumbs = [thumbnail_data_url(Path(image_path[image_id])) for image_id in dev["image_id"]]
    return pd.DataFrame(
        {
            "thumb": thumbs,
            "image_id": dev["image_id"].to_numpy(),
            "lighting": dev["lighting"].to_numpy(),
        }
    )


def fit_and_score(
    feats: Features,
    split_df: pd.DataFrame,
    picked: Sequence[str],
    *,
    features: tuple[FeatureBlock, ...],
    pca_dim: int | None,
    classifier: ClassifierName,
    device: str,
) -> tuple[pd.DataFrame, float]:
    """Fit a scorer on train normals plus the picked shots, then score the rest of dev.

    Delegates the fit/score/balance step to `trackb.pipeline.fit_and_score_shots`,
    the same helper `run_track_b`'s per-seed loop uses, so the two paths
    can't silently diverge.

    Args:
        feats: The scenario's cached features.
        split_df: The scenario's dev/lock split, as returned by `load_split`.
        picked: Image ids the user labelled anomalous (the `y=1` shots).
        features: Feature blocks to concatenate, e.g. `("cls", "mean_patch")`.
        pca_dim: PCA components to keep, or `None` when `features` is
            novelty-only.
        classifier: Classifier to fit.
        device: Device to run inference on (`resolve_device`'s result).

    Returns:
        `(rows, ms)`: one row per evaluation image, with `image_id`,
        `label`, `lighting`, `score` and `score_balanced` (NaN for a
        non-probabilistic classifier), and the fit-plus-predict wall time in
        milliseconds.
    """
    train_rows = np.flatnonzero(np.char.startswith(feats.image_id, "train/good/")).astype(np.int64)
    train_embedding = embedding(feats, train_rows, features)
    pca = None
    if train_embedding is not None:
        assert pca_dim is not None  # guaranteed by the caller, as in TrackBConfig validation
        pca = fit_pca(train_embedding, pca_dim)

    scorer = make_scorer(classifier, {}, seed=0, device=device)
    eval_df, score, score_balanced, fit_ms, predict_ms = fit_and_score_shots(
        feats, pca, train_rows, features, picked, split_df, "dev", scorer, device
    )
    rows = eval_df[["image_id", "label", "lighting"]].copy()
    rows["score"] = score
    rows["score_balanced"] = score_balanced
    return rows, fit_ms + predict_ms


def _cached_pairs(
    paths: Paths,
) -> list[tuple[Scenario, EncoderName, Literal["transformers", "timm"]]]:
    """List every (scenario, encoder, backend) with both a feature cache and a split.

    Args:
        paths: Run paths to scan.

    Returns:
        Every combination `cache_path(...)` resolves to an existing file
        for, restricted to scenarios that also have a dev/lock split on disk
        (both are needed to fit and score anything).
    """
    pairs: list[tuple[Scenario, EncoderName, Literal["transformers", "timm"]]] = []
    for scenario in Scenario:
        if not (paths.splits / f"{scenario}.csv").is_file():
            continue
        for encoder in _ENCODER_NAMES:
            for backend in _BACKENDS:
                if backend == "timm" and encoder == "siglip2":
                    continue
                if cache_path(encoder, scenario, paths, backend).is_file():
                    pairs.append((scenario, encoder, backend))
    return pairs


@st.cache_resource
def _cached_features(
    encoder: EncoderName, scenario: Scenario, backend: Literal["transformers", "timm"]
) -> Features:
    """Load a scenario's cached features once per Streamlit session.

    Args:
        encoder: Encoder whose cached features to load.
        scenario: Scenario to load.
        backend: Encoder backend that produced the cache.

    Returns:
        `load_features(encoder, scenario, Paths(), backend)`.
    """
    return load_features(encoder, scenario, Paths(), backend)


def main() -> None:
    """Run the Streamlit few-shot labelling demo page.

    Sidebar: pick a scenario, encoder and backend among cached feature
    pairs, a classifier (`tabpfn_fast` or `tabpfn`) and the gallery's
    lighting filter. Gallery tab: an unlabelled dev-image gallery; selecting
    rows refits the classifier on train normals plus the picks and shows a
    ranked table with balanced probabilities and the live regular/shifted
    AUROC gap. Run tab: runs one `configs/experiments/*.yaml` config, with
    `key=value` overrides, through `run_experiment`. Results tab: renders
    `reports/dev/figures/*.png`.
    """
    st.set_page_config(page_title="Anometa few-shot demo", layout="wide")
    st.title("Anometa few-shot demo")
    paths = Paths()

    pairs = _cached_pairs(paths)
    if not pairs:
        st.warning("No cached features found. Run `anometa extract` first.")
        return

    scenarios: list[Scenario] = sorted({p[0] for p in pairs}, key=list(Scenario).index)
    scenario = st.sidebar.selectbox("Scenario", scenarios)
    encoders: list[EncoderName] = sorted({p[1] for p in pairs if p[0] == scenario})
    encoder = st.sidebar.selectbox("Encoder", encoders)
    backends: list[Literal["transformers", "timm"]] = sorted(
        {p[2] for p in pairs if p[0] == scenario and p[1] == encoder}
    )
    backend = st.sidebar.selectbox("Backend", backends)
    classifier = st.sidebar.radio("Classifier", _CLASSIFIER_OPTIONS)
    lighting = st.sidebar.radio("Gallery lighting", _LIGHTING_OPTIONS)

    device = resolve_device("auto")
    feats = _cached_features(encoder, scenario, backend)
    split_df = load_split(scenario, paths)
    index = index_scenario(paths.data, scenario)

    tab_gallery, tab_run, tab_results = st.tabs(["Few-shot labelling", "Run", "Results"])

    with tab_gallery:
        frame = gallery_frame(split_df, index, lighting)
        event = st.dataframe(
            frame,
            column_config={"thumb": st.column_config.ImageColumn("Image")},
            on_select="rerun",
            selection_mode="multi-row",
            key="gallery",
        )
        picked: list[str] = frame.iloc[event.selection.rows]["image_id"].tolist()
        if not picked:
            st.info("Select one or more images above as few-shot anomaly examples.")
        else:
            rows, ms = fit_and_score(
                feats,
                split_df,
                picked,
                features=_FEATURES,
                pca_dim=_PCA_DIM,
                classifier=classifier,
                device=device,
            )
            st.caption(f"Refit + rescore: {ms:.0f} ms")
            regular = rows[rows["lighting"] == "regular"]
            shifted = rows[rows["lighting"] != "regular"]
            auroc_regular = auroc(
                regular["label"].to_numpy(dtype=np.int64),
                regular["score"].to_numpy(dtype=np.float64),
            )
            auroc_shifted = auroc(
                shifted["label"].to_numpy(dtype=np.int64),
                shifted["score"].to_numpy(dtype=np.float64),
            )
            cols = st.columns(3)
            cols[0].metric("Dev AUROC (regular)", f"{auroc_regular:.3f}")
            cols[1].metric("Dev AUROC (shifted)", f"{auroc_shifted:.3f}")
            cols[2].metric("Gap (regular - shifted)", f"{auroc_regular - auroc_shifted:.3f}")
            st.dataframe(rows.sort_values("score_balanced", ascending=False), hide_index=True)

    with tab_run:
        config_paths = sorted(Path("configs/experiments").glob("*.yaml"))
        if not config_paths:
            st.info("No experiment configs found under configs/experiments/.")
        else:
            config_path = st.selectbox("Experiment config", config_paths)
            overrides_text = st.text_area("Overrides (key=value, one per line)", "")
            if st.button("Run"):
                override_pairs = [line for line in overrides_text.splitlines() if line.strip()]
                data: dict[str, object] = yaml.safe_load(config_path.read_text())
                data = apply_overrides(data, override_pairs)
                cfg = TypeAdapter(ExperimentConfig).validate_python(data)
                if cfg.split == "lock":
                    st.error(f"{config_path}: split: lock configs run only through `anometa lock`")
                else:
                    result = run_experiment(cfg)
                    if result.status == "ok":
                        st.success(f"{result.run_id}: ok")
                    else:
                        error = result.error.strip().splitlines()[-1] if result.error else ""
                        st.error(f"{result.run_id}: failed: {error}")
                    st.json(result.metrics)

    with tab_results:
        figures = sorted(Path("reports/dev/figures").glob("*.png"))
        if not figures:
            st.info("No figures yet. Run `anometa report --split dev` first.")
        for figure in figures:
            st.image(str(figure), caption=figure.stem)


if __name__ == "__main__":
    main()
