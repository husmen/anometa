"""Streamlit few-shot dashboard: real images and anomaly maps front and center.

The Detect tab shows normal training parts, an unlabelled grid of dev images to
pick defect examples from by eye and, once at least one is picked, the rest of
the dev images ranked by the classifier's defect probability. Each result card
shows the image with its ground-truth defect outlined next to an anomaly map:
the cached DINOv3 patch distance map, or a PatchCore or EfficientAD-S map from
a finished dev Track A run. One image can be opened large below the grid. The
Run tab launches one `configs/experiments/*.yaml` config through
`run_experiment`; the Report tab renders `reports/dev/figures/*.png`.

`thumbnail`, `load_mask`, `outline`, `heat_overlay`, `map_range`, `map_runs`
and `fit_and_score` are pure and unit tested (`tests/test_dashboard.py`);
`main` is the Streamlit page itself.
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from matplotlib import colormaps
from numpy.typing import NDArray
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
_LIGHTING_OPTIONS: tuple[Literal["regular", "shifted", "all"], ...] = ("regular", "shifted", "all")
# Feature blocks and PCA width the dashboard fits with; not exposed as controls.
# (Plain comments: Streamlit "magic" would render bare strings on the page.)
_FEATURES: tuple[FeatureBlock, ...] = ("cls", "mean_patch")
_PCA_DIM = 16
# Map source name for the cached distance maps of the selected encoder.
_DISTANCE = "DINOv3 patch distance"
_GRID_PX, _FOCUS_PX = 360, 720
_DEFECT_RGB = (255, 59, 48)


def thumbnail(path: Path, size: int) -> NDArray[np.uint8]:
    """Load an image as RGB, at most `size` px on its longest side.

    Args:
        path: Image file.
        size: Longest side of the result in pixels.

    Returns:
        An `(h, w, 3)` uint8 array.
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    rgb.thumbnail((size, size))
    return np.asarray(rgb, dtype=np.uint8)


def load_mask(mask_path: object, size_hw: tuple[int, int]) -> NDArray[np.bool_]:
    """Load a ground-truth mask resized to `size_hw`, or all False for a good image.

    Args:
        mask_path: The index's `mask_path`; `None`/NaN for good images.
        size_hw: Target `(height, width)`.

    Returns:
        A boolean array, True on the defect.
    """
    if not isinstance(mask_path, str):
        return np.zeros(size_hw, dtype=bool)
    with Image.open(mask_path) as image:
        small = image.convert("L").resize((size_hw[1], size_hw[0]), Image.Resampling.NEAREST)
    return np.asarray(small) > 0


def outline(
    rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    color: tuple[int, int, int] = _DEFECT_RGB,
    width: int = 2,
) -> NDArray[np.uint8]:
    """Draw the boundary of `mask` onto a copy of `rgb`.

    Args:
        rgb: `(h, w, 3)` image.
        mask: `(h, w)` boolean defect mask.
        color: Boundary colour.
        width: Boundary width in pixels.

    Returns:
        The image with the mask's edge painted in `color`.
    """
    inner = mask.copy()
    for _ in range(width):
        shrunk = inner.copy()
        shrunk[1:, :] &= inner[:-1, :]
        shrunk[:-1, :] &= inner[1:, :]
        shrunk[:, 1:] &= inner[:, :-1]
        shrunk[:, :-1] &= inner[:, 1:]
        inner = shrunk
    out = rgb.copy()
    out[mask & ~inner] = color
    return out


def heat_overlay(
    rgb: NDArray[np.uint8],
    heat: NDArray[np.floating],
    lo: float,
    hi: float,
    alpha: float = 0.6,
) -> NDArray[np.uint8]:
    """Blend an anomaly map over an image with the inferno colormap.

    The map is resized bilinearly to the image and clipped to `[lo, hi]`; low
    values stay mostly transparent, so the part stays visible under the map.

    Args:
        rgb: `(h, w, 3)` image.
        heat: Anomaly map of any shape (patch grid or model resolution).
        lo: Map value shown as "normal" (fully dark).
        hi: Map value shown as "most anomalous" (brightest).
        alpha: Opacity of the map at its brightest.

    Returns:
        The blended `(h, w, 3)` uint8 image.
    """
    h, w = rgb.shape[:2]
    up = np.asarray(
        Image.fromarray(np.asarray(heat, dtype=np.float32)).resize(
            (w, h), Image.Resampling.BILINEAR
        )
    )
    t = np.clip((up - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    color = colormaps["inferno"](t)[..., :3] * 255.0
    weight = (alpha * (0.3 + 0.7 * t))[..., None]
    return (rgb * (1.0 - weight) + color * weight).astype(np.uint8)


def map_range(maps: Iterable[NDArray[np.floating]]) -> tuple[float, float]:
    """Return one display range for a set of maps: their 1st and 99.5th percentiles.

    A shared range keeps the cards comparable: the same colour means the
    same score on every image of the scenario.

    Args:
        maps: Anomaly maps of one scenario and source.

    Returns:
        `(lo, hi)`.
    """
    values = np.concatenate([np.asarray(m, dtype=np.float32).ravel() for m in maps])
    lo, hi = np.percentile(values, [1.0, 99.5])
    return float(lo), float(hi)


def map_runs(artifacts: Path, scenario: Scenario) -> dict[str, Path]:
    """Find finished dev Track A runs whose saved maps cover `scenario`.

    Args:
        artifacts: The artifacts folder.
        scenario: Scenario the maps must cover.

    Returns:
        One entry per run, from a label such as `"patchcore 512x512 (run id)"` to
        the run's `maps.npz`, sorted by label.
    """
    runs: dict[str, Path] = {}
    for manifest in artifacts.glob("*/manifest.json"):
        meta = json.loads(manifest.read_text())
        config = meta.get("config", {})
        maps = manifest.parent / "maps.npz"
        if not (
            meta.get("status") == "ok"
            and config.get("track") == "A"
            and config.get("split") == "dev"
            and config.get("model") in ("patchcore", "efficientad_s")
            and scenario in config.get("scenarios", [])
            and maps.is_file()
        ):
            continue
        size = "x".join(str(v) for v in config.get("image_size", (256, 256)))
        runs[f"{config['model']} {size} ({manifest.parent.name})"] = maps
    return dict(sorted(runs.items()))


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
    """Load a scenario's cached features once per Streamlit session."""
    return load_features(encoder, scenario, Paths(), backend)


@st.cache_data(max_entries=4096)
def _thumb(path: str, size: int) -> NDArray[np.uint8]:
    """Cached `thumbnail`."""
    return thumbnail(Path(path), size)


@st.cache_data(max_entries=8)
def _run_maps(maps_path: str, scenario: str) -> dict[str, NDArray[np.float32]]:
    """Load one Track A run's maps for one scenario, keyed by image id."""
    prefix = f"{scenario}/"
    with np.load(maps_path, allow_pickle=False) as saved:
        return {
            k.removeprefix(prefix): saved[k].astype(np.float32)
            for k in saved.files
            if k.startswith(prefix)
        }


def _maps_for(
    source: str, feats: Features, runs: dict[str, Path], scenario: Scenario
) -> dict[str, NDArray[np.float32]]:
    """Return every available map of `source`, keyed by image id."""
    if source == _DISTANCE:
        return {
            str(i): d.astype(np.float32) for i, d in zip(feats.image_id, feats.distmap, strict=True)
        }
    return _run_maps(str(runs[source]), str(scenario))


def _panel(
    path: object,
    mask_path: object,
    size: int,
    heat: NDArray[np.float32] | None,
    rng: tuple[float, float],
) -> NDArray[np.uint8]:
    """Image with its defect outlined, side by side with its map overlay (if any)."""
    rgb = _thumb(str(path), size)
    shown = outline(rgb, load_mask(mask_path, rgb.shape[:2]))
    if heat is None:
        return shown
    gap = np.full((rgb.shape[0], 6, 3), 255, dtype=np.uint8)
    return np.hstack([shown, gap, heat_overlay(rgb, heat, *rng)])


def _lighting_filter(frame: pd.DataFrame, lighting: str) -> pd.DataFrame:
    """Keep rows of the chosen lighting: regular, shifted (non-regular) or all."""
    if lighting == "regular":
        return frame[frame["lighting"] == "regular"]
    if lighting == "shifted":
        return frame[frame["lighting"] != "regular"]
    return frame


def _detect_tab(
    scenario: Scenario,
    feats: Features,
    split_df: pd.DataFrame,
    index: pd.DataFrame,
    runs: dict[str, Path],
    classifier: ClassifierName,
    lighting: str,
    source: str,
    show_maps: bool,
) -> None:
    """Normal parts, the pick grid, ranked results and the single-image view."""
    maps = _maps_for(source, feats, runs, scenario)
    rng = map_range(maps[i] for i in split_df["image_id"] if i in maps)
    st.subheader("Normal parts")
    normals = index[(index["source"] == "train") & (index["lighting"] == "regular")].head(6)
    for col, (_, row) in zip(st.columns(6), normals.iterrows(), strict=False):
        heat = maps.get(str(row["image_id"])) if show_maps else None
        col.image(_panel(row["path"], None, _GRID_PX // 2, heat, rng), width="stretch")

    st.subheader("Pick defect examples")
    candidates = _lighting_filter(split_df[split_df["split"] == "dev"], lighting)
    per_page = 18
    pages = max(1, -(-len(candidates) // per_page))
    page = st.number_input("Page", 1, pages, 1, key="page") if pages > 1 else 1
    shown = candidates.iloc[(page - 1) * per_page : page * per_page]
    cols = st.columns(6)
    for n, image_id in enumerate(shown["image_id"]):
        with cols[n % 6]:
            st.image(_thumb(str(index.at[image_id, "path"]), _GRID_PX // 2), width="stretch")
            st.checkbox("defect", key=f"pick|{scenario}|{image_id}")
    prefix = f"pick|{scenario}|"
    picked = [
        k.removeprefix(prefix)
        for k, v in st.session_state.items()
        if isinstance(k, str) and k.startswith(prefix) and v
    ]
    if not picked:
        st.info("Tick one or more images that look defective. Labels stay hidden until then.")
        return
    rows, ms = fit_and_score(
        feats,
        split_df,
        picked,
        features=_FEATURES,
        pca_dim=_PCA_DIM,
        classifier=classifier,
        device=resolve_device("auto"),
    )
    regular = rows[rows["lighting"] == "regular"]
    shifted = rows[rows["lighting"] != "regular"]
    stats = st.columns(4)
    stats[0].metric("Defect examples", len(picked))
    for col, name, part in ((stats[1], "regular", regular), (stats[2], "shifted", shifted)):
        value = auroc(
            part["label"].to_numpy(dtype=np.int64), part["score"].to_numpy(dtype=np.float64)
        )
        col.metric(f"Dev AUROC ({name} light)", f"{value:.3f}")
    stats[3].metric("Fit + score", f"{ms:.0f} ms")

    st.subheader("Results")
    order = st.radio("Show", ["Most anomalous", "Least anomalous"], horizontal=True, key="order")
    key = "score_balanced" if rows["score_balanced"].notna().all() else "score"
    ranked = _lighting_filter(rows, lighting).sort_values(key, ascending=order == "Least anomalous")
    top = ranked.head(9)
    cols = st.columns(3)
    for n, (_, r) in enumerate(top.iterrows()):
        image_id = str(r["image_id"])
        heat = maps.get(image_id) if show_maps else None
        verdict = "defect" if r["label"] == 1 else "normal"
        cols[n % 3].image(
            _panel(
                index.at[image_id, "path"], index.at[image_id, "mask_path"], _GRID_PX, heat, rng
            ),
            caption=f"P(defect) {r[key]:.2f} · truly {verdict} · {r['lighting']}",
            width="stretch",
        )

    st.subheader("Inspect one image")
    focus = st.selectbox("Image", ranked["image_id"].tolist(), key="focus")
    rgb = _thumb(str(index.at[focus, "path"]), _FOCUS_PX)
    mask = load_mask(index.at[focus, "mask_path"], rgb.shape[:2])
    panels = [("image, defect outlined", outline(rgb, mask))]
    sources = [
        (_DISTANCE, _maps_for(_DISTANCE, feats, runs, scenario)),
        *((label, _run_maps(str(path), str(scenario))) for label, path in runs.items()),
    ]
    for name, source_maps in sources:
        if focus in source_maps:
            panels.append(
                (name, heat_overlay(rgb, source_maps[focus], *map_range(source_maps.values())))
            )
    for col, (name, image) in zip(st.columns(len(panels)), panels, strict=True):
        col.image(image, caption=name, width="stretch")


def main() -> None:
    """Run the Streamlit few-shot dashboard page."""
    st.set_page_config(page_title="Anometa dashboard", layout="wide")
    paths = Paths()
    pairs = _cached_pairs(paths)
    if not pairs:
        st.title("Anometa dashboard")
        st.warning("No cached features found. Run `anometa extract` first.")
        return

    side = st.sidebar
    side.header("Setup")
    scenarios: list[Scenario] = sorted({p[0] for p in pairs}, key=list(Scenario).index)
    scenario = side.selectbox(
        "Scenario",
        scenarios,
        index=scenarios.index(Scenario.VIAL) if Scenario.VIAL in scenarios else 0,
        format_func=lambda s: str(s).replace("_", " ").title(),
    )
    encoders: list[EncoderName] = sorted({p[1] for p in pairs if p[0] == scenario})
    encoder = side.selectbox(
        "Encoder", encoders, index=encoders.index("dinov3_l") if "dinov3_l" in encoders else 0
    )
    backends: list[Literal["transformers", "timm"]] = sorted(
        {p[2] for p in pairs if p[0] == scenario and p[1] == encoder}
    )
    backend = side.selectbox("Backend", backends)
    classifier = side.radio("Classifier", _CLASSIFIER_OPTIONS)
    lighting = side.radio("Lighting shown", _LIGHTING_OPTIONS)
    runs = map_runs(paths.artifacts, scenario)
    source = side.selectbox("Anomaly map", [_DISTANCE, *runs])
    show_maps = side.toggle("Show maps", value=True)

    feats = _cached_features(encoder, scenario, backend)
    split_df = load_split(scenario, paths)
    index = index_scenario(paths.data, scenario).set_index("image_id", drop=False)
    st.title(f"Anometa · {str(scenario).replace('_', ' ').title()}")
    st.caption(
        "Pick a few defect examples by eye. TabPFN-3.5 reads them with the normal training "
        "parts as its context and scores every other dev image in one forward pass."
    )
    tab_detect, tab_run, tab_report = st.tabs(["Detect", "Run", "Report"])

    with tab_detect:
        _detect_tab(scenario, feats, split_df, index, runs, classifier, lighting, source, show_maps)

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

    with tab_report:
        figures = sorted(Path("reports/dev/figures").glob("*.png"))
        if not figures:
            st.info("No figures yet. Run `anometa report --split dev` first.")
        for figure in figures:
            st.image(str(figure), caption=figure.stem)


if __name__ == "__main__":
    main()
