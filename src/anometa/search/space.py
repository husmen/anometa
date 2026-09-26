"""The Track B unit-cube search space: `PARAMS`, its decoding and its config.

`PARAMS` is an ordered, fixed-dimension (`d = 7`) space over encoder,
feature set, PCA dimension and classifier choice, plus each few-shot
classifier's own hyperparameter. `decode_unit` turns a length-7 point in
`[0, 1]^7` into named values deterministically (used by the TabPFN-surrogate
Bayesian optimizer, which proposes points in the unit cube); `suggest` asks
an Optuna trial for the same values directly, so grid search, random search
and Optuna cover the identical flat space. `to_config` turns either kind of
point into a runnable dev-split `TrackBConfig`.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anometa.config import FeatureBlock, Paths, Scenario, TrackBConfig

if TYPE_CHECKING:
    import optuna


@dataclass(frozen=True)
class _Categorical:
    """One categorical `PARAMS` dimension: a fixed, ordered set of choices."""

    choices: tuple[str | int, ...]


@dataclass(frozen=True)
class _LogFloat:
    """One log-uniform float `PARAMS` dimension over `[low, high]`."""

    low: float
    high: float


@dataclass(frozen=True)
class _IntRange:
    """One integer `PARAMS` dimension over `[low, high]` inclusive."""

    low: int
    high: int


_Dimension = _Categorical | _LogFloat | _IntRange

PARAMS: dict[str, _Dimension] = {
    "encoder": _Categorical(("dinov3_s", "dinov3_l", "siglip2")),
    "features": _Categorical(("cls+mean_patch", "cls+mean_patch+novelty", "novelty")),
    "pca_dim": _Categorical((16, 32, 64, 128)),
    "classifier": _Categorical(("tabpfn", "tabpfn_fast", "logreg", "knn")),
    "n_estimators": _Categorical((4, 8, 16)),
    "C": _LogFloat(1e-3, 1e2),
    "n_neighbors": _IntRange(1, 15),
}
"""The unit-cube search space, `d = 7`, in cube-coordinate order."""

_FEATURE_TOKENS: dict[str, FeatureBlock] = {
    "cls": "cls",
    "mean_patch": "mean_patch",
    "novelty": "novelty",
}

_CLASSIFIER_PARAM_KEY: dict[str, str] = {
    "tabpfn": "n_estimators",
    "tabpfn_fast": "n_estimators",
    "logreg": "C",
    "knn": "n_neighbors",
}
"""Which `PARAMS` hyperparameter each few-shot classifier keeps in `classifier_params`."""


def parse_features(spec: str) -> tuple[FeatureBlock, ...]:
    """Parse a `"+"`-joined feature-block spec into `FeatureBlock`s.

    Args:
        spec: Feature blocks joined by `"+"`, e.g. `"cls+mean_patch"` or
            `"novelty"`.

    Returns:
        The parsed blocks, in the order written.

    Raises:
        KeyError: If a token isn't a known `FeatureBlock`.
    """
    return tuple(_FEATURE_TOKENS[token] for token in spec.split("+"))


def decode_unit(u: Sequence[float]) -> dict[str, Any]:
    """Decode a length-7 unit-cube point into `PARAMS`' named values.

    Each coordinate is first clamped to `[0, 1]`. Then a categorical
    dimension picks choice `min(floor(u * n), n - 1)`; a log-float dimension
    returns `low * (high / low) ** u`; an int dimension returns
    `low + min(floor(u * (high - low + 1)), high - low)`.

    Args:
        u: One coordinate per `PARAMS` entry, in `PARAMS` order; values
            outside `[0, 1]` are clamped.

    Returns:
        `PARAMS` names mapped to their decoded values.
    """
    values: dict[str, Any] = {}
    for (name, dim), raw in zip(PARAMS.items(), u, strict=True):
        ui = min(max(raw, 0.0), 1.0)
        if isinstance(dim, _Categorical):
            n = len(dim.choices)
            values[name] = dim.choices[min(math.floor(ui * n), n - 1)]
        elif isinstance(dim, _LogFloat):
            values[name] = dim.low * (dim.high / dim.low) ** ui
        else:
            span = dim.high - dim.low
            values[name] = dim.low + min(math.floor(ui * (span + 1)), span)
    return values


def suggest(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest one point of the unit-cube space from an Optuna trial.

    Every `PARAMS` dimension is suggested flat and independently
    (`suggest_categorical`, `suggest_float(log=True)` or `suggest_int`),
    identical to the unit cube `decode_unit` decodes.

    Args:
        trial: The trial to suggest each `PARAMS` dimension on.

    Returns:
        `PARAMS` names mapped to the trial's suggested value.
    """
    values: dict[str, Any] = {}
    for name, dim in PARAMS.items():
        if isinstance(dim, _Categorical):
            values[name] = trial.suggest_categorical(name, list(dim.choices))
        elif isinstance(dim, _LogFloat):
            values[name] = trial.suggest_float(name, dim.low, dim.high, log=True)
        else:
            values[name] = trial.suggest_int(name, dim.low, dim.high)
    return values


def restrict_scenarios(cfg: TrackBConfig, scenarios: tuple[Scenario, ...] | None) -> TrackBConfig:
    """Return `cfg` restricted to `scenarios`, revalidated.

    Args:
        cfg: The config to restrict.
        scenarios: Scenarios to run on, or `None` to keep `cfg`'s own
            (by default every scenario).

    Returns:
        `cfg` itself for `None`, else a validated copy with `scenarios` set.
    """
    if scenarios is None:
        return cfg
    return TrackBConfig.model_validate(cfg.model_dump() | {"scenarios": scenarios})


def scenario_suffix(scenarios: tuple[Scenario, ...] | None) -> str:
    """Return the study or file-name suffix for a scenario-restricted search.

    A restricted search gets its own study and trial frame, so it never
    resumes into, or mixes with, an all-scenario one.

    Args:
        scenarios: Scenarios the search runs on, or `None` for every scenario.

    Returns:
        `""` for `None`, else `"-"` and the scenario names joined by `"+"`.
    """
    return "" if scenarios is None else "-" + "+".join(scenarios)


def to_config(
    params: Mapping[str, Any],
    *,
    k: int,
    name: str = "search",
    paths: Paths = Paths(),
    scenarios: tuple[Scenario, ...] | None = None,
) -> TrackBConfig:
    """Build a dev-split `TrackBConfig` from one `decode_unit`/`suggest` point.

    Args:
        params: A `PARAMS`-keyed mapping, as `decode_unit` or `suggest` returns.
        k: Few-shot budget; not itself a search dimension.
        name: The config's `name`.
        paths: Filesystem layout for the run.
        scenarios: Scenarios to run on; `None` keeps `TrackBConfig`'s default
            (every scenario).

    Returns:
        A `TrackBConfig` with `split="dev"`, `pca_dim=None` when `features`
        is `"novelty"`, and `classifier_params` holding only the
        hyperparameter `params["classifier"]` uses.
    """
    features = parse_features(params["features"])
    classifier = params["classifier"]
    param_key = _CLASSIFIER_PARAM_KEY[classifier]
    cfg = TrackBConfig(
        name=name,
        split="dev",
        encoder=params["encoder"],
        features=features,
        pca_dim=None if features == ("novelty",) else params["pca_dim"],
        classifier=classifier,
        classifier_params={param_key: params[param_key]},
        k=k,
        paths=paths,
    )
    return restrict_scenarios(cfg, scenarios)
