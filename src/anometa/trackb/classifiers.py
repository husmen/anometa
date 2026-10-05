"""Track B scorers: scikit-learn/TabPFN classifiers and one-class controls.

`make_scorer` builds a `Scorer` for one `config.ClassifierName`: `logreg`
and `knn` fit a scikit-learn pipeline on labelled shots; `tabpfn` and
`tabpfn_fast` fit TabPFN-3.5 and 3.5-Fast the same way; `tabpfn_thinking`
fits TabPFN-3.5-Thinking via the Prior Labs API, caching its prediction to
disk so a rerun of the same (config, scenario, seed) never spends credits;
`mahalanobis` and `tabpfn_outlier` are one-class controls (`config.ONE_CLASS`)
that fit on normal rows only. Every scorer's `anomaly_score` is 1-D:
P(anomalous) in `[0, 1]` when `probabilistic`, otherwise unbounded with
higher meaning more anomalous. `tabpfn`, `tabpfn-extensions` and
`tabpfn_client` are imported inside their factory functions, so a run that
only uses the scikit-learn classifiers never pays for loading them.
"""

import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from anometa.config import ClassifierName

if TYPE_CHECKING:
    from tabpfn import TabPFNClassifier
    from tabpfn.base import ClassifierModelSpecs, RegressorModelSpecs
    from tabpfn.model_loading import ModelVersion
    from tabpfn_extensions import TabPFNUnsupervisedModel


class Scorer(Protocol):
    """A Track B anomaly scorer: fit on shots (or normals), then scored.

    Attributes:
        probabilistic: Whether `anomaly_score` returns P(anomalous) in
            `[0, 1]`. When `False`, the score is unbounded and only its
            order matters: higher means more anomalous.
    """

    probabilistic: bool

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> Self:
        """Fit the scorer.

        Args:
            X: `(n, d)` feature rows.
            y: `(n,)` 0/1 labels. One-class scorers fit on `X[y == 0]` only.

        Returns:
            `self`, so `fit` and `anomaly_score` chain.
        """
        ...

    def anomaly_score(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Score rows for anomalousness.

        Args:
            X: `(n, d)` feature rows.

        Returns:
            `(n,)` scores; see `probabilistic`.
        """
        ...


class _FittableClassifier(Protocol):
    """A binary classifier exposing the scikit-learn fit/predict_proba API."""

    @property
    def classes_(self) -> NDArray[np.int64]:
        """Classes seen at fit time, in the order `predict_proba` columns use."""
        ...

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> object:
        """Fit the classifier on `(X, y)`."""
        ...

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Return per-class probabilities, columns ordered as `classes_`."""
        ...


@dataclass
class _ClassifierScorer:
    """`Scorer` wrapping a binary classifier built fresh at fit time.

    Shared by `logreg`, `knn`, `tabpfn` and `tabpfn_fast`: `_build` receives
    the number of fit rows (`knn` uses it to cap `n_neighbors`) and returns
    an unfitted classifier. The anomaly score is P(y == 1), located in
    `predict_proba`'s output via `classes_` rather than assumed to be
    column 1.
    """

    _build: Callable[[int], _FittableClassifier]
    probabilistic: bool = True
    _clf: _FittableClassifier | None = field(default=None, init=False, repr=False)
    _positive_col: int = field(default=1, init=False, repr=False)

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> Self:
        """Build and fit a fresh classifier on `(X, y)`."""
        clf = self._build(len(X))
        clf.fit(X, y)
        self._clf = clf
        self._positive_col = list(clf.classes_).index(1)
        return self

    def anomaly_score(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Return P(y == 1) for each row of `X`."""
        if self._clf is None:
            raise RuntimeError("fit must be called before anomaly_score")
        return np.asarray(self._clf.predict_proba(X))[:, self._positive_col]


def _build_logreg(params: Mapping[str, int | float | str], seed: int, device: str) -> Scorer:
    """Build the `logreg` scorer: standardised `LogisticRegression`.

    Args:
        params: May hold `C` (default `1.0`).
        seed: `random_state` for `LogisticRegression`.
        device: Unused; scikit-learn classifiers run on CPU only.

    Returns:
        The scorer, unfitted.
    """
    c = float(params.get("C", 1.0))

    def build(_n_fit_rows: int) -> Pipeline:
        """Build an unfitted standardised logistic regression.

        Args:
            _n_fit_rows: Unused; logistic regression needs no size cap.

        Returns:
            A `StandardScaler` + `LogisticRegression` pipeline.
        """
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=seed)
        )

    return _ClassifierScorer(build)


def _build_knn(params: Mapping[str, int | float | str], seed: int, device: str) -> Scorer:
    """Build the `knn` scorer: standardised `KNeighborsClassifier`.

    Args:
        params: May hold `n_neighbors` (default `5`), capped to the number
            of fit rows instead of raising.
        seed: Unused; `KNeighborsClassifier` has no randomness.
        device: Unused; scikit-learn classifiers run on CPU only.

    Returns:
        The scorer, unfitted.
    """
    requested = int(params.get("n_neighbors", 5))

    def build(n_fit_rows: int) -> Pipeline:
        """Build an unfitted standardised k-NN classifier.

        Args:
            n_fit_rows: Number of fit rows; caps `n_neighbors`.

        Returns:
            A `StandardScaler` + `KNeighborsClassifier` pipeline.
        """
        return make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=min(requested, n_fit_rows))
        )

    return _ClassifierScorer(build)


def _n_estimators(params: Mapping[str, int | float | str]) -> int | Literal["auto"]:
    """Read TabPFN's `n_estimators` override, defaulting to `"auto"`.

    Args:
        params: Classifier params, optionally holding `n_estimators`.

    Returns:
        `"auto"`, or `params["n_estimators"]` narrowed to `int`.
    """
    value = params.get("n_estimators", "auto")
    return "auto" if value == "auto" else int(value)


@cache
def _model_specs(
    version: ModelVersion, which: Literal["classifier", "regressor"], device: str
) -> ClassifierModelSpecs | RegressorModelSpecs:
    """Load one TabPFN checkpoint once per process and device, as reusable model specs.

    Every `fit` otherwise rebuilds the network, randomly initialises it and
    loads the checkpoint over it, which is most of a small fit (0.76 s of
    0.88 s on the RTX 3090). Passed as `model_path`, the specs make TabPFN
    reuse the loaded network (`tabpfn.base.initialize_tabpfn_model` returns
    them as-is). The scores are unchanged
    (`test_model_specs_match_fresh_checkpoint_load`).

    Args:
        version: `ModelVersion.V3_5` or `ModelVersion.V3_5_FAST`.
        which: `"classifier"` or `"regressor"` (`tabpfn_outlier`).
        device: Device the network will run on; specs are cached per device,
            because fitting moves the shared network there.

    Returns:
        The specs `create_default_for_version(version)` would load, built once.
    """
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.base import ClassifierModelSpecs, RegressorModelSpecs, initialize_tabpfn_model

    estimator = (
        TabPFNClassifier.create_default_for_version(version, device=device)
        if which == "classifier"
        else TabPFNRegressor.create_default_for_version(version, device=device)
    )
    if not isinstance(estimator.model_path, str | Path):
        raise TypeError(
            f"{version} resolved to {estimator.model_path!r}, expected a checkpoint path"
        )
    # Every scorer keeps TabPFN's default fit mode, which caches no train-set representation.
    models, configs, bar_distribution, inference_config = initialize_tabpfn_model(
        model_path=estimator.model_path, which=which, fit_mode="fit_preprocessors"
    )
    if bar_distribution is None:
        return ClassifierModelSpecs(models[0], configs[0], inference_config)
    return RegressorModelSpecs(models[0], configs[0], inference_config, bar_distribution)


def _build_tabpfn_version(
    version: ModelVersion, params: Mapping[str, int | float | str], seed: int, device: str
) -> Scorer:
    """Build a TabPFN classifier scorer for one `ModelVersion`.

    Args:
        version: `ModelVersion.V3_5` or `ModelVersion.V3_5_FAST`.
        params: May hold `n_estimators` (default `"auto"`).
        seed: `random_state` for `TabPFNClassifier`.
        device: Device to run inference on.

    Returns:
        The scorer, unfitted.
    """
    from tabpfn import TabPFNClassifier

    n_estimators = _n_estimators(params)

    def build(_n_fit_rows: int) -> TabPFNClassifier:
        """Build an unfitted `TabPFNClassifier` pinned to `version`'s checkpoint.

        Args:
            _n_fit_rows: Unused; TabPFN needs no size cap.

        Returns:
            The classifier, from `create_default_for_version`.
        """
        return TabPFNClassifier.create_default_for_version(
            version,
            model_path=_model_specs(version, "classifier", device),
            device=device,
            random_state=seed,
            n_estimators=n_estimators,
        )

    return _ClassifierScorer(build)


def _build_tabpfn(params: Mapping[str, int | float | str], seed: int, device: str) -> Scorer:
    """Build the `tabpfn` scorer (TabPFN-3.5)."""
    from tabpfn.model_loading import ModelVersion

    return _build_tabpfn_version(ModelVersion.V3_5, params, seed, device)


def _build_tabpfn_fast(params: Mapping[str, int | float | str], seed: int, device: str) -> Scorer:
    """Build the `tabpfn_fast` scorer (TabPFN-3.5-Fast)."""
    from tabpfn.model_loading import ModelVersion

    return _build_tabpfn_version(ModelVersion.V3_5_FAST, params, seed, device)


@dataclass
class _MahalanobisScorer:
    """One-class `Scorer`: `StandardScaler` + `LedoitWolf` on normal rows.

    `anomaly_score` is the squared Mahalanobis distance to the fitted
    normal distribution: unbounded, higher means more anomalous.
    """

    probabilistic: bool = False
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _cov: LedoitWolf | None = field(default=None, init=False, repr=False)

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> Self:
        """Fit `StandardScaler` and `LedoitWolf` on `X[y == 0]`."""
        normals = X[y == 0]
        scaler = StandardScaler().fit(normals)
        self._scaler = scaler
        self._cov = LedoitWolf().fit(scaler.transform(normals))
        return self

    def anomaly_score(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Return the squared Mahalanobis distance of each row of `X`."""
        if self._scaler is None or self._cov is None:
            raise RuntimeError("fit must be called before anomaly_score")
        return np.asarray(self._cov.mahalanobis(self._scaler.transform(X)))


def _build_mahalanobis(params: Mapping[str, int | float | str], seed: int, device: str) -> Scorer:
    """Build the `mahalanobis` one-class scorer.

    Args:
        params: Unused; `mahalanobis` takes no hyperparameters.
        seed: Unused; `StandardScaler`/`LedoitWolf` have no randomness.
        device: Unused; runs on CPU only.

    Returns:
        The scorer, unfitted.
    """
    return _MahalanobisScorer()


@dataclass
class _TabPFNOutlierScorer:
    """One-class `Scorer`: `TabPFNUnsupervisedModel.outliers` on normal rows.

    `anomaly_score` negates `outliers`' output (there, lower means more
    anomalous), so higher here means more anomalous, matching every other
    scorer.
    """

    _model: TabPFNUnsupervisedModel
    _seed: int
    _n_permutations: int
    probabilistic: bool = False
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> Self:
        """Fit the unsupervised model on `X[y == 0]`."""
        self._model.fit(X[y == 0])
        self._fitted = True
        return self

    def anomaly_score(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Return the negated TabPFN outlier score for each row of `X`."""
        import random

        import torch

        if not self._fitted:
            raise RuntimeError("fit must be called before anomaly_score")
        random.seed(self._seed)
        torch.manual_seed(self._seed)
        scores = self._model.outliers(X, n_permutations=self._n_permutations)
        return np.asarray(-scores.detach().cpu().numpy())


def _build_tabpfn_outlier(
    params: Mapping[str, int | float | str], seed: int, device: str
) -> Scorer:
    """Build the `tabpfn_outlier` one-class scorer.

    Args:
        params: May hold `n_permutations` (default `10`), passed to
            `.outliers`.
        seed: Seeds `random` and `torch` before each `.outliers` call
            (required for reproducible permutations), and `TabPFNRegressor`'s
            `random_state`.
        device: Device to run inference on.

    Returns:
        The scorer, unfitted. Its `TabPFNRegressor` loads the pinned
        TabPFN-3.5 checkpoint (`tabpfn_revision("v3.5")`), not the
        environment-overridable `model_path="auto"`.
    """
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import ModelVersion
    from tabpfn_extensions import TabPFNUnsupervisedModel

    n_permutations = int(params.get("n_permutations", 10))
    reg = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5,
        model_path=_model_specs(ModelVersion.V3_5, "regressor", device),
        device=device,
        random_state=seed,
    )
    model = TabPFNUnsupervisedModel(tabpfn_clf=None, tabpfn_reg=reg)
    return _TabPFNOutlierScorer(model, seed, n_permutations)


_sleep: Callable[[float], None] = time.sleep
"""Waits between rate-limited Thinking calls; a module attribute so tests can replace it."""
_RATE_LIMIT_RETRIES = 5
_RETRY_IN = re.compile(r"Retry in (\d+)s")


def _with_rate_limit_retry[T](call: Callable[[], T]) -> T:
    """Run one Prior Labs API call, retrying after an HTTP 429 rate-limit reply.

    The API answers too many Thinking calls per minute with HTTP 429 and
    names the wait ("Retry in 33s"); this waits that long plus one second
    and retries, up to `_RATE_LIMIT_RETRIES` times. Any other error
    propagates at once, including the daily token limit (also HTTP 429, but
    with no wait named, since retrying cannot succeed before the reset).

    Args:
        call: The API call to run.

    Returns:
        The call's result.

    Raises:
        RuntimeError: The last rate-limit error after all retries, or any
            non-rate-limit error from the client.
    """
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            return call()
        except RuntimeError as exc:
            message = str(exc)
            wait = _RETRY_IN.search(message)
            if "HTTP 429" not in message or wait is None or attempt == _RATE_LIMIT_RETRIES:
                raise
            _sleep(int(wait.group(1)) + 1)
    raise AssertionError("unreachable")


@dataclass
class ThinkingScorer:
    """`Scorer` wrapping TabPFN-3.5-Thinking via the Prior Labs API, cached to disk.

    `fit` only stores the fit rows; the API call is deferred to
    `anomaly_score` and made at most once: when `cache_file` already holds a
    prediction, it is returned unchanged and no API call happens, so a
    rerun of the same (config, scenario, seed) never spends credits.
    `tabpfn_client` is resolved as a module attribute at call time (`import
    tabpfn_client` then `tabpfn_client.TabPFNClassifier`), so tests can
    monkeypatch it.
    """

    seed: int
    cache_file: Path
    effort: str = "medium"
    metric: str = "roc_auc"
    timeout_s: int = 600
    probabilistic: bool = field(default=True, init=False)
    _x: NDArray[np.float32] | None = field(default=None, init=False, repr=False)
    _y: NDArray[np.int64] | None = field(default=None, init=False, repr=False)

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> Self:
        """Store the fit rows; the API call happens lazily in `anomaly_score`."""
        self._x = X
        self._y = y
        return self

    def anomaly_score(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Return P(y == 1) for `X`, from `cache_file` or one Prior Labs API call.

        Args:
            X: `(n, d)` feature rows to score.

        Returns:
            `(n,)` P(anomalous), cached at `cache_file` after a live call
            (written to a temporary file first, then moved into place, so an
            interrupted write never leaves a truncated cache).

        Raises:
            ValueError: If a cached prediction's length doesn't match `len(X)`.
            RuntimeError: If `fit` was never called and there is no cache to read.
        """
        if self.cache_file.exists():
            with np.load(self.cache_file) as cached:
                p = np.asarray(cached["p"], dtype=np.float64)
            if len(p) != len(X):
                raise ValueError(
                    f"cached prediction at {self.cache_file} has {len(p)} rows, expected {len(X)}"
                )
            return p
        if self._x is None or self._y is None:
            raise RuntimeError("fit must be called before anomaly_score")

        import tabpfn_client

        clf = tabpfn_client.TabPFNClassifier.create_default_for_version(
            "v3.5",
            thinking_effort=self.effort,
            thinking_metric=self.metric,
            thinking_timeout_s=self.timeout_s,
            random_state=self.seed,
        )
        x_fit, y_fit = self._x, self._y
        _with_rate_limit_retry(lambda: clf.fit(x_fit, y_fit))
        proba = np.asarray(_with_rate_limit_retry(lambda: clf.predict_proba(X)))
        p = proba[:, list(clf.classes_).index(1)].astype(np.float64)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_file.with_name(self.cache_file.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez(f, p=p)
        os.replace(tmp, self.cache_file)
        return p


def thinking_cost(X_fit: NDArray[np.float32], X_eval: NDArray[np.float32]) -> int:
    """Estimate the Prior Labs token cost of one TabPFN-3.5-Thinking fit and predict.

    The fit estimate uses `thinking_effort="medium"`, matching `ThinkingScorer`'s
    default; the server accepts no effort for the predict estimate and no test
    rows for the fit estimate. Every operation costs at least 10,000 tokens against a 5M
    token/day default budget: call this before a real Thinking run and check
    the estimate against the remaining budget.

    Args:
        X_fit: `(n_fit, d)` feature rows a Thinking scorer would fit on.
        X_eval: `(n_eval, d)` feature rows it would then score.

    Returns:
        The summed `estimated_cost` of one `"thinking_fit"` and one
        `"thinking_predict"` operation.
    """
    import tabpfn_client

    fit_cost = tabpfn_client.estimate_cost(
        X_fit, None, operation="thinking_fit", thinking_effort="medium"
    ).estimated_cost
    predict_cost = tabpfn_client.estimate_cost(
        X_fit, X_eval, operation="thinking_predict"
    ).estimated_cost
    return fit_cost + predict_cost


def _build_tabpfn_thinking(seed: int, cache_key: str, cache_dir: Path) -> Scorer:
    """Build the `tabpfn_thinking` scorer: cached TabPFN-3.5-Thinking via the API.

    Args:
        seed: `random_state` for `ThinkingScorer`'s TabPFN classifier.
        cache_key: Cache-file stem, e.g. `f"{key}/{scenario}-{seed}"` (see `run_track_b`).
        cache_dir: Directory `cache_key`'s `.npz` cache file lives under.

    Returns:
        The scorer, unfitted.
    """
    return ThinkingScorer(seed, cache_dir / f"{cache_key}.npz")


_BUILDERS: dict[str, Callable[[Mapping[str, int | float | str], int, str], Scorer]] = {
    "logreg": _build_logreg,
    "knn": _build_knn,
    "tabpfn": _build_tabpfn,
    "tabpfn_fast": _build_tabpfn_fast,
    "mahalanobis": _build_mahalanobis,
    "tabpfn_outlier": _build_tabpfn_outlier,
}
"""Scorer builder per `ClassifierName` that only needs `params`/`seed`/`device`.

`tabpfn_thinking` isn't here: it also needs `cache_key`/`cache_dir`, so
`make_scorer` builds it directly instead. A name with no entry here and not
`tabpfn_thinking` makes `make_scorer` raise `KeyError`.
"""


def make_scorer(
    name: ClassifierName,
    params: Mapping[str, int | float | str],
    *,
    seed: int,
    device: str,
    cache_key: str | None = None,
    cache_dir: Path | None = None,
) -> Scorer:
    """Build an unfitted `Scorer` for one Track B classifier.

    Args:
        name: Classifier to build.
        params: Classifier-specific hyperparameters; see each `_build_*`.
        seed: Random seed for the classifier's own randomness.
        device: `"cpu"`, `"cuda"` or `"mps"` (see `config.resolve_device`).
        cache_key: Required for `"tabpfn_thinking"`: its cache file's stem,
            e.g. `f"{key}/{scenario}-{seed}"` (see `run_track_b`).
        cache_dir: Required for `"tabpfn_thinking"`: the directory its
            `.npz` cache file lives under.

    Returns:
        The scorer, unfitted.

    Raises:
        KeyError: If `name` has no scorer builder.
        ValueError: If `name` is `"tabpfn_thinking"` and `cache_key` or
            `cache_dir` is `None`.
    """
    if name == "tabpfn_thinking":
        if cache_key is None or cache_dir is None:
            raise ValueError("tabpfn_thinking requires cache_key and cache_dir")
        return _build_tabpfn_thinking(seed, cache_key, cache_dir)
    return _BUILDERS[name](params, seed, device)


@cache
def tabpfn_revision(version: Literal["v3.5", "v3.5-fast"]) -> dict[str, str]:
    """Describe the pinned TabPFN-3.5 checkpoint this host has cached.

    Reads the checkpoint file `TabPFNClassifier.create_default_for_version`
    (and, for `"v3.5"`, `TabPFNRegressor.create_default_for_version`)
    resolves for `version`, so the returned hash matches whatever weights a
    `tabpfn`/`tabpfn_fast`/`tabpfn_outlier` scorer actually fits with.
    Cached: hashing the checkpoint is the expensive part. The file exists
    only after a scorer has loaded it once, so call this after fitting.

    Args:
        version: `"v3.5"` or `"v3.5-fast"`.

    Returns:
        `tabpfn_version` (installed `tabpfn` package version), `checkpoint`
        (its filename) and `sha256` (hex digest of its file contents).
    """
    import hashlib

    import tabpfn
    from tabpfn.model_loading import ModelSource, get_cache_dir

    source = ModelSource.get_v3_5() if version == "v3.5" else ModelSource.get_v3_5_fast()
    checkpoint = get_cache_dir() / source.default_filename
    with checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {
        "tabpfn_version": tabpfn.__version__,
        "checkpoint": source.default_filename,
        "sha256": digest,
    }
