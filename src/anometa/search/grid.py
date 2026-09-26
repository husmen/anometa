"""A fixed grid of Track B configs, and running it end to end.

`GridSpec` describes the grid to sweep: embedding feature sets, PCA
dimensions, few-shot classifiers and shot budgets, plus each one-class
classifier's own PCA dimensions. `grid_configs` expands it into one
`TrackBConfig` per cell (see its docstring for the exact expansion);
`run_grid` runs every config through `run_experiment` and collects one row
per run, continuing past a config that raises.
"""

import traceback
from collections.abc import Sequence
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from anometa.config import ClassifierName, EncoderName, Paths, Scenario, TrackBConfig, run_id
from anometa.experiment import run_experiment
from anometa.search.space import parse_features, restrict_scenarios

_RESULT_COLUMNS: tuple[str, ...] = ("run_id", "status", "auroc", "ece_bal", "gap_auroc")
_DEFAULT_SEEDS: tuple[int, ...] = tuple(range(10))
"""Matches `TrackBConfig.seeds`'s own default; few-shot grid cells use it explicitly."""


class GridSpec(BaseModel):
    """A grid-search spec: the cross-product of dev Track B configs to run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    encoders: tuple[EncoderName, ...]
    feature_sets: tuple[str, ...]
    pca_dims: tuple[int, ...]
    classifiers: tuple[ClassifierName, ...]
    ks: tuple[int, ...]
    one_class_pca_dims: dict[Literal["mahalanobis", "tabpfn_outlier"], tuple[int, ...]]


def _cell(
    *,
    encoder: EncoderName,
    features: str,
    pca_dim: int | None,
    classifier: ClassifierName,
    k: int,
    seeds: tuple[int, ...],
    paths: Paths,
    scenarios: tuple[Scenario, ...] | None,
) -> TrackBConfig:
    """Build one grid cell's `TrackBConfig`.

    Args:
        encoder: Encoder for this cell.
        features: A `"+"`-joined feature-block spec (`parse_features`), or
            `"novelty"`.
        pca_dim: PCA dimension, or `None` when `features` is `"novelty"`.
        classifier: Classifier for this cell.
        k: Few-shot budget (`0` for a one-class classifier).
        seeds: Seeds for this cell (`(0,)` for a one-class classifier,
            `_DEFAULT_SEEDS` otherwise).
        paths: Filesystem layout for the run.
        scenarios: Scenarios to run on; `None` keeps `TrackBConfig`'s default
            (every scenario).

    Returns:
        The validated `TrackBConfig`, named `"grid"`, `split="dev"`.
    """
    cfg = TrackBConfig(
        name="grid",
        split="dev",
        encoder=encoder,
        features=parse_features(features),
        pca_dim=pca_dim,
        classifier=classifier,
        k=k,
        seeds=seeds,
        paths=paths,
    )
    return restrict_scenarios(cfg, scenarios)


def grid_configs(
    spec: GridSpec, paths: Paths = Paths(), scenarios: tuple[Scenario, ...] | None = None
) -> list[TrackBConfig]:
    """Expand a `GridSpec` into its cross-product of dev Track B configs.

    Per encoder: embedding feature sets x PCA dims x `spec.classifiers` x
    `spec.ks`, plus one `features="novelty"` cell per (classifier, k). Per
    one-class classifier in `spec.one_class_pca_dims`: encoders x embedding
    feature sets x that classifier's own PCA dims, at `k=0` and `seeds=(0,)`,
    plus one `features="novelty"` cell per encoder.

    Args:
        spec: The grid to expand.
        paths: Filesystem layout every config shares.
        scenarios: Scenarios every config runs on; `None` keeps
            `TrackBConfig`'s default (every scenario).

    Returns:
        One `TrackBConfig` per grid cell.
    """
    configs: list[TrackBConfig] = []
    for encoder in spec.encoders:
        for features in spec.feature_sets:
            for pca_dim in spec.pca_dims:
                for classifier in spec.classifiers:
                    for k in spec.ks:
                        configs.append(
                            _cell(
                                encoder=encoder,
                                features=features,
                                pca_dim=pca_dim,
                                classifier=classifier,
                                k=k,
                                seeds=_DEFAULT_SEEDS,
                                paths=paths,
                                scenarios=scenarios,
                            )
                        )
        for classifier in spec.classifiers:
            for k in spec.ks:
                configs.append(
                    _cell(
                        encoder=encoder,
                        features="novelty",
                        pca_dim=None,
                        classifier=classifier,
                        k=k,
                        seeds=_DEFAULT_SEEDS,
                        paths=paths,
                        scenarios=scenarios,
                    )
                )

    for classifier, pca_dims in spec.one_class_pca_dims.items():
        for encoder in spec.encoders:
            for features in spec.feature_sets:
                for pca_dim in pca_dims:
                    configs.append(
                        _cell(
                            encoder=encoder,
                            features=features,
                            pca_dim=pca_dim,
                            classifier=classifier,
                            k=0,
                            seeds=(0,),
                            paths=paths,
                            scenarios=scenarios,
                        )
                    )
            configs.append(
                _cell(
                    encoder=encoder,
                    features="novelty",
                    pca_dim=None,
                    classifier=classifier,
                    k=0,
                    seeds=(0,),
                    paths=paths,
                    scenarios=scenarios,
                )
            )
    return configs


def run_grid(configs: Sequence[TrackBConfig]) -> pd.DataFrame:
    """Run every config in `configs`, collecting one row per run.

    Prints one progress line per run (`[index/total] run_id status auroc`),
    flushed so a long grid can be followed live. Continues past a config
    whose `run_experiment` call raises: its traceback goes to stderr and it
    is recorded with status `"error"` and NaN metrics; `KeyboardInterrupt`
    still propagates. A dev config already complete on disk is read back by
    `run_experiment` itself rather than rerun, so rerunning after an
    interruption resumes.

    Args:
        configs: The configs to run, e.g. `grid_configs` output.

    Returns:
        One row per config, columns `run_id`, `status` (`"ok"`, `"failed"`
        or `"error"`), `auroc`, `ece_bal`, `gap_auroc` (NaN where a metric
        wasn't computed for that run).
    """
    rows: list[dict[str, str | float]] = []
    for index, cfg in enumerate(configs, start=1):
        try:
            result = run_experiment(cfg)
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
            row: dict[str, str | float] = {
                "run_id": run_id(cfg),
                "status": "error",
                "auroc": float("nan"),
                "ece_bal": float("nan"),
                "gap_auroc": float("nan"),
            }
        else:
            row = {
                "run_id": result.run_id,
                "status": result.status,
                "auroc": result.metrics.get("auroc", float("nan")),
                "ece_bal": result.metrics.get("ece_bal", float("nan")),
                "gap_auroc": result.metrics.get("gap_auroc", float("nan")),
            }
        rows.append(row)
        print(
            f"[{index}/{len(configs)}] {row['run_id']} {row['status']} auroc={row['auroc']}",
            flush=True,
        )
    return pd.DataFrame(rows, columns=list(_RESULT_COLUMNS))
