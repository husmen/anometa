"""TabPFN-surrogate Bayesian optimization over the Track B unit-cube space.

`run_tabpfn_bo` warm-starts from the same shared `initial_units` points every
search method starts from; afterwards each point comes from
`propose_next_point`, which fits a `TabPFNRegressor` surrogate on every
`(u, auroc)` pair observed so far and proposes the next unit-cube point to
try. A failed run, or one without an AUROC, contributes the chance value 0.5
to the surrogate, so one failure never stalls the loop. The trial frame is
rewritten after every trial, and a rerun resumes from it.
"""

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from tabpfn import TabPFNRegressor
from tabpfn_extensions.bayesian_optimization import propose_next_point

from anometa.config import Paths, Scenario, resolve_device
from anometa.experiment import run_experiment
from anometa.search.optuna_search import initial_units, write_frame
from anometa.search.space import decode_unit, scenario_suffix, to_config

_CHANCE_AUROC = 0.5
"""AUROC a failed (or AUROC-less) run contributes to the surrogate."""


def _surrogate_target(auroc: float) -> float:
    """Map a recorded trial AUROC to the surrogate's training target.

    Args:
        auroc: The trial's recorded `auroc` (already 0.5 for a failed run).

    Returns:
        `auroc`, or the chance value 0.5 when it is NaN.
    """
    return _CHANCE_AUROC if np.isnan(auroc) else auroc


def run_tabpfn_bo(
    n_trials: int,
    *,
    k: int = 2,
    seed: int = 0,
    n_init: int = 8,
    device: str = "auto",
    paths: Paths = Paths(),
    scenarios: tuple[Scenario, ...] | None = None,
) -> pd.DataFrame:
    """Run a TabPFN-surrogate Bayesian optimization loop over dev Track B configs.

    The first `n_init` points are `initial_units(n_init, seed)`, the same
    shared starting points `run_tpe` warm-starts from. Each later point comes
    from `propose_next_point` on a `TabPFNRegressor` fit on every `(u, auroc)`
    pair observed so far, where a failed run's or a missing/NaN `auroc` is
    the chance value 0.5; `torch` is seeded with `seed * 1000 + trial` right
    before each proposal, so a resumed run proposes what an uninterrupted
    one would. Every config keeps `to_config`'s default name `"search"`, so
    the shared starting points reuse the runs `run_tpe` already made.

    Resumes: the frame is rewritten atomically after every trial, and a
    rerun with an existing frame restores its rows and continues at the
    next trial.

    Args:
        n_trials: Total number of trials the frame should hold.
        k: Few-shot budget for every trial's config.
        seed: Seeds `initial_units`, the `TabPFNRegressor` and each proposal.
        n_init: Number of shared starting points.
        device: Device the `TabPFNRegressor` surrogate runs on.
        paths: Filesystem layout for every trial's run.
        scenarios: Scenarios every trial runs on; `None` for every scenario.
            A restricted run's parquet file name ends in
            `scenario_suffix(scenarios)`.

    Returns:
        One row per trial, columns `trial`, `u`, `run_id`, `status`,
        `auroc` (0.5 for a failed run), `best_so_far` (the running max of
        `auroc`) plus one column per `PARAMS` name; also written to
        `paths.artifacts / "search" / "bo-k{k}-seed{seed}.parquet"`.
    """
    resolved_device = resolve_device(device)
    reg = TabPFNRegressor(
        n_estimators=1,
        device=resolved_device,
        random_state=seed,
        inference_precision=torch.float32,
        differentiable_input=True,
    )
    starts = initial_units(n_init, seed)
    out_dir = paths.artifacts / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bo-k{k}-seed{seed}{scenario_suffix(scenarios)}.parquet"

    rows: list[dict[str, object]] = []
    us: list[NDArray[np.float64]] = []
    ys: list[float] = []
    best = float("nan")
    if out_path.exists():
        done = pd.read_parquet(out_path)
        us = [np.asarray(u, dtype=np.float64) for u in done["u"]]
        ys = [_surrogate_target(float(a)) for a in done["auroc"]]
        best = float(done["best_so_far"].iloc[-1])
        rows = [
            {str(col): value for col, value in record.items()} | {"u": tuple(u.tolist())}
            for record, u in zip(done.to_dict("records"), us, strict=True)
        ]

    for trial in range(len(rows), n_trials):
        if trial < n_init:
            u = starts[trial]
        else:
            train_x = torch.tensor(np.stack(us), dtype=torch.float32, device=resolved_device)
            train_y = torch.tensor(ys, dtype=torch.float32, device=resolved_device)
            torch.manual_seed(seed * 1000 + trial)
            proposed = propose_next_point(
                reg, train_x, train_y, n_candidates=512, top_k=4, n_refine_steps=8, refine_lr=0.05
            )
            u = proposed.cpu().numpy().astype(np.float64)

        params = decode_unit(u.tolist())
        result = run_experiment(to_config(params, k=k, paths=paths, scenarios=scenarios))
        auroc = (
            _CHANCE_AUROC if result.status != "ok" else result.metrics.get("auroc", float("nan"))
        )
        if not np.isnan(auroc):
            best = auroc if np.isnan(best) else max(best, auroc)

        rows.append(
            {
                "trial": trial,
                "u": tuple(float(x) for x in u),
                "run_id": result.run_id,
                "status": result.status,
                "auroc": auroc,
                "best_so_far": best,
                **params,
            }
        )
        us.append(u)
        ys.append(_surrogate_target(auroc))
        write_frame(pd.DataFrame(rows), out_path)

    return pd.DataFrame(rows)
