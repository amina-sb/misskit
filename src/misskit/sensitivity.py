"""How far from MAR would the mechanism have to be to change the answer?

Observed data cannot distinguish MAR from MNAR, so an analysis under MAR
should be accompanied by a sensitivity analysis. The delta-adjustment approach
(van Buuren 2018, section 3.8; Carpenter and Kenward 2013, chapter 10) imputes
under MAR and then shifts the imputed values of one column by ``delta``,
which is the same as assuming the unobserved values differ from comparable
observed ones by ``delta`` on average. Scanning a grid of deltas shows how
large that departure must be before the conclusion changes: the *tipping
point*.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from misskit.mice import Method, mice
from misskit.pooling import pool_ols

__all__ = ["delta_sensitivity", "tipping_point"]


def delta_sensitivity(
    df: pd.DataFrame,
    column: str,
    deltas: Sequence[float],
    outcome: str,
    predictors: Sequence[str],
    term: str,
    *,
    m: int = 20,
    n_iter: int = 10,
    method: Method = "norm",
    level: float = 0.95,
    seed: int | None = None,
) -> pd.DataFrame:
    """Pooled OLS estimate of ``term`` when ``column``'s imputations are shifted.

    For each delta, ``df`` is imputed ``m`` times with ``mice(..., delta={column:
    delta})``, ``outcome ~ 1 + predictors`` is fitted on each completed
    dataset and the coefficient on ``term`` is pooled. Every delta uses the
    same seed, so differences between rows come from the shift and not from
    Monte Carlo noise.
    """
    if term not in ("intercept", *predictors):
        raise ValueError(f"term {term!r} is not in the model")
    rows = []
    k = ["intercept", *predictors].index(term)
    for d in deltas:
        res = mice(df, m=m, n_iter=n_iter, method=method, delta={column: float(d)}, seed=seed)
        pooled = pool_ols(res.imputations, outcome, list(predictors))
        lo, hi = pooled.conf_int(level)[k]
        rows.append(
            {
                "delta": float(d),
                "estimate": float(pooled.estimate[k]),
                "std_error": float(pooled.std_error[k]),
                "lower": float(lo),
                "upper": float(hi),
            }
        )
    return pd.DataFrame(rows)


def tipping_point(table: pd.DataFrame, null: float = 0.0) -> float | None:
    """Smallest ``|delta|`` in ``table`` whose interval contains ``null``.

    Returns ``None`` when no delta in the grid reaches it, which says only
    that the grid was not wide enough.
    """
    inside = table[(table["lower"] <= null) & (null <= table["upper"])]
    if inside.empty:
        return None
    return float(inside["delta"].iloc[int(np.argmin(np.abs(inside["delta"].to_numpy())))])
