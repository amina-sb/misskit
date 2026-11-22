"""Remove values from complete data under a known missingness mechanism.

Simulation is the only way to know what an imputation method *should* have
recovered, so every method in this package is checked against data that was
complete before this module made holes in it.

The three mechanisms follow Rubin (1976):

``"MCAR"``
    Each target cell is dropped independently with probability ``rate``.
``"MAR"``
    The probability that a target cell is dropped is a logistic function of a
    standardised score built from *driver* columns, which are never amputed
    themselves. Missingness therefore depends only on values that remain
    observed.
``"MNAR"``
    The probability is a logistic function of the standardised target value
    itself, so missingness depends on exactly the information that is lost.

For MAR and MNAR the logistic intercept is solved numerically so the
*expected* missing rate on the supplied sample equals ``rate`` exactly; the
realised rate then differs only by Bernoulli noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from misskit._typing import FloatArray

__all__ = ["Mechanism", "ampute", "missing_probabilities"]

Mechanism = Literal["MCAR", "MAR", "MNAR"]


def _standardise(v: FloatArray) -> FloatArray:
    sd = float(np.std(v))
    if sd == 0.0:
        return np.zeros_like(v, dtype=float)
    return (v - float(np.mean(v))) / sd


def _calibrated(score: FloatArray, rate: float, strength: float) -> FloatArray:
    """Logistic probabilities ``expit(a + strength * score)`` with mean ``rate``."""

    def gap(a: float) -> float:
        return float(np.mean(expit(a + strength * score))) - rate

    a = brentq(gap, -60.0, 60.0, xtol=1e-12)
    return np.asarray(expit(a + strength * score), dtype=float)


def missing_probabilities(
    df: pd.DataFrame,
    target: str,
    rate: float,
    mechanism: Mechanism,
    drivers: Sequence[str] = (),
    strength: float = 2.0,
) -> FloatArray:
    """Per-row probability that ``target`` is removed under ``mechanism``.

    Exposed so tests and simulations can compare the realised mask against
    the probabilities that generated it.
    """
    if not 0.0 < rate < 1.0:
        raise ValueError(f"rate must lie strictly between 0 and 1, got {rate}")
    n = len(df)
    if mechanism == "MCAR":
        return np.full(n, rate, dtype=float)
    if mechanism == "MAR":
        if not drivers:
            raise ValueError("MAR amputation needs at least one driver column")
        if target in drivers:
            raise ValueError(f"target {target!r} cannot also be a driver under MAR")
        raw = df[list(drivers)].to_numpy(dtype=float)
        if np.isnan(raw).any():
            raise ValueError("driver columns must be complete")
        score = _standardise(np.column_stack([_standardise(c) for c in raw.T]).sum(axis=1))
        return _calibrated(score, rate, strength)
    if mechanism == "MNAR":
        values = df[target].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError("target column must be complete before amputation")
        return _calibrated(_standardise(values), rate, strength)
    raise ValueError(f"unknown mechanism {mechanism!r}")


def ampute(
    df: pd.DataFrame,
    targets: Sequence[str],
    rate: float,
    mechanism: Mechanism = "MCAR",
    *,
    drivers: Sequence[str] = (),
    strength: float = 2.0,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with values removed from ``targets``.

    Parameters
    ----------
    df:
        Complete numeric data. It is not modified.
    targets:
        Columns to make holes in.
    rate:
        Expected fraction of each target column to remove.
    mechanism:
        ``"MCAR"``, ``"MAR"`` or ``"MNAR"``; see the module docstring.
    drivers:
        Columns whose values drive MAR missingness. Must be disjoint from
        ``targets`` so they stay fully observed.
    strength:
        Logistic slope on the standardised score. Larger values make the
        mechanism more selective; ignored for MCAR.
    seed:
        Seed or generator for the Bernoulli draws.
    """
    targets = list(targets)
    missing_cols = [c for c in [*targets, *drivers] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"columns not in frame: {missing_cols}")
    overlap = set(targets) & set(drivers)
    if overlap:
        raise ValueError(f"columns cannot be both target and driver: {sorted(overlap)}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    out = df.copy()
    for col in targets:
        p = missing_probabilities(df, col, rate, mechanism, drivers, strength)
        drop = rng.random(len(df)) < p
        out[col] = out[col].astype(float).mask(drop)
    return out
