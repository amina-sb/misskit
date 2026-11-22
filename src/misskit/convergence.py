"""Convergence diagnostics for MICE chains.

Each of the ``m`` imputations is a separate Gibbs chain, so the classical
potential scale reduction factor of Gelman and Rubin (1992) applies to any
scalar tracked along the chains — here the mean of the imputed values in each
column, which is what van Buuren (2018, section 6.5.2) recommends plotting.
Values close to 1 indicate the chains have mixed; values above about 1.1 mean
more iterations are needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from misskit.mice import MiceResult

__all__ = ["chain_trace", "rhat", "rhat_table"]


def rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin potential scale reduction for an ``(m, n)`` array of draws.

    Uses the original (non-split) estimator::

        W = mean of within-chain variances
        B / n = variance of chain means
        var+ = (n - 1) / n * W + B / n
        R = sqrt(var+ / W)
    """
    c = np.asarray(chains, dtype=float)
    if c.ndim != 2 or c.shape[0] < 2 or c.shape[1] < 2:
        raise ValueError("need an (m, n) array with m >= 2 chains of n >= 2 draws")
    n = c.shape[1]
    w = float(np.mean(np.var(c, axis=1, ddof=1)))
    b_over_n = float(np.var(np.mean(c, axis=1), ddof=1))
    if w == 0.0:
        return 1.0 if b_over_n == 0.0 else float("inf")
    var_plus = (n - 1) / n * w + b_over_n
    return float(np.sqrt(var_plus / w))


def chain_trace(result: MiceResult) -> dict[str, np.ndarray]:
    """``(m, n_iter)`` array of imputed-value means for each imputed column."""
    return {
        col: np.array([[step[col] for step in chain] for chain in result.chain_means])
        for col in result.visit_order
    }


def rhat_table(result: MiceResult, burn_in: float = 0.5) -> pd.Series:
    """R-hat per imputed column, discarding the first ``burn_in`` of iterations."""
    if result.m < 2:
        raise ValueError("R-hat needs at least two imputations")
    start = int(np.floor(result.n_iter * burn_in))
    if result.n_iter - start < 2:
        raise ValueError("too few iterations left after burn-in; increase n_iter")
    traces = chain_trace(result)
    return pd.Series({col: rhat(t[:, start:]) for col, t in traces.items()}, name="rhat")
