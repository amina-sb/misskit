"""Little's (1988) test of missing completely at random.

Little, R. J. A. (1988). A test of missing completely at random for
multivariate data with missing values. *JASA* 83(404), 1198-1202.

Under MCAR, the mean of the observed variables within each missingness
pattern should agree with the overall mean. With ``mu`` and ``Sigma`` the EM
maximum-likelihood estimates, the statistic

    d2 = sum_j n_j (ybar_j - mu_oj)' Sigma_oj^{-1} (ybar_j - mu_oj)

is asymptotically chi-squared with ``sum_j p_j - p`` degrees of freedom, where
``p_j`` is the number of observed variables in pattern ``j``.

What the test can and cannot say
--------------------------------
Rejecting MCAR is informative: missingness is related to the data. Failing to
reject is weak evidence; the test has no power at all against MNAR mechanisms
that leave pattern means unchanged, and it can never distinguish MAR from MNAR,
because that distinction depends on values that were not observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from misskit.em import _group_patterns, em_mvn

__all__ = ["LittleResult", "little_mcar_test"]


@dataclass(frozen=True)
class LittleResult:
    statistic: float
    df: int
    p_value: float
    n_patterns: int


def little_mcar_test(data: pd.DataFrame | np.ndarray) -> LittleResult:
    """Run Little's MCAR test on numeric data with ``NaN`` for missing values.

    Rows that are entirely missing are dropped before testing, as they carry
    no information about the observed means.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data must be two-dimensional")
    x = x[~np.isnan(x).all(axis=1)]
    est = em_mvn(x)
    miss = np.isnan(x)
    groups = _group_patterns(miss)
    d2 = 0.0
    dof = -x.shape[1]
    for rows, obs in groups:
        diff = x[np.ix_(rows, obs)].mean(axis=0) - est.mean[obs]
        s = est.cov[np.ix_(obs, obs)]
        d2 += len(rows) * float(diff @ np.linalg.solve(s, diff))
        dof += int(obs.sum())
    if dof <= 0:
        # a single pattern: every row has the same missingness, nothing to compare
        return LittleResult(statistic=0.0, df=0, p_value=1.0, n_patterns=len(groups))
    p = float(stats.chi2.sf(d2, dof))
    return LittleResult(statistic=d2, df=dof, p_value=p, n_patterns=len(groups))
