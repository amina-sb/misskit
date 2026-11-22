"""Combine estimates across multiply imputed datasets with Rubin's rules.

Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
Barnard, J. and Rubin, D. B. (1999). Small-sample degrees of freedom with
multiple imputation. *Biometrika* 86(4), 948-955.

For ``m`` imputations with estimates ``q_l`` and squared standard errors
``u_l`` of the same quantity:

- pooled estimate ``qbar = mean(q_l)``
- within-imputation variance ``ubar = mean(u_l)``
- between-imputation variance ``b = var(q_l, ddof=1)``
- total variance ``t = ubar + (1 + 1/m) b``

Intervals use a t reference distribution with the Barnard-Rubin degrees of
freedom, which never exceed the complete-data degrees of freedom ``df_com``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["PooledEstimate", "pool", "pool_ols"]


@dataclass(frozen=True)
class PooledEstimate:
    """Rubin-pooled estimates, one entry per parameter."""

    estimate: np.ndarray
    within: np.ndarray
    between: np.ndarray
    total: np.ndarray
    df: np.ndarray
    m: int
    names: tuple[str, ...]

    @property
    def std_error(self) -> np.ndarray:
        return np.sqrt(self.total)

    @property
    def riv(self) -> np.ndarray:
        """Relative increase in variance due to nonresponse, ``(1 + 1/m) b / ubar``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray((1 + 1 / self.m) * self.between / self.within, dtype=float)

    @property
    def lambda_(self) -> np.ndarray:
        """Proportion of total variance attributable to the missing data."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray((1 + 1 / self.m) * self.between / self.total, dtype=float)

    def conf_int(self, level: float = 0.95) -> np.ndarray:
        """``(k, 2)`` array of lower and upper bounds."""
        crit = stats.t.ppf(0.5 + level / 2, self.df)
        half = crit * self.std_error
        return np.column_stack([self.estimate - half, self.estimate + half])

    def summary(self, level: float = 0.95) -> pd.DataFrame:
        ci = self.conf_int(level)
        return pd.DataFrame(
            {
                "estimate": self.estimate,
                "std_error": self.std_error,
                "lower": ci[:, 0],
                "upper": ci[:, 1],
                "df": self.df,
                "lambda": self.lambda_,
            },
            index=list(self.names),
        )


def pool(
    estimates: Sequence[Sequence[float]] | np.ndarray,
    variances: Sequence[Sequence[float]] | np.ndarray,
    df_com: float = np.inf,
    names: Sequence[str] | None = None,
) -> PooledEstimate:
    """Pool ``(m, k)`` estimates and squared standard errors with Rubin's rules.

    ``df_com`` is the complete-data residual degrees of freedom (``n - k``
    for OLS). The default of infinity gives Rubin's (1987) large-sample
    degrees of freedom.
    """
    q = np.atleast_2d(np.asarray(estimates, dtype=float))
    u = np.atleast_2d(np.asarray(variances, dtype=float))
    if q.shape != u.shape:
        raise ValueError(f"estimates {q.shape} and variances {u.shape} differ in shape")
    m, k = q.shape
    if m < 2:
        raise ValueError("pooling needs at least two imputations")
    if np.any(u < 0):
        raise ValueError("variances must be non-negative")
    qbar = q.mean(axis=0)
    ubar = u.mean(axis=0)
    b = q.var(axis=0, ddof=1)
    t = ubar + (1 + 1 / m) * b
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(t > 0, (1 + 1 / m) * b / t, 0.0)
        df_old = np.where(lam > 0, (m - 1) / lam**2, np.inf)
        if np.isinf(df_com):
            df = df_old
        else:
            df_obs = (df_com + 1) / (df_com + 3) * df_com * (1 - lam)
            df = np.where(np.isinf(df_old), df_obs, df_old * df_obs / (df_old + df_obs))
    labels = tuple(names) if names is not None else tuple(f"q{i}" for i in range(k))
    if len(labels) != k:
        raise ValueError(f"got {len(labels)} names for {k} parameters")
    return PooledEstimate(
        estimate=qbar,
        within=ubar,
        between=b,
        total=t,
        df=np.asarray(df, dtype=float),
        m=m,
        names=labels,
    )


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    sigma2 = float(resid @ resid) / (len(y) - x.shape[1])
    return beta, np.diag(xtx_inv) * sigma2


def pool_ols(
    datasets: Sequence[pd.DataFrame], outcome: str, predictors: Sequence[str]
) -> PooledEstimate:
    """Fit ``outcome ~ 1 + predictors`` by OLS on each dataset and pool."""
    est, var = [], []
    n = None
    for d in datasets:
        if d[[outcome, *predictors]].isna().to_numpy().any():
            raise ValueError("datasets passed to pool_ols must be complete")
        x = np.column_stack([np.ones(len(d)), d[list(predictors)].to_numpy(dtype=float)])
        beta, v = _ols(d[outcome].to_numpy(dtype=float), x)
        est.append(beta)
        var.append(v)
        n = len(d)
    assert n is not None
    return pool(
        np.array(est),
        np.array(var),
        df_com=n - len(predictors) - 1,
        names=["intercept", *predictors],
    )
