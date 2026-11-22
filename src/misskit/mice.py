"""Multiple imputation by chained equations.

van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.).
CRC Press. Section numbers below refer to that book.

Each incomplete column is imputed in turn from all other columns, using the
current filled-in values of those columns as predictors, and the cycle is
repeated ``n_iter`` times. Running the whole procedure ``m`` times from
independent random starts gives ``m`` completed datasets whose spread carries
the uncertainty due to the missing values.

Two univariate methods are provided:

``"norm"``
    Bayesian linear regression (Algorithm 3.1). Regression parameters are
    *drawn* from their posterior before predicting, so imputations reflect
    parameter uncertainty as well as residual noise. Imputing the fitted
    value plus noise without the parameter draw ("improper" imputation)
    understates the between-imputation variance and gives intervals that are
    too narrow.
``"norm.nob"``
    Least-squares fitted value plus normal noise, *without* the parameter
    draw (R ``mice``'s method of the same name). Provided as the contrast
    that shows why the draw matters: its pooled intervals undercover, by an
    amount the test suite measures.
``"pmm"``
    Predictive mean matching (Algorithm 3.3, type-1 matching). The same
    posterior draw produces predicted means; each missing value is replaced
    by the observed value of a donor drawn at random from the ``donors``
    observed rows whose predicted means are closest. Imputations are always
    values actually seen in the column.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["Method", "MiceResult", "mice"]

Method = Literal["norm", "norm.nob", "pmm"]


@dataclass(frozen=True)
class MiceResult:
    """The ``m`` completed datasets and the trace of imputed column means."""

    imputations: list[pd.DataFrame]
    method: Method
    n_iter: int
    visit_order: tuple[str, ...]
    # chain_means[l][it][col] = mean of imputed values in column col after
    # iteration it of imputation l; useful for judging convergence
    chain_means: list[list[dict[str, float]]] = field(repr=False)

    @property
    def m(self) -> int:
        return len(self.imputations)


def _posterior_draw(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, ridge: float
) -> tuple[np.ndarray, float]:
    """Draw ``(beta, sigma)`` from the posterior of ``y ~ x`` (Algorithm 3.1)."""
    n, q = x.shape
    xtx = x.T @ x
    penalty = ridge * np.diag(np.diag(xtx))
    v = np.linalg.inv(xtx + penalty)
    v = (v + v.T) / 2
    beta_hat = v @ x.T @ y
    resid = y - x @ beta_hat
    dof = max(n - q, 1)
    sigma = float(np.sqrt(float(resid @ resid) / rng.chisquare(dof)))
    chol = np.linalg.cholesky(v)
    beta = beta_hat + sigma * chol @ rng.standard_normal(q)
    return beta, sigma


def _impute_column(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_mis: np.ndarray,
    method: Method,
    rng: np.random.Generator,
    donors: int,
    ridge: float,
) -> np.ndarray:
    if method == "norm.nob":
        beta_hat = np.linalg.lstsq(x_obs, y_obs, rcond=None)[0]
        resid = y_obs - x_obs @ beta_hat
        sd = float(np.sqrt(float(resid @ resid) / max(len(y_obs) - x_obs.shape[1], 1)))
        return np.asarray(x_mis @ beta_hat + sd * rng.standard_normal(len(x_mis)), dtype=float)
    beta, sigma = _posterior_draw(x_obs, y_obs, rng, ridge)
    if method == "norm":
        return np.asarray(x_mis @ beta + sigma * rng.standard_normal(len(x_mis)), dtype=float)
    # type-1 matching: observed predictions use beta_hat, missing use the draw
    beta_hat = np.linalg.lstsq(x_obs, y_obs, rcond=None)[0]
    yhat_obs = x_obs @ beta_hat
    yhat_mis = x_mis @ beta
    k = min(donors, len(y_obs))
    dist = np.abs(yhat_mis[:, None] - yhat_obs[None, :])
    nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
    pick = nearest[np.arange(len(x_mis)), rng.integers(0, k, len(x_mis))]
    return np.asarray(y_obs[pick], dtype=float)


def mice(
    df: pd.DataFrame,
    m: int = 5,
    n_iter: int = 10,
    method: Method = "norm",
    *,
    predictors: dict[str, Sequence[str]] | None = None,
    donors: int = 5,
    ridge: float = 1e-5,
    seed: int | np.random.Generator | None = None,
) -> MiceResult:
    """Impute ``df`` ``m`` times by chained equations.

    Parameters
    ----------
    df:
        Numeric data with ``NaN`` for missing values. Not modified.
    m:
        Number of imputed datasets.
    n_iter:
        Gibbs iterations per imputation. Ten is usually ample for data with
        moderate missingness; inspect ``chain_means`` if in doubt.
    method:
        ``"norm"``, ``"norm.nob"`` or ``"pmm"``; see the module docstring.
    predictors:
        Optional map from each incomplete column to the columns used to
        predict it. Defaults to every other column.
    donors:
        Donor pool size for ``"pmm"``.
    ridge:
        Ridge penalty, relative to the diagonal of ``X'X``, guarding against
        collinear predictors (van Buuren's ``ridge`` argument).
    seed:
        Seed or generator. A fixed seed reproduces every imputation exactly.
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    if method not in ("norm", "norm.nob", "pmm"):
        raise ValueError(f"unknown method {method!r}")
    data = df.astype(float)
    miss = data.isna()
    rates = miss.mean()
    if (rates == 1).any():
        raise ValueError(f"columns entirely missing: {list(rates.index[rates == 1])}")
    # van Buuren's default "monotone-ish" order: least missing first
    order = tuple(str(c) for c in rates[rates > 0].sort_values(kind="mergesort").index)
    preds = {c: [p for p in data.columns if p != c] for c in order}
    if predictors is not None:
        for c, ps in predictors.items():
            if c not in preds:
                raise KeyError(f"{c!r} has no missing values to impute")
            preds[c] = list(ps)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    imputations: list[pd.DataFrame] = []
    traces: list[list[dict[str, float]]] = []
    for _ in range(m):
        filled = data.copy()
        for c in order:  # initialise by random draws from the observed values
            obs = data[c].dropna().to_numpy()
            holes = miss[c].to_numpy()
            filled.loc[holes, c] = rng.choice(obs, size=int(holes.sum()))
        trace: list[dict[str, float]] = []
        for _ in range(n_iter):
            step: dict[str, float] = {}
            for c in order:
                holes = miss[c].to_numpy()
                x = np.column_stack([np.ones(len(filled)), filled[preds[c]].to_numpy(dtype=float)])
                y = filled[c].to_numpy(dtype=float)
                draws = _impute_column(x[~holes], y[~holes], x[holes], method, rng, donors, ridge)
                filled.loc[holes, c] = draws
                step[c] = float(draws.mean())
            trace.append(step)
        imputations.append(filled)
        traces.append(trace)
    return MiceResult(
        imputations=imputations,
        method=method,
        n_iter=n_iter,
        visit_order=order,
        chain_means=traces,
    )
