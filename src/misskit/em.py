"""Maximum-likelihood mean and covariance of incomplete multivariate normal data.

The EM algorithm of Dempster, Laird and Rubin (1977), as laid out in Little and
Rubin, *Statistical Analysis with Missing Data* (3rd ed., 2019), section 11.2.
Rows are grouped by missingness pattern so the conditional regressions are
computed once per pattern rather than once per row.

The result is the *maximum-likelihood* covariance, i.e. divided by ``n`` and
not ``n - 1``; on complete data it equals ``np.cov(X, rowvar=False, ddof=0)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["EMResult", "em_mvn"]


@dataclass(frozen=True)
class EMResult:
    mean: np.ndarray
    cov: np.ndarray
    n_iter: int
    converged: bool
    loglik: float


def _group_patterns(miss: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(row_indices, observed_mask)`` for each distinct pattern."""
    uniq, inverse = np.unique(miss, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    return [(np.flatnonzero(inverse == k), ~uniq[k]) for k in range(len(uniq))]


def _loglik(
    x: np.ndarray, groups: list[tuple[np.ndarray, np.ndarray]], mu: np.ndarray, sigma: np.ndarray
) -> float:
    total = 0.0
    for rows, obs in groups:
        k = int(obs.sum())
        if k == 0:
            continue
        s = sigma[np.ix_(obs, obs)]
        d = x[np.ix_(rows, obs)] - mu[obs]
        _, logdet = np.linalg.slogdet(s)
        quad = np.einsum("ij,ij->i", d @ np.linalg.inv(s), d)
        total += float(-0.5 * (len(rows) * (k * np.log(2 * np.pi) + logdet) + quad.sum()))
    return total


def em_mvn(
    x: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> EMResult:
    """Estimate the mean vector and covariance matrix of ``x`` by EM.

    Parameters
    ----------
    x:
        ``(n, p)`` array with ``NaN`` for missing values. Rows that are
        entirely missing carry no information and are ignored.
    max_iter, tol:
        Iteration stops when the largest absolute change in any mean or
        covariance entry falls below ``tol``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional")
    miss = np.isnan(x)
    keep = ~miss.all(axis=1)
    x, miss = x[keep], miss[keep]
    n, p = x.shape
    if n == 0:
        raise ValueError("every row is entirely missing")
    if miss.all(axis=0).any():
        raise ValueError("a column is entirely missing; its moments are not identified")

    mu = np.nanmean(x, axis=0)
    filled = np.where(miss, mu, x)
    sigma = np.cov(filled, rowvar=False, ddof=0).reshape(p, p)
    sigma = sigma + np.eye(p) * 1e-9 * max(1.0, float(np.trace(sigma)))
    groups = _group_patterns(miss)

    converged = False
    n_iter = 0
    for _ in range(max_iter):
        n_iter += 1
        sum_x = np.zeros(p)
        sum_xx = np.zeros((p, p))
        for rows, obs in groups:
            mis = ~obs
            xo = x[np.ix_(rows, obs)]
            block = np.empty((len(rows), p))
            block[:, obs] = xo
            if mis.any():
                s_oo = sigma[np.ix_(obs, obs)]
                s_mo = sigma[np.ix_(mis, obs)]
                beta = np.linalg.solve(s_oo, s_mo.T).T if obs.any() else np.zeros((mis.sum(), 0))
                block[:, mis] = mu[mis] + (xo - mu[obs]) @ beta.T
                cond = sigma[np.ix_(mis, mis)] - beta @ s_mo.T
                sum_xx[np.ix_(mis, mis)] += len(rows) * cond
            sum_x += block.sum(axis=0)
            sum_xx += block.T @ block
        new_mu = sum_x / n
        new_sigma = sum_xx / n - np.outer(new_mu, new_mu)
        new_sigma = (new_sigma + new_sigma.T) / 2
        delta = max(float(np.max(np.abs(new_mu - mu))), float(np.max(np.abs(new_sigma - sigma))))
        mu, sigma = new_mu, new_sigma
        if delta < tol:
            converged = True
            break
    return EMResult(
        mean=mu, cov=sigma, n_iter=n_iter, converged=converged, loglik=_loglik(x, groups, mu, sigma)
    )
