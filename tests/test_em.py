from __future__ import annotations

import itertools

import numpy as np
import pytest

from misskit._typing import FloatArray
from misskit.em import em_mvn


def _mvn(n: int, seed: int, p: int = 3) -> FloatArray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((p, p))
    cov = a @ a.T + np.eye(p)
    return rng.multivariate_normal(np.arange(p, dtype=float), cov, size=n)


def test_complete_data_gives_sample_moments() -> None:
    x = _mvn(500, 0)
    r = em_mvn(x)
    assert r.converged
    np.testing.assert_allclose(r.mean, x.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(r.cov, np.cov(x, rowvar=False, ddof=0), atol=1e-10)


def test_bivariate_monotone_matches_anderson_closed_form() -> None:
    """x complete, y partly missing: Anderson (1957) closed-form ML estimates.

    mu_y = ybar_c + b (xbar - xbar_c), with b the complete-case regression slope
    of y on x; sigma_yy = s_yy.x + b^2 s_xx; sigma_xy = b s_xx
    (Little & Rubin 2019, section 7.2.1).
    """
    rng = np.random.default_rng(1)
    n = 400
    x = rng.standard_normal(n)
    y = 1.0 + 0.8 * x + 0.5 * rng.standard_normal(n)
    y[rng.random(n) < 1 / (1 + np.exp(-2 * x))] = np.nan  # MAR on x
    data = np.column_stack([x, y])
    r = em_mvn(data, tol=1e-12)

    c = ~np.isnan(y)
    xc, yc = x[c], y[c]
    s_xx_c = np.mean((xc - xc.mean()) ** 2)
    b = np.mean((xc - xc.mean()) * (yc - yc.mean())) / s_xx_c
    s_yy_x = np.mean((yc - yc.mean() - b * (xc - xc.mean())) ** 2)
    s_xx = np.mean((x - x.mean()) ** 2)
    mu_y = yc.mean() + b * (x.mean() - xc.mean())

    np.testing.assert_allclose(r.mean, [x.mean(), mu_y], atol=1e-8)
    np.testing.assert_allclose(r.cov[0, 0], s_xx, atol=1e-8)
    np.testing.assert_allclose(r.cov[0, 1], b * s_xx, atol=1e-8)
    np.testing.assert_allclose(r.cov[1, 1], s_yy_x + b**2 * s_xx, atol=1e-8)


def test_loglik_does_not_decrease_with_more_iterations() -> None:
    x = _mvn(300, 2, p=4)
    rng = np.random.default_rng(3)
    x[rng.random(x.shape) < 0.25] = np.nan
    lls = [em_mvn(x, max_iter=k, tol=0).loglik for k in (1, 2, 5, 20, 100)]
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(lls))


def test_recovers_truth_under_mar_where_complete_case_does_not() -> None:
    rng = np.random.default_rng(4)
    n = 20_000
    x = rng.standard_normal(n)
    y = 0.9 * x + np.sqrt(1 - 0.81) * rng.standard_normal(n)
    y[x > 0.3] = np.where(rng.random((x > 0.3).sum()) < 0.8, np.nan, y[x > 0.3])
    r = em_mvn(np.column_stack([x, y]))
    assert abs(r.mean[1]) < 0.03  # truth is 0
    assert np.nanmean(y) < -0.2  # complete-case mean is badly biased


def test_covariance_is_symmetric_positive_definite() -> None:
    x = _mvn(200, 5, p=5)
    x[np.random.default_rng(6).random(x.shape) < 0.3] = np.nan
    r = em_mvn(x)
    np.testing.assert_allclose(r.cov, r.cov.T)
    assert np.linalg.eigvalsh(r.cov).min() > 0


def test_ignores_all_missing_rows_and_rejects_unidentified_columns() -> None:
    x = _mvn(50, 7)
    x[0] = np.nan
    r_with = em_mvn(x)
    r_without = em_mvn(x[1:])
    np.testing.assert_allclose(r_with.mean, r_without.mean)
    x[:, 1] = np.nan
    with pytest.raises(ValueError):
        em_mvn(x)
