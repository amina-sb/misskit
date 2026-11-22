from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from misskit.pooling import pool, pool_ols


def _by_hand(q: list[float], u: list[float], df_com: float) -> tuple[float, float, float]:
    """Explicit-loop Rubin / Barnard-Rubin, written independently of the module."""
    m = len(q)
    qbar = 0.0
    for v in q:
        qbar += v
    qbar /= m
    ubar = 0.0
    for v in u:
        ubar += v
    ubar /= m
    b = 0.0
    for v in q:
        b += (v - qbar) ** 2
    b /= m - 1
    t = ubar + (1 + 1 / m) * b
    lam = (1 + 1 / m) * b / t
    nu_old = (m - 1) / lam**2
    nu_obs = (df_com + 1) / (df_com + 3) * df_com * (1 - lam)
    return qbar, t, nu_old * nu_obs / (nu_old + nu_obs)


def test_matches_hand_computation() -> None:
    q = [1.10, 0.95, 1.30, 1.02, 1.18]
    u = [0.040, 0.050, 0.045, 0.038, 0.047]
    r = pool(np.array(q)[:, None], np.array(u)[:, None], df_com=40)
    qbar, t, nu = _by_hand(q, u, 40)
    assert r.estimate[0] == pytest.approx(qbar, rel=1e-12)
    assert r.total[0] == pytest.approx(t, rel=1e-12)
    assert r.df[0] == pytest.approx(nu, rel=1e-12)
    lo, hi = r.conf_int()[0]
    crit = stats.t.ppf(0.975, nu)
    assert (lo, hi) == pytest.approx((qbar - crit * np.sqrt(t), qbar + crit * np.sqrt(t)))


def test_zero_between_variance_reduces_to_complete_data() -> None:
    q = np.full((4, 2), 3.0)
    u = np.full((4, 2), 0.25)
    r = pool(q, u, df_com=30)
    np.testing.assert_allclose(r.total, 0.25)
    np.testing.assert_allclose(r.df, 31 / 33 * 30)  # Barnard-Rubin df_obs with lambda = 0
    np.testing.assert_allclose(r.lambda_, 0.0)


def test_large_sample_df_is_rubin_1987() -> None:
    q = np.array([[1.0], [2.0], [1.5]])
    u = np.array([[0.1], [0.1], [0.1]])
    r = pool(q, u)
    lam = (1 + 1 / 3) * 0.25 / (0.1 + (1 + 1 / 3) * 0.25)
    assert r.df[0] == pytest.approx(2 / lam**2)


@settings(max_examples=80, deadline=None)
@given(
    st.integers(2, 20),
    st.floats(1.0, 500.0),
    st.integers(0, 2**32 - 1),
)
def test_df_bounded_by_complete_data_df(m: int, df_com: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    r = pool(rng.standard_normal((m, 3)), rng.uniform(0.01, 2.0, (m, 3)), df_com=df_com)
    assert np.all(r.df > 0)
    assert np.all(r.df <= df_com + 1e-9)
    assert np.all((r.lambda_ >= 0) & (r.lambda_ < 1))
    assert np.all(r.total >= r.within)


def test_pool_ols_single_dataset_repeated_matches_statsmodels() -> None:
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.standard_normal(80), "z": rng.standard_normal(80)})
    d["y"] = 1 + 2 * d["x"] - d["z"] + rng.standard_normal(80)
    r = pool_ols([d, d.copy()], "y", ["x", "z"])
    fit = sm.OLS(d["y"], sm.add_constant(d[["x", "z"]])).fit()
    np.testing.assert_allclose(r.estimate, fit.params.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(r.std_error, fit.bse.to_numpy(), rtol=1e-10)
    assert r.names == ("intercept", "x", "z")
    assert list(r.summary().index) == ["intercept", "x", "z"]


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        pool([[1.0]], [[0.1]])
    with pytest.raises(ValueError):
        pool([[1.0], [2.0]], [[0.1]])
    with pytest.raises(ValueError):
        pool([[1.0], [2.0]], [[0.1], [-0.1]])
    d = pd.DataFrame({"y": [1.0, np.nan, 3.0], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        pool_ols([d, d], "y", ["x"])
