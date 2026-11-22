from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from misskit.amputation import ampute
from misskit.em import em_mvn
from misskit.mice import mice
from misskit.pooling import pool


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
    return pd.DataFrame(rng.multivariate_normal(np.zeros(3), cov, size=n), columns=["a", "b", "c"])


@settings(max_examples=25, deadline=None)
@given(st.sampled_from(["norm", "pmm"]), st.floats(0.05, 0.6), st.integers(0, 2**32 - 1))
def test_observed_values_never_change_and_holes_are_filled(
    method: str, rate: float, seed: int
) -> None:
    holed = ampute(_frame(120, seed % 997), ["b", "c"], rate, "MAR", drivers=["a"], seed=seed)
    res = mice(holed, m=2, n_iter=3, method=method, seed=seed)  # type: ignore[arg-type]
    obs = holed.notna().to_numpy()
    for imp in res.imputations:
        assert imp.notna().all().all()
        np.testing.assert_array_equal(imp.to_numpy()[obs], holed.to_numpy()[obs])


def test_pmm_only_imputes_observed_values() -> None:
    holed = ampute(_frame(300, 1), ["b"], 0.4, "MAR", drivers=["a"], seed=2)
    seen = set(holed["b"].dropna().tolist())
    res = mice(holed, m=3, n_iter=5, method="pmm", seed=3)
    for imp in res.imputations:
        assert set(imp.loc[holed["b"].isna(), "b"].tolist()) <= seen


def test_seed_reproducible_and_imputations_differ() -> None:
    holed = ampute(_frame(200, 4), ["b"], 0.3, "MCAR", seed=5)
    a = mice(holed, m=3, n_iter=4, seed=6)
    b = mice(holed, m=3, n_iter=4, seed=6)
    for x, y in zip(a.imputations, b.imputations, strict=True):
        pd.testing.assert_frame_equal(x, y)
    assert not a.imputations[0].equals(a.imputations[1])


def test_visit_order_least_missing_first_and_trace_shape() -> None:
    df = _frame(200, 7)
    holed = ampute(df, ["c"], 0.5, "MCAR", seed=8)
    holed = ampute(holed, ["b"], 0.1, "MCAR", seed=9)
    res = mice(holed, m=2, n_iter=4, seed=10)
    assert res.visit_order == ("b", "c")
    assert len(res.chain_means) == 2
    assert len(res.chain_means[0]) == 4
    assert set(res.chain_means[0][0]) == {"b", "c"}


def test_agrees_with_em_under_mar() -> None:
    """Two different estimators of the same quantity on a large sample."""
    df = _frame(5000, 11)
    holed = ampute(df, ["b", "c"], 0.4, "MAR", drivers=["a"], seed=12)
    em = em_mvn(holed.to_numpy())
    res = mice(holed, m=10, n_iter=10, seed=13)
    mice_mean = np.mean([imp.mean().to_numpy() for imp in res.imputations], axis=0)
    np.testing.assert_allclose(mice_mean, em.mean, atol=0.02)
    # and both are far from the complete-case mean, which MAR biases
    assert abs(holed["b"].mean() - em.mean[1]) > 0.2


@pytest.mark.parametrize("method", ["norm", "pmm"])
def test_pooled_interval_for_mean_covers_at_nominal_rate(method: str) -> None:
    reps, n, m, hits = 150, 150, 5, 0
    for r in range(reps):
        holed = ampute(_frame(n, 1000 + r), ["b"], 0.4, "MAR", drivers=["a"], seed=r)
        res = mice(holed, m=m, n_iter=5, method=method, seed=r)  # type: ignore[arg-type]
        q = [[imp["b"].mean()] for imp in res.imputations]
        u = [[imp["b"].var(ddof=1) / n] for imp in res.imputations]
        lo, hi = pool(q, u, df_com=n - 1).conf_int()[0]
        hits += lo <= 0.0 <= hi
    coverage = hits / reps
    # 150 reps: binomial sd at 0.95 is ~0.018
    assert 0.89 <= coverage <= 0.995, coverage


def test_custom_predictors_and_validation() -> None:
    holed = ampute(_frame(100, 14), ["b"], 0.3, "MCAR", seed=15)
    res = mice(holed, m=1, n_iter=2, predictors={"b": ["a"]}, seed=16)
    assert res.m == 1
    with pytest.raises(KeyError):
        mice(holed, predictors={"a": ["b"]})
    with pytest.raises(KeyError):
        mice(holed, predictors={"b": ["a", "nope"]})
    with pytest.raises(ValueError):
        mice(holed, m=0)
    with pytest.raises(ValueError):
        mice(holed, method="mean")  # type: ignore[arg-type]
    allgone = holed.copy()
    allgone["c"] = np.nan
    with pytest.raises(ValueError):
        mice(allgone)


def _coverage(method: str, reps: int = 300, n: int = 60) -> float:
    hits = 0
    for r in range(reps):
        holed = ampute(_frame(n, 2000 + r), ["b"], 0.6, "MAR", drivers=["a"], seed=r)
        res = mice(holed, m=5, n_iter=5, method=method, seed=r)  # type: ignore[arg-type]
        q = [[imp["b"].mean()] for imp in res.imputations]
        u = [[imp["b"].var(ddof=1) / n] for imp in res.imputations]
        lo, hi = pool(q, u, df_com=n - 1).conf_int()[0]
        hits += lo <= 0.0 <= hi
    return hits / reps


def test_dropping_the_parameter_draw_undercovers() -> None:
    """Naive expectation: the parameter draw is a refinement that barely matters.

    Measured at n = 60 with 60% MAR missingness: proper imputation covers at
    about the nominal 95%, the same imputer without the draw at about 80%.
    The between-imputation variance is what Rubin's rules rely on, and
    leaving out parameter uncertainty removes a large part of it.
    """
    proper, improper = _coverage("norm"), _coverage("norm.nob")
    assert proper >= 0.91, proper
    assert improper <= 0.88, improper
    assert proper - improper >= 0.07
