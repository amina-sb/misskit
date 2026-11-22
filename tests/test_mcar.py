"""Little's test is checked by what it is for: its rejection rate.

Size: under MCAR the test should reject about ``alpha`` of the time.
Power: under a clear MAR mechanism it should reject nearly always.
Blind spot: under a symmetric MNAR mechanism that leaves pattern means
unchanged, it has no power — which is a property of the test, not a bug,
and is pinned here so the documentation's warning stays true.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from misskit.amputation import ampute
from misskit.mcar import little_mcar_test

ALPHA = 0.05


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    return pd.DataFrame(rng.multivariate_normal(np.zeros(3), cov, size=n), columns=["a", "b", "c"])


def _rejection_rate(mechanism: str, reps: int, n: int = 200) -> float:
    rejections = 0
    for r in range(reps):
        df = _frame(n, seed=r)
        out = ampute(df, ["b", "c"], 0.25, mechanism, drivers=["a"], seed=10_000 + r)  # type: ignore[arg-type]
        rejections += little_mcar_test(out).p_value < ALPHA
    return rejections / reps


def test_size_under_mcar() -> None:
    reps = 400
    rate = _rejection_rate("MCAR", reps)
    se = np.sqrt(ALPHA * (1 - ALPHA) / reps)
    assert abs(rate - ALPHA) < 3.5 * se, rate


def test_power_under_mar() -> None:
    assert _rejection_rate("MAR", 100) > 0.9


def test_blind_to_symmetric_mnar() -> None:
    # Remove the most extreme |b| values: pattern means stay centred, so the
    # test cannot see a mechanism that is plainly not MCAR.
    reps, rejections = 300, 0
    for r in range(reps):
        df = _frame(200, seed=r)
        rng = np.random.default_rng(50_000 + r)
        p = 1 / (1 + np.exp(-(2.0 * np.abs(df["b"]) - 2.5)))
        df.loc[rng.random(len(df)) < p, "b"] = np.nan
        rejections += little_mcar_test(df).p_value < ALPHA
    assert rejections / reps < 0.12


def test_complete_data_has_no_degrees_of_freedom() -> None:
    res = little_mcar_test(_frame(50, 0))
    assert (res.statistic, res.df, res.p_value, res.n_patterns) == (0.0, 0, 1.0, 1)


def test_degrees_of_freedom_by_hand() -> None:
    df = _frame(60, 1)
    df.loc[0:9, "b"] = np.nan  # pattern (a, c) observed: 2
    df.loc[10:19, ["b", "c"]] = np.nan  # pattern (a) observed: 1
    res = little_mcar_test(df)
    assert res.n_patterns == 3
    assert res.df == 3 + 2 + 1 - 3
    assert 0.0 <= res.p_value <= 1.0
    assert res.statistic >= 0.0


def test_accepts_arrays_and_frames_equally() -> None:
    out = ampute(_frame(150, 2), ["b"], 0.3, "MAR", drivers=["a"], seed=3)
    a = little_mcar_test(out)
    b = little_mcar_test(out.to_numpy())
    assert a.statistic == pytest.approx(b.statistic, rel=1e-12)
