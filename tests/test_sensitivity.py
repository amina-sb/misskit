from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from misskit.amputation import ampute
from misskit.mice import mice
from misskit.sensitivity import delta_sensitivity, tipping_point


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n)
    y = 1 + 0.5 * x[:, 0] + 0.8 * x[:, 1] + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1]})


def test_zero_delta_reproduces_unadjusted_imputations() -> None:
    holed = ampute(_frame(), ["x2"], 0.4, "MCAR", seed=1)
    a = mice(holed, m=3, n_iter=4, seed=2)
    b = mice(holed, m=3, n_iter=4, delta={"x2": 0.0}, seed=2)
    for x, y in zip(a.imputations, b.imputations, strict=True):
        pd.testing.assert_frame_equal(x, y)


@pytest.mark.parametrize("method", ["norm", "pmm"])
def test_single_column_is_shifted_by_exactly_delta(method: str) -> None:
    # Only x2 is imputed and its model is fitted on observed rows alone, so the
    # random draws are identical and the shift must come through unchanged.
    holed = ampute(_frame(), ["x2"], 0.4, "MAR", drivers=["y"], seed=3)
    holes = holed["x2"].isna().to_numpy()
    base = mice(holed, m=2, n_iter=3, method=method, seed=4)  # type: ignore[arg-type]
    shifted = mice(holed, m=2, n_iter=3, method=method, delta={"x2": 0.75}, seed=4)  # type: ignore[arg-type]
    for b, s in zip(base.imputations, shifted.imputations, strict=True):
        np.testing.assert_allclose(s["x2"].to_numpy()[holes] - b["x2"].to_numpy()[holes], 0.75)
        np.testing.assert_array_equal(s["x2"].to_numpy()[~holes], holed["x2"].to_numpy()[~holes])
        pd.testing.assert_frame_equal(s[["y", "x1"]], holed[["y", "x1"]])


def test_delta_for_complete_column_is_rejected() -> None:
    holed = ampute(_frame(), ["x2"], 0.4, "MCAR", seed=5)
    with pytest.raises(KeyError):
        mice(holed, delta={"x1": 1.0})


def test_sensitivity_curve_is_not_monotone() -> None:
    """Naive expectation: shifting imputed x2 moves its slope one way.

    It does not. Shifting the imputed subset (indicator I) of x2 by delta
    changes cov(x2, y) linearly, by delta * cov(I, y), but var(x2) by
    delta**2 * var(I) + 2 * delta * cov(x2, I). The slope is a ratio of the
    two, so it rises and then falls. Here the curve peaks near delta = 0 and
    two different deltas give similar estimates, which is why the sensitivity
    analysis reports the whole grid and not only where the interval first
    crosses a null.

    The design is MNAR with large x2 more likely missing, so the true
    departure is positive: delta = +1 moves the estimate towards 0.8 and
    delta = -1 moves it the other way.
    """
    holed = ampute(_frame(n=400, seed=6), ["x2"], 0.4, "MNAR", seed=7)
    table = delta_sensitivity(
        holed, "x2", [-1.0, -0.5, 0.0, 0.5, 1.0], "y", ["x1", "x2"], "x2", m=10, n_iter=5, seed=8
    )
    assert table["delta"].tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert np.all(table["lower"] < table["estimate"])
    assert np.all(table["estimate"] < table["upper"])
    est = dict(zip(table["delta"], table["estimate"], strict=True))
    assert est[0.0] > est[-1.0] and est[0.0] > est[1.0]
    assert abs(est[1.0] - 0.8) < abs(est[0.0] - 0.8)
    assert est[-1.0] < 0.7


def test_tipping_point() -> None:
    t = pd.DataFrame(
        {
            "delta": [0.0, 0.5, 1.0, 1.5],
            "lower": [0.3, 0.1, -0.05, -0.2],
            "upper": [0.9, 0.7, 0.5, 0.4],
        }
    )
    assert tipping_point(t) == 1.0
    assert tipping_point(t, null=5.0) is None
    with pytest.raises(ValueError):
        delta_sensitivity(_frame(), "x2", [0.0], "y", ["x1"], "x2")
