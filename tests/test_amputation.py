from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from misskit.amputation import ampute, missing_probabilities


def _data(n: int = 20_000, rho: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y, "z": rng.standard_normal(n)})


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR", "MNAR"])
@pytest.mark.parametrize("rate", [0.1, 0.3, 0.6])
def test_expected_rate_is_exact_and_realised_rate_is_close(mechanism: str, rate: float) -> None:
    df = _data()
    p = missing_probabilities(df, "y", rate, mechanism, drivers=["x"])  # type: ignore[arg-type]
    assert p.mean() == pytest.approx(rate, abs=1e-9)
    out = ampute(df, ["y"], rate, mechanism, drivers=["x"], seed=1)  # type: ignore[arg-type]
    realised = out["y"].isna().mean()
    se = np.sqrt(rate * (1 - rate) / len(df))
    assert abs(realised - rate) < 5 * se


def test_drivers_and_untargeted_columns_stay_complete() -> None:
    out = ampute(_data(), ["y"], 0.4, "MAR", drivers=["x"], seed=2)
    assert out["x"].notna().all()
    assert out["z"].notna().all()


@settings(max_examples=40, deadline=None)
@given(
    st.sampled_from(["MCAR", "MAR", "MNAR"]),
    st.floats(0.05, 0.9),
    st.integers(0, 2**32 - 1),
)
def test_observed_values_are_untouched(mechanism: str, rate: float, seed: int) -> None:
    df = _data(n=300, rho=0.5, seed=seed % 1000)
    out = ampute(df, ["y", "z"], rate, mechanism, drivers=["x"], seed=seed)  # type: ignore[arg-type]
    kept = out.notna()
    np.testing.assert_array_equal(out.to_numpy()[kept.to_numpy()], df.to_numpy()[kept.to_numpy()])


def test_does_not_modify_input() -> None:
    df = _data(n=100)
    before = df.copy()
    ampute(df, ["y"], 0.5, "MNAR", seed=0)
    pd.testing.assert_frame_equal(df, before)


def test_seed_reproducible() -> None:
    df = _data(n=500)
    a = ampute(df, ["y"], 0.3, "MAR", drivers=["x"], seed=7)
    b = ampute(df, ["y"], 0.3, "MAR", drivers=["x"], seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_mar_ignores_own_value_and_mnar_does_not() -> None:
    # y independent of the driver x: under MAR the observed y should have the
    # same mean as the full y; under MNAR (high y more likely lost) it falls.
    df = _data(rho=0.0, seed=3)
    full_mean = df["y"].mean()
    se = df["y"].std() / np.sqrt(len(df))
    mar = ampute(df, ["y"], 0.4, "MAR", drivers=["x"], seed=4)
    mnar = ampute(df, ["y"], 0.4, "MNAR", seed=4)
    assert abs(mar["y"].mean() - full_mean) < 6 * se
    assert mnar["y"].mean() < full_mean - 0.3


def test_mar_missingness_tracks_driver() -> None:
    df = _data(seed=5)
    out = ampute(df, ["y"], 0.3, "MAR", drivers=["x"], strength=2.0, seed=6)
    miss = out["y"].isna()
    # positive slope: rows with large x lose y more often
    assert df.loc[miss, "x"].mean() > df.loc[~miss, "x"].mean() + 0.5


def test_mcar_independent_of_everything() -> None:
    df = _data(seed=8)
    out = ampute(df, ["y"], 0.3, "MCAR", seed=9)
    miss = out["y"].isna()
    se = np.sqrt(1 / miss.sum() + 1 / (~miss).sum())
    for col in ["x", "y", "z"]:
        assert abs(df.loc[miss, col].mean() - df.loc[~miss, col].mean()) < 5 * se


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"targets": ["y"], "rate": 0.0}, ValueError),
        ({"targets": ["y"], "rate": 1.0}, ValueError),
        ({"targets": ["y"], "rate": 0.2, "mechanism": "MAR"}, ValueError),
        ({"targets": ["y"], "rate": 0.2, "mechanism": "MAR", "drivers": ["y"]}, ValueError),
        ({"targets": ["nope"], "rate": 0.2}, KeyError),
        ({"targets": ["y"], "rate": 0.2, "mechanism": "MXAR"}, ValueError),
    ],
)
def test_rejects_bad_arguments(kwargs: dict[str, object], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        ampute(_data(n=50), **kwargs)  # type: ignore[arg-type]
