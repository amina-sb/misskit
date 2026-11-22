from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from misskit.patterns import co_missingness, missing_rates, patterns, summarize


def _frame() -> pd.DataFrame:
    nan = np.nan
    return pd.DataFrame(
        {
            "a": [1.0, nan, 3.0, nan, 5.0],
            "b": [1.0, 2.0, nan, nan, 5.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_missing_rates_by_hand() -> None:
    rates = missing_rates(_frame())
    assert rates.to_dict() == {"a": 0.4, "b": 0.4, "c": 0.0}


def test_patterns_by_hand() -> None:
    p = patterns(_frame())
    # complete (2 rows), a-only, b-only, a-and-b (1 row each)
    assert p["n_rows"].tolist() == [2, 1, 1, 1]
    assert p[["a", "b", "c"]].iloc[0].tolist() == [False, False, False]
    assert p["n_missing"].tolist()[0] == 0
    assert p["n_missing"].tolist()[-1] == 2
    assert p["n_rows"].sum() == 5


def test_co_missingness_by_hand() -> None:
    cm = co_missingness(_frame())
    assert cm.loc["a", "b"] == pytest.approx(0.2)
    assert cm.loc["a", "c"] == 0.0
    np.testing.assert_allclose(np.diag(cm.to_numpy()), missing_rates(_frame()).to_numpy())


def test_summary() -> None:
    s = summarize(_frame())
    assert (s.n_rows, s.n_cols, s.n_missing, s.n_complete_rows, s.n_patterns) == (5, 3, 4, 2, 4)
    assert s.cell_rate == pytest.approx(4 / 15)
    assert s.complete_row_rate == pytest.approx(0.4)


def test_treats_none_and_pd_na_as_missing() -> None:
    df = pd.DataFrame({"x": pd.array([1, None, 3], dtype="Int64"), "y": ["a", None, "c"]})
    assert missing_rates(df).tolist() == pytest.approx([1 / 3, 1 / 3])


def test_empty_frame() -> None:
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    assert np.isnan(missing_rates(df)["a"])
    assert len(patterns(df)) == 0


def test_rejects_non_frame() -> None:
    with pytest.raises(TypeError):
        missing_rates(np.zeros((2, 2)))  # type: ignore[arg-type]


masks = arrays(np.bool_, st.tuples(st.integers(1, 40), st.integers(1, 5)))


@settings(max_examples=60, deadline=None)
@given(masks)
def test_pattern_counts_partition_rows(mask: np.ndarray) -> None:
    df = pd.DataFrame(np.where(mask, np.nan, 1.0), columns=[f"c{i}" for i in range(mask.shape[1])])
    p = patterns(df)
    assert p["n_rows"].sum() == mask.shape[0]
    assert len(p) == len({tuple(r) for r in mask.tolist()})


@settings(max_examples=60, deadline=None)
@given(masks, st.integers(0, 2**32 - 1))
def test_row_shuffle_invariance(mask: np.ndarray, seed: int) -> None:
    df = pd.DataFrame(np.where(mask, np.nan, 1.0), columns=[f"c{i}" for i in range(mask.shape[1])])
    perm = np.random.default_rng(seed).permutation(len(df))
    shuffled = df.iloc[perm].reset_index(drop=True)
    pd.testing.assert_series_equal(missing_rates(df), missing_rates(shuffled))
    cm = co_missingness(df).to_numpy()
    np.testing.assert_allclose(cm, co_missingness(shuffled).to_numpy())
    np.testing.assert_allclose(cm, cm.T)
    assert np.all((cm >= 0) & (cm <= 1))
