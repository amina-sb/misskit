from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from misskit.amputation import ampute
from misskit.convergence import chain_trace, rhat, rhat_table
from misskit.mice import mice


def test_rhat_by_hand() -> None:
    c = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    w = 1.0  # both chains have variance 1
    b_over_n = 0.5  # variance of means (2, 3) with ddof=1
    expected = np.sqrt((2 / 3 * w + b_over_n) / w)
    assert rhat(c) == pytest.approx(expected)


def test_rhat_near_one_for_iid_chains_and_large_for_separated() -> None:
    rng = np.random.default_rng(0)
    iid = rng.standard_normal((5, 2000))
    assert rhat(iid) == pytest.approx(1.0, abs=0.01)
    shifted = iid + np.arange(5)[:, None] * 3.0
    assert rhat(shifted) > 2.0


def test_rhat_degenerate_cases() -> None:
    assert rhat(np.ones((3, 4))) == 1.0
    assert rhat(np.array([[1.0, 1.0], [2.0, 2.0]])) == float("inf")
    with pytest.raises(ValueError):
        rhat(np.ones((1, 10)))


def test_mice_chains_mix() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 400), columns=["a", "b"]
    )
    holed = ampute(df, ["b"], 0.3, "MAR", drivers=["a"], seed=2)
    res = mice(holed, m=5, n_iter=40, seed=3)
    assert chain_trace(res)["b"].shape == (5, 40)
    table = rhat_table(res)
    assert table.index.tolist() == ["b"]
    assert table["b"] < 1.1
    with pytest.raises(ValueError):
        rhat_table(mice(holed, m=1, n_iter=4, seed=4))
    with pytest.raises(ValueError):
        rhat_table(mice(holed, m=2, n_iter=2, seed=4))
