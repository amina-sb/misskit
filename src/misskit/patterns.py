"""Describe where values are missing.

Everything here works on the boolean *missingness indicator* ``M`` where
``M[i, j]`` is true when row ``i`` has no value in column ``j``. The functions
accept a :class:`pandas.DataFrame` and treat ``NaN``/``None``/``pd.NA`` as
missing, which is what :meth:`pandas.DataFrame.isna` reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["MissingSummary", "co_missingness", "missing_rates", "patterns", "summarize"]


def _indicator(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
    return df.isna()


def missing_rates(df: pd.DataFrame) -> pd.Series:
    """Fraction of missing values in each column, in column order."""
    m = _indicator(df)
    if len(m) == 0:
        return pd.Series(np.nan, index=df.columns, dtype=float)
    return m.mean(axis=0).astype(float)


def patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct missingness patterns with how many rows follow each one.

    Returns one row per pattern. The pattern columns hold ``True`` where the
    value is *missing*; ``n_rows`` counts rows with that pattern and
    ``n_missing`` counts how many columns the pattern leaves empty. Rows are
    sorted by ``n_rows`` descending, ties broken by fewer missing columns, so
    the complete-case pattern (if present) leads its tie group.
    """
    m = _indicator(df)
    cols = list(m.columns)
    if len(m) == 0:
        return pd.DataFrame(columns=[*cols, "n_rows", "n_missing"])
    grouped = m.value_counts(sort=False).rename("n_rows").reset_index()
    grouped["n_missing"] = grouped[cols].sum(axis=1).astype(int)
    grouped["n_rows"] = grouped["n_rows"].astype(int)
    grouped = grouped.sort_values(
        ["n_rows", "n_missing"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    return grouped


def co_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise probability that two columns are missing together.

    Entry ``(a, b)`` is the fraction of rows where both ``a`` and ``b`` are
    missing, so the diagonal equals :func:`missing_rates`. The matrix is
    symmetric by construction.
    """
    m = _indicator(df).to_numpy(dtype=float)
    n = m.shape[0]
    out = m.T @ m / n if n else np.full((m.shape[1], m.shape[1]), np.nan)
    return pd.DataFrame(out, index=df.columns, columns=df.columns)


@dataclass(frozen=True)
class MissingSummary:
    """Headline numbers about a dataset's missingness."""

    n_rows: int
    n_cols: int
    n_missing: int
    n_complete_rows: int
    n_patterns: int

    @property
    def cell_rate(self) -> float:
        """Fraction of all cells that are missing."""
        cells = self.n_rows * self.n_cols
        return self.n_missing / cells if cells else float("nan")

    @property
    def complete_row_rate(self) -> float:
        """Fraction of rows a complete-case analysis would keep."""
        return self.n_complete_rows / self.n_rows if self.n_rows else float("nan")


def summarize(df: pd.DataFrame) -> MissingSummary:
    """Count missing cells, complete rows and distinct patterns."""
    m = _indicator(df)
    row_missing = m.sum(axis=1)
    return MissingSummary(
        n_rows=int(m.shape[0]),
        n_cols=int(m.shape[1]),
        n_missing=int(m.to_numpy().sum()),
        n_complete_rows=int((row_missing == 0).sum()),
        n_patterns=len(patterns(df)),
    )
