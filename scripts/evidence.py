"""Simulation study: which analysis recovers a regression coefficient, and when.

Data-generating model, fixed for every replicate::

    (x1, x2) ~ N(0, [[1, 0.5], [0.5, 1]])
    y = 1 + 0.5 x1 + 0.8 x2 + e,   e ~ N(0, 1)

The estimand is the coefficient on ``x2`` (true value 0.8). ``x2`` loses 40%
of its values under one of three mechanisms:

- MCAR: at random;
- MAR: more often when ``y`` is large (driven by the outcome, which stays
  observed);
- MNAR: more often when ``x2`` itself is large.

Analyses compared on the same holed data:

- full data: OLS before any values were removed (the oracle);
- complete case: OLS on rows with ``x2`` observed;
- mean imputation: fill ``x2`` with its observed mean, then OLS with the
  naive standard error;
- MICE: ``misskit.mice`` with ``norm``, m = 10, pooled by Rubin's rules;
- MICE without y: the same, but ``x2`` is imputed from ``x1`` only.

Run ``python scripts/evidence.py`` to print the tables, ``--check`` to
compare them with README.md and exit 1 on any difference.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from misskit._typing import FloatArray
from misskit.amputation import Mechanism, ampute
from misskit.mice import mice
from misskit.pooling import pool_ols

TRUE_BETA = 0.8
N = 200
RATE = 0.4
STRENGTH = 2.0
M = 10
N_ITER = 5
REPS = 200
MAIN_SEED = 20261122
SPREAD_SEEDS = (1, 2, 3, 4, 5)
SPREAD_REPS = 100
MECHANISMS: tuple[Mechanism, ...] = ("MCAR", "MAR", "MNAR")
README = Path(__file__).resolve().parents[1] / "README.md"


def generate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    x = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], size=n)
    y = 1.0 + 0.5 * x[:, 0] + TRUE_BETA * x[:, 1] + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1]})


def make_holes(df: pd.DataFrame, mechanism: Mechanism, rng: np.random.Generator) -> pd.DataFrame:
    drivers = ["y"] if mechanism == "MAR" else []
    return ampute(df, ["x2"], RATE, mechanism, drivers=drivers, strength=STRENGTH, seed=rng)


Estimate = tuple[float, float, float]  # estimate, lower, upper


def _ols_ci(d: pd.DataFrame) -> Estimate:
    x = np.column_stack([np.ones(len(d)), d["x1"].to_numpy(), d["x2"].to_numpy()])
    y = d["y"].to_numpy()
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    dof = len(y) - 3
    se = float(np.sqrt(float(resid @ resid) / dof * xtx_inv[2, 2]))
    crit = float(stats.t.ppf(0.975, dof))
    b = float(beta[2])
    return b, b - crit * se, b + crit * se


def _mice_ci(holed: pd.DataFrame, rng: np.random.Generator, with_outcome: bool) -> Estimate:
    preds: dict[str, Sequence[str]] | None = None if with_outcome else {"x2": ["x1"]}
    res = mice(holed, m=M, n_iter=N_ITER, method="norm", predictors=preds, seed=rng)
    pooled = pool_ols(res.imputations, "y", ["x1", "x2"])
    lo, hi = pooled.conf_int()[2]
    return float(pooled.estimate[2]), float(lo), float(hi)


Arm = Callable[[pd.DataFrame, pd.DataFrame, np.random.Generator], Estimate]

ARMS: dict[str, Arm] = {
    "full data": lambda full, holed, rng: _ols_ci(full),
    "complete case": lambda full, holed, rng: _ols_ci(holed.dropna()),
    "mean imputation": lambda full, holed, rng: _ols_ci(holed.fillna({"x2": holed["x2"].mean()})),
    "MICE": lambda full, holed, rng: _mice_ci(holed, rng, with_outcome=True),
    "MICE without y": lambda full, holed, rng: _mice_ci(holed, rng, with_outcome=False),
}


@dataclass(frozen=True)
class Cell:
    bias: float
    bias_half: float  # half-width of the 95% Monte Carlo interval for the bias
    rmse: float
    coverage: float
    cov_lo: float
    cov_hi: float
    width: float


def summarise(est: FloatArray, lo: FloatArray, hi: FloatArray) -> Cell:
    reps = len(est)
    err = est - TRUE_BETA
    hits = int(np.sum((lo <= TRUE_BETA) & (TRUE_BETA <= hi)))
    ci = stats.binomtest(hits, reps).proportion_ci(0.95, method="wilson")
    return Cell(
        bias=float(err.mean()),
        bias_half=float(1.96 * err.std(ddof=1) / np.sqrt(reps)),
        rmse=float(np.sqrt(np.mean(err**2))),
        coverage=hits / reps,
        cov_lo=float(ci.low),
        cov_hi=float(ci.high),
        width=float(np.mean(hi - lo)),
    )


def run(seed: int, reps: int) -> dict[tuple[str, str], Cell]:
    """Every arm on the same ``reps`` datasets for each mechanism."""
    out: dict[tuple[str, str], Cell] = {}
    for k, mech in enumerate(MECHANISMS):
        rng = np.random.default_rng([seed, k])
        results: dict[str, list[Estimate]] = {a: [] for a in ARMS}
        for _ in range(reps):
            full = generate(N, rng)
            holed = make_holes(full, mech, rng)
            for name, arm in ARMS.items():
                results[name].append(arm(full, holed, rng))
        for name, rows in results.items():
            arr = np.array(rows)
            out[(mech, name)] = summarise(arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def main_table(cells: dict[tuple[str, str], Cell]) -> str:
    lines = [
        "| mechanism | analysis | bias (95% MC interval) | RMSE | coverage of 95% CI "
        "(Wilson interval) | mean CI width |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mech in MECHANISMS:
        for arm in ARMS:
            c = cells[(mech, arm)]
            lines.append(
                f"| {mech} | {arm} | {c.bias:+.3f} (±{c.bias_half:.3f}) | {c.rmse:.3f} | "
                f"{c.coverage:.3f} ({c.cov_lo:.3f}-{c.cov_hi:.3f}) | {c.width:.3f} |"
            )
    return "\n".join(lines)


def spread_table(runs: list[dict[tuple[str, str], Cell]]) -> str:
    lines = [
        "| mechanism | analysis | bias: min to max over seeds | coverage: min to max over seeds |",
        "| --- | --- | --- | --- |",
    ]
    for mech in MECHANISMS:
        for arm in ARMS:
            b = [r[(mech, arm)].bias for r in runs]
            c = [r[(mech, arm)].coverage for r in runs]
            lines.append(
                f"| {mech} | {arm} | {min(b):+.3f} to {max(b):+.3f} "
                f"| {min(c):.2f} to {max(c):.2f} |"
            )
    return "\n".join(lines)


def render() -> dict[str, str]:
    main = main_table(run(MAIN_SEED, REPS))
    spread = spread_table([run(s, SPREAD_REPS) for s in SPREAD_SEEDS])
    return {"main": main, "spread": spread}


def readme_blocks(text: str) -> dict[str, str]:
    blocks = {}
    for name in ("main", "spread"):
        m = re.search(
            rf"<!-- evidence:{name}:start -->\n(.*?)\n<!-- evidence:{name}:end -->", text, re.S
        )
        blocks[name] = m.group(1).strip() if m else ""
    return blocks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="fail if README.md differs")
    args = parser.parse_args(argv)
    tables = render()
    for name, table in tables.items():
        print(f"<!-- evidence:{name}:start -->\n{table}\n<!-- evidence:{name}:end -->\n")
    if args.check:
        published = readme_blocks(README.read_text())
        stale = [k for k in tables if published[k] != tables[k]]
        if stale:
            print(f"README evidence out of date: {', '.join(stale)}", file=sys.stderr)
            return 1
        print("README evidence matches.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
