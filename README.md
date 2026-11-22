# misskit

Missing-data diagnostics and multiple imputation for tabular data, with a
simulation study that shows when imputation changes the answer and when it
cannot.

[![CI](https://github.com/amina-sb/misskit/actions/workflows/ci.yml/badge.svg)](https://github.com/amina-sb/misskit/actions/workflows/ci.yml)

```bash
git clone https://github.com/amina-sb/misskit && cd misskit
pip install -e ".[dev]"
pytest                               # includes the README evidence check
```

Requires Python 3.10 or later, numpy, pandas and scipy.

## Diagnose

```python
import numpy as np, pandas as pd
import misskit

rng = np.random.default_rng(0)
df = pd.DataFrame(rng.standard_normal((500, 3)), columns=["age", "income", "score"])
holed = misskit.ampute(df, ["income"], rate=0.3, mechanism="MAR", drivers=["age"], seed=1)

misskit.patterns(holed)  # distinct patterns with row counts
misskit.missing_rates(holed)  # per-column fraction missing
misskit.little_mcar_test(
    holed
)  # LittleResult(statistic=151.8, df=2, p_value=1.1e-33, n_patterns=2)
```

## Simulate a mechanism

`ampute` removes values from complete data under MCAR, MAR (driven by
columns that stay observed) or MNAR (driven by the value itself). The logistic
intercept is solved so the *expected* missing rate on your sample is exactly
the rate you asked for. Every method in this package is tested against data
that was complete before `ampute` made holes in it, so the right answer is
known.

## Impute and pool

```python
rng = np.random.default_rng(0)
cov = [[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]]
df = pd.DataFrame(rng.multivariate_normal([0, 0, 0], cov, 500), columns=["age", "income", "score"])
df["score"] = 1 + 0.5 * df["age"] + 0.8 * df["income"] + rng.standard_normal(500)
holed = misskit.ampute(df, ["income"], rate=0.3, mechanism="MAR", drivers=["age"], seed=1)

res = misskit.mice(holed, m=20, n_iter=10, seed=2)
misskit.rhat_table(res)  # income    1.054
misskit.pool_ols(res.imputations, "score", ["age", "income"]).summary()
```

```
           estimate  std_error  lower  upper       df  lambda
intercept     0.959      0.052  0.857  1.062  225.364   0.192
age           0.572      0.065  0.444  0.701  108.867   0.341
income        0.707      0.061  0.587  0.826  208.945   0.206
```

The true coefficients are 1, 0.5 and 0.8; both slope intervals contain them.
`lambda` is the share of each estimate's variance that is due to the missing
data.

`mice` supports three univariate methods:

- `"norm"` — Bayesian linear regression with a posterior draw of the
  coefficients and residual variance before predicting (van Buuren 2018,
  Algorithm 3.1). This is the default.
- `"pmm"` — predictive mean matching: impute a value actually observed in a
  donor whose predicted mean is close. Never produces out-of-range values.
- `"norm.nob"` — fitted value plus noise *without* the parameter draw. It is
  there as the contrast: at n = 60 with 60% MAR missingness, the pooled 95%
  interval for a mean covers the truth about 80% of the time against about
  95% for `"norm"` (300 simulated datasets, pinned in the test suite).

## What Little's test can tell you

The test suite measures the test's rejection rate at alpha = 0.05 rather than
asserting it is "about right":

- under MCAR it rejects at close to the nominal rate (checked to within 3.5
  binomial standard errors over 400 simulated datasets);
- under MAR it rejects in over 90% of 100 simulated datasets;
- under an MNAR mechanism that removes the most extreme values of a column in
  both tails, it rejects in fewer than 12% — the pattern means do not move,
  so the test cannot see it.

So a small p-value is evidence against MCAR, and a large one is not evidence
for it. No test on observed data can separate MAR from MNAR.

## When imputation changes the answer

`scripts/evidence.py` simulates `y = 1 + 0.5 x1 + 0.8 x2 + e` with
`corr(x1, x2) = 0.5`, n = 200, and removes 40% of `x2` under three
mechanisms: MCAR; MAR, more often when the outcome `y` is large; and MNAR,
more often when `x2` itself is large. Five analyses estimate the coefficient
on `x2` (true value 0.8) from the same 200 datasets per mechanism. "Full
data" is the oracle fitted before any values were removed. MICE uses
`method="norm"`, m = 10, and is pooled by Rubin's rules.

<!-- evidence:main:start -->
| mechanism | analysis | bias (95% MC interval) | RMSE | coverage of 95% CI (Wilson interval) | mean CI width |
| --- | --- | --- | --- | --- | --- |
| MCAR | full data | -0.003 (±0.012) | 0.084 | 0.940 (0.898-0.965) | 0.324 |
| MCAR | complete case | -0.001 (±0.014) | 0.104 | 0.950 (0.910-0.973) | 0.424 |
| MCAR | mean imputation | -0.097 (±0.015) | 0.146 | 0.870 (0.816-0.910) | 0.439 |
| MCAR | MICE | -0.008 (±0.013) | 0.094 | 0.980 (0.950-0.992) | 0.399 |
| MCAR | MICE without y | -0.324 (±0.010) | 0.333 | 0.190 (0.142-0.250) | 0.494 |
| MAR | full data | +0.002 (±0.011) | 0.080 | 0.950 (0.910-0.973) | 0.323 |
| MAR | complete case | -0.139 (±0.015) | 0.177 | 0.725 (0.659-0.782) | 0.395 |
| MAR | mean imputation | -0.303 (±0.015) | 0.322 | 0.285 (0.227-0.351) | 0.486 |
| MAR | MICE | -0.009 (±0.014) | 0.103 | 0.950 (0.910-0.973) | 0.411 |
| MAR | MICE without y | -0.409 (±0.010) | 0.415 | 0.045 (0.024-0.083) | 0.538 |
| MNAR | full data | +0.010 (±0.011) | 0.078 | 0.960 (0.923-0.980) | 0.325 |
| MNAR | complete case | +0.034 (±0.017) | 0.124 | 0.955 (0.917-0.976) | 0.502 |
| MNAR | mean imputation | -0.105 (±0.016) | 0.157 | 0.945 (0.904-0.969) | 0.545 |
| MNAR | MICE | +0.075 (±0.016) | 0.138 | 0.895 (0.845-0.930) | 0.489 |
| MNAR | MICE without y | -0.305 (±0.011) | 0.315 | 0.460 (0.392-0.529) | 0.592 |
<!-- evidence:main:end -->

What the table shows:

- **Leaving the outcome out of the imputation model is the worst choice in
  every scenario, MCAR included** (bias about -0.32 on a coefficient of 0.8).
  Imputing `x2` from `x1` alone builds in the assumption that `x2` and `y`
  are unrelated, and the analysis then finds that they are less related than
  they are. The outcome belongs in the imputation model.
- **Under MCAR**, complete-case analysis is unbiased. MICE is also unbiased
  and gives narrower intervals (0.399 against 0.424), because it keeps the
  rows where only `x2` is missing.
- **Mean imputation is biased even under MCAR** (-0.097). Filling a
  correlated covariate with a constant weakens its relationship with `y`.
- **Under MAR driven by the outcome**, complete-case analysis is biased
  (-0.139, coverage 0.725) and MICE recovers the coefficient (-0.009,
  coverage 0.950).
- **Under MNAR driven by `x2` itself**, complete-case analysis does better
  than MICE. When missingness depends only on a covariate, dropping those
  rows does not bias the regression of `y` on the covariates. MICE assumes
  MAR, which is false here, and its estimate is biased upward. No analysis
  of the observed data can tell you which of these two worlds you are in.

The bias interval is ±1.96 Monte Carlo standard errors; the coverage interval
is a Wilson interval over the 200 replicates. Repeating the study with five
further seeds (100 datasets each) gives this spread:

<!-- evidence:spread:start -->
| mechanism | analysis | bias: min to max over seeds | coverage: min to max over seeds |
| --- | --- | --- | --- |
| MCAR | full data | -0.016 to +0.012 | 0.97 to 1.00 |
| MCAR | complete case | -0.020 to +0.021 | 0.95 to 0.98 |
| MCAR | mean imputation | -0.120 to -0.076 | 0.83 to 0.94 |
| MCAR | MICE | -0.026 to +0.009 | 0.94 to 0.98 |
| MCAR | MICE without y | -0.333 to -0.314 | 0.14 to 0.24 |
| MAR | full data | -0.010 to +0.004 | 0.95 to 0.98 |
| MAR | complete case | -0.156 to -0.142 | 0.61 to 0.79 |
| MAR | mean imputation | -0.324 to -0.309 | 0.23 to 0.28 |
| MAR | MICE | -0.017 to -0.010 | 0.94 to 0.97 |
| MAR | MICE without y | -0.414 to -0.407 | 0.02 to 0.06 |
| MNAR | full data | -0.001 to +0.006 | 0.92 to 0.94 |
| MNAR | complete case | -0.018 to +0.008 | 0.93 to 0.97 |
| MNAR | mean imputation | -0.153 to -0.125 | 0.82 to 0.88 |
| MNAR | MICE | +0.039 to +0.056 | 0.89 to 0.94 |
| MNAR | MICE without y | -0.336 to -0.317 | 0.33 to 0.46 |
<!-- evidence:spread:end -->

The conclusions above hold at every seed. The MNAR complete-case bias in the
main table (+0.034 ± 0.017) is just outside zero, but no other seed puts it
outside -0.018 to +0.008, so it reads as Monte Carlo noise and not as a bias.

## How far from MAR before the answer changes?

The MNAR row above has no fix that uses only the observed data. What you can
do is ask how large a departure from MAR would have to be before the
conclusion moves. `mice(..., delta={"x2": d})` imputes under MAR and adds
`d` to every imputed `x2` inside each iteration, which is the same as assuming
the missing values are on average `d` higher than comparable observed ones.
`misskit.delta_sensitivity` scans a grid of `d` and pools the coefficient at
each one.

```python
holed = misskit.ampute(df, ["x2"], 0.4, "MNAR", seed=1)  # df: y, x1, x2, n = 300
misskit.delta_sensitivity(holed, "x2", [0, 0.5, 1.0], "y", ["x1", "x2"], term="x2", m=20, seed=2)
```

```
 delta  estimate  std_error  lower  upper
   0.0     0.822      0.103  0.618  1.025
   0.5     0.827      0.081  0.666  0.988
   1.0     0.719      0.068  0.585  0.853
```

On the MNAR scenario (200 datasets; each row uses the same datasets and
imputation seeds):

<!-- evidence:delta:start -->
| delta added to imputed x2 | bias (95% MC interval) | RMSE | coverage of 95% CI (Wilson interval) |
| --- | --- | --- | --- |
| -0.50 | -0.096 (±0.018) | 0.160 | 0.925 (0.880-0.954) |
| +0.00 | +0.059 (±0.017) | 0.134 | 0.945 (0.904-0.969) |
| +0.25 | +0.072 (±0.015) | 0.132 | 0.930 (0.886-0.958) |
| +0.50 | +0.048 (±0.014) | 0.109 | 0.960 (0.923-0.980) |
| +0.75 | -0.002 (±0.012) | 0.085 | 0.960 (0.923-0.980) |
| +1.00 | -0.064 (±0.010) | 0.098 | 0.945 (0.904-0.969) |
| +1.50 | -0.192 (±0.008) | 0.201 | 0.205 (0.155-0.266) |
<!-- evidence:delta:end -->

The missing `x2` values really are higher than MAR predicts, and a shift of
about +0.75 standard deviations removes the bias. The curve is **not
monotone**, though. Bias rises from +0.059 to +0.072 before it falls,
because shifting a subset of a covariate changes its covariance with `y`
linearly in `d` and its variance quadratically. Two different deltas can
therefore give the same estimate. Report the whole grid; a single "tipping
point" is not enough.

```bash
python scripts/evidence.py           # print the tables (about 65 s)
python scripts/evidence.py --check   # exit 1 if README.md has drifted
```

## Layout

| module | contents |
| --- | --- |
| `patterns` | pattern matrix, per-column rates, co-missingness |
| `amputation` | MCAR / MAR / MNAR generators with known ground truth |
| `em` | EM maximum-likelihood mean and covariance for incomplete normal data, checked against Anderson's closed form |
| `mcar` | Little's (1988) MCAR test |
| `mice` | chained-equation imputation: `norm`, `pmm`, `norm.nob` |
| `pooling` | Rubin's rules with Barnard-Rubin degrees of freedom; pooled OLS |
| `convergence` | Gelman-Rubin R-hat across imputation chains |
| `sensitivity` | delta-adjustment scan for departures from MAR |

See [ROADMAP.md](ROADMAP.md) for what comes next.
