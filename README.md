# misskit

Missing-data diagnostics and multiple imputation for tabular data, with a
simulation study that shows when imputation changes the answer and when it
cannot.

```bash
pip install -e ".[dev]"
pytest
```

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
cov = [[1, .5, .3], [.5, 1, .4], [.3, .4, 1]]
df = pd.DataFrame(rng.multivariate_normal([0, 0, 0], cov, 500), columns=["age", "income", "score"])
df["score"] = 1 + 0.5 * df["age"] + 0.8 * df["income"] + rng.standard_normal(500)
holed = misskit.ampute(df, ["income"], rate=0.3, mechanism="MAR", drivers=["age"], seed=1)

res = misskit.mice(holed, m=20, n_iter=10, seed=2)
misskit.rhat_table(res)                       # income    1.054
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

See [ROADMAP.md](ROADMAP.md) for what comes next.
