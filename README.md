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

See [ROADMAP.md](ROADMAP.md) for what comes next.
