# Roadmap

## Phase 1 — Missingness patterns and amputation
- [x] Pattern matrix, per-column rates and co-missingness summary
- [x] Amputation generators for MCAR, MAR and MNAR with a documented mechanism
- [x] Little's MCAR test, validated by simulated size and power

## Phase 2 — Multiple imputation
- [x] MICE with Bayesian linear regression draws (proper imputation)
- [x] Rubin's rules pooling with Barnard–Rubin degrees of freedom
- [x] Property tests: observed values are never altered
- [x] Chain convergence diagnostic (R-hat)

## Phase 3 — Evidence: when imputation changes the answer
- [x] Simulation of bias and CI coverage for complete-case, mean and MICE under each mechanism
- [x] Evidence script that prints the README table and fails when it drifts
- [x] Seed-spread report

## Phase 4 — Release
- [x] Continuous integration: lint, type check, tests, evidence
- [x] Release workflow via trusted publishing

## Phase 5 — Beyond the linear model
- [ ] Logistic-regression imputation for binary columns
- [x] Delta-adjustment sensitivity analysis for MNAR (how far the answer moves per unit of departure from MAR)
- [ ] Documentation site
