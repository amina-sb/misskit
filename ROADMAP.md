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
- [ ] Simulation of bias and CI coverage for complete-case, mean and MICE under each mechanism
- [ ] Evidence script that prints the README table and fails when it drifts
- [ ] Seed-spread report

## Phase 4 — Release
- [ ] Continuous integration: lint, type check, tests, evidence
- [ ] Release workflow via trusted publishing
