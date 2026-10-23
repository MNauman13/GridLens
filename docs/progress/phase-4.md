# Phase 4 results

Implemented last-available-day persistence, weekly persistence, and a 100-feature, 48-output ridge baseline with checked JSON model bundles. All candidates share 1,460 eligible training origins and 364 point-validation origins. Label-maturity embargoes prevent fitting or model selection from consuming observations unavailable at the following role's first issue.

| Baseline | MAE, MW | RMSE, MW |
| --- | ---: | ---: |
| Last available day | 2734.75 | 3646.32 |
| Weekly persistence | 2037.39 | 2807.36 |
| Ridge, alpha 1 | 1411.82 | 1916.25 |
| Ridge, alpha 10 | 1390.80 | 1895.45 |
| Ridge, alpha 100 | 1401.82 | 1913.63 |

The predeclared penalties are 1, 10, and 100; alpha 10 minimises validation RMSE. These are selection-set results, not final test results. The 2024 evaluation remains locked. `reports/baseline-validation-v1.json` includes per-slot errors, peaks, timings, runtime, and model identities.

The packaged real validation forecast for 2022-02-10 has a typed 48-point response, exact input/cutoff/target boundaries, source/model/dataset/truth identifiers, and a verified content ID. `/forecasts` provides the chart, model selector, slot readout, and provenance.

Tests verify lag eligibility, hand metrics, fitted output, save/reload equality, corrupt-artifact rejection, HTTP responses, and real desktop/mobile browser interactions. Live CI independently rebuilds the source-frozen dataset, reruns all five candidates, and requires the same selected penalty and absolute metric agreement within 1e-6 MW.

[Walkthrough](../learning/phase-4.md) · [Acceptance record](phases-3-5.md)
