# Phases 3–5 implementation and verification

## Delivered

| Phase | Result |
| --- | --- |
| 3 | Frozen eligible observations, captured-time filtering, causal fills/missingness/age, exact context/gap/targets, training-only scaling, chronology and label-maturity guards, reproducible manifests |
| 4 | Persistence and ridge baselines, checked bundles, full eligible validation comparison, typed saved-forecast API, real-data browser chart and provenance |
| 5 | Manual scaled/multi-head attention, causal/padding masks, positional buffers, numerical/gradient oracles, exact interactive calculations and head/position animations |

## Data and fitting boundaries

The annual sources have 140,256 actual intervals. Accepted origins are 1,460 point-training, 364 point-validation, 180 calibrator-training, 91 calibrator-validation, 92 initialisation, and 366 locked-test. Four fitting/selection boundary origins are excluded because labels would mature after the next role's first issue. No dates were excluded based on model errors.

The final 2024 evaluation remains locked. Annual data is revised historical outturn with unknown historical publication vintages.

| Baseline | MAE, MW | RMSE, MW |
| --- | ---: | ---: |
| Last available day | 2734.75 | 3646.32 |
| Weekly persistence | 2037.39 | 2807.36 |
| Ridge, alpha 1 | 1411.82 | 1916.25 |
| Ridge, alpha 10 | 1390.80 | 1895.45 |
| Ridge, alpha 100 | 1401.82 | 1913.63 |

Ridge selects alpha 10 from the predeclared 1/10/100 candidates. These are validation/model-selection results, not final generalisation metrics. The saved demo target is 2022-02-10.

The source-to-baseline workflow requires exact equality with the approved dataset manifest, original annual hashes, identical eligible origins, the same selected penalty, and absolute error agreement within 1e-6 MW. Reference generation never pushes automatically. See [live data/baseline acceptance](https://github.com/MNauman13/GridLens/actions/workflows/neso-audit.yml).

The [Windows/Linux calendar run](https://github.com/MNauman13/GridLens/actions/runs/40020171081) verifies every phase of the daily, weekly, and annual cycles exactly. Fixed Decimal arithmetic resolves detected OS math-library rounding differences without weakening hash checks.

## Acceptance gates

The [foundation workflow](https://github.com/MNauman13/GridLens/actions/workflows/ci.yml) records checks for each exact commit. Required gates are 64 Python checks (only conditional CUDA may skip), sixteen real Chromium tests, three Node tests, schema and visual-fixture freshness, lint, strict Python/TypeScript types, the production build, and CPU PyTorch verification.

[The source verification run](https://github.com/MNauman13/GridLens/actions/runs/40020171125) passed the source tests and all sixteen browser scenarios. The final live-reference run additionally verifies the committed embargoed artifacts.

The browser scenarios cover real contract requests, timeline controls, fixture outage, settlement conversion, real saved forecasts/provenance, forecast outage, attention masks/head/query controls, and positional animation, each at desktop/mobile widths. No manual screenshot review or CUDA execution is claimed.

## Use and limits

- `/forecasts`: real saved baseline forecast.
- `/learn/settlement`: 46/48/50-period conversion.
- `/learn/attention`: checked attention arithmetic and positional animation.

The workspace disconnected during the final phase. Completion continued through normal GitHub commits and hosted verification. Preserve and compare uncommitted files in the older local checkout before reconciling it with remote main; no reset, clean, or force-push was performed.

Teaching attention uses identity projections; its patterns are not learned electricity explanations. The complete encoder/decoder and sequence trainer are Phase 6. No trained Transformer result is claimed.

[Phase 3](../learning/phase-3.md) · [Phase 4](../learning/phase-4.md) · [Phase 5](../learning/phase-5.md)
