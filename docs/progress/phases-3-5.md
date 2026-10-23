# Phases 3–5 implementation and verification

## Delivered

| Phase | Result |
| --- | --- |
| 3 | Eligible frozen observation versions; captured-time filtering; causal two-slot filling with missingness/age; fixed feature order; exact 336/48-gap/48-target windows; training-only scaling; split ownership; reproducible manifests |
| 4 | Last-available-day, weekly, and ridge baselines; checked model bundles; complete 2022 validation comparison; typed saved-forecast API; real-data browser chart and provenance |
| 5 | Manual scaled/multi-head attention, causal/padding masks, positional buffers, numerical and gradient oracles, interactive head/query/mask calculator, attention/position animations |

## Data and model evidence

The annual data has 140,256 actual intervals. Dataset membership has 1,461 point-training and 365 point-validation origins with no drops. Calibration roles and 366 final-test origins are frozen separately. The final 2024 model evaluation remains locked.

| Baseline | Validation MAE, MW | Validation RMSE, MW |
| --- | ---: | ---: |
| Last available day | 2732.066210 | 3642.848166 |
| Weekly persistence | 2035.021632 | 2804.568086 |
| Selected ridge, alpha 10 | 1389.199691 | 1892.542181 |

These are validation/model-selection results. Ridge uses a fixed 100-feature lag/calendar design and fits only 2018–2021. The saved demo target is 2022-02-10.

The [live data-to-baseline verification](https://github.com/MNauman13/GridLens/actions/runs/40018056952) passed at commit 24d55ceceba4c98eb842459ca3d185f67ae2e17e. It independently retrieved original CSVs, verified frozen hashes, rebuilt the exact dataset manifest, fitted all five baseline candidates, verified bundle reloads, and reproduced all reference errors within 1e-6 MW.

The [Windows/Linux calendar verification](https://github.com/MNauman13/GridLens/actions/runs/40018057101) passed at the same commit. Labels and measurement scaling were already identical; fixed-precision calendar arithmetic resolves the detected platform drift without relaxing dataset SHA checks. The manual review workflow cannot push and no longer runs automatically.

## Test acceptance

At the preceding implementation commit, all 61 Python tests completed successfully except the explicitly conditional CUDA check. CPU attention parity/gradients, HTTP behavior, artifact integrity, leakage invariants, and dataset/ingestion fixtures passed. The current source additionally checks genuine self-attention oracle parity and exact calendar quadrants.

The browser suite contains eight scenarios at desktop/mobile widths: contract round-trip, timeline, fixture outage, settlement lesson, saved forecast/provenance, forecast outage, attention controls/masks, and position animation. The final 16-test browser acceptance and source checks are tracked by the [foundation workflow](https://github.com/MNauman13/GridLens/actions/workflows/ci.yml). Final run evidence is recorded after this source is verified.

## Use and limits

- `/forecasts`: saved real-data baseline forecast.
- `/learn/settlement`: 46/48/50-period conversion.
- `/learn/attention`: checked numerical attention and positional animation.

The workspace disconnected during the final phase; completion continued through normal GitHub Git-object commits and clean hosted runners. The earlier disconnected checkout may contain uncommitted lesson files. Preserve and compare those files before fetching/reconciling the updated main branch; no reset or clean was attempted. Shared history was not force-pushed.

No manual screenshot review or CUDA execution is claimed. Annual observations are revised historical data with unknown historical publication vintages. The attention patterns are transparent teaching fixtures, not learned electricity explanations. Full encoder/decoder construction is Phase 6; no trained Transformer result is claimed.

[Phase 3 walkthrough](../learning/phase-3.md) · [Phase 4 walkthrough](../learning/phase-4.md) · [Phase 5 walkthrough](../learning/phase-5.md)
