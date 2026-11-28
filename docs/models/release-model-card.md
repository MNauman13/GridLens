# Release model and evaluation card

GridLens predicts NESO National Demand in MW for 48 half-hours on the following UTC day. The issue time is 09:30 UTC. Inputs use seven complete days ending at the start of the issue day. The entire issue day is excluded from observed model inputs.

## Selection

Training uses 2018–2021. All four neural configurations were fitted with seeds 17, 42 and 101, at most 50 epochs and patience eight. Configuration selection used mean full-2022 validation RMSE across seeds. TimeXer with 24-slot patches ranked first, with mean RMSE 1808.29 MW and sample seed SD 7.05 MW, compared with 1895.45 MW for the validation-selected ridge baseline. The lowest-validation-RMSE fitted seed within that configuration was frozen, as declared in the study plan.

The numerical Transformer, LSTM, TimeXer patch 48, all seeds, ridge penalties 1/10/100 and persistence baselines remain in the reports. The selected model was not changed after test inspection.

## Frozen test

The final manifest has identity `9ba01343407c54d217aa641da0cad8d200ab3d269b8054a18e082ac6b5ac403f`. Commit `cc39b120ac2c6bb90914ffd26af37e0fdae12984` contains the exact freeze and evaluation code before the test campaign. All seventeen candidates were evaluated on 366 eligible 2024 days, without refitting.

The selected TimeXer achieved RMSE 1967.52 MW, compared with 2003.71 MW for the selected ridge model, a 1.81% reduction. Its MAE was 1518.39 MW, compared with ridge's 1516.58 MW. The paired daily MAE difference was +1.81 MW. Descriptive 95% circular block-bootstrap intervals were −71.60 to +68.01 MW with seven-day blocks and −75.50 to +80.82 MW with fourteen-day blocks. Both include zero. These results do not establish broad forecasting superiority.

## Uncertainty

Point weights were frozen before calibrator fitting. January–June 2023 fitted the NCC calibrator, July–September selected the interval method, and October–December initialised delayed feedback without refitting weights. Rolling residual bands were selected. Although NCC passed validation coverage checks, its 0.22% WIS improvement over rolling missed the declared 5% promotion threshold.

| Method | Test 80% coverage | Test 90% coverage | Test 95% coverage | Test WIS MW |
| --- | --- | --- | --- | --- |
| Fixed residual | 73.38% | 84.11% | 90.93% | 617.33 |
| Rolling residual, preselected | 78.59% | 88.81% | 95.18% | 595.21 |
| NCC, experimental | 77.63% | 86.60% | 91.01% | 595.74 |

Coverage is marginal per half-hour, not simultaneous daily coverage. Actuals mature on the day after the target at 09:00 UTC. The implementation rejects future feedback, conflicting duplicates and backwards clocks. Independent NCC code includes documented application adaptations, bounded test-time correction and explicit nesting repair. No original-paper coverage guarantee is transferred.

## Scope and limitations

Historical data is revised outturn data. It does not prove which versions were available at the original issue time. A separate prospective capture workflow records actual retrieval times and checks availability before producing an operational ridge forecast. Missing or quarantined windows cause a labelled skip, rather than fabricated observations.

The live public preview and offline selected model have separate identities until the new fitted archive is authorised for public publication. Runtime studies use the exact I0 weights and do not automatically change either model selection or precision. CPU measurements do not establish GPU, energy, carbon or multi-region availability benefits. Fixed-date holiday slices are explicitly labelled proxies, not a complete UK bank-holiday evaluation.

Raw test curves and metrics are in `reports/release/locked-test.json.gz`. Decompress with Python's gzip module. The report references the committed freeze, preserves every candidate and records no test refitting.
