# Phase 4 results

Implemented last-available-day persistence, weekly persistence, and a 100-feature, 48-output ridge baseline with JSON model bundles. All five candidates use the identical 1,461 training and 365 point-validation origins; ridge penalties 1, 10, and 100 are compared on validation RMSE. Alpha 10 is selected.

| Model | Validation MAE, MW | Validation RMSE, MW |
| --- | ---: | ---: |
| Last available day | 2732.0662 | 3642.8482 |
| Weekly persistence | 2035.0216 | 2804.5681 |
| Selected ridge | 1389.1997 | 1892.5422 |

The full report is `reports/baseline-validation-v1.json`, including per-slot errors, peak error definitions, timings, and runtime. These are model-selection results. No final 2024 test metrics have been computed.

The real saved validation forecast for target 2022-02-10 is packaged with its input window, source hashes, model/dataset/truth versions, and content ID. FastAPI returns a typed 48-point artifact. `/forecasts` renders the chart, model selector, exact boundaries, slot readout, and provenance. Missing/corrupt artifacts fail explicitly.

Local checks verify exact lag positions, hand-computed metrics, fitted ridge output, reload equality, corruption, and edited forecast rejection. HTTP tests require the API extra and run in CI. Browser tests use both real servers at desktop/mobile sizes. The live data workflow reruns the whole frozen dataset and baseline comparison, permitting 1e-6 MW numerical error across CPU/BLAS platforms.

See the [walkthrough](../learning/phase-4.md). Final verification run links are recorded in the Phases 3–5 completion report.
