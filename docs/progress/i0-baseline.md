# I0: recovered executable baseline and frozen upgrade protocol

Implemented and measured on 24 November 2026. The original Phase 8 source/data/model decisions remain intact; I0 establishes a new executable starting point for subsequent work.

## What changed

- Added a recovery runner that verifies all eight pinned raw source hashes and the exact rebuilt dataset freeze, refits the selected ridge and regenerates only the existing numerical Transformer and patch-48 TimeXer configurations.
- Added a frozen 32-origin validation corpus, warm benchmark protocol, metric registry and raw-observation schema.
- Measured 1,000 uncached warm bundle predictions per model, 1,000 prepared-tensor forward calls per neural model and 20 fresh-process starts per model.
- Added explicit-file model archiving, checksum inventories, restored-model prediction checks and a verified separate persistent copy of the deployable archive.
- Added regression tests for metrics, warmup exclusion, date eligibility, immutable reports and archive safety; CI now checks I0 lineage and the new scripts.
- Updated the design decision to allow verified accelerated runtimes alongside the manual reference and require a public real-inference release in I1.

The accepted original training run lists no remaining downloadable artifacts. The new weights are regenerated, not recovered byte-for-byte from that expired artifact. They have actual distinct model identities. No original report, public example or champion was replaced.

## Measurements

Environment: Linux, Python 3.12.14, Torch 2.14.1+cpu, NumPy 2.5.4, two Torch threads, nine visible/affinity CPUs. CPU model and observed memory are recorded in the report. Results are environment-specific.

| Model | Validation MAE (MW) | Validation RMSE (MW) | Warm bundle p50 (ms) | Prepared engine p50 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Ridge alpha 10 | 1,390.80 | 1,895.45 | 1.982 | N/A |
| Numerical Transformer | 1,853.65 | 2,548.36 | 37.809 | 39.406 |
| TimeXer patch 48 | 1,391.24 | 1,877.71 | 1.619 | 1.107 |

Bundle and engine columns are separate benchmark runs, not nested stage timings from the same calls; measurement noise can make the numerical Transformer's engine median slightly higher. Do not subtract these medians to estimate preprocessing overhead. Warm bundle timing includes feature transformation, tensor preparation, inverse scaling and contract validation. Its scope differs from the original five-call engine-only pilot.

Fresh-process medians including startup/import/load/predict/exit were approximately 2.44-2.54 seconds. Verified bundle load alone was approximately 2.38 ms for ridge, 9.25 ms for the Transformer and 11.26 ms for TimeXer. These measurements use local artifacts; they do not establish cold network/disk-cache performance. All 20 new-process trials per model had identical prediction fingerprints.

Ridge reproduced the original validation RMSE to floating-point precision. Regenerated Transformer and TimeXer RMSE differ from their original pilot values by about +0.51 MW and +1.08 MW respectively. Maximum drift on the committed demonstration is about 87.97 MW and 50.51 MW. These are different fitted weights, not a violation of a same-weight runtime parity claim. The new bundles passed exact save/reload checks in the original trainer and <=0.05 MW single-origin versus batch checks on all 32 benchmark origins.

The final archive contains 17 explicitly listed files, is 946,429 bytes and has SHA-256 `860c78208df49838edaa998b98c3b3ca49798a9538c0f1796adf7e1243d6b9c4`. It omits raw source CSVs, training checkpoints and secrets. Extraction, member hashes, model loading and actual prediction fingerprints were verified for all three models. A separate persistent copy is retained for the owner; other developers can regenerate it using the runbook rather than needing access to that private copy.

## Evidence

- [Baseline reproduction and full metrics](../../reports/i0-baseline-reproduction-v2.json)
- [Cold-process and prepared-engine measurements](../../reports/i0-runtime-supplement-v2.json)
- [Archive inventory and member hashes](../../reports/i0-artifact-inventory-v2.json)
- [Verification record](../../reports/i0-verification-v2.json)
- [Frozen upgrade manifest](../../configs/upgrade_experiment_v2.json)
- [Metric definitions](../../configs/metric_registry_v2.json)
- [Reproduction operations](../operations/i0-baseline.md)

Local checks: 81 Python tests (80 passed, one CUDA-only skip), strict typing for 24 source files, Ruff, lock verification, contract-schema verification, original neural-report integrity checks, I0 lineage checks, three Node unit tests and the archive restore smoke check passed. No frontend code changed; production build and browser checks were not rerun locally. The existing full CI workflow remains in place.

## Limits and next increment

There is no new cache optimisation, quantised runtime, live forecast endpoint, calibration or multi-seed superiority claim here. Process-wide memory and fresh-process memory have explicit scopes. The dataset builder verifies all original role fingerprints, but only 2022 point validation receives model predictions/metrics. The 2024 test is still locked and unscored.

I1 should load these verified bundles into a bounded worker and let a visitor request an actual forecast from the web application. The saved examples remain available as labelled examples while that live path is implemented.
