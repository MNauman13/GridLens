# I2: checked transformer inference optimisation

Implemented 25 November 2026. Current code was confirmed clean and pushed at b19a787 before starting I2. I2 preserves the original model/training/bundle sources, recovered model identities, feature snapshots, scaling and 2024 test lock.

## Result

The numerical Transformer improved on this host; TimeXer did not show a convincing gain. The reference stays the default. The optional Python worker can expose checked SDPA after an explicit opt-in and a startup parity gate. The public browser continues using its original WASM runtime, with the Python measurements shown separately at `/experiments#inference`.

| Complete bundle prediction | Numerical Transformer | TimeXer |
| --- | ---: | ---: |
| Manual p50, ms | 44.047 | 1.649 |
| SDPA p50, ms | 37.535 | 1.644 |
| Manual p95, ms | 51.698 | 2.546 |
| SDPA p95, ms | 43.275 | 2.467 |
| Median latency reduction | 14.8% | 0.3% |
| p50 speed ratio, manual / SDPA | 1.173× | 1.003× |
| Mean-speedup descriptive 95% interval | 1.167–1.183× | 0.977–1.032× |
| Sequential forecasts/s, manual | 23.19 | 533.98 |
| Sequential forecasts/s, SDPA | 27.25 | 537.28 |
| Maximum prediction drift, MW | 0.015625 | 0.00390625 |
| 32-origin RMSE, manual, MW | 2601.786130 | 1899.053829 |
| 32-origin RMSE, SDPA, MW | 2601.785528 | 1899.053845 |

These are fresh paired comparisons on the same AMD EPYC 9V74 host, PyTorch 2.14.1+cpu, two Torch threads, float32 and batch size one. They are not comparisons against I0 measurements on another host. Complete bundle timing includes preprocessing, model execution, inverse scaling and output contract validation; HTTP, queueing and browser rendering are excluded. Sequential capacity is 1000/mean latency and is not concurrent service throughput. Tiny error changes are floating-point differences, not an accuracy improvement.

Prepared-tensor p50 changed from 42.752 to 36.415 ms for the numerical Transformer and from 1.150 to 1.158 ms for TimeXer. Prepared-tensor p95 changed from 51.989 to 42.211 ms and from 1.597 to 1.603 ms respectively. TimeXer's intervals include a ratio of one in both scopes, so a speed improvement is not established for that model.

Each runtime/model/scope has 10 excluded warmups and five repeats of 200 measured calls, with seed-17 runtime order shuffled within each pair. The same 32 frozen validation origins are cycled in matched order. There are 8,000 raw timing samples in the final experiment. Descriptive intervals resample five paired repeat blocks with 2,000 bootstrap draws; they do not establish general performance across hardware or model-accuracy significance.

## Why the second candidate helped

The first candidate retained overflow rejection by computing a full score-matrix preflight before SDPA. That duplicated work: complete-bundle p50 ratios were 0.959× for the numerical Transformer and 0.991× for TimeXer. The unsuccessful trial retains its source, complete predictions, memory trials and 8,000 timing samples under [preflight-trial](../../reports/i2/preflight-trial).

The revised adapter proves ordinary projected values cannot overflow using a conservative maximum-range bound for larger score grids. If the bound cannot prove safety, it executes the exact original preflight rather than rejecting the input or dropping validation. Small grids keep the preflight directly. The numerical Transformer attends over 336 historical tokens and repeatedly decodes a growing prefix, so avoiding redundant large score checks matters. The patch-48 TimeXer has just eight endogenous/global tokens and tiny cross-attention queries; its workload is too small to justify the same speed claim. No decoder cache or architecture change is included.

Profiling observed `aten::scaled_dot_product_attention` and `aten::_scaled_dot_product_flash_attention_for_cpu`. These are CPU observations; no CUDA benchmark or GPU speed claim is made. Projection weights, parameter/state names and trained-model IDs remain identical; runtime identity is independently versioned.

## Memory, checks and limitations

Median process lifetime peak RSS across three fresh interpreters per condition was 320.71 MiB manual versus 315.60 MiB SDPA for the numerical Transformer, and 315.05 versus 315.38 MiB for TimeXer. These include imports, the corpus and model loading/inference. They do not isolate tensor allocation or establish a robust memory saving. Full per-process measurements, model/feature identities and every forecast are in [the report](../../reports/i2/benchmark.json); raw paired timings are in [timings.jsonl.gz](../../reports/i2/timings.jsonl.gz).

The original sources still pass their SHA-256 pins. The measured adapter and driver hashes are checked before publishing the web snapshot and in CI. The report records the I1 parent commit plus a dirty working tree during I2 development; its exact new adapter/driver source hashes pin the measured implementation. No unmeasured commit identity is substituted into that record.

Tests cover masks and future/padding invariance, optional attention inspection, whole-model and token generation parity, unchanged independent weights, training rejection, nonfinite/overflow inputs, an inconclusive range bound with valid scores, optional worker runtime selection, session ownership and migration of an existing I1 SQLite database. The range certificate is conservative; score-matrix fallback remains possible, so quadratic validation memory has not been universally eliminated. The Python service remains single-process and separately hosted, as in I1.

See [ADR 0009](../decisions/0009-checked-sdpa-inference.md) and [the I2 runbook](../operations/i2-sdpa-inference.md). This experiment supports an optional numerical-Transformer inference optimisation and explicit rejection of a blanket TimeXer speed claim. It changes neither the selected forecasting champion nor the original validation conclusions.

Raw timing streams are stored losslessly as `timings.jsonl.gz`; report SHA-256 values refer to the decompressed JSONL bytes. The evidence checker verifies that stream before recomputing its aggregates.

Local verification: 95 Python tests (94 passed, one CUDA-only skip), 35 desktop/mobile browser tests passed (one duplicate mobile parity matrix skipped), three Node tests, lint/types, source/schema checks, evidence verification, production build and zero production dependency audit findings. Browser tests run actual SDPA jobs through the optional Python worker and inspect the separate runtime identity.
