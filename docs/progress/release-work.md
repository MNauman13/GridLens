# End-to-end release status

Updated 28 November 2026. Executed evidence is retained under `reports/release`, with model/training artifacts kept separately from Git.

| Phase | Executed outcome | Remaining gate |
| --- | --- | --- |
| I0–I1 | Verified baseline recovery, frozen datasets and real browser/server inference | None for the original preview |
| I2 | Request-local full KV decoding, B1/B4 parity, cancellation/isolation tests and 11,000 retained timings. B1 prepared median 34.04 to 22.07 ms, 1.54× ratio, maximum drift 0.023 MW | No GPU claim |
| I3 | Eager, TorchScript, FP32 ONNX and dynamic INT8 at B1/B4 and one/two threads. 3,200 warm timings and 24 cold process trials. FP32 ONNX B1 median about 79% lower at two threads | Runtime scopes remain separate from public-network latency |
| I4 | Four neural configurations, seeds 17/42/101, twelve fits, three ridge penalties and two persistence baselines. Full 2022 validation selected TimeXer with 24-slot patches | No post-test reselection |
| I5 | Fixed/rolling/NCC causal intervals, matured feedback, checkpoint restoration, bounded TTA and raw validation replay. Rolling selected; NCC's 0.22% WIS gain missed its 5% promotion threshold | NCC remains experimental |
| I6 | Durable PostgreSQL/SQLite queue, real process-death recovery, fencing, owner tickets, retries/cancellation, verified SQLite and PostgreSQL backup/restore. All 1,800 requests completed in a 15-minute offered-load test; POST-to-result p95 77 ms, read p95 8.3 ms | Compose compute, ownership, restart and idempotency passed in CI run 46141993636. Local load is not public availability |
| I7 | Live computation, causal replay with browser residual quantiles, CSV download, results and architecture pages. Desktop/mobile browser tests and automated WCAG checks | Public desktop/mobile inference, replay restoration, frozen results and WCAG scans passed at commit 86b0f444 |
| I8 | Freeze committed in `cc39b120` before scoring 366 days from 2024. All seventeen declared models evaluated once, with no refitting. Selected TimeXer RMSE 1967.52 MW versus selected ridge 2003.71 MW; MAE slightly worse and bootstrap intervals include zero | No broad accuracy-superiority claim |
| I9 | Public smoke, source capture/forecast scheduling and release-aware probe-window assessment implemented. A private 136-member archive restored the selected model and all calibration states successfully | Seven real days of monitoring cannot be completed in this session |

The selected rolling intervals achieved test marginal coverage of 78.59%, 88.81% and 95.18% against 80%, 90% and 95% nominal levels. The method was selected before test evaluation. Fixed and NCC methods missed test coverage tolerances; they are not silently promoted or retuned.

Publishing the newly fitted public model archive was blocked by automatic approval review because the earlier approval covered the original baseline archive. The public live preview retains its original pinned models until explicit approval for the new archive arrives. The uncertainty replay and recorded study results identify their own separately frozen selected model.
