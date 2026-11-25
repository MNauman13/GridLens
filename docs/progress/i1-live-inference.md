# I1: bounded live inference preview

Implemented 25 November 2026. I1 adds fresh point inference; intervals, calibration and model selection remain later work. The public deployment uses browser CPU inference; the separately runnable Python reference service has not been provisioned on a public model host.

Public preview: https://gridlens-mnaumansiddiqui06-6472s-projects.vercel.app/live

Anonymous hosted-browser checks confirmed two TimeXer dates produce new execution IDs and different forecasts, the ridge baseline also runs, the unconnected Python option is disabled, and no browser JavaScript errors occurred. [Recorded deployment verification](../evidence/i1/public_smoke.json) pins the tested deployment/commit; public endpoints also returned HTTP 200 without authentication.

## Delivered behavior

Visitors select a trained TimeXer transformer or ridge baseline and one of 32 fixed 2022 target dates. Each click computes 48 predictions, records a unique execution ID, and shows execution location, runtime, model identity, logical issue time/history cutoff, prediction time, pipeline time, model initialisation status and selected-day MAE. Observed target values are displayed only for comparison. Saved-example pages remain explicitly labelled. Browser inputs and model files are SHA-256 checked before use; corrupt assets fail visibly without a replacement forecast. The 2024 test is not selectable or scored.

The Python service loads and verifies original I0 bundles once, executes through a bounded single-worker queue, persists state/results in SQLite and limits access by random HttpOnly session cookie. Readiness fails for missing/corrupt artifacts. Queued/running jobs interrupted by a restart fail explicitly. Eight active jobs globally, one/session, 60 submissions/session/minute and 240 global/minute bound this preview; these are not production identity-based abuse controls.

## Recorded evidence

| Measurement | Ridge | TimeXer |
| --- | ---: | ---: |
| Worker predictions (32 origins × 3 repetitions) | 96 | 96 |
| Failures | 0 | 0 |
| Prediction p50, ms | 2.332 | 1.826 |
| Prediction p95, ms | 3.299 | 2.664 |
| Queue wait p50, ms | 0.433 | 0.490 |
| Queue wait p95, ms | 0.532 | 0.910 |
| Submit to persisted result p50, ms | 4.426 | 3.073 |
| Submit to persisted result p95, ms | 5.558 | 4.845 |

This is one local Linux CPU process with PyTorch 2.14.1+cpu, Python 3.12.14 and two Torch threads. Three repetitions include first predictions; no warmup exclusion. Submission timing includes polling and persistence, excludes HTTP/network/rendering, and is not public service latency. Verified service initialisation took 81.566 ms after importing dependencies. Process peak RSS was 254,676 KiB, including imports and both resident models. These are initial serving measurements, not speedups over I0 or a load test.

Portable ONNX CPU versus original PyTorch maximum drift was **0.006913 MW across all 32 dates** against a 0.5 MW acceptance threshold. See [raw export parity](../evidence/i1/portable_parity.json) and [raw worker samples](../evidence/i1/live_worker.json). Browser WASM parity was also checked end to end against the Python service on all 32 dates: maximum drift **0.009688 MW** using UI values rounded to 0.01 MW. See [browser parity and visible-result samples](../evidence/i1/browser_parity.json). The local desktop first click took 927 ms; subsequent click-to-visible samples were 51–85 ms, including automation/rendering overhead. These are not public network measurements. Artifact SHA-256, manifest SHA-256, catalog identity and immutable download URL are pinned in [the artifact specification](../../configs/i1_demo_artifact.json). The ZIP is 1,020,577 bytes; the browser also downloads the separately installed ORT runtime (approximately 15 MB uncompressed WASM). No performance claim excludes that first-load cost from pipeline timing.

Validation includes 87 Python tests (86 passed, one CUDA-only skip), model/asset integrity failures, session isolation, bounded queue admission, worker failures, persisted results and restart handling. Lint, types, schema/source-pin checks, Node tests and the production build pass. Browser tests also cover server-mode UI and cross-origin admission. Asset fetches use digest-specific cache keys and 30-second network deadlines; server polling shares one 30-second request deadline.

The inherited Next.js 16.4.2 dependency triggered a build security advisory and was updated to 16.4.6 with its matching ESLint configuration. Production dependency audit then reported zero vulnerabilities.

## Boundaries and next work

This preview meets anonymous real inference through the browser, with a tested reference API available for separate hosting. It does not claim a publicly hosted Python worker, contemporary source capture, intervals, uncertainty calibration, reduced model error or proven acceleration. SQLite is single-process only; concurrent worker leases, stronger admission controls, periodic retention cleanup, cancellation and production service SLOs remain I6. See [ADR 0008](../decisions/0008-portable-live-preview.md) and [the runbook](../operations/i1-live-inference.md).

I2 should introduce established transformer attention/runtime candidates beside the frozen manual reference, evaluate numerical parity first, and compare latency/memory on the same hardware, feature windows and measurement scopes. A new runtime must not receive an accuracy improvement claim from altered data, training or selection. Preserve all original source and model identities while adding separately identified adapters.
