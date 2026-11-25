# I1 live inference

## Public browser preview

```bash
npm ci
npm run build
npm start
```

Open `/live`. The prebuild script downloads the pinned 1,020,577-byte demo archive, verifies its SHA-256 and manifest, and copies verified features/model files and the installed ORT WASM runtime to public assets. No Python or training job runs during the public page request. The public artifact contains historical NESO inputs, observed targets, fitted ridge coefficients, the TimeXer graph and reference bundles. Never put private data into this artifact.

A click computes 48 new values, assigns a new execution ID, and displays model execution time, pipeline time, model-load status and selected-day MAE. These are validation examples, not evidence of generalisation on the locked test set. Browser model execution excludes preprocessing/model initialisation; pipeline time includes input verification, model initialisation if needed and prediction, but excludes React rendering. Browser and server numbers must not be compared as speedups.

For Vercel, use root directory `apps/web`, Node 24, and enable source files outside the root directory. The checked-in Vercel configuration installs from the workspace root so build-time tools and the shared export preparation script are available. No reference-service URL is configured for the browser-only public preview.

## Python reference mode

```bash
node scripts/prepare_live_assets.mjs
uv sync --frozen --extra api --extra ml --group dev
GRIDLENS_LIVE_ASSETS=artifacts/i1 GRIDLENS_JOB_DB=reports/local/i1/jobs.sqlite3 uv run --frozen --extra api --extra ml uvicorn gridlens.api:app --host 127.0.0.1 --port 8000 --workers 1
# In another terminal:
GRIDLENS_API_URL=http://127.0.0.1:8000 npm start
```

The Next catalog enables the Python selector only if the reference catalog matches its own. Liveness is `/v1/health/live`; readiness is `/v1/health/ready`. Missing/corrupt bundles give readiness 503. Jobs use POST `/v1/forecast-jobs`, then GET `/v1/jobs/{id}`; ownership is a hashed random HttpOnly session cookie. Do not expose the reference port directly without host-level origin/access controls. The Next proxy rejects cross-origin POST requests and forwards Retry-After. On HTTPS deployments set GRIDLENS_COOKIE_SECURE=1 for the reference service. Mount the SQLite path on persistent writable storage and use one process only; see ADR 0008 for limitations.

## Verification and measurement

```bash
uv run --frozen --extra api --extra ml python -m unittest discover -s tests -v
npm run lint
npm run typecheck
npm run test:unit
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
uv run --frozen --extra api --extra ml python -m scripts.benchmark_live
```

For an existing compatible Chromium binary, set PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH. Browser tests include real models on desktop/mobile, changed origins and execution IDs, checksum failures, and all 32 browser TimeXer origins against the actual Python service. CI installs the pinned demo artifact before Python integration tests.

Re-exporting requires the retained I0 bundles and data plus `uv sync --frozen --extra ml --extra export`. Run `python -m scripts.prepare_live_demo` through that environment into a new output directory if immutable reports already exist. Export dependencies are not required for serving. Publish a new immutable archive and update its pinned specification only after parity passes; do not overwrite existing model IDs or source pins.

The single-worker benchmark records every prediction, queue wait and submission-to-persisted-result duration, plus startup time, thread count, software, process peak RSS and failures. It excludes HTTP/network/rendering and uses three runs per origin, including each model's first prediction. Retain raw samples, not just aggregates. Production request SLOs, cancellation/restart recovery beyond explicit failure, concurrent-load throughput and distributed scheduling remain later phases.
