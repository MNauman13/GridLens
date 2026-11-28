# Durable serving

The Python service uses `DurableLiveService` by default. Set `GRIDLENS_DATABASE_URL` to PostgreSQL for multi-process deployment. Without it, `GRIDLENS_JOB_DB` selects a SQLite file for a single-host deployment. The original I1 `LiveService` remains as a compatibility reference and model loader.

Admission creates an owner ticket and an immutable work payload. `Idempotency-Key` reuses a ticket only for the same owner and payload. Different owners requesting identical active work coalesce onto one task. A completed task is never used as a prediction cache for a new key. Result reads require the anonymous HttpOnly session cookie. Changing the requested model/catalog identity does not silently execute old queued work under new weights.

Transactions are short. SQLite uses `BEGIN IMMEDIATE`; PostgreSQL uses an advisory transaction lock for queue transitions. Model computation runs outside database transactions. The default worker renews a ten-second lease every third of a lease. Every claim increments a fencing token. Heartbeat, completion and failure writes require an active, unexpired lease with the same token. Process death leaves work reclaimable. Three failed attempts terminate the task. Explicit failures use bounded exponential backoff. Cancelling the last ticket fences any running task, so a late result cannot become visible.

Limits are eight distinct active tasks, one active ticket per owner, 60 admitted tickets per owner per minute and 240 globally per minute. Rejections return 429 and Retry-After. Terminal jobs are retained until 24 hours after admission; hourly cleanup never removes active jobs. These are admission controls, not a guarantee of resistance to anonymous cookie rotation.

Readiness checks verified models and database access. Liveness only reports process responsiveness. A missing dependency or invalid artifact fails closed with 503. No saved forecast replaces failed inference.

## Local run

```
node scripts/prepare_live_assets.mjs
uv sync --frozen --extra api --extra ml --group dev
GRIDLENS_LIVE_ASSETS=artifacts/i1 uv run --frozen --extra api --extra ml python -m uvicorn gridlens.api:app --host 127.0.0.1 --port 8000
```

For Compose, set a strong URL-safe `GRIDLENS_POSTGRES_PASSWORD`, prepare the verified assets first, then run `docker compose up --build`. Use `GRIDLENS_COOKIE_SECURE=0` only for local HTTP. Put production traffic behind HTTPS and keep PostgreSQL private. The API image runs as UID 10001. Persist the PostgreSQL volume and worker state. Container execution must be verified on a Docker host; a successful Python test is not a container smoke test.

## Verification

```
uv run --frozen --extra api --extra ml python -m unittest tests.test_jobs tests.test_worker -v
uv run --extra api --extra ml --with embedded-postgres python -m scripts.check_postgres
uv run --frozen --extra api --extra ml python -m scripts.load_service
```

The PostgreSQL helper creates a temporary UTF-8 server with loopback TCP and trust authentication only inside the disposable test directory. It shuts the server down after the contract suite. Never use its trust configuration for deployment. CI runs the same queue tests against its dedicated PostgreSQL service.

The load driver launches its own API subprocess, offers two requests per second from five session clients for 900 seconds, records every rejection and admitted completion, and reports read latency separately. Its raw events are retained alongside aggregates. It must not be presented as a seven-day uptime measurement.

## Public web adapter

`POST /api/compute` provides real synchronous ridge or ONNX WASM inference within the deployed Next.js server. It verifies input/model digests, permits one active inference per server instance and returns a fresh execution identifier. This adapter does not use the Python durable queue or claim global distributed admission limits. It is useful without provisioning a Python host. Browser execution and the optional Python service remain separately selectable. All three execution locations are labelled in results.

## Backups and rollback

Use `scripts.backup_jobs` to take a consistent SQLite backup while workers run. The helper uses SQLite's backup API and verifies integrity, records task counts and pins the archive digest. Restoration refuses an existing target and a changed archive. Point `GRIDLENS_JOB_DB` to the restored file only after stopping the old worker. Pending work can be reclaimed after its lease expires; a prior worker cannot complete through an expired fencing token.

For PostgreSQL, use `pg_dump -Fc` on the private database and `pg_restore` into a new database, then verify result counts and immutable result payloads before changing `GRIDLENS_DATABASE_URL`. Keep database backups private because they contain session-owned job records. The ephemeral PostgreSQL verification helper now performs an actual dump and restore and checks that an owned result remains identical.

Deployments retain immutable source commits and pinned model archives. To roll the website back, redeploy the last verified commit through Vercel and rerun `scripts.public_smoke`. The original I1 demo archive remains independently pinned; `GRIDLENS_DEMO_SPEC=configs/i1_demo_artifact.json` restores that model catalog at build time. Source capture archives and public model artifacts must not be overwritten under existing hashes.

The optional `GRIDLENS_ONNX_ENABLED=1` CPU path runs the verified FP32 graph through ONNX Runtime. Startup compares every allowlisted case against the matching PyTorch bundle and refuses readiness if drift exceeds 0.5 MW. Its graph, model, library and thread configuration have a distinct runtime identity. Compose enables this path; the original manual and SDPA paths remain available for comparison. The runtime study uses a separate re-exported I0 graph and records its full-2022 parity independently. Do not conflate Python ONNX CPU execution with Node/browser WASM timings.
