# ADR 0008: actual browser inference and a separate Python reference worker

Accepted 25 November 2026 for I1.

The first anonymous live preview executes the recovered patch-48 TimeXer model in the visitor's browser using ONNX Runtime WebAssembly, and the fitted ridge coefficients using JavaScript float64. These are fresh computations over 32 allowlisted 2022 validation origins. Models and inputs may be cached; predictions are never substituted from saved examples. Historical observations are displayed after inference for comparison and are not model inputs. No 2024 origin is exposed.

The existing Next.js application can therefore run on Vercel without requiring a separately provisioned GPU or Python model host. A standalone FastAPI reference worker also loads the verified original PyTorch/ridge bundles. Its bounded queue, session ownership, durable SQLite job records and readiness endpoints are exercised locally. The public preview does not claim that this Python service has been hosted. Connecting an independently hosted reference service requires GRIDLENS_API_URL and the exact matching catalog.

The portable export is introduced ahead of I3 for deployment, not as an inference speed improvement. Keep original model/training source pins unchanged. Verify all 32 origins against PyTorch before publishing, then verify browser WASM outputs separately. Runtime identities and browser/server timings must stay distinct; device-dependent browser timings are not server benchmark results. ONNX has fixed validated input shapes; exported tracing guards do not replace boundary validation.

Pin the downloadable artifact's archive and manifest hashes in Git; verify them during build and verify each model/input asset again in the browser. Store bundles outside Git. Prepared assets contain revised historical data, not evidence of what was captured at the original issue time.

SQLite is an explicit single-process I1 choice, not the I6 distributed scheduler. Run exactly one API worker with a persistent writable database volume. The queue permits eight active jobs globally and one per session, with 60 submissions/session/minute and 240 globally/minute. Session limits are not identity-based abuse prevention. Restarted pending jobs fail explicitly; multi-process leases, cancellation, stronger abuse controls, migrations and scheduled cleanup remain I6 work. Completed records older than 24 hours are pruned at startup; disk retention requires operational monitoring before continuous public server hosting.

Saved baseline/neural pages retain their explicit saved-example labels and can read the same checked-in fixtures when no backend is configured. The contract lesson can calculate synthetic UTC boundaries locally. Neither path is a fallback for live model execution failures.
