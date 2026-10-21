# Phase 1 results

Status: implemented and verified on a clean GitHub-hosted runner. Local loopback browser access remains restricted; no claim is made that the chat's in-app browser bridge is fixed.

## Delivered against the plan

| Task | Implementation | Verification |
| --- | --- | --- |
| P1.1: workspace and lockfiles | Installable src/gridlens package; Next.js app; uv.lock; npm lock; Python 3.12 and Node 24 | Fresh CI installation, installed fixture command, production build |
| P1.2: shared interfaces | Dataset, training example, point forecast, interval forecast, matured feedback, model/calibrator protocols; seven JSON Schema exports | Shape, time, nesting, finite-value, and schema checks |
| P1.3: CI | Python/TypeScript lint and type checks, unit tests, real desktop/mobile Chromium tests | Full workflow passed |
| P1.4: artifact integrity | Canonical manifest IDs, byte size and SHA-256 verification, safe member paths | Stable ID, same-size corruption, traversal rejection tests |
| P1.5: reproducible fixture and explanations | gridlens fixture; API fixture; web contract inspector; architecture ADR; learning guide; browser runbook | CLI output and real browser round trip |

## Reproducible evidence

Verified dependency-refresh run: https://github.com/MNauman13/GridLens/actions/runs/39599523323

The run uses source commit `148d128e9d7fc86e78b47af75c1722cba71fd481`. It generated the reviewed updated npm lock committed at the end of this phase. The final workflow uses `npm ci` and `uv sync --frozen`, with no lock regeneration. The latest Actions run additionally verifies that final committed lock.

| Check | Result |
| --- | --- |
| Python unit tests | 15 passed |
| Node timeline logic tests | 3 passed |
| Real Chromium tests | 6 passed: three scenarios at desktop/mobile sizes |
| Python lint | Passed |
| Python strict types | Passed; seven source modules checked |
| TypeScript types and ESLint | Passed |
| Next.js production build | Passed |
| Deterministic installed fixture | Passed |
| JSON Schema freshness and Phase 0 consistency | Passed |
| CPU PyTorch forward and backward | Passed with 2.14.1+cpu |
| Production npm dependency audit | Zero vulnerabilities reported |

## Browser checks, precisely

The success scenario calls the real Python API through the real Next.js route, changes the origin to a leap-day case, checks the rendered boundaries, checks browser JavaScript errors, and checks horizontal overflow. The other scenarios click timeline controls and simulate an API outage.

The runner starts and stops both servers, saves desktop/mobile screenshots, and retains traces on failure. Artifacts are available from each Actions run for seven days. These are automated browser checks; screenshot pixel layout has not been manually inspected in this restricted environment.

An early run exposed an overly broad alert selector: Next.js has its own accessibility alert outside the application. Scoping the outage assertion to main content preserved the intended check and passed on both viewports.

## Dependency audit scope

The initial scaffold versions produced production advisories. Next.js, React, the browser runner, and lint tooling were refreshed using official npm registry metadata, and the production-only audit then reported zero vulnerabilities.

A development-only advisory remains for `braces` <=3.0.3: https://github.com/advisories/GHSA-vfj7-8cjw-p6xm . The official registry's latest release was still 3.0.3 at this audit, so no fictional patched override was added. npm reports five high-severity dependency entries through that development dependency chain. This is recorded rather than hidden; production dependencies are checked separately in CI. Recheck available patches during the next dependency review.

## Local environment versus clean-install evidence

Local package tests used a project virtual environment sharing the installed primary runtime's Pydantic and timezone packages. Editable package installation and the installed CLI succeeded after temporary build files were directed to the task workspace. This is not described as an isolated fresh dependency installation; the hosted runner provides that evidence.

Loopback requests in the restricted shell failed with Windows socket error 10013. The in-app browser timed out reaching the test server. That local route is still blocked. The verified workaround is the committed Playwright suite on an ordinary developer machine or GitHub-hosted runner, not weakened network permissions or a public tunnel.

The old probe server was stopped. The web page is not publicly deployed.

## Next phase

Phase 2 implements NESO retrieval, immutable raw bytes/checksums, schema validation, correct settlement-to-UTC conversion, actual/forecast discrimination, and the full coverage audit. A trained Transformer and measured forecasting accuracy remain later milestones.
