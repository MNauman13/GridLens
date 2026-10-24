# Browser verification runbook

## Normal local workflow

Prerequisites: Python 3.12, uv, Node 24, npm, and access to package registries and the Chromium download. Run commands from the repository root.

```bash
uv sync --frozen --extra api --group dev
npm ci
npx playwright install chromium
npm run build
npm run test:e2e
```

On Linux, the browser installation may also require `npx playwright install --with-deps chromium`. CI handles this on its ephemeral runner. Do not change system network/security settings merely to make a test pass.

The test runner starts FastAPI on `127.0.0.1:8000` and the production Next.js app on `127.0.0.1:3000`. It waits for HTTP readiness before opening Chromium and stops its servers afterward. A pre-existing local server may be reused outside CI; stop an outdated server when testing a new build.

## What is actually verified

There are eight scenarios at two viewport sizes, giving sixteen browser tests:

1. Open the page, call the real Python fixture API, validate its boundaries, and change to a leap-day origin.
2. Click Next/Previous through context, gap, target, and delayed feedback; check boundary buttons.
3. Simulate an API outage and check that the application displays an error instead of a success state.

The main scenario checks uncaught browser JavaScript errors and horizontal overflow. The suite saves full-page desktop/mobile screenshots; failures also retain traces and a browser report. An outage mock is used only for the explicit failure scenario; the successful fixture scenario goes through both real servers.

Tests scope application alerts to `main`: Next.js also has an alert region for route announcements. This avoids confusing the framework's accessibility mechanism with the product's error message.

## Evidence

Local outputs are under `reports/local/browser-results` and `reports/local/playwright-report`. Open the report with:

```bash
npx playwright show-report reports/local/playwright-report
```

GitHub Actions uploads these outputs to the workflow run's artifact, retained for seven days. The commit and run URL identify the source under test. Automated assertions passing do not mean every pixel has been manually reviewed; report visual review separately.

## When the in-app browser cannot connect

A local server process alone is insufficient. Verify all three links:

- The server has started and reports readiness.
- An HTTP client can reach its loopback URL.
- The controlled browser can reach that same URL.

In the restricted task environment used for Phase 1, direct loopback HTTP was rejected with socket error 10013 and the in-app browser timed out. The same committed suite ran on GitHub's runner, where both servers were reachable. That proves the application can be tested in a real browser without asserting that this chat's local browser bridge works.

Do not relabel Node DOM mocks as browser tests, or a launched process as a verified page. Record the blocked layer and use a verified environment for the full test.

## Phases 4–5 additions

The suite also verifies real saved baseline forecasts, exact issue/target/cutoff boundaries, model switching, lag values, provenance, and forecast API errors. Attention checks compare displayed weights/output with the PyTorch-verified fixture, test causal masking and head/query controls, and play/pause attention and positional animations. Eight scenarios at two Chromium viewport sizes give sixteen tests.

Install the ML extra for the complete repository checks: `uv sync --frozen --extra api --extra ml --group dev`. The web server still serves saved baseline artifacts without running training during a request.

## Phases 6–8 additions

`transformer.spec.ts` checks shifted versus generated prefix controls and the opt-in TimeXer operator animation. `neural.spec.ts` checks saved trained forecasts, model switching, all-slot inspection, provenance, failure messaging and mobile overflow. The complete suite runs these alongside the existing scenarios at desktop and mobile Chromium sizes. Screenshots are generated for the neural forecast view; these are automated browser evidence, not a claim of manual visual review.

The experiments page displays committed, measured evidence rather than generating new training work in the browser. Neural HTTP responses enforce exactly 48 finite values and a content identity; corrupt or missing resources return 503. The viewer labels each example as retrospective validation and does not present it as a live issue.
