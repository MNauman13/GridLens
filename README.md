# GridLens

Electricity demand forecasting with a Transformer implemented from scratch in PyTorch, a TimeXer application adaptation, and Neural Conformal Control.

**Stage:** Phase 1: installable Python package, Next.js foundation, typed interfaces, and repeatable browser tests. No model is trained yet.

GridLens will predict NESO National Demand in MW for the following UTC day and evaluate prediction accuracy, interval coverage, width, and runtime. It will retain source and model versions so forecasts can be reproduced.

## Start here

- [Full specification](docs/specification.md)
- [Forecasting decisions](docs/decisions/0001-forecast-contract.md)
- [Evidence modes and truth](docs/decisions/0002-evidence-and-truth.md)
- [Phase 0 walkthrough](docs/learning/phase-0.md)
- [Phase 1 walkthrough](docs/learning/phase-1.md)
- [Source audit](docs/data/source-audit.md)
- [Research mapping](docs/research/mapping.md)
- [Phase status](docs/progress/phase-0.md)

The default contract issues at 09:30 UTC, uses seven complete UTC days of history ending at the start of the issue day, skips observed values from the issue day, and predicts 48 half-hours of the following UTC day.

## Install and run

Use Python 3.12 and Node 24. Core/API, development, and CPU ML dependencies are separated. Lockfiles record the actual dependency resolution.

```bash
uv sync --frozen --extra api --group dev
uv run --frozen --extra api gridlens fixture
npm ci
npm run build
npx playwright install chromium
npm run test:e2e
```

Use the absolute path to a working interpreter if `python` resolves to the Windows Store launcher. The audit records the interpreter actually used; no personal directory or machine name is required in committed reports.

For manual inspection, run `uv run --frozen --extra api python -m uvicorn gridlens.api:app --host 127.0.0.1 --port 8000` and `npm run dev` in separate terminals, then open `http://127.0.0.1:3000`.

`npm run test:e2e` starts both servers automatically and checks real Chromium interactions at desktop and mobile sizes. Reports, screenshots, and failure traces go to `reports/local/`. GitHub Actions runs the same checks without relying on this chat's loopback access.

```bash
uv run --frozen --extra api python -m unittest discover -s tests -v
uv run --frozen --extra api python scripts/phase0_check.py
uv run --frozen --extra api python scripts/export_schemas.py --check
uv run --frozen --extra api ruff check src tests/test_foundation.py
uv run --frozen --extra api mypy src/gridlens
npm run typecheck
npm run lint
npm run test:unit
```

## Contribution and push cadence

Commit each coherent, verified change: contracts, source evidence, research references, checks, and walkthrough. Push that commit immediately when a remote is available. Do not force-push, rewrite shared history, commit raw datasets/checkpoints/secrets, or claim a successful push before verifying the remote commit.

## Data attribution

Supported by National Energy SO Open Data.

The data licence applies to the relevant NESO datasets, not automatically to project code. GridLens is independent and is not endorsed by NESO. Reference repositories are not vendored; their reuse permissions must be checked separately.
