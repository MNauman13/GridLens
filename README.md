# GridLens

An auditable electricity forecasting project with manual PyTorch attention, a complete encoder-decoder Transformer, real-data baselines, and a documented TimeXer application. Neural Conformal Control is a later phase.

**Stage:** Through Phase 8: frozen causal datasets, trained encoder-decoder reproduction, numerical Transformer/LSTM comparisons, a bounded TimeXer validation pilot, verified model bundles, and interactive browser lessons. Prediction intervals and online calibration begin in Phase 9.

GridLens will predict NESO National Demand in MW for the following UTC day and evaluate prediction accuracy, interval coverage, width, and runtime. It will retain source and model versions so forecasts can be reproduced.

## Start here

- [Full specification](docs/specification.md)
- [Forecasting decisions](docs/decisions/0001-forecast-contract.md)
- [Evidence modes and truth](docs/decisions/0002-evidence-and-truth.md)
- [Phase 0 walkthrough](docs/learning/phase-0.md)
- [Phase 1 walkthrough](docs/learning/phase-1.md)
- [Phase 2 walkthrough and animated lesson](docs/learning/phase-2.md)
- [Phase 3: causal datasets](docs/learning/phase-3.md)
- [Phase 4: baseline forecast explorer](docs/learning/phase-4.md)
- [Phase 5: manual attention and animations](docs/learning/phase-5.md)
- [Phase 6: complete encoder-decoder Transformer](docs/learning/phase-6.md)
- [Phase 7: numerical forecasting and resumable training](docs/learning/phase-7.md)
- [Phase 8: TimeXer operator mapping and application](docs/learning/phase-8.md)
- [Phases 6-8 results and limitations](docs/progress/phases-6-8.md)
- [Neural validation evidence](reports/neural-validation-v1.json)
- [Frozen point champion](configs/point_champion_v1.json)
- [Phases 3–5 results](docs/progress/phases-3-5.md)
- [Frozen full-data audit](reports/neso-audit-v1.json)
- [Source audit](docs/data/source-audit.md)
- [Research mapping](docs/research/mapping.md)
- [Phase status](docs/progress/phase-0.md)
- [Phase 1 results](docs/progress/phase-1.md)
- [Phase 2 results](docs/progress/phase-2.md)
- [Browser verification runbook](docs/operations/browser-verification.md)

The default contract issues at 09:30 UTC, uses seven complete UTC days of history ending at the start of the issue day, skips observed values from the issue day, and predicts 48 half-hours of the following UTC day.

## Install and run

Use Python 3.12 and Node 24. Core/API, development, and CPU ML dependencies are separated. Lockfiles record the actual dependency resolution.

```bash
uv sync --frozen --extra api --extra ml --group dev
uv run --frozen --extra api --extra ml gridlens fixture
npm ci
npm run build
npx playwright install chromium
npm run test:e2e
```

Use the absolute path to a working interpreter if `python` resolves to the Windows Store launcher. The audit records the interpreter actually used; no personal directory or machine name is required in committed reports.

For manual inspection, run `uv run --frozen --extra api --extra ml python -m uvicorn gridlens.api:app --host 127.0.0.1 --port 8000` and `npm run dev` in separate terminals, then open `http://127.0.0.1:3000`.

`npm run test:e2e` starts both servers automatically and checks real Chromium interactions at desktop and mobile sizes. Reports, screenshots, and failure traces go to `reports/local/`. GitHub Actions runs the same checks without relying on this chat's loopback access.

```bash
uv run --frozen --extra api --extra ml python -m unittest discover -s tests -v
uv run --frozen --extra api --extra ml python scripts/phase0_check.py
uv run --frozen --extra api --extra ml python scripts/export_schemas.py --check
uv run --frozen --extra api --extra ml ruff check src tests
uv run --frozen --extra api --extra ml mypy src/gridlens
npm run typecheck
npm run lint
npm run test:unit
```

Fetch original data with `uv run --frozen --no-dev gridlens ingest-neso --report reports/local/live-audit.json`. Add `--sources 2017 2018 2019 2020 2021 2022 2023 2024 daily` to include the mixed daily update. Raw bytes, retrieval receipts, observation versions, and per-snapshot audits are stored under `data/neso/`. The animated settlement-time lesson is at `/learn/settlement`.

## Contribution and push cadence

Commit each coherent, verified change: contracts, source evidence, research references, checks, and walkthrough. Push that commit immediately when a remote is available. Do not force-push, rewrite shared history, commit raw datasets/checkpoints/secrets, or claim a successful push before verifying the remote commit.

## Data attribution

Supported by National Energy SO Open Data.

The data licence applies to the relevant NESO datasets, not automatically to project code. GridLens is independent and is not endorsed by NESO. Reference repositories are not vendored; their reuse permissions must be checked separately.

## Explore the current implementation

- `/forecasts`: real saved 2022 validation forecast, three baselines, exact time boundaries, and provenance.
- `/learn/settlement`: British clock-change conversion animation.
- `/learn/attention`: checked numerical attention calculator and positional animation.
- `/learn/transformer`: shifted training versus generated inference, plus the TimeXer operator animation.
- `/forecasts/neural`: verified saved Transformer/LSTM/selected-TimeXer validation forecasts.
- `/experiments`: model comparison, ablations, reversal learning curve and limitations.

The 2022 selected ridge baseline has MAE 1,390.80 MW and RMSE 1,895.45 MW. These are validation results; the final 2024 evaluation remains locked. See `reports/baseline-validation-v1.json`. The serving API validates packaged results and does not fit models during a page request.

Run `uv run --frozen --extra ml python scripts/build_dataset.py` and `uv run --frozen --extra ml python -m scripts.run_baselines` after ingestion. The latter rebuilds and verifies the frozen dataset before fitting. Hosted CI verifies the complete source-to-baseline path and tests the app in Chromium. A separate Windows/Linux job requires identical fingerprints for all annual, weekly, and daily calendar phases.

Label-maturity embargoes retain 1,460 training and 364 validation origins. Four fitting/selection boundary dates are excluded because their labels would arrive after the next role's first issue; exclusions are frozen and do not depend on model error.

## Reproduce the neural pilot

```bash
uv run --frozen --extra ml python scripts/train_reversal.py
uv run --frozen --extra ml python -m scripts.train_neural
uv run --frozen --extra ml python scripts/check_neural_reference.py
```

Ingest the pinned annual data first. Generated weights, training checkpoints and aligned full-validation predictions stay under ignored `reports/local/`. Hosted experiment artifacts preserve those bundles; the frozen champion record links the exact run and expected manifest hash. The committed JSON examples serve without importing Torch. The numeric models require the complete `FeatureSnapshot`, since the legacy demand-only input contract cannot represent wind, solar and missingness. `NeuralBundle.load_bundle(...).predict(snapshot)` validates and returns the shared `PointForecast` contract.

This is a single-seed, bounded CPU pilot, not a completed multi-seed study or a reproduction of published benchmark scores. The original small Transformer learns disjoint same-length reversal sequences but fails longer-length generalization. All 2024 accuracy evaluation remains locked. See the reports for actual measurements and the walkthroughs for application deviations.
