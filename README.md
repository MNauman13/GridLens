# GridLens

Electricity demand forecasting with a Transformer implemented from scratch in PyTorch, a TimeXer application adaptation, and Neural Conformal Control.

**Stage:** Phase 0: verify sources and freeze contracts. No model is trained yet.

GridLens will predict NESO National Demand in MW for the following UTC day and evaluate prediction accuracy, interval coverage, width, and runtime. It will retain source and model versions so forecasts can be reproduced.

## Start here

- [Full specification](docs/specification.md)
- [Forecasting decisions](docs/decisions/0001-forecast-contract.md)
- [Evidence modes and truth](docs/decisions/0002-evidence-and-truth.md)
- [Phase 0 walkthrough](docs/learning/phase-0.md)
- [Source audit](docs/data/source-audit.md)
- [Research mapping](docs/research/mapping.md)
- [Phase status](docs/progress/phase-0.md)

The default contract issues at 09:30 UTC, uses seven complete UTC days of history ending at the start of the issue day, skips observed values from the issue day, and predicts 48 half-hours of the following UTC day.

## Run the Phase 0 checks

Phase 0 uses Python 3.11+ and only the standard library. It does not install ML packages.

```bash
python scripts/forecast_contract.py --issue-date 2024-02-10
python -m unittest discover -s tests -v
python scripts/phase0_check.py
python scripts/environment_audit.py --output reports/local/environment.json
```

Use the absolute path to a working interpreter if `python` resolves to the Windows Store launcher. The audit records the interpreter actually used; no personal directory or machine name is required in committed reports.

## Contribution and push cadence

Commit each coherent, verified change: contracts, source evidence, research references, checks, and walkthrough. Push that commit immediately when a remote is available. Do not force-push, rewrite shared history, commit raw datasets/checkpoints/secrets, or claim a successful push before verifying the remote commit.

## Data attribution

Supported by National Energy SO Open Data.

The data licence applies to the relevant NESO datasets, not automatically to project code. GridLens is independent and is not endorsed by NESO. Reference repositories are not vendored; their reuse permissions must be checked separately.
