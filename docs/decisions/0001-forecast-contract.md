# ADR 0001: forecast timing and target

Status: accepted for contract v1. Raw data completeness is audited separately.

## Decision

Predict NESO's National Demand (`ND`) in MW, with 48 half-hour intervals of the next UTC day. Issue daily at 09:30 UTC. Use seven complete UTC days ending at 00:00 at the start of the issue day. The entire issue day is a 48-interval gap between context and target.

The machine-readable authority is `configs/forecast_contract_v1.json`; the executable example is `scripts/forecast_contract.py`.

## Why this choice

ND and TSD are different fields. Fixing ND prevents a model comparison from accidentally mixing targets. UTC gives every target day 48 points. Europe/London remains a display and source-conversion timezone, where local days can contain 46 or 50 settlement periods.

NESO describes its daily update as usually available by 08:30 UTC. We choose 09:30 as a buffer, but availability must be checked. Missing data causes a failure, not a silent cutoff change.

## Worked origin

| Boundary | UTC value |
| --- | --- |
| Issue | 2024-02-10 09:30 |
| Context start | 2024-02-03 00:00 |
| Context end, exclusive | 2024-02-10 00:00 |
| Last observed interval start | 2024-02-09 23:30 |
| Target start | 2024-02-11 00:00 |
| Last target interval start | 2024-02-11 23:30 |
| Target end, exclusive | 2024-02-12 00:00 |

The first output starts 870 minutes after issue. The last starts 2,280 minutes after issue. Horizon index and lead time are different quantities.

## Consequences

Every sampler and baseline must implement the gap. Weekly persistence uses a target timestamp minus seven days; last-available-day persistence uses the same slot of the day before issue. The numeric decoder must receive actual target timestamps or lead features. We do not get to use February 10 demand merely because we are predicting February 11.

Changing issue time, cutoff, target definition, interval size, gap, or target day creates a new contract version. Scaling, forecasting, and calibration must advertise compatible contracts.

## Sources

- https://www.neso.energy/data-portal/daily-demand-update/demand_data_update
- https://www.neso.energy/data-portal/historic-demand-data
- https://www.elexon.co.uk/bsc/settlement/
