# Phase 0 source audit

This is the historical Phase 0 evidence record. Phase 2 has now captured and audited the complete original CSVs. See the [Phase 2 walkthrough](../learning/phase-2.md) and [frozen full-data audit](../../reports/neso-audit-v1.json) for current coverage and checksums. The limitations below describe the earlier preview-only stage.

## What has been checked

Official NESO resource pages exist for the 2017 warmup and every proposed training, validation, calibration, and test year through 2024. Their advertised resource IDs are saved in `configs/source_catalog_v1.json`. The current daily-update resource is separately identified.

The public API's five-row previews for 2018, 2024, and the daily update were inspected through the web retrieval service. The first two report totals of 17,520 and 17,568 records respectively, consistent with 365 and 366 half-hourly days. A matching count alone does not prove no gaps or duplicates.

**No complete source CSV has been downloaded in this environment.** Shell DNS/network access failed. Preview evidence is transcribed into the catalog with its query URL; it is not a raw-byte snapshot and has no invented checksum. Full completeness, null, duplicate, range, and DST audits remain Phase 2 gates. The proposed experiment split remains conditional on that audit.

## Selected schema

| Field | Role | Rule |
| --- | --- | --- |
| SETTLEMENT_DATE | Source local settlement date | Parse source-specific date representation; retain original |
| SETTLEMENT_PERIOD | Half-hour ordinal within that date | Resolve through Europe/London clock changes |
| ND | Label and historical target input | MW; do not replace with TSD |
| EMBEDDED_WIND_GENERATION | Candidate historical covariate | No future observed values |
| EMBEDDED_SOLAR_GENERATION | Candidate historical covariate | No future observed values |
| FORECAST_ACTUAL_INDICATOR | Daily update type discriminator | Require A for observed inputs/labels; reject unknown indicator |

Annual historical outturn resources do not have to contain the daily indicator column. Their source adapter must explicitly classify them as historic outturn based on the registered dataset type; absence of the column is not a generic actual-value inference for arbitrary sources.

Interconnector and newer fields vary across years. Require only the stable selected schema, report additive columns, and reject incompatible changes to mandatory fields.

## Availability

NESO's historic dataset is delayed and revised. Use frozen revised history for the first retrospective experiments and label it accordingly. Start collecting daily snapshots later for captured-availability evidence. Daily previews are current rolling content and cannot stand in for archived historical releases.

## Licence and attribution

The historic collection advertises the NESO Open Data Licence. Its current terms require attribution and do not imply endorsement. Use the exact short statement: **Supported by National Energy SO Open Data**. Verify the licence association for each resource when materialising it and retain that metadata. Do not reuse provider logos.

No external code is copied into GridLens. Dataset permissions do not determine the licence of research reference code.

## Phase 2 acceptance checks

1. Retrieve and checksum original CSV bytes for each registered resource, retaining metadata and retrieval time.
2. Audit required fields, parse rates, nulls, duplicate keys, and range boundaries.
3. Map local settlement dates/periods to UTC; verify 46/48/50-period transition fixtures and annual edges.
4. Report target origins dropped by missing context/targets; never impute held-out labels.
5. Include December 24-31, 2017 warmup, and any adjacent source intervals required by UTC/local boundaries.
6. Verify every selected historical covariate across all years before freezing feature order.
7. Freeze real raw checksums, eligible-origin lists, and split manifests before training.

## Sources

- https://www.neso.energy/data-portal/historic-demand-data
- https://www.neso.energy/data-portal/daily-demand-update/demand_data_update
- https://www.neso.energy/data-portal/neso-open-licence
- https://www.elexon.co.uk/bsc/settlement/

Resource-specific pages and preview queries are in the catalog. The primary specification contains the rationale for revised and captured modes.
