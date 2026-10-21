# ADR 0005: content-addressed CSVs and an offline observation index

Status: accepted for Phase 2.

## Decision

Pin the provider's original CSV URLs by resource ID. Retrieve full CSV responses, not a paginated mutable DataStore view. Preserve response bytes before validation; identify each content object by SHA-256. Keep a receipt for each retrieval, even if its bytes were previously captured.

NESO currently redirects downloads to a public Cloudflare bucket. Validate the expected resource path and observed bucket host. Keep the final origin/path in provenance, but omit expiring signed query parameters from stored receipts. The pinned URL remains the official NESO URL. A changed redirect fails explicitly and requires review.

Use the lockfile's tzdata package on every OS. Convert each settlement-day ordinal by adding elapsed half-hours to its local-midnight instant in UTC. Retain original date and period. Two-digit years require the explicitly declared resource year, with matching final digits.

The local, reproducible Phase 2 data product uses SQLite schema version 1 for retrieval receipts, snapshot validation status, and actual observation versions. Its transactions prevent a partial row set being visible. This is an offline ingestion index, not the project's eventual multi-user PostgreSQL serving database. Subsequent schema changes require a migration rather than recreating a populated index. Unknown schema versions fail.

Observation identity includes the source, raw checksum, source interpretation, and transform version. Changed bytes retain a separate version. A repeat download preserves the original first-seen time. Historical annual rows are revised outturn; retrieval today does not establish publication availability at a historical forecast issue.

## Failure behavior

Missing required columns, unknown actual/forecast flags, malformed rows, impossible period ordinals, incompatible nonblank numbers, negative ND, and conflicting duplicate rows quarantine the complete snapshot. Raw bytes and the failure report remain. No canonical rows from that snapshot are inserted. Blank numeric values remain explicit nulls and reduce coverage; no label interpolation is performed.

Additional columns are logged. Optional wind/solar/TSD fields are separately audited. High positive values are retained; zero ND and negative embedded generation are flagged for investigation rather than clipped. Exact duplicate rows collapse; differing extra-column contents also cause a duplicate conflict.

## Consequences

Raw and canonical products are independently inspectable, ordinary runs need no database server, and later feature snapshots can refer to frozen observation versions. A provider's valid correction changes the checksum: the live audit stops at a hash-drift check and requires a new reviewed freeze. Daily updates are expected to change and are audited separately from annual completeness.

The filesystem and SQLite are not one distributed transaction. Complete raw bytes and a retrieval receipt intentionally survive later validation/index failure. Files are published atomically without replacing existing content objects. Abandoned temporary files may be cleaned only after confirming that no active writer owns them.
