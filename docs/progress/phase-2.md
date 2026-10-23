# Phase 2 results

Implemented and verified. All eight original annual CSVs (2017–2024) contain 140,256 actual half-hours, with zero missing ND slots and no duplicate conflicts. The captured daily update contains 2,415 actual rows and excludes 321 forecast rows. All four audited numeric fields have complete annual coverage.

The [full-data audit](../../reports/neso-audit-v1.json) freezes checksums, columns, field ranges, exclusions, and transition-day counts. [Retrieval receipts](../../reports/neso-capture-receipts-v1.json) record original-byte capture provenance. The [walkthrough](../learning/phase-2.md) explains the implementation and animated lesson.

Verification at source commit `4ee0f3931eda17c6b8b4cb27e5ee7ecb60c1aec3`:

- [Foundation CI](https://github.com/MNauman13/GridLens/actions/runs/39675487320): successful Python tests, lint/types, web build and desktop/mobile browser checks, and CPU PyTorch smoke test.
- [Live original-CSV audit](https://github.com/MNauman13/GridLens/actions/runs/39675487301): successful Python HTTP retrieval, raw checksums equal the frozen annual files, indexed counts verified; full data product retained as an artifact for 30 days.
- Local suite: 37 tests passed, including every 2017–2024 settlement interval, malformed/schema/flag fixtures, bounded retries, idempotency, corruption, revisions, and injected transaction rollback. Local strict types passed for all six ingestion modules.

The live workflow verifies the actual Python adapter independently of the original Node HTTP captures used locally. The local in-app browser path remains blocked; no manual screenshot review is claimed. The animated lesson at `/learn/settlement` is tested in Chromium through hosted CI.

Annual data is revised history. Its historical publication vintages are unknown; passing coverage does not establish a live-faithful backtest. Phase 3 must enforce feature cutoffs, captured-time eligibility, target ownership, and training-only scaling before models are fitted.
