# Phase 3 results

Implemented frozen observation selection, captured-time eligibility, causal filling with missingness/age, explicit feature order, deterministic UTC calendar features, exact 336-context/48-gap/48-target windows, training-only scaling, split ownership, and content-derived manifests.

The full annual data produces 1,461 training origins, 365 point-validation origins, 181 calibrator-training origins, 92 calibrator-validation origins, 92 initialisation origins, and 366 reserved test origins. No origins are dropped. The final test is materialised for membership verification but has not been evaluated.

Eight new perturbation/ownership tests passed locally; the combined suite passed 45 tests. Strict typing passed for all seven data modules. Rebuilding the source-frozen dataset must reproduce `reports/dataset-freeze-v1.json`; no targets are interpolated and no validation/test values fit scaling. The [walkthrough](../learning/phase-3.md) explains the data path and feature/dataset identity distinction.
