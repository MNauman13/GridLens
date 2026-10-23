# Phase 3 results

Implemented frozen source selection, captured-time eligibility, causal two-slot filling with missingness/age, explicit feature order, deterministic UTC calendar features, exact 336-context/48-gap/48-target windows, training-only scaling, split ownership, and content-derived manifests.

| Role | Accepted origins | Boundary exclusions |
| --- | ---: | ---: |
| Point training | 1460 | 1 |
| Point validation | 364 | 1 |
| Calibrator training | 180 | 1 |
| Calibrator validation | 91 | 1 |
| Initialisation | 92 | 0 |
| Locked test | 366 | 0 |

Coverage is complete. The four exclusions are label-maturity embargoes, not missing data: labels for 2021-12-31, 2022-12-31, 2023-06-30, and 2023-09-30 would be released after the next role's first issue. Under the declared Mode A rule, actual release is target day + 1 at 09:00 UTC. Captured-mode fitting labels also require first-seen eligibility at the fitting deadline. Initialisation observations remain subject to delayed-feedback handling when that later phase is implemented.

Perturbation tests prove feature and scaler invariance; additional tests reject reversed chronology, overlapping target ownership, and unavailable fitting labels. Exact calendar arithmetic has identical complete-phase fingerprints on Windows and Linux.

The frozen manifest is `784a934d96749bbc8105c63ae32e4cc63fd7d817557adbaf045f0625f15a0f26`. The final 2024 test is materialised for membership only and has not been evaluated. See the [walkthrough](../learning/phase-3.md) and [combined verification](phases-3-5.md).
