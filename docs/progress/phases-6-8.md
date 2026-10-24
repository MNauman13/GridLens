# Phases 6–8: accepted implementation and evidence

The project is implemented through the Phase 8 exit gates as a **bounded single-seed pilot**, with the broader multi-seed experiment explicitly deferred. Phases 9 onward are not complete.

## Phase 6

Implemented the original-style post-norm encoder and decoder with manual attention, cross-attention, ReLU feed-forward layers, embedding scaling, sinusoidal positions, tied vocabulary weights, PAD/BOS/EOS handling and bounded greedy generation. Source padding and decoder causality have invariance tests. Save/reload reproduces evaluation outputs exactly.

The accepted seeded run reaches 100% exact accuracy on an eight-sequence overfit fixture and 256 same-length evaluation sequences that are explicitly excluded from training. Longer lengths 9–12 score 0%. The failed default-initialization run remains committed as a diagnostic. The accepted run uses 4,000 steps and takes approximately 263 seconds for full training plus 3 seconds for tiny overfit on its hosted CPU.

- [Accepted report](../../reports/reversal-validation-v1.json)
- [Actual training run and weights artifact](https://github.com/MNauman13/GridLens/actions/runs/40091076428)
- [Walkthrough](../learning/phase-6.md)

## Phase 7

Implemented numerical history/target projections, explicit Transformer position offset across the 48-slot gap, autoregressive inference, LSTM comparison, early stopping, training-only scaling, checkpoint manifests and exact optimizer/RNG continuation tests. `NeuralBundle` loads real state dictionaries and predicts the common 48-value `PointForecast` from a complete immutable feature snapshot. Its final identity includes weight bytes; training identity includes source/data/configuration/seed/Torch version.

The common trainer fits 1,460 origins and evaluates all 364 validation origins in inference mode. It saves best-epoch bundles and full aligned predictions. Transformer and LSTM models both perform worse than ridge in this pilot; they are retained rather than removed from the comparison.

## Phase 8

Implemented a separately named gap-free TimeXer reference path with patch/global/variable tokens, global-only external queries, post-norm feed-forward blocks, final normalization and a channel-before-token flattened head with reference output dropout. The independent implementation does not vendor upstream code. The GridLens adapter forecasts 96 slots, discards 48 gap slots and adds known target-calendar residuals.

Six TimeXer configurations compare external histories, 24/48-slot patches, four/one heads, positions and target-calendar residuals. Patch length 12 is deferred to respect the six-configuration budget. The selected point champion is the patch-48 configuration. All candidates use the same seed, data and epoch budget.

| Candidate | MAE (MW) | RMSE (MW) |
| --- | ---: | ---: |
| Frozen selected ridge | 1,390.80 | 1,895.45 |
| Numeric Transformer | 1,854.48 | 2,547.85 |
| LSTM | 1,865.31 | 2,437.00 |
| TimeXer default | 1,411.03 | 1,900.04 |
| TimeXer without external route | 1,435.91 | 1,932.07 |
| TimeXer patch 48 — selected | 1,390.05 | 1,876.62 |
| TimeXer one head | 1,429.14 | 1,896.52 |
| TimeXer without positions | 1,398.66 | 1,882.37 |
| TimeXer without target calendar | 1,451.98 | 1,932.74 |

The selected validation RMSE difference versus ridge is −18.83 MW. Paired circular bootstrap 95% percentile intervals are **[−91.64, 61.07] MW** for seven-day blocks and **[−95.64, 59.61] MW** for fourteen-day blocks. Both include zero. No statistically established accuracy advantage is claimed. The bootstrap also excludes seed and selection uncertainty.

Earlier diagnostic scores near 1,803 MW used an incomplete head-dropout path and are superseded. Only the corrected run below supplies accepted neural evidence.

- [Complete corrected validation report](../../reports/neural-validation-v1.json)
- [Frozen champion and bundle manifest hash](../../configs/point_champion_v1.json)
- [Corrected training run, checkpoints, predictions and model bundles](https://github.com/MNauman13/GridLens/actions/runs/40091539589)
- [TimeXer mapping and deviations](../learning/phase-8.md)
- [Model card](../models/neural-pilot-model-card.md)
- [Experiment operations](../operations/neural-experiments.md)

The accepted neural artifact expires **2026-11-23**. Download it before expiry for long-term archival. Future workflow runs request 90-day retention. Its archive digest and exact selected bundle manifest hash are frozen in the champion record.

## Teaching and browser surface

`/learn/transformer` explains shifted targets and generated prefixes, then animates TimeXer's operator route. `/forecasts/neural` serves real saved forecasts from reloaded model bundles with issue/cutoff/target times, 48-slot inspection and hash provenance. `/experiments` presents the measured comparison and reversal learning curve. Diagrams and animations are explicitly schematic where they are not model-derived values.

The [accepted verification run](https://github.com/MNauman13/GridLens/actions/runs/40092440862) passes **73 Python tests**, with one CUDA-only skip, strict typing for **23 source files**, and **24 desktop/mobile Chromium checks**, along with lint, static evidence checks and the production build. See the [machine-readable verification record](../../reports/phases-6-8-verification.json). The local restricted environment cannot run the full dependency/browser stack; no local-browser or manual-screenshot-review claim is made.

## Remaining limits

This is not the full three-seed, 50-epoch study. The 2024 test set remains locked. Uncertainty intervals, delayed-feedback replay, NCC, production jobs and deployment are later phases. The serving slice is an immutable validation demonstration, not a live forecasting service.
