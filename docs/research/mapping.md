# Research-to-implementation mapping

The project uses the original Transformer and exactly two additional methods. This is an application and reproduction project, not a new architecture claim. Reference code is inspected for interpretation; it is not vendored or installed.

## Immutable references

| Method | Paper | Repository commit |
| --- | --- | --- |
| Original Transformer | User-supplied 2017 paper, sections 3.1-3.5 | No third-party implementation dependency |
| TimeXer | arXiv 2402.19072 v4 / NeurIPS 2024 | 76011909357972bd55a27adba2e1be994d81b327 |
| NCC | arXiv 2412.18144 v1 / AAAI 2025 | 1430cd38bc5ac7331b7f1d7b31867d27074efddf |

`configs/research_pins.json` records commit/tree IDs and the relevant source file blob IDs, so future upstream changes cannot silently change our reference.

## Mapping skeleton

| Paper responsibility | Planned local module | Verification planned | GridLens deviation |
| --- | --- | --- | --- |
| Scaled dot-product attention | models/attention | Independent numbers and PyTorch oracle | None in authoritative manual operation |
| Multiple heads and projection | models/attention | Self/cross shapes, masks, gradients | Smaller model width |
| Sine/cosine positions | models/attention.SinusoidalPositionalEncoding | Known positions, offsets, reload and device/dtype | None in original reproduction |
| Encoder and masked decoder | models/transformer | Causal invariance and learned sequence task | Smaller layers/data budget |
| Numeric forecasting interface | models/numeric, models/bundle | 336-context/48-gap/48-label timestamps; generated inference; checkpoint continuation | Numeric projections and regression |
| TimeXer patches/global/variable tokens | models/timexer | Operator-order and token-route checks | Forecast gap and target-calendar head |
| NCC score threshold predictor | calibration/ncc | Residual-quantile and gradient fixtures | Shared weights across 48 slot streams |
| NCC controller and interval ordering | calibration/ncc | Delayed feedback, nesting, state restoration | Delayed daily feedback and bounded TTA |

Replace planned paths with actual classes/functions and test names as each phase is built. Do not mark rows implemented before the code exists.

Through Phase 8, the attention, encoder-decoder, numeric and TimeXer rows are implemented. `tests/test_transformer.py`, `tests/test_numeric.py` and `tests/test_training.py` check masks, shifts, global-only external queries, gradients, save/reload and exact optimizer/RNG continuation. `docs/learning/phase-8.md` maps the pinned TimeXer classes to the implemented operators and separates the gap/calendar adapter. The NCC rows remain planned.

## Reuse decision

Recursive trees at both pinned commits were retrieved successfully without truncation. No path named LICENSE, LICENCE, COPYING, or NOTICE was found. The inspected implementation file headers did not supply a licence grant. This audit does not prove every file has no licence text; it establishes that no explicit repository-wide licence was located.

The chosen action is independent implementation of the published methods with paper citations. We will not copy, redistribute, or relicence upstream source. This resolves the build approach without requiring permission to copy code. If code reuse is later proposed, obtain and record an explicit grant first.

## Model boundaries

The original module follows post-normalisation, ReLU feed-forward blocks, sinusoidal positions, shifted targets, and causal masks. The numeric and TimeXer adaptations have distinct names/configurations. NCC's weights and mutable feedback state remain separate from point-model weights. Published performance and guarantees do not automatically transfer to the application.

The exact NCC loss transcription and proof assumptions are implementation gates in Phase 10. Phase 0 pins references; it does not pretend that loss code already exists.

## Primary links

- https://papers.nips.cc/paper/7181-attention-is-all-you-need
- https://arxiv.org/abs/2402.19072v4
- https://github.com/thuml/TimeXer/tree/76011909357972bd55a27adba2e1be994d81b327
- https://arxiv.org/abs/2412.18144v1
- https://github.com/complex-ai-lab/ncc/tree/1430cd38bc5ac7331b7f1d7b31867d27074efddf
