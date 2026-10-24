# Phase 8: TimeXer and the GridLens adapter

This is an independent implementation of published operators, with no vendored reference code. The pinned comparison source is [thuml/TimeXer, commit 7601190](https://github.com/thuml/TimeXer/blob/76011909357972bd55a27adba2e1be994d81b327/models/TimeXer.py). The paper and source identifiers are frozen in `configs/research_pins.json`.

| Reference operator | GridLens implementation | Tensor route |
| --- | --- | --- |
| Endogenous patch embedding | `TimeXerReference.patch_projection` | `[B,336] → [B,14,24] → [B,14,32]` |
| Positional embedding and global token | `positions`, `global_token` | 14 patches plus one learned token |
| Inverted external embedding | `variable_projection` | `[B,336,14] → [B,14,32]` |
| Endogenous self-attention | `TimeXerBlock.self_attention` | All 15 endogenous tokens |
| Global external cross-attention | `external_attention` | One global query, 14 external keys/values |
| Feed-forward and three normalizations | `TimeXerBlock` | Linear-ReLU-linear; equivalent to reference 1×1 convolutions |
| Final normalization and flattened head | `norm`, `head` | Channel-before-token flatten order |
| Per-context normalization | `TimeXerReference.forward` | Detached means; variance epsilon 1e-5; invert target scale |

The gap-free reference supports one endogenous target in channel zero and one or more external historical channels. Channel order is explicit and differs from the reference MS path’s target-last convention. The reference's optional calendar-variable embedding is represented here by the already frozen historical calendar channels. This is an operator-level adaptation, not a claim of matching the authors' benchmark results.

Only the global token queries external variables. Within a block the patch tokens therefore cannot change when only external tokens change; a dedicated test checks this. Subsequent blocks can propagate updated global information through self-attention. Attention itself reuses the independently tested manual implementation.

The **GridLens application adapter** changes the head to forecast 96 slots, discards slots 0–47 across the issue-day gap, and adds a learned six-calendar-feature residual to slots 48–95. Gap outputs have no separate loss. No future measurements are fabricated. The remaining historical features include wind/solar, missing indicators, ages and calendar channels. Normalization is applied per channel; constant channels remain finite through the variance floor.

The bounded pilot trains six configurations: default, no external route, patch length 48 instead of 24, one head instead of four, no positional encoding, and no target-calendar residual. They share the same data, seed and epoch budget. This uses all six allowed TimeXer configurations and two of the three allowed numeric configurations. Patch length 12 is deferred to respect that budget; the specification's broader 12/24/48 comparison is not claimed complete. Ablation differences combine stochastic optimization and architecture effects; one seed cannot establish a statistically reliable ranking. The reference head includes output dropout during training, followed by inversion of the context normalization.

Point champion selection compares full validation RMSE across all pilot candidates and the existing frozen baselines. Ridge is eligible to remain champion. The selected identity is frozen in the report; its model bundle is preserved in the corresponding experiment artifact (or the existing baseline artifact). No test score participates in selection. Uncertainty, delayed-feedback calibration and NCC remain Phases 9–10.

The interactive `/learn/transformer` page animates patching, self-attention, the global external query and the gap-aware head. It is a schematic, not an attention explanation for a specific prediction.
