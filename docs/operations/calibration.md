# Causal uncertainty intervals

The interval engine produces marginal symmetric 80%, 90% and 95% prediction bands. Coverage is per half-hour. No daily simultaneous coverage or exchangeability claim is made.

`IntervalReplay` owns a deep copy of its session state. At an issue, only feedback whose release time is at or before that issue can have been applied. Targets release at 09:00 UTC on the following day; issues occur at 09:30 UTC on the day before the target. A future observation, conflicting duplicate, backwards clock or unconsumed matured feedback is rejected before issuing. Rewind uses a content-verified checkpoint, including method, controller state, scale, window and fixed thresholds. Point model and calibration identities are part of the state.

Fixed residual widths are frozen before validation. Rolling widths use the latest 90 eligible daily residual curves. With fewer than 60 days, the engine explicitly labels pooled-slot fallback as degraded. Finite-sample quantile ranks are ceil((n+1)*coverage), capped at the available maximum when that rank exceeds n. This finite fallback is not a distribution-free guarantee for a dependent electricity series.

The independent NCC implementation follows the pinned paper's quantile, coverage, efficiency and monotonicity objectives. It encodes recent scores, previously issued thresholds and errors using three GRUs, combines these with calendar/slot context through attention, and predicts positive cumulative increments. Its controller updates only when feedback matures. Per-forecast TTA changes a local correction variable, never the frozen neural weights. TTA is capped at 32 steps. Any remaining negative width or crossing receives a recorded deterministic repair.

The paper and source interpretation are pinned in `configs/research_pins.json`; the code is independently written. Application adaptations, fixed training budgets and selection criteria are declared in `configs/calibration_study_v1.json`. Delayed 48-slot feedback, shared parameters, bounded TTA and nesting repair mean the paper's original guarantee is not transferred to this application.

Training uses January–June 2023, method selection uses July–September, and October–December only initialises the released state. Weights are frozen before that initialisation. The point champion is frozen first. The final 2024 test must remain untouched until a complete release manifest is committed.

Record coverage, absolute coverage error, interval width and interval score at each level, weighted interval score, slot/daily/rolling coverage, crossings, repair magnitude and generation latency. NCC is promoted only if all coverage gates pass and its weighted interval score improves at least 5% over the eligible residual champion. Otherwise it remains explicitly experimental. Failed gates remain visible.
