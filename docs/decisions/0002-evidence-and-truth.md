# ADR 0002: what the system knew, and which actual is truth

Status: accepted.

## Two evidence modes

`retrospective_revised`: use a frozen historical download. Enforce event-time cutoffs, the gap, chronological splits, and training-only transforms. Historical revision availability is unknown. Do not call the result a live-faithful backtest.

`captured_availability`: use immutable snapshots captured prospectively. A selected value must have `first_seen_at <= issued_at`. Future forecast covariates additionally need an eligible source issue time. Do not fill older origins with snapshots first retrieved later.

An interval's event time is when electricity was measured. Its first-seen time is when our collector received that version. They answer different questions. Retrieving an old event today does not make it available to a forecast issued last year.

## Two truth roles

Operational feedback uses the first eligible captured actual to update calibration. Final evaluation uses an identified, frozen actual snapshot, possibly containing corrections. A correction can change a new evaluation report but must not rewrite the state behind an already-issued forecast.

Mode A simulates each target day's actual release at 09:00 UTC the next day. This is our assumption, not proof of historical publication times. Mode B uses actual captured release events.

For a forecast issued February 10 targeting February 11, full actual feedback is unavailable at the February 11 issue. Under the simulation, it becomes available February 12 at 09:00, before that day's 09:30 issue.

## Calibration guarantees

Bands are marginal prediction intervals for each output slot. A nominal 90% band is not a 90% guarantee that the whole 48-point curve will fit. NCC's theoretical conclusions are not automatically inherited after changing feedback delays, horizon sharing, or interval repairs. Evaluate final served bands empirically.

## Train/validation/test

Freeze the point model before generating calibrator examples. Fit the calibrator on out-of-sample point predictions, not residuals from point-model training. Test neural weights remain frozen except for explicitly documented NCC test-time adaptation. Matured-feedback controller state updates are allowed.

See `configs/experiment_manifest_v1.json`. The dataset needs December 2017 warmup to produce the first January 2018 target; checking only calendar-year training files would miss this requirement.
