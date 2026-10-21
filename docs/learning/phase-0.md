# Phase 0: decide exactly what the model is allowed to know

We begin with the prediction problem, before neural network code. A perfectly implemented Transformer can still produce meaningless results if it sees tomorrow's answers during training or evaluation.

## 1. What is one prediction?

Our label is National Demand (`ND`) from NESO, measured in megawatts. Each label describes one half-hour interval. The finished model emits 48 values, one for each half-hour of a UTC day.

ND is a defined metered demand measure, not a promise to predict every unit of electricity consumed in the UK. Fixing the field name and units stops different models from accidentally solving different problems.

## 2. Draw the time boundaries

For the example issue on February 10 at 09:30 UTC:

| Region | Time range | May become an observed input? |
| --- | --- | --- |
| Seven-day context | February 3 00:00 to February 10 00:00, exclusive end | Yes, subject to evidence-mode availability |
| Issue-day gap | February 10 00:00 to February 11 00:00 | No |
| Target day | February 11 00:00 to February 12 00:00 | Labels for later evaluation only |

The last input interval starts February 9 at 23:30. The first forecast interval starts February 11 at 00:00. There is a complete missing day between them.

This is deliberate: the daily provider update does not give us a complete issue day at 09:30. We must not train on information our eventual application cannot receive.

The companion `forecast-timeline.html` is a step-through explanation. It contains no demand measurements or model predictions; its boxes represent documented time regions.

## 3. Why seven days gives 336 inputs

Each UTC day has 24 hours. Each hour has two half-hours. Therefore:

`7 days * 24 hours/day * 2 intervals/hour = 336 intervals`.

The issue-day gap contains 48 intervals and the target contains another 48. A sampler that takes the next 48 entries immediately after the input would be wrong: it would predict February 10 instead of February 11.

## 4. Horizon index is not lead time

Slot 0 is the first interval of the target day. Its start is 14 hours 30 minutes after issue, or 870 minutes. Slot 47 starts 38 hours after issue, or 2,280 minutes. Its end is another 30 minutes later.

Storing both slot and lead time makes charts and evaluation interpretable. Calling slot 0 a zero-minute-ahead prediction would misstate the problem.

## 5. A baseline can leak too

Weekly persistence means predicting a target half-hour with the value seven days earlier. February 11 00:00 uses February 4 00:00, which is inside our eligible context.

The last-available-day baseline uses February 9, not February 10. Although February 10 is the day before the target, its complete demand is unknown at our issue time. A seemingly reasonable 'yesterday' baseline can therefore cheat.

The executable checks verify every baseline reference interval falls inside the allowed context.

## 6. Event time is different from availability time

Suppose the February 9 23:30 reading is corrected after our February 10 issue. Its event time is still February 9. The corrected version was not available at issue.

In revised-data mode we use one frozen historical download and disclose that it contains revisions. In captured-availability mode we select only versions retrieved before issue. We cannot manufacture historical versions by downloading today's file.

That is why the machine contract includes a `first_seen_at` rule in addition to the interval cutoff. The first limits versions; the second limits when electricity was measured.

## 7. Why the calendar needs care

The source reports British settlement dates and period ordinals. When clocks change, those local days can contain 46 or 50 periods. Our target is a UTC day, which still contains 48 half-hours. Source conversion must handle local transitions before joining observations to the model grid.

The current checks prove that the UTC forecast contract stays fixed around clock changes. They do not yet implement or validate source settlement conversion; that belongs to Phase 2 and needs a timezone database.

## 8. Why a calibrator cannot update immediately

The uncertainty component learns from mistakes, but a mistake is only known after the actual arrives. A forecast issued February 10 targets February 11. Its full actual day is not available at the February 11 morning issue.

Our revised-data replay explicitly simulates those actuals arriving February 12 at 09:00. Captured replay uses real first-seen times. The state update queue must follow those release events, not the convenient ordering of rows in a notebook.

## 9. What has been built now

`configs/forecast_contract_v1.json` fixes the target, timing, gap, and interval interpretation. `scripts/forecast_contract.py` converts them into exact timestamps. Its tests exercise the worked origin, baseline eligibility, leap/year boundaries, clock-change dates, and incompatible contract edits.

The source catalog and research pins make external dependencies explicit. The experiment plan declares split roles before any model scores are examined. Phase 1 will install and lock the application environment; Phase 2 will capture full source bytes and perform the full quality audit.

## Check your understanding

1. Can we use February 10 evening demand for the February 10 morning issue? Why?
2. Why does downloading a February 9 measurement on February 12 not prove it was available on February 10?
3. Which date supplies the weekly baseline for February 11 00:00?
4. Do 48 marginal 90% intervals guarantee that the whole day fits inside the band?

Answers: 1. No; it is after issue and outside context. 2. The version's first-seen timestamp is too late. 3. February 4. 4. No; per-point coverage and whole-curve coverage are different objectives.
