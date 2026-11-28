# Retained release evidence

- `point-study.json` retains configurations, all fitted seed metrics, learning curves and the original full-study report identity. Large aligned training/validation artifacts are retained separately.
- `validation-supplement.json` retains seasonal, weekday/weekend, fixed-date holiday-proxy, training-threshold high-demand and missing-history slices plus descriptive paired block intervals.
- `calibration-study.json` records fixed/rolling/NCC fitting, coverage gates, WIS selection, source identities and the rejected NCC promotion.
- `runtime-study.json` contains all 3,200 warm timings, 24 fresh-process trials, full-validation precision results, graph sizes and operator counts.
- `service-load.json` preserves every offered request and read, not only successful latency aggregates. The measured service used the checkpoint implementation before the optional ONNX path was added; that path has separate correctness and browser tests. The load host was shared with training work.
- `locked-test.json.gz` contains all seventeen candidates' aligned 2024 predictions, truth curves, metrics/slices, interval metrics and bootstrap comparisons. Its freeze was committed before scoring. Gzip compression preserves the JSON report identity.
- `postgres-recovery.json` records an actual dump/restore comparison.
- `public-smoke.json` records the earlier checkpoint's public fresh-compute probe. Final deployment evidence is retained separately when executed.

Run `python -m scripts.check_release_evidence` within the locked project environment to reconcile identities, raw timing aggregates, method gates and the public snapshots. Never interpret a running monitoring workflow as an observed seven-day availability record.
