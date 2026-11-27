# Incremental decoder evidence

Executed on 27 November 2026 against the exact I0 numerical Transformer, CPU float32, two threads. `configs/i2_cache_experiment_v1.json` declares the experiment. No training weights or original operators changed.

`benchmark.json` contains source identities, aligned parity results at B1/B4, all predictions and forecast metrics, per-slot instrumentation, operator counts, estimated linear-layer MACs and three fresh-process memory trials per variant. `timings.jsonl.gz` retains all 11,000 uninstrumented timings, with both compressed and uncompressed digests. Five runtime variants are interleaved in random order per matched case.

B1 prepared medians were 34.039 ms reference, 30.496 ms SDPA, 32.954 ms cross-only, 22.073 ms full cache, and 21.548 ms full cache with SDPA. Manual full caching meets the 1.5× stretch target. The extra B1 gain from SDPA is below 10%, so the simpler manual cache is the batch-one candidate. B4 benefits more from SDPA. This does not change the deployed point model or browser runtime.

Maximum MW drift was 0.022935 across 32 benchmark dates and eight disjoint 2022 dates. Normalised tensors also pass atol/rtol 1e-5. Tests exercise multilayer B1/B4, repeated calls, interleaved sessions, cancellation, early close, input/output mutation, poisoned unused slots, save/reload and concurrent requests. Cache tensors are request-local, including self-attention keys and values at the exact 384+slot positional offset. The SDPA single-query cached path deliberately does not use a non-square causal mask.

Batch-one full key/value allocation is 98,304 bytes. This is not process RSS or peak activation memory. Linear MAC estimates exclude non-linear operators and Python validation. Per-step instrumentation and process RSS are measured separately from primary timing.

Reproduce after installing the verified baseline and live archives with the preparation scripts.

```
uv run --frozen --extra api --extra ml python -m scripts.benchmark_cached --output reports/local/cache-reproduction
uv run --frozen --extra api --extra ml python -m unittest tests.test_cached -v
uv run --frozen --extra api --extra ml python -m scripts.export_cache_evidence --check
```

The public experiments page shows these recorded CPU measurements. Its browser inference has a separate execution scope. The 2024 test role was not evaluated. The earlier interrupted run is not used as benchmark evidence.
