# I2 checked SDPA inference

## Reproduce measurements

Keep the recovered I0 neural bundles in reports/local/i0. The I1 public archive supplies the same frozen 32 feature windows and observed validation curves; it does not supply numerical Transformer weights. Recover I0 first if those retained weights are absent.

```bash
uv sync --frozen --extra api --extra ml --group dev
npm ci
node scripts/prepare_live_assets.mjs
uv run --frozen --extra api --extra ml python -m scripts.benchmark_sdpa --output reports/local/i2-new-run
```

Use a fresh output directory because evidence writes are immutable. The benchmark verifies source/data pins, model identities and artifact digests before comparing. It aborts if normalised prediction agreement, the 0.5 MW maximum drift gate or the forecast-RMSE regression gate fails. It records 8,000 paired timing observations per experiment, both scopes, complete prediction arrays, forecast metrics, actual SDPA profiler operators, hardware/software and three fresh-process memory trials per runtime/model. No training or test evaluation occurs. Avoid other sustained CPU workloads while measuring.

The retained preflight trial used the same protocol but a slower, unconditional score-validation matmul. Its source is archived next to its report. The later range-certificate experiment is the main report. Do not overwrite the rejected trial or compare separate hosts/runs as a paired speedup.

## Use the adapter

```python
from pathlib import Path
from gridlens.models.bundle import NeuralBundle
from gridlens.inference.sdpa import adapt_bundle, runtime_record

reference = NeuralBundle.load_bundle(Path("path/to/verified/neural/bundle"))
optimised, replacements = adapt_bundle(reference)
# snapshot must be a complete, validated FeatureSnapshot.
forecast = optimised.predict(snapshot)
print(replacements, runtime_record())
```

No training API is provided by this adapter. Original source modules are untouched, and the independent copy retains the same weights and state keys. Explicit attention inspection uses `need_weights=True` on the adapted attention module and takes the manual path. That inspection mode is excluded from SDPA timings.

## Optional worker integration

```bash
GRIDLENS_LIVE_ASSETS=artifacts/i1 GRIDLENS_SDPA_ENABLED=1 GRIDLENS_JOB_DB=reports/local/i1/jobs.sqlite3 uv run --frozen --extra api --extra ml uvicorn gridlens.api:app --host 127.0.0.1 --port 8000 --workers 1
# Separate terminal:
GRIDLENS_API_URL=http://127.0.0.1:8000 npm start
```

In `/live`, select Python reference service, TimeXer, then an enabled attention runtime. The runtime capability is returned separately from the immutable model/input catalog. All 32 startup forecasts must pass normalised and MW parity before the SDPA option is advertised. Jobs record the chosen runtime in SQLite and the result includes its source/software identity. Ridge has no attention runtime option. Missing/disabled/unknown runtimes return 422; there is no automatic promotion or substitution.

The I1 one-process/SQLite, admission, ownership and hosting constraints still apply. Enabling SDPA keeps both reference and candidate TimeXer resident, so the serving process has a different memory scope from a single-engine child benchmark. The public deployment has no Python host attached and does not enable this selector. Its browser predictions still run through WASM/JavaScript.

## Verify retained evidence

```bash
uv run --frozen --extra api --extra ml python -m scripts.export_i2_evidence --check
```

This checks source hashes, frozen model/corpus identities, timing digests and aggregate agreement, the retained rejected trial and the web snapshot. The public `/experiments#inference` section distinguishes bundle versus prepared timing, CPU versus browser runtime and process RSS versus tensor memory. Benchmark speedups are conditional on this host and workload, not general API throughput or improved forecast accuracy.

Committed raw timing streams use lossless gzip with a zero timestamp. Decompress `reports/i2/timings.jsonl.gz` (and the corresponding preflight-trial file) to analyse every call; report hashes identify the uncompressed streams. The checker handles decompression automatically.
