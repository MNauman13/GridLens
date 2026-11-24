# I0: baseline recovery and measurement

I0 rebuilds verified ridge, numerical Transformer and selected patch-48 TimeXer bundles. It changes no forecasting architecture and never scores 2024. Model identities from a fresh training run may differ from the original pilot, even with matching seed and software, because floating-point training is sensitive to execution environment.

## Install and recover

Use the repository's frozen Python environment:

```bash
uv sync --frozen --extra api --extra ml --group dev
uv run --frozen --extra ml gridlens ingest-neso --sources 2017 2018 2019 2020 2021 2022 2023 2024 --report reports/local/i0-source-audit.json
uv run --frozen --extra ml python -m scripts.reproduce_baseline --rebuild-neural --output reports/local/i0
uv run --frozen --extra ml python -m scripts.benchmark_baseline_cold --output reports/local/i0
```

The recovery command checks actual raw CSV hashes and the full rebuilt freeze before fitting. It fits ridge on point training and runs the original `scripts.train_neural.train` for precisely two existing configurations, seed 17, at most 12 epochs and the original patience. It does not execute the complete eight-candidate pilot, expand the search or select another champion.

The dataset builder rebuilds role membership and label fingerprints across the original freeze, including test membership. This is integrity verification, not inference or scoring on the test role: predictions, metrics, benchmark inputs and exported actuals come exclusively from 2022 point validation.

Outputs are immutable. Use a new output directory for another run. To reuse verified neural bundles rather than train again, copy the `bundles/` tree and `bundle_index.json` into a fresh output directory and omit `--rebuild-neural`. The warm runner refits ridge under the frozen contract and verifies all recovered neural metadata/weight manifests. It fails explicitly if the required files are missing. It never substitutes random weights or saved JSON for inference.

`bundle_index.json` is a resumable recovery pointer, not accepted public evidence by itself. Only the completed content-addressed reports and deployable bundle manifests establish what actually ran. Checkpoint files are excluded from the retained deployable archive.

## Timing scope

The first runner makes 10 warmup calls, then five repetitions of 200 timed calls per model across 32 frozen validation origins. It records every sample, repeat summaries and p50/p95/p99. Prediction-result caching is disabled. Loaded models are reused.

Its warm timing is `bundle.predict`: feature transformation, tensor creation, model forward, inverse scaling and contract validation are included. It is not directly comparable to the Phase 8 five-call engine-only timing. The supplement measures prepared-tensor neural forward separately, with the same warm protocol.

Cold measurements run 20 new interpreters per model and record parent startup-to-exit wall time, import/setup, verified model load and first prediction. They are process-cold, not proof of cold disk/network caches. All trials must reproduce an identical prediction fingerprint. RSS from the warm parent includes dataset/training allocation; fresh-process child memory is reported separately. Missing GPU/queue/calibration metrics are N/A with reasons.

No speedup claim follows merely from recording a faster number on another host. The output explicitly stores source commit and dirty-tree status, package versions, CPU visibility/affinity, memory limits where observable, Torch threads, artifact sizes and measurement scope. Regenerated-versus-original forecast differences are diagnostic; baseline comparisons for later optimisations use the new frozen weights.

## Output artifacts

| File | Meaning |
| --- | --- |
| `environment.json` | Machine/software facts without user paths or host names |
| `benchmark_manifest_v2.json` | Exact 32 dates, feature IDs and warm protocol |
| `baseline_reproduction.json` | Validation metrics, recovery identities, warm bundle timing and limitations |
| `baseline_runtime_supplement.json` | Cold-process samples and prepared-engine timing |
| `timings.jsonl`, `engine_timings.jsonl` | Raw timed observations |
| `validation-predictions.json` | Aligned actuals and predictions on 2022 only |
| `bundle_index.json`, `bundles/` | Recovery pointers and verified deployable bundles |
| `baseline-models-v2.zip`, `archive_inventory.json` | Explicit-file archive and member hashes from warm runner |

After the supplement, retain it and its raw traces alongside the archive, or create a new complete archive using `gridlens.benchmark.archive_files` with an explicit member list. Check the final ZIP SHA-256 after transfer. A file in an ignored local directory alone is not a durable archival copy.

## Verification

```bash
uv run --frozen --extra api --extra ml python -m unittest discover -s tests -v
uv run --frozen --extra api --extra ml ruff check src tests scripts/reproduce_baseline.py scripts/benchmark_baseline_cold.py
uv run --frozen --extra api --extra ml mypy src/gridlens
uv run --frozen --extra ml python scripts/check_neural_reference.py
uv run --frozen --extra ml python -m scripts.check_i0_reference
```

The new unit fixtures test percentile definitions, exclusion of warmups, corpus eligibility, manifest pins, immutability and explicit-file archive safety. They do not claim measured forecasting quality. Full data recovery and executed benchmarks provide that separate evidence.
