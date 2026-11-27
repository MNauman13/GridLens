"""Frozen same-weight cache ablations, real B1/B4 parity and request memory."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gridlens.artifacts import sha256
from gridlens.benchmark import environment_record, latency_summary, memory_record, write_report
from gridlens.data.store import content_id, write_once
from gridlens.inference.cached import CacheDiagnostics, adapt_cached_bundle, cache_runtime_record
from gridlens.inference.sdpa import adapt_bundle
from scripts.benchmark_sdpa import ROOT, load_inputs, load_model
from scripts.reproduce_baseline import extended_metrics, frozen_data, verify_pins

VARIANTS = ("reference", "sdpa_only", "cross_kv", "full_kv", "full_kv_sdpa")


def engine(baseline: Path, variant: str):
    if variant not in VARIANTS:
        raise ValueError("Unknown cache variant")
    reference, _ = load_model(baseline, "transformer", "manual")
    if variant == "reference":
        return reference
    if variant == "sdpa_only":
        return adapt_bundle(reference)[0]
    return adapt_cached_bundle(
        reference,
        "cross_kv" if variant == "cross_kv" else "full_kv",
        "sdpa" if variant == "full_kv_sdpa" else "manual",
    )


def prepared(bundle, snapshots, batch):
    return [
        (
            torch.tensor(
                [
                    bundle.scaling.transform(snapshots[(start + j) % len(snapshots)].history)
                    for j in range(batch)
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [snapshots[(start + j) % len(snapshots)].target_calendar for j in range(batch)],
                dtype=torch.float32,
            ),
        )
        for start in range(0, len(snapshots), batch)
    ]


def memory_child(args):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    _, snapshots, _ = load_inputs(args.assets)
    before = memory_record()
    bundle = engine(args.baseline, args.variant)
    gc.collect()
    loaded = memory_record()
    h, c = prepared(bundle, snapshots, 1)[0]
    stats = CacheDiagnostics()
    with torch.no_grad():
        if args.variant in ("reference", "sdpa_only"):
            bundle.model(h, c)
        else:
            bundle.model(h, c, diagnostics=stats)
    print(
        json.dumps(
            {
                "before_model": before,
                "loaded": loaded,
                "after_forecast": memory_record(),
                "cache_allocated_bytes": stats.cache_allocated_bytes,
                "cache_used_bytes": stats.cache_used_bytes,
            }
        )
    )


def bootstrap(rows, candidate):
    means = np.asarray(
        [
            [
                np.mean(
                    [
                        r["elapsed_ms"]
                        for r in rows
                        if r["repeat"] == repeat and r["variant"] == name
                    ]
                )
                for name in ("reference", candidate)
            ]
            for repeat in range(5)
        ]
    )
    rng = np.random.default_rng(17)
    resamples = means[rng.integers(0, 5, (2000, 5))].mean(axis=1)
    return np.quantile(resamples[:, 0] / resamples[:, 1], [0.025, 0.975]).tolist()


def diagnostics(bundle, inputs):
    curves = []
    walls = []
    cache = []
    # Identical boundaries for cached and reference: target projection entry to
    # forecast-head exit, excluding argument preparation and yielded-value cloning.
    for h, c in inputs:
        slots = []
        state = {"started": 0, "slots": slots}

        def begin(module, args, *, state=state):
            state["started"] = time.perf_counter_ns()

        def finish(module, args, output, *, state=state):
            state["slots"].append((time.perf_counter_ns() - state["started"]) / 1e6)

        handles = [
            bundle.model.target_projection.register_forward_pre_hook(begin),
            bundle.model.output.register_forward_hook(finish),
        ]
        stats = CacheDiagnostics()
        try:
            tick = time.perf_counter_ns()
            with torch.no_grad():
                if hasattr(bundle.model, "iter_forecast"):
                    bundle.model(h, c, diagnostics=stats)
                else:
                    bundle.model(h, c)
            walls.append((time.perf_counter_ns() - tick) / 1e6)
        finally:
            for handle in handles:
                handle.remove()
        if len(slots) != 48:
            raise ValueError("Incomplete decoder-step diagnostic")
        curves.append(slots)
        cache.append(
            {
                "allocated_bytes": stats.cache_allocated_bytes,
                "used_bytes": stats.cache_used_bytes,
                "closed": stats.closed,
                "stage_ms": stats.stage_ms,
            }
        )
    # Operator self times are exclusive profiler observations; the unattributed
    # wall residual includes Python AND profiling overhead, not pure loop cost.
    macs = 0

    def linear(module, args, output):
        nonlocal macs
        macs += (args[0].numel() // module.in_features) * module.in_features * module.out_features

    handles = [
        m.register_forward_hook(linear)
        for m in bundle.model.modules()
        if isinstance(m, torch.nn.Linear)
    ]
    try:
        with (
            torch.no_grad(),
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile,
        ):
            bundle.model(*inputs[0])
    finally:
        for handle in handles:
            handle.remove()
    operators = [
        {
            "name": e.key,
            "count": e.count,
            "self_cpu_us": e.self_cpu_time_total,
            "inclusive_cpu_us": e.cpu_time_total,
        }
        for e in profile.key_averages()
    ]
    return {
        "scope": "diagnostic hooks enabled; target-projection entry to output-head exit; excludes pre-call concatenation and yield copy; not headline timings",
        "step_curves_ms": curves,
        "step_p50_ms": np.median(curves, axis=0).tolist(),
        "step_p95_ms": np.quantile(curves, 0.95, axis=0).tolist(),
        "diagnostic_forecast_wall_ms": walls,
        "cache_samples": cache,
        "linear_multiply_accumulates_one_forecast": macs,
        "operator_profile": operators,
    }


def run(args):
    parent = json.loads((ROOT / "configs/upgrade_experiment_v2.json").read_text())
    plan = json.loads((ROOT / "configs/i2_cache_experiment_v1.json").read_text())
    verify_pins(ROOT, parent)
    if plan["parent_plan_id"] != content_id(parent):
        raise ValueError("Frozen parent changed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    catalog, snapshots, actual = load_inputs(args.assets)
    expected = json.loads((ROOT / "reports/dataset-freeze-v1.json").read_text())
    datasets, _, _ = frozen_data(ROOT, args.sources, expected)
    validation = datasets["point_validation"]
    membership = {str(s.inputs.target_start.date()): s for s in validation.features}
    heldout = [membership[day] for day in plan["heldout_parity_dates"]]
    del datasets, validation
    bundles = {name: engine(args.baseline, name) for name in VARIANTS}
    tensors = {batch: prepared(bundles["reference"], snapshots, batch) for batch in (1, 4)}
    parity = []
    predictions = {name: [] for name in VARIANTS}
    for corpus, windows in [("benchmark", snapshots), ("heldout", heldout)]:
        for batch in (1, 4):
            for index, (h, c) in enumerate(prepared(bundles["reference"], windows, batch)):
                with torch.no_grad():
                    reference = bundles["reference"].model(h, c)
                for name, bundle in bundles.items():
                    with torch.no_grad():
                        candidate = bundle.model(h, c)
                    torch.testing.assert_close(candidate, reference, atol=1e-5, rtol=1e-5)
                    drift = float((candidate - reference).abs().max()) * bundle.scaling.scales[0]
                    if drift > 0.5:
                        raise ValueError("Cache MW parity gate failed")
                    parity.append(
                        {
                            "corpus": corpus,
                            "batch_size": batch,
                            "batch_index": index,
                            "variant": name,
                            "normalised_max_abs_drift": float((candidate - reference).abs().max()),
                            "max_abs_drift_mw": drift,
                        }
                    )
                    if corpus == "benchmark" and batch == 1:
                        predictions[name].append(bundle.predict(windows[index]).prediction_mw)
    conditions = []
    raw = []
    for batch, scope, calls in [
        (1, "prepared_engine", 200),
        (1, "bundle_predict", 200),
        (4, "prepared_engine", 40),
    ]:

        def call(name, case, *, bs=batch, sc=scope):
            if sc == "bundle_predict":
                bundles[name].predict(snapshots[case])
            else:
                with torch.no_grad():
                    bundles[name].model(*tensors[bs][case])

        cases = len(snapshots) if scope == "bundle_predict" else len(tensors[batch])
        for name in VARIANTS:
            for i in range(10):
                call(name, i % cases)
        rows = []
        rng = random.Random(17)
        for repeat in range(5):
            for index in range(calls):
                case = (repeat * calls + index) % cases
                order = list(VARIANTS)
                rng.shuffle(order)
                for position, name in enumerate(order):
                    tick = time.perf_counter_ns()
                    call(name, case)
                    rows.append(
                        {
                            "batch_size": batch,
                            "scope": scope,
                            "variant": name,
                            "repeat": repeat,
                            "case_index": case,
                            "order_in_block": position,
                            "elapsed_ms": (time.perf_counter_ns() - tick) / 1e6,
                        }
                    )
        summaries = {}
        for name in VARIANTS:
            subset = [r for r in rows if r["variant"] == name]
            values = [r["elapsed_ms"] for r in subset]
            summaries[name] = {
                **latency_summary(values),
                "forecasts_per_second": batch * 1000 / float(np.mean(values)),
                "output_points_per_second": batch * 48 * 1000 / float(np.mean(values)),
                "per_repeat": [
                    latency_summary([r["elapsed_ms"] for r in subset if r["repeat"] == i])
                    for i in range(5)
                ],
            }
        ratios = {
            name: {
                "p50_speedup": summaries["reference"]["p50_ms"] / summaries[name]["p50_ms"],
                "p95_speedup": summaries["reference"]["p95_ms"] / summaries[name]["p95_ms"],
                "mean_speedup_95pct_block_bootstrap": bootstrap(rows, name),
            }
            for name in VARIANTS
            if name != "reference"
        }
        conditions.append(
            {
                "batch_size": batch,
                "scope": scope,
                "calls_per_runtime": 5 * calls,
                "runtimes": summaries,
                "comparisons": ratios,
            }
        )
        raw.extend(rows)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "in_progress.json").write_text(
            json.dumps({"conditions": conditions, "rows": raw}, separators=(",", ":"))
        )
        print(
            json.dumps(
                {
                    "batch_size": batch,
                    "scope": scope,
                    "p50_ms": {k: v["p50_ms"] for k, v in summaries.items()},
                }
            ),
            flush=True,
        )
    profiles = {name: diagnostics(bundle, tensors[1]) for name, bundle in bundles.items()}
    memory = {}
    for name in VARIANTS:
        memory[name] = []
        for _ in range(3):
            p = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.benchmark_cached",
                    "--memory-child",
                    "--variant",
                    name,
                    "--baseline",
                    str(args.baseline),
                    "--assets",
                    str(args.assets),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
                timeout=60,
            )
            memory[name].append(json.loads(p.stdout))
    encoded = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in raw).encode()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "timings.jsonl.gz"
    write_once(raw_path, gzip.compress(encoded, mtime=0))
    import hashlib

    environment = environment_record(ROOT)
    environment["torch_threads"] = torch.get_num_threads()
    report = {
        "schema_version": "gridlens_i2_cache_benchmark_v1",
        "phase": "I2-cache",
        "plan_id": content_id(plan),
        "model_id": bundles["reference"].metadata["identity"],
        "dataset_manifest_id": catalog["dataset_manifest_id"],
        "catalog_id": catalog["catalog_id"],
        "target_dates": [r["target_date"] for r in catalog["cases"]],
        "heldout_parity_dates": plan["heldout_parity_dates"],
        "runtime_records": {
            name: cache_runtime_record(
                "cross_kv" if name == "cross_kv" else "full_kv",
                "sdpa" if name == "full_kv_sdpa" else "manual",
            )
            for name in VARIANTS
            if name not in ("reference", "sdpa_only")
        },
        "source_hashes": {
            p: sha256(ROOT / p)
            for p in (
                "src/gridlens/inference/cached.py",
                "scripts/benchmark_cached.py",
                "src/gridlens/inference/sdpa.py",
            )
        },
        "environment": environment,
        "evaluation_role": "point_validation",
        "evidence_mode": "retrospective_revised",
        "locked_test_evaluated": False,
        "parity": parity,
        "conditions": conditions,
        "diagnostics": profiles,
        "isolated_memory_trials": memory,
        "predictions_mw": predictions,
        "forecast_metrics": {
            name: extended_metrics(actual, np.asarray(values))
            for name, values in predictions.items()
        },
        "raw_timings": {
            "file": "timings.jsonl.gz",
            "uncompressed_sha256": hashlib.sha256(encoded).hexdigest(),
            "compressed_sha256": sha256(raw_path),
            "rows": len(raw),
        },
        "limitations": [
            "Batch-4 performance is a preregistered 200-call secondary probe; batch-1 has 1000 calls per runtime/scope.",
            "Cache metrics count request-local self/cross K/V tensors only, not weights, encoder memory, interpreter or result tensors.",
            "Diagnostic hooks/profiling are separate from headline timings. Operator self times are not whole validation or Python-loop costs.",
            "Local sequential CPU capacity is not HTTP throughput; no GPU or public-browser speedup claim.",
        ],
    }
    write_report(args.output / "benchmark.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports/local/i0")
    parser.add_argument("--assets", type=Path, default=ROOT / "artifacts/i1")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/neso")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/local/i2-cache")
    parser.add_argument("--memory-child", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, default="full_kv")
    args = parser.parse_args()
    if args.memory_child:
        memory_child(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
