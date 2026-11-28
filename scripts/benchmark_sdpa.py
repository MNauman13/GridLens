"""Paired I2 CPU benchmark on verified I0 weights and the frozen I1 corpus."""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gridlens.artifacts import ArtifactManifest, safe_member, sha256, verify_manifest
from gridlens.benchmark import environment_record, latency_summary, memory_record, write_report
from gridlens.contracts import InputWindow
from gridlens.data.dataset import FeatureSnapshot
from gridlens.data.store import content_id
from gridlens.inference.sdpa import adapt_bundle, runtime_record
from gridlens.models.bundle import NeuralBundle
from scripts.reproduce_baseline import extended_metrics, verify_pins

ROOT = Path(__file__).resolve().parents[1]


def load_inputs(assets: Path) -> tuple[dict[str, Any], list[FeatureSnapshot], np.ndarray[Any, Any]]:
    manifest = ArtifactManifest.model_validate_json((assets / "manifest.json").read_text())
    if manifest.kind != "gridlens_i1_demo_v1":
        raise ValueError("Unexpected demo artifact")
    verify_manifest(assets, manifest)
    catalog = json.loads((assets / "client/catalog.json").read_text())
    if catalog["catalog_id"] != content_id({k: v for k, v in catalog.items() if k != "catalog_id"}):
        raise ValueError("Catalog digest mismatch")
    parent = json.loads((ROOT / "configs/upgrade_experiment_v2.json").read_text())
    freeze = json.loads((ROOT / "reports/dataset-freeze-v1.json").read_text())
    dates = [r["target_date"] for r in catalog["cases"]]
    if dates != parent["benchmark_target_dates"] or not set(dates).issubset(
        freeze["splits"]["point_validation"]["target_dates"]
    ):
        raise ValueError("I2 must use exactly the frozen 32 validation origins")
    if (
        catalog["dataset_manifest_id"] != parent["dataset_manifest_id"]
        or catalog["evaluation_role"] != "point_validation"
        or catalog["evidence_mode"] != "retrospective_revised"
    ):
        raise ValueError("Unexpected corpus lineage")
    snapshots, actual = [], []
    for entry in catalog["cases"]:
        path = safe_member(assets / "client", entry["asset"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Case digest mismatch")
        case = json.loads(path.read_text())
        inputs = InputWindow.model_validate(case["inputs"])
        if (
            inputs.target_start.date().isoformat() != entry["target_date"]
            or inputs.feature_snapshot_id != entry["feature_snapshot_id"]
        ):
            raise ValueError("Case timing/identity mismatch")
        snapshots.append(
            FeatureSnapshot(
                inputs,
                tuple(map(tuple, case["history"])),
                tuple(map(tuple, case["target_calendar"])),
            )
        )
        actual.append(case["observed_mw"])
    return catalog, snapshots, np.asarray(actual)


def load_model(baseline: Path, kind: str, runtime: str) -> tuple[NeuralBundle, int]:
    report = json.loads((baseline / "baseline_reproduction.json").read_text())
    row = next(r for r in report["results"] if r["kind"] == kind)
    committed = json.loads((ROOT / "reports/i0-baseline-reproduction-v2.json").read_text())
    if row["model_id"] != next(r["model_id"] for r in committed["results"] if r["kind"] == kind):
        raise ValueError("I2 cannot compare freshly retrained weights")
    bundle = NeuralBundle.load_bundle(safe_member(baseline, row["bundle"]))
    if bundle.metadata["identity"] != row["model_id"]:
        raise ValueError("Bundle identity mismatch")
    if runtime == "manual":
        return bundle, 0
    if runtime != "sdpa":
        raise ValueError("Unknown runtime")
    return adapt_bundle(bundle)


def memory_child(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    _, snapshots, _ = load_inputs(args.assets)
    before = memory_record()
    start = time.perf_counter_ns()
    bundle, count = load_model(args.baseline, args.kind, args.runtime)
    load_ms = (time.perf_counter_ns() - start) / 1e6
    gc.collect()
    loaded = memory_record()
    tick = time.perf_counter_ns()
    bundle.predict(snapshots[0])
    first_ms = (time.perf_counter_ns() - tick) / 1e6
    for snapshot in snapshots:
        bundle.predict(snapshot)
    print(
        json.dumps(
            {
                "before_model": before,
                "after_model": loaded,
                "after_corpus": memory_record(),
                "load_and_adapt_ms": load_ms,
                "first_predict_ms": first_ms,
                "replaced_attention_modules": count,
            }
        )
    )


def block_interval(rows: list[dict[str, Any]], seed: int) -> list[float]:
    means = np.asarray(
        [
            [
                np.mean(
                    [
                        r["elapsed_ms"]
                        for r in rows
                        if r["repeat"] == repeat and r["runtime"] == runtime
                    ]
                )
                for runtime in ("manual", "sdpa")
            ]
            for repeat in range(5)
        ]
    )
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, 5, size=(2000, 5))].mean(axis=1)
    return np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975]).tolist()


def benchmark(args: argparse.Namespace) -> None:
    parent = json.loads((ROOT / "configs/upgrade_experiment_v2.json").read_text())
    plan = json.loads((ROOT / "configs/i2_runtime_experiment_v1.json").read_text())
    verify_pins(ROOT, parent)
    if plan["parent_plan_id"] != content_id(parent):
        raise ValueError("Parent experiment changed")
    torch.set_num_threads(parent["torch_threads"])
    torch.use_deterministic_algorithms(True)
    catalog, snapshots, actual = load_inputs(args.assets)
    environment = environment_record(ROOT)
    environment.update(
        {
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    )
    results = []
    all_rows = []
    for kind in ("transformer", "timexer"):
        manual, _ = load_model(args.baseline, kind, "manual")
        candidate, count = adapt_bundle(manual)
        bundles = {"manual": manual, "sdpa": candidate}
        prepared = [
            (
                torch.tensor([manual.scaling.transform(s.history)], dtype=torch.float32),
                torch.tensor([s.target_calendar], dtype=torch.float32),
            )
            for s in snapshots
        ]
        predictions = {runtime: [] for runtime in bundles}
        normal_drift = []
        for index, snapshot in enumerate(snapshots):
            with torch.no_grad():
                reference = manual.model(*prepared[index])
                optimised = candidate.model(*prepared[index])
            torch.testing.assert_close(
                optimised,
                reference,
                atol=parent["exact_runtime_normalized_atol"],
                rtol=parent["exact_runtime_normalized_rtol"],
            )
            normal_drift.append(float((optimised - reference).abs().max()))
            for runtime, bundle in bundles.items():
                predictions[runtime].append(bundle.predict(snapshot).prediction_mw)
        arrays = {k: np.asarray(v) for k, v in predictions.items()}
        mw_drift = float(np.abs(arrays["sdpa"] - arrays["manual"]).max())
        if mw_drift > parent["exact_runtime_max_mw_drift"]:
            raise ValueError("MW parity gate failed")
        metrics = {
            runtime: extended_metrics(actual, prediction) for runtime, prediction in arrays.items()
        }
        if metrics["sdpa"]["rmse_mw"] > metrics["manual"]["rmse_mw"] * (
            1 + parent["approximate_runtime_max_relative_rmse_degradation"]
        ):
            raise ValueError("Accuracy regression gate failed")
        comparisons = {}
        for scope in ("prepared_engine", "bundle_predict"):

            def call(
                runtime: str,
                case: int,
                *,
                measured_scope: str = scope,
                engines: dict[str, NeuralBundle] = bundles,
                tensors: list = prepared,
            ) -> None:
                if measured_scope == "bundle_predict":
                    engines[runtime].predict(snapshots[case])
                else:
                    with torch.no_grad():
                        engines[runtime].model(*tensors[case])

            for runtime in bundles:
                for index in range(parent["warmup_calls"]):
                    call(runtime, index % len(snapshots))
            rows = []
            rng = random.Random(plan["order_seed"])
            for repeat in range(parent["repetitions"]):
                for index in range(parent["calls_per_repetition"]):
                    case = (repeat * parent["calls_per_repetition"] + index) % len(snapshots)
                    order = ["manual", "sdpa"]
                    rng.shuffle(order)
                    for position, runtime in enumerate(order):
                        tick = time.perf_counter_ns()
                        call(runtime, case)
                        rows.append(
                            {
                                "model": kind,
                                "scope": scope,
                                "runtime": runtime,
                                "repeat": repeat,
                                "case_index": case,
                                "order_in_pair": position,
                                "elapsed_ms": (time.perf_counter_ns() - tick) / 1e6,
                            }
                        )
            summaries = {}
            for runtime in bundles:
                selected = [r for r in rows if r["runtime"] == runtime]
                times = [r["elapsed_ms"] for r in selected]
                summaries[runtime] = {
                    **latency_summary(times),
                    "sequential_capacity_forecasts_per_second": 1000 / float(np.mean(times)),
                    "per_repeat": [
                        latency_summary(
                            [r["elapsed_ms"] for r in selected if r["repeat"] == repeat]
                        )
                        for repeat in range(parent["repetitions"])
                    ],
                }
            comparisons[scope] = {
                "runtimes": summaries,
                "p50_speedup": summaries["manual"]["p50_ms"] / summaries["sdpa"]["p50_ms"],
                "p95_speedup": summaries["manual"]["p95_ms"] / summaries["sdpa"]["p95_ms"],
                "mean_speedup_95pct_block_bootstrap": block_interval(rows, plan["order_seed"]),
            }
            all_rows.extend(rows)
        with (
            torch.no_grad(),
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile,
        ):
            candidate.model(*prepared[0])
        kernels = sorted(
            {event.key for event in profile.key_averages() if "scaled_dot_product" in event.key}
        )
        memory = {}
        for runtime in bundles:
            trials = []
            for _ in range(plan["memory_processes_per_runtime"]):
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.benchmark_sdpa",
                        "--memory-child",
                        "--kind",
                        kind,
                        "--runtime",
                        runtime,
                        "--baseline",
                        str(args.baseline),
                        "--assets",
                        str(args.assets),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
                trials.append(json.loads(process.stdout))
            memory[runtime] = trials
        results.append(
            {
                "kind": kind,
                "model_id": manual.metadata["identity"],
                "replaced_attention_modules": count,
                "parity": {
                    "normalised_max_abs_drift": max(normal_drift),
                    "max_abs_drift_mw": mw_drift,
                    "per_origin_normalised_max_abs_drift": normal_drift,
                    "passed": True,
                },
                "forecast_metrics": metrics,
                "comparisons": comparisons,
                "observed_cpu_sdpa_operators": kernels,
                "isolated_memory_trials": memory,
                "predictions_mw": predictions,
            }
        )
        print(
            json.dumps(
                {
                    "kind": kind,
                    "max_abs_drift_mw": mw_drift,
                    "bundle_p50_speedup": comparisons["bundle_predict"]["p50_speedup"],
                    "bundle_p95_speedup": comparisons["bundle_predict"]["p95_speedup"],
                }
            ),
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "timings.jsonl"
    from gridlens.data.store import write_once

    write_once(
        raw, "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in all_rows).encode()
    )
    report = {
        "schema_version": "gridlens_i2_runtime_benchmark_v1",
        "phase": "I2",
        "experiment_plan_id": content_id(plan),
        "parent_plan_id": content_id(parent),
        "dataset_manifest_id": catalog["dataset_manifest_id"],
        "catalog_id": catalog["catalog_id"],
        "target_dates": [r["target_date"] for r in catalog["cases"]],
        "feature_snapshot_ids": [s.inputs.feature_snapshot_id for s in snapshots],
        "runtime": runtime_record(),
        "environment": environment,
        "driver_sha256": sha256(Path(__file__)),
        "evaluation_role": "point_validation",
        "evidence_mode": "retrospective_revised",
        "locked_test_evaluated": False,
        "batch_size": 1,
        "prediction_result_cache": "disabled",
        "warmup_calls_excluded_per_condition": parent["warmup_calls"],
        "timing_scope": "paired sequential CPU calls; bundle includes preprocessing/denormalisation/contract; prepared excludes those; neither is HTTP/browser throughput",
        "memory_scope": "three fresh interpreters per condition; imports and corpus included; peak is process lifetime; OS file cache may be warm; no GPU metrics",
        "raw_timings": {"path": "timings.jsonl", "sha256": sha256(raw), "rows": len(all_rows)},
        "results": results,
    }
    write_report(args.output / "benchmark.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports/local/i0")
    parser.add_argument("--assets", type=Path, default=ROOT / "artifacts/i1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/local/i2")
    parser.add_argument("--memory-child", action="store_true")
    parser.add_argument("--kind", choices=("transformer", "timexer"), default="timexer")
    parser.add_argument("--runtime", choices=("manual", "sdpa"), default="manual")
    args = parser.parse_args()
    if args.memory_child:
        memory_child(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
