"""Verify I2 report/source/timing lineage and render a small web evidence snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from gridlens.artifacts import sha256
from gridlens.data.store import content_id
from gridlens.inference.sdpa import runtime_record
from scripts.reproduce_baseline import verify_pins

ROOT = Path(__file__).resolve().parents[1]


def timing_bytes(folder: Path) -> bytes:
    return gzip.decompress((folder / "timings.jsonl.gz").read_bytes())


def checked_summary() -> dict:
    folder = ROOT / "reports/i2"
    report = json.loads((folder / "benchmark.json").read_text())
    if report["report_id"] != content_id({k: v for k, v in report.items() if k != "report_id"}):
        raise ValueError("I2 report digest mismatch")
    plan = json.loads((ROOT / "configs/i2_runtime_experiment_v1.json").read_text())
    parent = json.loads((ROOT / "configs/upgrade_experiment_v2.json").read_text())
    verify_pins(ROOT, parent)
    if report["experiment_plan_id"] != content_id(plan) or report["parent_plan_id"] != content_id(
        parent
    ):
        raise ValueError("I2 plan changed")
    runtime = runtime_record()
    # CI may install the same pinned version on a different host; source must still match.
    if report["runtime"]["adapter_sha256"] != runtime["adapter_sha256"] or report[
        "driver_sha256"
    ] != sha256(ROOT / "scripts/benchmark_sdpa.py"):
        raise ValueError("I2 source differs from benchmarked source; remeasure before publishing")
    if report["runtime"]["runtime_id"] != content_id(
        {k: v for k, v in report["runtime"].items() if k != "runtime_id"}
    ):
        raise ValueError("I2 runtime identity mismatch")
    raw = timing_bytes(folder)
    if (
        report["raw_timings"]["path"] != "timings.jsonl"
        or hashlib.sha256(raw).hexdigest() != report["raw_timings"]["sha256"]
    ):
        raise ValueError("I2 raw timing digest mismatch")
    if (
        report["target_dates"] != parent["benchmark_target_dates"]
        or report["locked_test_evaluated"]
        or report["evaluation_role"] != "point_validation"
        or report["dataset_manifest_id"] != parent["dataset_manifest_id"]
    ):
        raise ValueError("I2 corpus changed or locked-test evaluation occurred")
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    expected = parent["repetitions"] * parent["calls_per_repetition"]
    if len(rows) != report["raw_timings"]["rows"] or len(rows) != expected * 2 * 2 * 2:
        raise ValueError("Unexpected I2 timing count")
    baseline = json.loads((ROOT / "reports/i0-baseline-reproduction-v2.json").read_text())
    identities = {r["kind"]: r["model_id"] for r in baseline["results"]}
    trial_folder = folder / "preflight-trial"
    trial = json.loads((trial_folder / "benchmark.json").read_text())
    if (
        trial["report_id"] != content_id({k: v for k, v in trial.items() if k != "report_id"})
        or trial["runtime"]["adapter_sha256"] != sha256(trial_folder / "adapter-source.txt")
        or trial["raw_timings"]["sha256"] != hashlib.sha256(timing_bytes(trial_folder)).hexdigest()
    ):
        raise ValueError("Initial unsuccessful trial evidence changed")
    models = []
    import statistics

    from gridlens.benchmark import latency_summary

    for result in report["results"]:
        kind = result["kind"]
        if (
            result["model_id"] != identities[kind]
            or result["parity"]["max_abs_drift_mw"] > parent["exact_runtime_max_mw_drift"]
            or not result["parity"]["passed"]
        ):
            raise ValueError("I2 model identity/parity failure")
        for scope, comparison in result["comparisons"].items():
            for runtime_name, record in comparison["runtimes"].items():
                values = [
                    r["elapsed_ms"]
                    for r in rows
                    if r["model"] == kind and r["scope"] == scope and r["runtime"] == runtime_name
                ]
                if len(values) != expected or any(
                    record[k] != v for k, v in latency_summary(values).items()
                ):
                    raise ValueError("I2 timing aggregates disagree with raw samples")
        peaks = {
            name: statistics.median(
                t["after_corpus"]["process_lifetime_peak_rss_bytes"] for t in trials
            )
            for name, trials in result["isolated_memory_trials"].items()
        }
        models.append(
            {
                "kind": kind,
                "label": "Numerical Transformer" if kind == "transformer" else "TimeXer",
                "model_id": result["model_id"],
                "comparisons": result["comparisons"],
                "max_abs_drift_mw": result["parity"]["max_abs_drift_mw"],
                "rmse_mw": {k: v["rmse_mw"] for k, v in result["forecast_metrics"].items()},
                "median_process_peak_rss_bytes": peaks,
                "observed_cpu_sdpa_operators": result["observed_cpu_sdpa_operators"],
            }
        )
    return {
        "schema_version": "gridlens_i2_web_evidence_v1",
        "report_id": report["report_id"],
        "runtime_id": report["runtime"]["runtime_id"],
        "source_commit_at_measurement": report["environment"]["source_commit"],
        "cpu": report["environment"]["cpu_model"],
        "torch": report["environment"]["packages"]["torch"],
        "torch_threads": report["environment"]["torch_threads"],
        "target_dates": report["target_dates"],
        "calls_per_condition": expected,
        "raw_timing_count": len(rows),
        "warmup_calls_excluded": report["warmup_calls_excluded_per_condition"],
        "memory_trials_per_condition": plan["memory_processes_per_runtime"],
        "initial_trial": {
            "report_id": trial["report_id"],
            "bundle_p50_speedups": {
                r["kind"]: r["comparisons"]["bundle_predict"]["p50_speedup"]
                for r in trial["results"]
            },
        },
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "apps/web/data/i2-runtime.json"
    rendered = json.dumps(checked_summary(), indent=2, allow_nan=False) + "\n"
    if args.check:
        if target.read_text() != rendered:
            raise ValueError("I2 web evidence is stale")
        print("I2 sources, frozen weights/corpus, raw timings and web evidence verified")
    else:
        target.write_text(rendered)


if __name__ == "__main__":
    main()
