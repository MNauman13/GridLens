"""Reconcile retained cache measurements and export the visitor-facing summary."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from gridlens.artifacts import sha256
from gridlens.benchmark import latency_summary
from gridlens.data.store import content_id

ROOT = Path(__file__).resolve().parents[1]


def summary():
    folder = ROOT / "reports/i2-cache"
    report = json.loads((folder / "benchmark.json").read_text())
    if report["report_id"] != content_id({k: v for k, v in report.items() if k != "report_id"}):
        raise ValueError("Cache report digest mismatch")
    plan = json.loads((ROOT / "configs/i2_cache_experiment_v1.json").read_text())
    if report["plan_id"] != content_id(plan) or report["locked_test_evaluated"]:
        raise ValueError("Cache plan/role mismatch")
    for path, digest in report["source_hashes"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Measured cache source changed")
    blob = (folder / "timings.jsonl.gz").read_bytes()
    raw = gzip.decompress(blob)
    if (
        hashlib.sha256(blob).hexdigest() != report["raw_timings"]["compressed_sha256"]
        or hashlib.sha256(raw).hexdigest() != report["raw_timings"]["uncompressed_sha256"]
    ):
        raise ValueError("Raw cache timing digest mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    if len(rows) != report["raw_timings"]["rows"] or len(rows) != 11000:
        raise ValueError("Incomplete cache timing matrix")
    for row in report["parity"]:
        if row["max_abs_drift_mw"] > 0.5 or row["batch_size"] not in (1, 4):
            raise ValueError("Cache parity failure")
    if set(report["heldout_parity_dates"]) & set(report["target_dates"]):
        raise ValueError("Heldout cache corpus overlaps")
    for condition in report["conditions"]:
        for name, metrics in condition["runtimes"].items():
            values = [
                r["elapsed_ms"]
                for r in rows
                if r["batch_size"] == condition["batch_size"]
                and r["scope"] == condition["scope"]
                and r["variant"] == name
            ]
            if len(values) != condition["calls_per_runtime"] or any(
                metrics[k] != v for k, v in latency_summary(values).items()
            ):
                raise ValueError("Cache summary disagrees with raw timings")
    return {
        "report_id": report["report_id"],
        "model_id": report["model_id"],
        "cpu": report["environment"]["cpu_model"],
        "torch": report["environment"]["packages"]["torch"],
        "conditions": report["conditions"],
        "max_abs_drift_mw": max(r["max_abs_drift_mw"] for r in report["parity"]),
        "origins": len(report["target_dates"]),
        "heldout_origins": len(report["heldout_parity_dates"]),
        "raw_samples": len(rows),
        "cache_bytes": {
            name: values["cache_samples"][0]["allocated_bytes"]
            for name, values in report["diagnostics"].items()
        },
        "step_p50_ms": {
            name: values["step_p50_ms"] for name, values in report["diagnostics"].items()
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    path = ROOT / "apps/web/data/i2-cache.json"
    text = json.dumps(summary(), indent=2) + "\n"
    if args.check:
        if path.read_text() != text:
            raise ValueError("Cache web snapshot is stale")
        print("Cache source, parity, raw timings and web snapshot verified")
    else:
        path.write_text(text)


if __name__ == "__main__":
    main()
