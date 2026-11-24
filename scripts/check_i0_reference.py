"""Verify committed I0 evidence and frozen policies without loading test outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from gridlens.benchmark import validate_development_dates
from gridlens.data.store import content_id
from scripts.reproduce_baseline import verify_pins

ROOT = Path(__file__).resolve().parents[1]


def checked_report(path: Path) -> dict:
    record = json.loads(path.read_text())
    if record.get("report_id") != content_id({k: v for k, v in record.items() if k != "report_id"}):
        raise ValueError(f"Invalid report digest: {path.name}")
    return record


def main() -> None:
    plan = json.loads((ROOT / "configs/upgrade_experiment_v2.json").read_text())
    verify_pins(ROOT, plan)
    freeze = json.loads((ROOT / "reports/dataset-freeze-v1.json").read_text())
    allowed = freeze["splits"]["point_validation"]["target_dates"]
    validate_development_dates(plan["benchmark_target_dates"], allowed)
    report = checked_report(ROOT / "reports/i0-baseline-reproduction-v2.json")
    supplemental = checked_report(ROOT / "reports/i0-runtime-supplement-v2.json")
    if report["experiment_plan_id"] != content_id(plan) or report["dataset_manifest_id"] != freeze["manifest_id"]:
        raise ValueError("I0 evidence does not match frozen experiment/data")
    if report["locked_test_evaluated"] or supplemental["locked_test_evaluated"]:
        raise ValueError("I0 must never evaluate locked test")
    if report["validation_target_dates"] != allowed or report["evaluated_roles"] != ["point_validation"]:
        raise ValueError("I0 evaluated unexpected origins/roles")
    if supplemental["baseline_report_id"] != report["report_id"]:
        raise ValueError("Runtime supplement has different baseline lineage")
    if supplemental["experiment_plan_id"] != content_id(plan):
        raise ValueError("Runtime supplement has a different experiment plan")
    rows = report["results"]
    if len(rows) != 3 or {row["kind"] for row in rows} != {"ridge", "transformer", "timexer"}:
        raise ValueError("Missing/duplicate baseline model evidence")
    if len(supplemental["results"]) != 3 or {
        row["kind"]: row["model_id"] for row in supplemental["results"]
    } != {row["kind"]: row["model_id"] for row in rows}:
        raise ValueError("Runtime supplement has missing or incompatible model identities")
    if report["split_origin_counts"] != {
        role: len(split["target_dates"]) for role, split in freeze["splits"].items()
    }:
        raise ValueError("Dataset membership counts changed")
    expected_calls = plan["repetitions"] * plan["calls_per_repetition"]
    for row in rows:
        summary = row["bundle_predict_latency"]
        if row["metrics"]["origins"] != len(allowed) or summary["calls"] != expected_calls:
            raise ValueError("Unexpected validation/benchmark sample count")
        if not 0 <= summary["p50_ms"] <= summary["p95_ms"] <= summary["p99_ms"]:
            raise ValueError("Invalid latency distribution")
    for row in supplemental["results"]:
        if row["cold_process_count"] != 20 or len(row["cold_raw_trials"]) != 20:
            raise ValueError("Missing cold-process trials")
        if len({item["prediction_fingerprint"] for item in row["cold_raw_trials"]}) != 1:
            raise ValueError("Cold-process predictions differ")
    print("I0 pins, three recovered baselines, warm/cold samples and locked-test policy verified")


if __name__ == "__main__":
    main()
