"""Reconcile retained release evidence, raw timings and the public summaries."""

import gzip
import json

from gridlens.artifacts import sha256
from gridlens.benchmark import latency_summary
from gridlens.data.store import content_id
from scripts.reproduce_baseline import ROOT


def checked(path):
    report = json.loads(
        gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_text()
    )
    if report.get("report_id") != content_id({k: v for k, v in report.items() if k != "report_id"}):
        raise ValueError(f"Invalid report identity: {path.name}")
    return report


def main():
    folder = ROOT / "reports/release"
    point = checked(folder / "point-study.json")
    cal = checked(folder / "calibration-study.json")
    runtime = checked(folder / "runtime-study.json")
    load = checked(folder / "service-load.json")
    checked(folder / "validation-supplement.json")
    frozen = json.loads((ROOT / "configs/final_release_freeze_v1.json").read_text())
    if frozen["freeze_id"] != content_id({k: v for k, v in frozen.items() if k != "freeze_id"}):
        raise ValueError("Invalid final freeze")
    web = json.loads((ROOT / "apps/web/public/replay/summary.json").read_text())
    replay_path = ROOT / "apps/web/public/replay/evidence.json"
    replay = json.loads(replay_path.read_text())
    if replay["replay_id"] != content_id(
        {k: v for k, v in replay.items() if k != "replay_id"}
    ) or web["replay"]["sha256"] != sha256(replay_path):
        raise ValueError("Invalid public replay")
    if (
        point["original_report_id"] != frozen["point_study_report_id"]
        or point["champion_model_id"] != frozen["point_model_id"]
        or cal["point_model_id"] != frozen["point_model_id"]
    ):
        raise ValueError("Point/calibration freeze mismatch")
    if (
        web["point_study"]["report_id"] != point["original_report_id"]
        or web["calibration"]["report_id"] != cal["report_id"]
        or web["runtime"]["report_id"] != runtime["report_id"]
    ):
        raise ValueError("Stale public study summaries")
    neural = [row for row in point["candidate_metrics"] if row["seed"] is not None]
    if (
        len(neural) != 12
        or sorted(set(row["seed"] for row in neural)) != [17, 42, 101]
        or point["locked_test_evaluated"]
    ):
        raise ValueError("Incomplete declared point study")
    if len(runtime["raw_timings"]) != 3200 or len(runtime["cold_process_trials"]) != 24:
        raise ValueError("Incomplete runtime matrix")
    for condition in runtime["conditions"]:
        for name, metrics in condition["runtimes"].items():
            values = [
                row["elapsed_ms"]
                for row in runtime["raw_timings"]
                if row["runtime"] == name
                and row["threads"] == condition["threads"]
                and row["batch"] == condition["batch"]
            ]
            if len(values) != 200 or latency_summary(values) != metrics:
                raise ValueError("Runtime aggregates do not match raw timings")
    if len(load["raw_events"]) != load["offered"] or load["duration_seconds"] < 900:
        raise ValueError("Incomplete offered-load campaign")
    completed = [
        row["elapsed_ms"]
        for row in load["raw_events"]
        if row["status"] == 202 and row["state"] == "succeeded"
    ]
    reads = [row["elapsed_ms"] for row in load["raw_reads"] if row["status"] == 200]
    if (
        latency_summary(completed) != load["post_to_result"]
        or latency_summary(reads) != load["reads"]
    ):
        raise ValueError("Service aggregates do not match raw requests")
    if cal["selected"] == "ncc" and (
        not cal["coverage_gate_passed"] or cal["ncc_wis_improvement_fraction"] < 0.05
    ):
        raise ValueError("NCC promoted without its declared gates")
    if web["test"] is not None:
        test = checked(folder / "locked-test.json.gz")
        if (
            test["freeze_id"] != frozen["freeze_id"]
            or test["test_refitting"]
            or test["evaluation_campaign_count"] != 1
        ):
            raise ValueError("Invalid frozen test lineage")
        if {row["model_id"] for row in test["rows"]} != set(frozen["all_candidate_model_ids"]):
            raise ValueError("Test candidate registry changed")
        if web["test"]["report_id"] != test["report_id"]:
            raise ValueError("Stale public test summary")
    print(
        "Release identities, declared candidates, raw timing aggregates and public summaries verified"
    )


if __name__ == "__main__":
    main()
