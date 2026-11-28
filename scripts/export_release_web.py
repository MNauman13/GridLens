"""Derive public summaries from measured reports; do not invent pending results."""

import argparse
import json

from gridlens.artifacts import sha256
from gridlens.data.store import content_id
from scripts.reproduce_baseline import ROOT


def checked(path):
    report = json.loads(path.read_text())
    if report["report_id"] != content_id({k: v for k, v in report.items() if k != "report_id"}):
        raise ValueError("Invalid measured report")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    release = ROOT / "artifacts/release"
    study = checked(release / "study.json")
    cal = checked(release / "calibration/study.json")
    rows = json.loads((release / "calibration/replay-validation.json").read_text())
    checkpoints = json.loads((release / "calibration/validation-checkpoints.json").read_text())
    payload = {
        "schema_version": "gridlens_public_replay_v1",
        "point_model_id": cal["point_model_id"],
        "calibration_id": cal["calibration_id"],
        "selected": cal["selected"],
        "ncc_status": cal["ncc_status"],
        "coverage_gate_passed": cal["coverage_gate_passed"],
        "levels": [0.8, 0.9, 0.95],
        "window": 90,
        "minimum": 60,
        "evidence_mode": "retrospective_revised",
        "point_source": "Frozen point forecasts computed during the 2023 calibration-validation campaign; this page replays those points. Use /live for fresh model execution.",
        "rows": [
            {"target_date": r["target_date"], "point": r["point"], "actual": r["actual"]}
            for r in rows["rows"]
        ],
        "initial": {
            method: {
                "scores": row["state"]["scores"],
                "feedback_watermark": row["state"]["feedback_watermark"],
                "fixed": row["fixed"],
            }
            for method, row in checkpoints.items()
        },
        "thresholds": {
            method: [r["thresholds"] for r in values]
            for method, values in rows["intervals"].items()
        },
        "comparisons": cal["comparisons"],
    }
    payload["replay_id"] = content_id(payload)
    path = ROOT / "apps/web/public/replay/evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    if args.replay_only:
        print(json.dumps({"replay_bytes": path.stat().st_size, "replay_id": payload["replay_id"]}))
        return
    runtime = checked(release / "runtime/benchmark.json")
    summary = {
        "schema_version": "gridlens_release_summary_v1",
        "point_study": {
            "report_id": study["report_id"],
            "ranking": study["ranking"],
            "completed_seed_fits": study["completed_seed_fits"],
            "champion_model_id": study["champion_model_id"],
            "training_origins": study["training_origins"],
            "validation_origins": len(study["target_dates"]),
            "candidates": [
                {
                    "kind": r["kind"],
                    "config": r["config"],
                    "seed": r["seed"],
                    "model_id": r["model_id"],
                    "metrics": r["metrics"],
                    "epochs": r.get("epochs"),
                    "fit_seconds": r.get("fit_seconds"),
                    "history": r.get("history", []),
                }
                for r in study["rows"]
            ],
        },
        "calibration": {k: v for k, v in cal.items() if k not in ("point_freeze", "source_hashes")},
        "runtime": {k: v for k, v in runtime.items() if k != "raw_timings"},
        "replay": {
            "id": payload["replay_id"],
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        },
        "test": None,
    }
    test = release / "test/evaluation.json"
    if test.exists():
        report = checked(test)
        summary["test"] = {k: v for k, v in report.items() if k not in ("rows", "actual_mw")}
        summary["test"]["candidates"] = [
            {k: v for k, v in r.items() if k != "prediction_mw"} for r in report["rows"]
        ]
    (ROOT / "apps/web/public/replay/summary.json").write_text(
        json.dumps(summary, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "replay_bytes": path.stat().st_size,
                "validation_origins": len(payload["rows"]),
                "test_evaluated": summary["test"] is not None,
            }
        )
    )


if __name__ == "__main__":
    main()
