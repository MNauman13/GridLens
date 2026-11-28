"""Create the final manifest before any locked-test scoring. Commit it next."""

import json
import subprocess
from datetime import UTC, datetime

from gridlens.artifacts import sha256
from gridlens.data.store import content_id, write_once
from scripts.reproduce_baseline import ROOT


def main():
    study = json.loads((ROOT / "artifacts/release/study.json").read_text())
    point = json.loads((ROOT / "artifacts/release/point-model-freeze.json").read_text())
    cal = json.loads((ROOT / "artifacts/release/calibration/freeze.json").read_text())
    runtime = json.loads((ROOT / "artifacts/release/runtime/benchmark.json").read_text())
    if study["locked_test_evaluated"] or runtime["locked_test_evaluated"]:
        raise ValueError("Test was unlocked before final freeze")
    files = [
        *sorted((ROOT / "src/gridlens").rglob("*.py")),
        ROOT / "scripts/evaluate_release.py",
        ROOT / "scripts/calibration_study.py",
        ROOT / "scripts/release_study.py",
        ROOT / "scripts/benchmark_runtime.py",
    ]
    result = {
        "schema_version": "gridlens_final_release_freeze_v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "parent_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dataset_manifest_id": study["dataset_manifest_id"],
        "point_study_report_id": study["report_id"],
        "point_model_id": point["champion_model_id"],
        "all_candidate_model_ids": [row["model_id"] for row in study["rows"]],
        "model_bundles": point["model_bundles"],
        "calibration": cal,
        "runtime_report_id": runtime["report_id"],
        "serving_policy": {
            "champion": "verified original bundle eager CPU; no automatic precision promotion",
            "public_live_preview": "Pinned I0 public preview remains active. The selected TimeXer export has passed parity but its public archive requires separate approval; this does not change offline model selection.",
            "interval_replay": "Frozen 2023 calibration-validation examples; no novel coverage guarantee",
        },
        "evaluation": {
            "point_models": "All preregistered fitted candidates once on full eligible 2024 origins",
            "interval_models": ["fixed", "rolling", "ncc"],
            "calibration_target_model": point["champion_model_id"],
            "bootstrap": {"seed": 17, "draws": 2000, "block_days": [7, 14]},
            "slices": [
                "season",
                "weekday_weekend",
                "fixed_date_holiday_proxy",
                "high_demand_training_threshold",
                "missing_history",
            ],
            "test_refitting": False,
        },
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        "test_unlock_authorised": True,
        "test_evaluated": False,
    }
    result["freeze_id"] = content_id(result)
    write_once(
        ROOT / "configs/final_release_freeze_v1.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "freeze_id": result["freeze_id"],
                "next": "Commit this manifest before invoking evaluate_release",
            }
        )
    )


if __name__ == "__main__":
    main()
