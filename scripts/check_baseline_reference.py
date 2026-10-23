"""Allow CPU/BLAS rounding differences while requiring the same frozen comparison."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("reports/baseline-validation-v1.json"))
    args = parser.parse_args()
    actual = json.loads(args.report.read_text())
    reference = json.loads(args.reference.read_text())
    for key in ("dataset_manifest_id", "training_dataset_id", "validation_dataset_id", "validation_target_dates", "locked_test_evaluated"):
        if actual[key] != reference[key]:
            raise SystemExit(f"Different baseline comparison: {key}")
    expected = {(r["model_kind"], r["alpha"]): r for r in reference["results"]}
    for row in actual["results"]:
        ref = expected.pop((row["model_kind"], row["alpha"]))
        for metric in ("mae_mw", "rmse_mw", "mean_daily_peak_absolute_error_mw"):
            if not math.isclose(row["metrics"][metric], ref["metrics"][metric], rel_tol=0, abs_tol=1e-6):
                raise SystemExit(f"Baseline reference drift: {row['model_kind']} {metric}")
        if row["metrics"]["origins"] != len(reference["validation_target_dates"]):
            raise SystemExit("Validation origins differ")
    actual_selected = next(r for r in actual["results"] if r["model_version"] == actual["selected_ridge_model"])
    reference_selected = next(r for r in reference["results"] if r["model_version"] == reference["selected_ridge_model"])
    if (actual_selected["model_kind"], actual_selected["alpha"]) != (reference_selected["model_kind"], reference_selected["alpha"]):
        raise SystemExit("Selected ridge configuration differs")
    if expected:
        raise SystemExit("Missing baseline candidates")
    print("All five frozen baseline candidates reproduce validation errors within 1e-6 MW; final test remains locked")


if __name__ == "__main__":
    main()
