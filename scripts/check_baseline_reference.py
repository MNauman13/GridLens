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
            if not math.isclose(row["metrics"][metric], ref["metrics"][metric], rel_tol=1e-9, abs_tol=1e-6):
                raise SystemExit(f"Baseline reference drift: {row['model_kind']} {metric}")
        if row["metrics"]["origins"] != 365:
            raise SystemExit("Validation origins differ")
    if expected:
        raise SystemExit("Missing baseline candidates")
    print("All five frozen baseline candidates reproduce validation errors within 1e-6 MW; final test remains locked")


if __name__ == "__main__":
    main()
