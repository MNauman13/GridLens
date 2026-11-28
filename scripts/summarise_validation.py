"""Additional full-validation slices and block intervals; never reads test role."""

import json

import numpy as np

from gridlens.benchmark import write_report
from scripts.evaluate_release import block_interval, slices
from scripts.reproduce_baseline import ROOT, frozen_data


def main():
    study = json.loads((ROOT / "artifacts/release/study.json").read_text())
    datasets, _, _ = frozen_data(
        ROOT, ROOT / "data/neso", json.loads((ROOT / "reports/dataset-freeze-v1.json").read_text())
    )
    data = datasets["point_validation"]
    actual = np.asarray([e.target_mw for e in data.examples])
    threshold = float(
        np.quantile([max(e.target_mw) for e in datasets["point_train"].examples], 0.9)
    )
    champion = next(row for row in study["rows"] if row["model_id"] == study["champion_model_id"])
    ridge = next(
        row for row in study["rows"] if row["model_id"] == study["deployment_choices"]["ridge"]
    )
    d = np.mean(
        np.abs(actual - np.asarray(champion["validation_predictions_mw"]))
        - np.abs(actual - np.asarray(ridge["validation_predictions_mw"])),
        axis=1,
    )
    report = {
        "schema_version": "gridlens_full_validation_supplement_v1",
        "point_study_report_id": study["report_id"],
        "role": "point_validation",
        "origins": len(actual),
        "aggregate_recorded_fit_seconds": sum(row.get("fit_seconds", 0) for row in study["rows"]),
        "study_execution_elapsed_seconds": study["elapsed_seconds"],
        "clock_note": "Execution elapsed time excludes earlier completed/resumed fits. The aggregate fit figure sums recorded fit durations including checkpointed work.",
        "training_high_demand_threshold_mw": threshold,
        "slices": {
            row["model_id"]: slices(
                data, actual, np.asarray(row["validation_predictions_mw"]), threshold
            )
            for row in study["rows"]
        },
        "champion_minus_ridge_daily_mae_mw": float(d.mean()),
        "paired_daily_mae_block_intervals": {
            str(block): block_interval(d, block) for block in (7, 14)
        },
        "interpretation": "Descriptive validation intervals conditional on the selected configuration and seed. Selection uncertainty is not included. No independent test superiority claim.",
        "locked_test_evaluated": False,
    }
    write_report(ROOT / "artifacts/release/validation-supplement.json", report)
    print(
        json.dumps(
            {
                "origins": len(actual),
                "models": len(study["rows"]),
                "fit_seconds": report["aggregate_recorded_fit_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
