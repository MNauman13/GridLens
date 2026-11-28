"""One frozen 2024 campaign. Requires the final freeze to be committed first."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import torch

from gridlens.artifacts import sha256
from gridlens.benchmark import write_report
from gridlens.calibration.metrics import interval_metrics
from gridlens.calibration.ncc import NCCNetwork
from gridlens.calibration.replay import IntervalReplay
from gridlens.data.store import content_id
from gridlens.models.baselines import Baseline
from gridlens.models.bundle import NeuralBundle
from scripts.calibration_study import campaign_with_history, point_values
from scripts.reproduce_baseline import ROOT, extended_metrics, frozen_data

OUT = ROOT / "artifacts/release/test"


def slices(dataset, actual, predicted, high_threshold):
    groups = {}
    for i, (example, feature) in enumerate(zip(dataset.examples, dataset.features, strict=True)):
        day = example.inputs.target_start.date()
        season = {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }[day.month]
        labels = [
            season,
            "weekend" if day.weekday() >= 5 else "weekday",
            "fixed_date_holiday_proxy"
            if (day.month, day.day) in ((1, 1), (12, 25), (12, 26))
            else "other_dates",
            "high_demand" if max(example.target_mw) >= high_threshold else "ordinary_demand",
            "missing_history"
            if any(any(row[3:6]) for row in feature.history)
            else "complete_history",
        ]
        for label in labels:
            groups.setdefault(label, []).append(i)
    return {
        name: {
            "origins": len(indices),
            "metrics": extended_metrics(actual[indices], predicted[indices]),
        }
        for name, indices in groups.items()
    }


def block_interval(daily_difference, block, draws=2000):
    rng = np.random.default_rng(17)
    n = len(daily_difference)
    values = []
    for _ in range(draws):
        starts = rng.integers(0, n, size=(n + block - 1) // block)
        indices = np.concatenate([(np.arange(block) + start) % n for start in starts])[:n]
        values.append(float(daily_difference[indices].mean()))
    return np.quantile(values, [0.025, 0.975]).tolist()


def main():
    freeze_path = ROOT / "configs/final_release_freeze_v1.json"
    frozen = freeze_path.read_bytes()
    freeze = json.loads(frozen)
    if freeze["freeze_id"] != content_id({k: v for k, v in freeze.items() if k != "freeze_id"}):
        raise ValueError("Invalid release identity")
    committed = subprocess.check_output(
        ["git", "show", "HEAD:configs/final_release_freeze_v1.json"], cwd=ROOT
    )
    if committed != frozen:
        raise ValueError("Commit the exact final freeze before test evaluation")
    for path, digest in freeze["source_hashes"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Frozen implementation changed before evaluation")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "evaluation.json").exists():
        raise ValueError(
            "This frozen test campaign has already been recorded; inspect the retained report"
        )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    datasets, _, scaling = frozen_data(
        ROOT, ROOT / "data/neso", json.loads((ROOT / "reports/dataset-freeze-v1.json").read_text())
    )
    data = datasets["locked_test"]
    actual = np.asarray([e.target_mw for e in data.examples])
    dates = [str(e.inputs.target_start.date()) for e in data.examples]
    threshold = float(
        np.quantile([max(e.target_mw) for e in datasets["point_train"].examples], 0.9)
    )
    study = json.loads((ROOT / "artifacts/release/study.json").read_text())
    if set(freeze["all_candidate_model_ids"]) != {row["model_id"] for row in study["rows"]}:
        raise ValueError("Candidate registry changed")
    predictions = {}
    rows = []
    for row in study["rows"]:
        folder = ROOT / freeze["model_bundles"][row["model_id"]]
        model = (
            Baseline.load_bundle(folder)
            if row["kind"] in ("ridge", "weekly", "last_day")
            else NeuralBundle.load_bundle(folder)
        )
        values = point_values(model, data, scaling)
        predictions[row["model_id"]] = values
        result = {
            "model_id": row["model_id"],
            "kind": row["kind"],
            "config": row["config"],
            "seed": row["seed"],
            "metrics": extended_metrics(actual, values),
            "slices": slices(data, actual, values, threshold),
            "prediction_mw": values.tolist(),
        }
        rows.append(result)
        print(
            json.dumps(
                {"kind": row["kind"], "seed": row["seed"], "rmse_mw": result["metrics"]["rmse_mw"]}
            ),
            flush=True,
        )
    champion = predictions[freeze["point_model_id"]]
    ridge_id = study["deployment_choices"]["ridge"]
    ridge = predictions[ridge_id]
    daily_difference = np.mean(np.abs(actual - champion) - np.abs(actual - ridge), axis=1)
    comparisons = {
        "champion_minus_ridge_daily_mae_mw": float(daily_difference.mean()),
        "block_bootstrap_95pct": {
            str(block): block_interval(daily_difference, block) for block in (7, 14)
        },
        "interpretation": "Paired daily MAE differences conditional on frozen validation selection; not a superiority claim based on selected test winners",
    }
    calroot = ROOT / "artifacts/release/calibration"
    calreport = json.loads((calroot / "study.json").read_text())
    calfreeze = freeze["calibration"]
    if (
        sha256(calroot / "ncc.pt") != calfreeze["weights_sha256"]
        or sha256(calroot / "checkpoints.json") != calfreeze["checkpoints_sha256"]
    ):
        raise ValueError("Calibration checkpoint/weights changed")
    network = NCCNetwork()
    network.load_state_dict(torch.load(calroot / "ncc.pt", weights_only=True))
    network.eval()
    checkpoints = json.loads((calroot / "checkpoints.json").read_text())
    combined = {
        str(e.inputs.target_start.date()): list(e.target_mw)
        for role in ("calibration_initialisation", "locked_test")
        for e in datasets[role].examples
    }
    cases = [
        {
            "target_date": day,
            "point": value.tolist(),
            "actual": list(e.target_mw),
            "context": [list(c) for c in feature.target_calendar],
        }
        for day, value, e, feature in zip(
            dates, champion, data.examples, data.features, strict=True
        )
    ]
    intervals = {}
    interval_predictions = {}
    for method in freeze["evaluation"]["interval_models"]:
        replay = IntervalReplay.restore(
            checkpoints[method], network=network if method == "ncc" else None
        )
        records, _ = campaign_with_history(replay, cases, combined)
        intervals[method] = interval_metrics(actual, champion, [r["thresholds"] for r in records])
        interval_predictions[method] = records
    result = {
        "schema_version": "gridlens_locked_test_release_v1",
        "freeze_id": freeze["freeze_id"],
        "freeze_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "role": "locked_test",
        "target_dates": dates,
        "actual_mw": actual.tolist(),
        "rows": rows,
        "point_model_id": freeze["point_model_id"],
        "selected_interval_method": calreport["selected"],
        "intervals": intervals,
        "comparisons": comparisons,
        "training_high_demand_threshold_mw": threshold,
        "test_refitting": False,
        "evaluation_campaign_count": 1,
    }
    write_report(OUT / "evaluation.json", result)
    (OUT / "interval-predictions.json").write_text(
        json.dumps(interval_predictions, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "origins": len(dates),
                "models": len(rows),
                "freeze_id": freeze["freeze_id"],
                "test_refitting": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
