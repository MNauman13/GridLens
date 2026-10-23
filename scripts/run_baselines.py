"""Fit on 2018–2021, select ridge on 2022, and export a real validation forecast."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from gridlens.contracts import BaselineForecast
from gridlens.data.store import content_id
from gridlens.metrics import point_metrics
from gridlens.models.baselines import Baseline
from scripts.build_dataset import ROOT, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/neso")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/baseline-validation-v1.json")
    parser.add_argument("--forecast-output", type=Path, default=ROOT / "src/gridlens/resources/baseline_forecasts_v1.json")
    parser.add_argument("--bundle-root", type=Path, default=ROOT / "artifacts/baselines")
    args = parser.parse_args()
    datasets, freeze, _ = prepare(args.root)
    committed = json.loads((ROOT / "reports/dataset-freeze-v1.json").read_text())
    if json.loads(json.dumps(freeze)) != committed:
        raise ValueError("Dataset rebuild differs from the approved freeze")
    train, validation = datasets["point_train"], datasets["point_validation"]
    actual = [e.target_mw for e in validation.examples]
    results, fitted = [], {}
    candidates = [Baseline("last_day"), Baseline("weekly")] + [Baseline("ridge", alpha=a) for a in (1.0, 10.0, 100.0)]
    for model in candidates:
        start = time.perf_counter()
        model.fit(train)
        fit_seconds = time.perf_counter() - start
        predicted, latencies = [], []
        for example in validation.examples:
            start = time.perf_counter()
            predicted.append(model.predict(example.inputs).prediction_mw)
            latencies.append((time.perf_counter() - start) * 1000)
        metrics = point_metrics(actual, predicted)
        bundle = args.bundle_root / model.model_version
        model.save_bundle(bundle)
        restored = Baseline.load_bundle(bundle)
        np.testing.assert_allclose(restored.predict(validation.examples[0].inputs).prediction_mw, predicted[0], rtol=0, atol=1e-9)
        record = {"model_kind": model.kind, "alpha": model.alpha if model.kind == "ridge" else None, "model_version": model.model_version, "metrics": metrics, "fit_seconds": fit_seconds, "prediction_p50_ms": statistics.median(latencies), "prediction_p95_ms": sorted(latencies)[int(.95 * (len(latencies) - 1))], "fitted_coefficient_count": 4848 if model.kind == "ridge" else 0}
        results.append(record)
        fitted[model.model_version] = model
        print(json.dumps({"model": model.kind, "alpha": record["alpha"], "mae_mw": metrics["mae_mw"], "rmse_mw": metrics["rmse_mw"]}))
    best = min((r for r in results if r["model_kind"] == "ridge"), key=lambda r: (r["metrics"]["rmse_mw"], r["alpha"]))
    report = {"schema_version": "gridlens_baseline_validation_v1", "dataset_manifest_id": freeze["manifest_id"], "training_dataset_id": train.manifest.dataset_id, "validation_dataset_id": validation.manifest.dataset_id, "validation_target_dates": freeze["splits"]["point_validation"]["target_dates"], "selection_rule": "ridge alpha chosen by minimum 2022 validation RMSE; ties choose smaller alpha", "selected_ridge_model": best["model_version"], "results": results, "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.system(), "device": "CPU"}, "locked_test_evaluated": False, "evidence_mode": "retrospective_revised"}
    report["report_id"] = content_id(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    demo = next(e for e in validation.examples if str(e.inputs.target_start.date()) == "2022-02-10")
    public = {}
    for record in [r for r in results if r["model_kind"] != "ridge"] + [best]:
        model = fitted[record["model_version"]]
        payload = {"schema_version": "gridlens_baseline_forecast_v1", "inputs": demo.inputs.model_dump(mode="json"), "point": model.predict(demo.inputs).model_dump(mode="json"), "model_kind": model.kind, "observed_mw": demo.target_mw, "dataset_id": validation.manifest.dataset_id, "source_sha256": validation.manifest.source_sha256, "truth_version": demo.truth_version, "evaluation_role": "point_validation", "validation_report_id": report["report_id"]}
        artifact = BaselineForecast.model_validate({**payload, "forecast_id": content_id(payload)})
        public[model.kind] = artifact.model_dump(mode="json")
    args.forecast_output.parent.mkdir(parents=True, exist_ok=True)
    args.forecast_output.write_text(json.dumps(public, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_id": report["report_id"], "selected_alpha": best["alpha"], "locked_test_evaluated": False}))


if __name__ == "__main__":
    main()
