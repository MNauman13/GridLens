"""Rebuild/verify I0 CPU baselines without scoring or training on locked test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gridlens.artifacts import sha256
from gridlens.benchmark import (
    archive_files,
    environment_record,
    memory_record,
    select_origins,
    timed_calls,
    validate_development_dates,
    write_report,
)
from gridlens.data.store import content_id, write_once
from gridlens.metrics import point_metrics
from gridlens.models.baselines import Baseline
from gridlens.models.bundle import NeuralBundle
from scripts.build_dataset import ROOT, prepare
from scripts.train_neural import predict, tensors, train


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_pins(root: Path, plan: dict[str, Any]) -> None:
    for relative, expected in plan["reference_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Pinned reference changed: {relative}")
    if plan["locked_test_evaluated"] is not False or plan["allowed_inference_role"] != "point_validation":
        raise ValueError("I0 must preserve locked test and use point_validation only")
    for field in ("warmup_calls", "repetitions", "calls_per_repetition", "torch_threads", "corpus_origins"):
        if type(plan[field]) is not int or plan[field] < (0 if field == "warmup_calls" else 1):
            raise ValueError(f"Invalid benchmark budget: {field}")
    if plan["recovery_epochs"] != 12 or plan["seed"] != 17:
        raise ValueError("Recovery must retain the accepted pilot's seed and epoch budget")


def extended_metrics(actual: np.ndarray[Any, Any], predicted: np.ndarray[Any, Any]) -> dict[str, Any]:
    metrics = point_metrics(actual.tolist(), predicted.tolist())
    errors = predicted - actual
    metrics.update({
        "bias_mw": float(errors.mean()),
        "ramp_mae_mw": float(np.abs(np.diff(predicted, axis=1) - np.diff(actual, axis=1)).mean()),
        "daily_mae_p95_mw": float(np.quantile(np.abs(errors).mean(axis=1), .95)),
        "daily_rmse_p95_mw": float(np.quantile(np.square(errors).mean(axis=1) ** .5, .95)),
    })
    return metrics


def frozen_data(root: Path, source_root: Path, expected: dict[str, Any]) -> tuple[Any, Any, Any]:
    audit = read_json(root / "reports/neso-audit-v1.json")
    for row in audit["resources"]:
        if row["source_kind"] == "historic_outturn":
            path = source_root / "raw" / (row["raw_sha256"] + ".csv")
            if not path.is_file() or sha256(path) != row["raw_sha256"]:
                raise ValueError("Missing or changed frozen source; never silently re-freeze data")
    datasets, freeze, scaling = prepare(source_root)
    if json.loads(json.dumps(freeze)) != expected:
        raise ValueError("Rebuilt data differs from the accepted freeze")
    return datasets, freeze, scaling


def run(output: Path, source_root: Path, rebuild: bool) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(ROOT / "configs/upgrade_experiment_v2.json")
    verify_pins(ROOT, plan)
    torch.set_num_threads(plan["torch_threads"])
    torch.use_deterministic_algorithms(True)
    environment = environment_record(ROOT)
    environment["torch_threads"] = torch.get_num_threads()
    environment["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    write_report(output / "environment.json", environment)
    data_start = time.perf_counter()
    datasets, freeze, scaling = frozen_data(ROOT, source_root, read_json(ROOT / "reports/dataset-freeze-v1.json"))
    data_seconds = time.perf_counter() - data_start
    training, validation = datasets["point_train"], datasets["point_validation"]
    dates = [str(example.inputs.target_start.date()) for example in validation.examples]
    corpus_dates = plan["benchmark_target_dates"]
    validate_development_dates(corpus_dates, dates)
    if corpus_dates != select_origins(dates, plan["corpus_origins"]):
        raise ValueError("Benchmark corpus no longer matches its frozen selection rule")
    corpus = [dates.index(day) for day in corpus_dates]
    benchmark_manifest = {
        "schema_version": "gridlens_benchmark_manifest_v2",
        "experiment_plan_id": content_id(plan), "environment_id": content_id(environment),
        "dataset_manifest_id": freeze["manifest_id"], "role": "point_validation",
        "target_dates": corpus_dates,
        "feature_snapshot_ids": [validation.features[index].inputs.feature_snapshot_id for index in corpus],
        "protocol": {key: plan[key] for key in ("warmup_calls", "repetitions", "calls_per_repetition", "torch_threads")},
        "locked_test_evaluated": False,
    }
    write_report(output / "benchmark_manifest_v2.json", benchmark_manifest)
    actual = np.asarray([example.target_mw for example in validation.examples])
    baseline_reference = read_json(ROOT / "reports/baseline-validation-v1.json")
    ridge_ref = next(row for row in baseline_reference["results"] if row["model_version"] == baseline_reference["selected_ridge_model"])
    ridge_folder = output / "bundles" / "ridge"
    fit_start = time.perf_counter()
    ridge = Baseline("ridge", alpha=ridge_ref["alpha"])
    ridge.fit(training)
    fit_seconds = time.perf_counter() - fit_start
    ridge.save_bundle(ridge_folder)
    ridge = Baseline.load_bundle(ridge_folder)
    ridge_prediction = np.asarray([ridge.predict(example.inputs).prediction_mw for example in validation.examples])
    baseline_demo = read_json(ROOT / "src/gridlens/resources/baseline_forecasts_v1.json")["ridge"]
    demo_index = dates.index(baseline_demo["inputs"]["target_start"][:10])
    rows: list[dict[str, Any]] = [{
        "kind": "ridge", "model_id": ridge.model_version,
        "bundle": str(ridge_folder.relative_to(output)), "recovery": "refitted_from_frozen_training_data",
        "fit_seconds": fit_seconds, "metrics": extended_metrics(actual, ridge_prediction),
        "reference_rmse_mw": ridge_ref["metrics"]["rmse_mw"],
        "reference_demo_max_abs_drift_mw": float(np.abs(ridge_prediction[demo_index] - baseline_demo["point"]["prediction_mw"]).max()),
    }]
    all_predictions = {ridge.model_version: ridge_prediction.tolist()}
    bundle_index = read_json(output / "bundle_index.json") if (output / "bundle_index.json").exists() else {}
    train_data, validation_data = tensors(training, scaling), tensors(validation, scaling)
    neural_reference = read_json(ROOT / "reports/neural-validation-v1.json")
    public = read_json(ROOT / "src/gridlens/resources/neural_forecasts_v1.json")
    for name, config in (("transformer", {"kind": "transformer"}), ("timexer", {"kind": "timexer", "patch": 48})):
        if rebuild:
            row, _ = train(config, plan["seed"], train_data, validation_data, scaling, freeze, output / "bundles/neural", plan["recovery_epochs"], False)
            bundle_index[name] = {"path": "bundles/neural/" + row["bundle_directory"], "recovery": "regenerated_with_original_training_code", "fit_seconds": row["fit_seconds"]}
            # Persist successful recoveries so an interrupted later benchmark need not retrain.
            (output / "bundle_index.json").write_text(json.dumps(bundle_index, indent=2) + "\n")
        if name not in bundle_index:
            raise ValueError("No verified neural bundle index; run once with --rebuild-neural")
        folder = (output / bundle_index[name]["path"]).resolve()
        if not folder.is_relative_to((output / "bundles").resolve()):
            raise ValueError("Bundle index escaped the local bundle directory")
        bundle = NeuralBundle.load_bundle(folder)
        if bundle.metadata["config"] != config or bundle.metadata["dataset_manifest_id"] != freeze["manifest_id"]:
            raise ValueError("Bundle architecture or training dataset mismatch")
        with torch.no_grad():
            prediction = (predict(bundle.model, validation_data[0], validation_data[1]) * scaling.scales[0] + scaling.means[0]).numpy()
        for index in corpus:
            np.testing.assert_allclose(bundle.predict(validation.features[index]).prediction_mw, prediction[index], rtol=0, atol=.05)
        ref = next(item for item in neural_reference["results"] if item["config"] == config)
        reference_demo = public[name]
        index = dates.index(reference_demo["inputs"]["target_start"][:10])
        drift = float(np.abs(np.asarray(bundle.predict(validation.features[index]).prediction_mw) - reference_demo["prediction_mw"]).max())
        rows.append({
            "kind": name, "model_id": bundle.metadata["identity"], "bundle": str(folder.relative_to(output.resolve())),
            "recovery": bundle_index[name]["recovery"], "fit_seconds": bundle_index[name]["fit_seconds"],
            "metrics": extended_metrics(actual, prediction), "reference_rmse_mw": ref["metrics"]["rmse_mw"],
            "reference_demo_max_abs_drift_mw": drift,
            "reference_model_identity_equal": bundle.metadata["identity"] == ref["identity"],
            "bundle_single_vs_batch_max_drift_ceiling_mw": .05,
        })
        all_predictions[bundle.metadata["identity"]] = prediction.tolist()
    timings = []
    # The reporting/validation above is outside the timed calls.
    for row in rows:
        folder = output / row["bundle"]
        load_start = time.perf_counter_ns()
        if row["kind"] == "ridge":
            predictor: Any = Baseline.load_bundle(folder)
        else:
            predictor = NeuralBundle.load_bundle(folder)
        load_ms = (time.perf_counter_ns() - load_start) / 1_000_000

        def call(index: int, loaded: Any = predictor, kind: str = row["kind"]) -> object:
            inputs = validation.examples[corpus[index]].inputs if kind == "ridge" else validation.features[corpus[index]]
            return loaded.predict(inputs)

        before = memory_record()
        summary, traces = timed_calls(call, cases=len(corpus), warmup=plan["warmup_calls"], repetitions=plan["repetitions"], calls=plan["calls_per_repetition"])
        row["bundle_predict_latency"] = summary
        row["bundle_reload_ms"] = load_ms
        row["load_scope"] = "warm process and filesystem; not a cold-start measurement"
        row["memory_before"] = before
        row["memory_after"] = memory_record()
        row["bundle_bytes"] = sum(path.stat().st_size for path in folder.iterdir() if path.name in ("model.json", "metadata.json", "weights.pt", "manifest.json"))
        row["reference_rmse_difference_mw"] = row["metrics"]["rmse_mw"] - row["reference_rmse_mw"]
        row["reference_demo_within_0_5_mw"] = row["reference_demo_max_abs_drift_mw"] <= .5
        row["runtime_scope"] = "warm batch-one full bundle.predict, includes preprocessing and output contract validation"
        timings.extend({"model_id": row["model_id"], **trace} for trace in traces)
    report = {
        "schema_version": "gridlens_baseline_reproduction_v2", "phase": "I0",
        "experiment_plan_id": content_id(plan), "environment": environment,
        "dataset_manifest_id": freeze["manifest_id"], "source_hashes_verified": len(freeze["source_sha256"]),
        "split_origin_counts": {role: len(data.examples) for role, data in datasets.items()},
        "data_rebuild_seconds": data_seconds, "benchmark_manifest_id": content_id(benchmark_manifest),
        "validation_target_dates": dates, "results": rows,
        "locked_test_evaluated": False, "evaluated_roles": ["point_validation"],
        "limitations": [
            "Neural weights regenerated unless an explicitly verified recovery index says otherwise; original reports unchanged.",
            "Warm bundle.predict includes feature transformation; not comparable to old engine-only five-call pilot timing.",
            "Memory is process-wide, includes training/data; not isolated per-model memory.",
            "No cold-process, GPU, concurrent service, calibration or final-test claim in I0.",
            "This is baseline recovery, not the I4 multi-seed model-selection study.",
        ],
    }
    write_once(output / "timings.jsonl", ("\n".join(json.dumps(t) for t in timings) + "\n").encode())
    write_report(output / "validation-predictions.json", {"target_dates": dates, "actual_mw": actual.tolist(), "predictions": all_predictions, "role": "point_validation"})
    write_report(output / "baseline_reproduction.json", report)
    members = [output / row["bundle"] / member for row in rows for member in (("manifest.json", "model.json") if row["kind"] == "ridge" else ("manifest.json", "metadata.json", "weights.pt"))]
    members += [output / name for name in ("bundle_index.json", "environment.json", "benchmark_manifest_v2.json", "baseline_reproduction.json", "validation-predictions.json", "timings.jsonl")]
    archive = archive_files(output, members, output / "baseline-models-v2.zip")
    write_report(output / "archive_inventory.json", archive)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/neso")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/local/i0")
    parser.add_argument("--rebuild-neural", action="store_true", help="Regenerate exactly the existing Transformer and patch-48 TimeXer pilot configurations")
    args = parser.parse_args()
    report = run(args.output.resolve(), args.root.resolve(), args.rebuild_neural)
    print(json.dumps({"phase": "I0", "locked_test_evaluated": False, "models": [{"kind": row["kind"], "rmse_mw": row["metrics"]["rmse_mw"], "p50_bundle_ms": row["bundle_predict_latency"]["p50_ms"]} for row in report["results"]]}))


if __name__ == "__main__":
    main()
