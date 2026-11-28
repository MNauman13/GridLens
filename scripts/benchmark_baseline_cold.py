"""Supplement I0 with fresh-process load timing and prepared-tensor engine timing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gridlens.benchmark import latency_summary, memory_record, timed_calls, write_report
from gridlens.data.store import content_id

ROOT = Path(__file__).resolve().parents[1]


def child(bundle: Path, feature: Path, kind: str) -> dict[str, Any]:
    started = time.perf_counter_ns()
    import torch

    from gridlens.contracts import InputWindow
    from gridlens.data.dataset import FeatureSnapshot
    from gridlens.models.baselines import Baseline
    from gridlens.models.bundle import NeuralBundle

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    imports_ms = (time.perf_counter_ns() - started) / 1e6
    payload = json.loads(feature.read_text())
    snapshot = FeatureSnapshot(
        InputWindow.model_validate(payload["inputs"]),
        tuple(tuple(row) for row in payload["history"]),
        tuple(tuple(row) for row in payload["target_calendar"]),
    )
    freeze = json.loads((ROOT / "reports/dataset-freeze-v1.json").read_text())
    if str(snapshot.inputs.target_start.date()) not in freeze["splits"]["point_validation"]["target_dates"]:
        raise ValueError("Cold benchmark is restricted to frozen point_validation dates")
    before = memory_record()
    start = time.perf_counter_ns()
    loaded: Any = Baseline.load_bundle(bundle) if kind == "ridge" else NeuralBundle.load_bundle(bundle)
    load_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    result = loaded.predict(snapshot.inputs if kind == "ridge" else snapshot)
    first_predict_ms = (time.perf_counter_ns() - start) / 1e6
    return {
        "model_id": result.model_version, "imports_ms": imports_ms,
        "load_ms": load_ms, "first_bundle_predict_ms": first_predict_ms,
        "memory_before_load": before, "memory_after_first_predict": memory_record(),
        "feature_snapshot_id": result.feature_snapshot_id,
        "prediction_fingerprint": content_id(result.prediction_mw),
    }


def supplement(output: Path, source_root: Path) -> None:
    import torch

    from gridlens.artifacts import sha256
    from gridlens.models.bundle import NeuralBundle
    from scripts.build_dataset import prepare
    from scripts.reproduce_baseline import verify_pins

    plan = json.loads((ROOT / "configs/upgrade_experiment_v2.json").read_text())
    verify_pins(ROOT, plan)
    report = json.loads((output / "baseline_reproduction.json").read_text())
    if report["locked_test_evaluated"] or report["experiment_plan_id"] != content_id(plan):
        raise ValueError("Supplement must match the frozen I0 report")
    torch.set_num_threads(plan["torch_threads"])
    torch.use_deterministic_algorithms(True)
    datasets, freeze, scaling = prepare(source_root)
    if freeze["manifest_id"] != report["dataset_manifest_id"]:
        raise ValueError("Frozen dataset mismatch")
    data = datasets["point_validation"]
    dates = [str(example.inputs.target_start.date()) for example in data.examples]
    indices = [dates.index(day) for day in plan["benchmark_target_dates"]]
    feature = data.features[indices[0]]
    path = output / "cold_feature.json"
    payload = {
        "inputs": feature.inputs.model_dump(mode="json"),
        "history": feature.history, "target_calendar": feature.target_calendar,
    }
    path.write_text(json.dumps(payload) + "\n")
    rows = []
    traces = []
    for record in report["results"]:
        folder = output / record["bundle"]
        trials = []
        for _ in range(20):
            start = time.perf_counter_ns()
            command = [sys.executable, "-m", "scripts.benchmark_baseline_cold", "--child", "--bundle", str(folder), "--feature", str(path), "--kind", record["kind"]]
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True, timeout=60)
            observation = json.loads(completed.stdout)
            observation["fresh_process_wall_ms"] = (time.perf_counter_ns() - start) / 1e6
            if observation["model_id"] != record["model_id"] or observation["feature_snapshot_id"] != feature.inputs.feature_snapshot_id:
                raise ValueError("Cold subprocess used incompatible model/features")
            trials.append(observation)
        if len({trial["prediction_fingerprint"] for trial in trials}) != 1:
            raise ValueError("Fresh-process prediction fingerprint changed")
        row = {
            "kind": record["kind"], "model_id": record["model_id"],
            "cold_process_count": 20,
            "cold_scope": "new interpreter and model instance; OS file cache may remain warm; local artifact bytes",
            "cold_load": latency_summary([trial["load_ms"] for trial in trials]),
            "cold_import": latency_summary([trial["imports_ms"] for trial in trials]),
            "cold_first_predict": latency_summary([trial["first_bundle_predict_ms"] for trial in trials]),
            "cold_process_wall": latency_summary([trial["fresh_process_wall_ms"] for trial in trials]),
            "cold_raw_trials": trials,
        }
        if record["kind"] != "ridge":
            loaded = NeuralBundle.load_bundle(folder)
            cases = []
            preparation = []
            for index in indices:
                stamp = time.perf_counter_ns()
                history = torch.tensor([scaling.transform(data.features[index].history)], dtype=torch.float32)
                calendar = torch.tensor([data.features[index].target_calendar], dtype=torch.float32)
                preparation.append((time.perf_counter_ns() - stamp) / 1e6)
                cases.append((history, calendar))
            before = memory_record()

            def engine(index: int, model: Any = loaded.model, prepared: Any = cases) -> object:
                with torch.no_grad():
                    return model(*prepared[index])

            engine_summary, raw = timed_calls(engine, cases=len(cases), warmup=plan["warmup_calls"], repetitions=plan["repetitions"], calls=plan["calls_per_repetition"])
            row["prepared_engine_latency"] = engine_summary
            row["engine_scope"] = "model forward only; excludes feature transformation, scaling, inverse scaling and output validation"
            row["preparation_latency"] = latency_summary(preparation)
            row["memory_before_engine"] = before
            row["memory_after_engine"] = memory_record()
            traces.extend({"model_id": record["model_id"], **trace} for trace in raw)
        rows.append(row)
    write_report(output / "baseline_runtime_supplement.json", {
        "schema_version": "gridlens_baseline_runtime_supplement_v1",
        "baseline_report_id": report["report_id"], "experiment_plan_id": content_id(plan),
        "protocol": {"cold_processes_per_model": 20, "torch_threads": 2, "device": "cpu"},
        "script_sha256": sha256(Path(__file__)),
        "locked_test_evaluated": False, "results": rows,
        "not_applicable": {
            "gpu": "CPU-only baseline", "decoder_kv_cache_bytes": "Reference path has no incremental cache",
            "http_and_concurrent_load": "I1/I6 implement the live service",
            "isolated_encoder_decoder_spans": "Forward-only aggregate here; fine-grained profiler belongs to I2",
            "statistical_model_superiority": "Single-seed reproduction, no new selection claim",
        },
    })
    (output / "engine_timings.jsonl").write_text("\n".join(json.dumps(item) for item in traces) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/local/i0")
    parser.add_argument("--root", type=Path, default=ROOT / "data/neso")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--feature", type=Path)
    parser.add_argument("--kind", choices=("ridge", "transformer", "timexer"))
    args = parser.parse_args()
    if args.child:
        if args.bundle is None or args.feature is None or args.kind is None:
            parser.error("Child requires bundle, feature and model kind")
        print(json.dumps(child(args.bundle, args.feature, args.kind)))
    else:
        supplement(args.output.resolve(), args.root.resolve())


if __name__ == "__main__":
    main()
