"""I3 eager/export/INT8 comparison on verified I0 weights and 2022 only."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from gridlens.artifacts import sha256
from gridlens.benchmark import environment_record, latency_summary, write_report
from gridlens.data.store import content_id
from gridlens.models.bundle import NeuralBundle
from scripts.reproduce_baseline import ROOT, frozen_data
from scripts.train_neural import predict, tensors

OUT = ROOT / "artifacts/release/runtime"


def main():
    plan = json.loads((ROOT / "configs/i3_runtime_experiment_v1.json").read_text())
    baseline = ROOT / "reports/local/i0-public"
    report = json.loads((baseline / "baseline_reproduction.json").read_text())
    row = next(r for r in report["results"] if r["kind"] == "timexer")
    bundle = NeuralBundle.load_bundle(baseline / row["bundle"])
    datasets, _, _ = frozen_data(
        ROOT, ROOT / "data/neso", json.loads((ROOT / "reports/dataset-freeze-v1.json").read_text())
    )
    data = datasets["point_validation"]
    h, c, y = tensors(data, bundle.scaling)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with torch.no_grad():
        reference = predict(bundle.model, h, c).numpy()
        traced = torch.jit.trace(bundle.model, (h[:1], c[:1]), check_trace=True)
        traced.save(str(OUT / "timexer.ts"))
        torch.onnx.export(
            bundle.model,
            (h[:1], c[:1]),
            str(OUT / "timexer.onnx"),
            input_names=["history", "calendar"],
            output_names=["prediction"],
            dynamic_axes={
                "history": {0: "batch"},
                "calendar": {0: "batch"},
                "prediction": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(str(OUT / "timexer.onnx"))
    quantize_dynamic(
        str(OUT / "timexer.onnx"),
        str(OUT / "timexer-int8.onnx"),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=True,
    )
    build_seconds = time.perf_counter() - start
    graph = onnx.load(str(OUT / "timexer-int8.onnx"))
    operators = {
        node.op_type: sum(n.op_type == node.op_type for n in graph.graph.node)
        for node in graph.graph.node
    }
    dates = [str(e.inputs.target_start.date()) for e in data.examples]
    selected = json.loads((ROOT / "configs/upgrade_experiment_v2.json").read_text())[
        "benchmark_target_dates"
    ]
    indices = [dates.index(day) for day in selected]
    actual = y.numpy()
    converted = reference.astype(np.float64) * bundle.scaling.scales[0] + bundle.scaling.means[0]
    reference_rmse = float(np.sqrt(np.mean((actual - converted) ** 2)))
    conditions = []
    parity = []
    raw = []
    initialisations = []
    for threads in plan["threads"]:
        torch.set_num_threads(threads)
        sessions = {}
        for name, file in [("onnx_fp32", "timexer.onnx"), ("onnx_int8", "timexer-int8.onnx")]:
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            tick = time.perf_counter()
            sessions[name] = ort.InferenceSession(
                str(OUT / file), options, providers=["CPUExecutionProvider"]
            )
            initialisations.append(
                {
                    "runtime": name,
                    "threads": threads,
                    "session_init_ms": (time.perf_counter() - tick) * 1000,
                }
            )

        def execute(name, a, b, _sessions=sessions):
            if name == "eager":
                return bundle.model(a, b).numpy()
            if name == "torchscript":
                return traced(a, b).numpy()
            return _sessions[name].run(
                ["prediction"], {"history": a.numpy(), "calendar": b.numpy()}
            )[0]

        with torch.no_grad():
            for name in plan["runtimes"]:
                values = np.concatenate(
                    [execute(name, h[i : i + 4], c[i : i + 4]) for i in range(0, len(h), 4)]
                )
                mw = values.astype(np.float64) * bundle.scaling.scales[0] + bundle.scaling.means[0]
                drift = float(np.abs(mw - converted).max())
                rmse = float(np.sqrt(np.mean((actual - mw) ** 2)))
                passed = (
                    (rmse <= reference_rmse * 1.01)
                    if name == "onnx_int8"
                    else bool(np.allclose(values, reference, atol=1e-5, rtol=1e-5) and drift <= 0.5)
                )
                parity.append(
                    {
                        "runtime": name,
                        "threads": threads,
                        "origins": len(values),
                        "max_abs_drift_mw": drift,
                        "rmse_mw": rmse,
                        "rmse_increase_fraction": rmse / reference_rmse - 1,
                        "accuracy_gate_passed": passed,
                    }
                )
            for batch in plan["batches"]:
                prepared = [
                    (
                        h[[indices[(i + j) % len(indices)] for j in range(batch)]],
                        c[[indices[(i + j) % len(indices)] for j in range(batch)]],
                    )
                    for i in range(len(indices))
                ]
                for name in plan["runtimes"]:
                    for i in range(plan["warmups"]):
                        execute(name, *prepared[i % len(prepared)])
                rng = random.Random(plan["seed"])
                samples = {name: [] for name in plan["runtimes"]}
                for repeat in range(plan["repeats"]):
                    for case in range(plan["calls_per_repeat"]):
                        order = list(plan["runtimes"])
                        rng.shuffle(order)
                        for name in order:
                            tick = time.perf_counter_ns()
                            execute(name, *prepared[case % len(prepared)])
                            elapsed = (time.perf_counter_ns() - tick) / 1e6
                            samples[name].append(elapsed)
                            raw.append(
                                {
                                    "runtime": name,
                                    "threads": threads,
                                    "batch": batch,
                                    "repeat": repeat,
                                    "case": case,
                                    "elapsed_ms": elapsed,
                                }
                            )
                summary = {name: latency_summary(values) for name, values in samples.items()}
                conditions.append(
                    {
                        "threads": threads,
                        "batch": batch,
                        "scope": "prepared tensor forward; includes numpy bridge for ONNX",
                        "runtimes": summary,
                    }
                )
    decisions = []
    for name in plan["runtimes"][1:]:
        matching = [p for p in parity if p["runtime"] == name]
        benefits = [
            1 - c["runtimes"][name]["p50_ms"] / c["runtimes"]["eager"]["p50_ms"]
            for c in conditions
            if c["batch"] == 1
        ]
        decisions.append(
            {
                "runtime": name,
                "accuracy_gate_passed": all(p["accuracy_gate_passed"] for p in matching),
                "b1_latency_reductions": benefits,
                "eligible": all(p["accuracy_gate_passed"] for p in matching)
                and max(benefits) >= 0.1,
                "public_promotion": False,
            }
        )
    write_report(
        OUT / "benchmark.json",
        {
            "schema_version": "gridlens_i3_runtime_v1",
            "plan_id": content_id(plan),
            "model_id": bundle.metadata["identity"],
            "environment": environment_record(),
            "build_seconds": build_seconds,
            "initialisations": initialisations,
            "artifacts": {
                file: {"sha256": sha256(OUT / file), "bytes": (OUT / file).stat().st_size}
                for file in ["timexer.ts", "timexer.onnx", "timexer-int8.onnx"]
            },
            "quantized_operator_counts": operators,
            "parity": parity,
            "conditions": conditions,
            "decisions": decisions,
            "raw_timings": raw,
            "locked_test_evaluated": False,
            "source_sha256": sha256(Path(__file__)),
        },
    )
    print(json.dumps(decisions), flush=True)


if __name__ == "__main__":
    main()
