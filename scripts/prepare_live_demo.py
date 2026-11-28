"""Package verified I0 models and eligible inputs; export a parity-checked browser model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from gridlens.artifacts import build_manifest, sha256
from gridlens.benchmark import archive_files, validate_development_dates, write_report
from gridlens.data.store import content_id, write_once
from gridlens.models.baselines import Baseline
from gridlens.models.bundle import NeuralBundle
from scripts.build_dataset import ROOT, prepare


def json_file(path: Path, value: Any) -> None:
    write_once(path, (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())


def prepare_demo(baseline: Path, output: Path, sources: Path) -> dict[str, Any]:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    report = json.loads((baseline / "baseline_reproduction.json").read_text())
    plan = json.loads((ROOT / "configs/upgrade_experiment_v2.json").read_text())
    datasets, freeze, _ = prepare(sources)
    if freeze["manifest_id"] != report["dataset_manifest_id"]:
        raise ValueError("Live demo must use the verified I0 dataset")
    validation = datasets["point_validation"]
    dates = [str(item.inputs.target_start.date()) for item in validation.examples]
    selected = plan["benchmark_target_dates"]
    validate_development_dates(selected, dates)
    output.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    loaded: dict[str, Any] = {}
    for kind in ("ridge", "timexer"):
        row = next(item for item in report["results"] if item["kind"] == kind)
        source = baseline / row["bundle"]
        model = Baseline.load_bundle(source) if kind == "ridge" else NeuralBundle.load_bundle(source)
        identity = model.model_version if kind == "ridge" else model.metadata["identity"]
        if identity != row["model_id"]:
            raise ValueError("I0 bundle identity changed")
        loaded[kind] = model
        target = output / "server" / kind
        target.mkdir(parents=True, exist_ok=True)
        for name in (("model.json", "manifest.json") if kind == "ridge" else ("metadata.json", "weights.pt", "manifest.json")):
            write_once(target / name, (source / name).read_bytes())
        models[kind] = {"kind": kind, "model_id": identity, "bundle": f"server/{kind}", "reference_runtime": "numpy_cpu" if kind == "ridge" else "pytorch_cpu"}
    model_path = output / "client/models/timexer.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    example = validation.features[dates.index(selected[0])]
    neural = loaded["timexer"]
    history = torch.tensor([neural.scaling.transform(example.history)], dtype=torch.float32)
    calendar = torch.tensor([example.target_calendar], dtype=torch.float32)
    torch.onnx.export(neural.model, (history, calendar), str(model_path), input_names=["history", "calendar"], output_names=["prediction"], opset_version=17, dynamo=False, external_data=False)
    onnx.checker.check_model(onnx.load(str(model_path)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    models["timexer"].update({"browser_runtime": "onnx_wasm_fp32", "asset": "models/timexer.onnx", "sha256": sha256(model_path), "scaling": {"means": neural.scaling.means, "scales": neural.scaling.scales}})
    ridge_path = output / "client/models/ridge.json"
    json_file(ridge_path, loaded["ridge"].parameters())
    models["ridge"].update({"browser_runtime": "javascript_float64", "asset": "models/ridge.json", "sha256": sha256(ridge_path)})
    cases, parity = [], []
    for day in selected:
        index = dates.index(day)
        features = validation.features[index]
        case = {"inputs": features.inputs.model_dump(mode="json"), "history": features.history, "target_calendar": features.target_calendar, "observed_mw": validation.examples[index].target_mw}
        path = output / "client/cases" / f"{day}.json"
        json_file(path, case)
        cases.append({"target_date": day, "issue_date": str(features.inputs.issued_at.date()), "asset": f"cases/{day}.json", "sha256": sha256(path), "feature_snapshot_id": features.inputs.feature_snapshot_id})
        h = np.asarray([neural.scaling.transform(features.history)], dtype=np.float32)
        c = np.asarray([features.target_calendar], dtype=np.float32)
        normalised = session.run(["prediction"], {"history": h, "calendar": c})[0][0]
        converted = normalised.astype(np.float64) * neural.scaling.scales[0] + neural.scaling.means[0]
        reference = np.asarray(neural.predict(features).prediction_mw)
        maximum = float(np.abs(converted - reference).max())
        if not np.isfinite(converted).all() or maximum > .5:
            raise ValueError("Portable model failed reference prediction parity")
        parity.append({"target_date": day, "max_abs_drift_mw": maximum})
    catalog = {
        "schema_version": "gridlens_live_catalog_v1", "dataset_manifest_id": freeze["manifest_id"],
        "baseline_report_id": report["report_id"], "contract_version": "gb_nd_day_ahead_v1",
        "evidence_mode": "retrospective_revised", "evaluation_role": "point_validation",
        "models": models, "cases": cases, "horizon": 48, "context_slots": 336,
        "attribution": "Supported by National Energy SO Open Data",
        "license_url": "https://www.neso.energy/data-portal/neso-open-licence",
        "browser_parity_max_mw": max(row["max_abs_drift_mw"] for row in parity),
        "note": "Browser and server predictions are freshly computed. Historical revised-data replay is not a contemporaneously captured forecast.",
    }
    catalog["catalog_id"] = content_id(catalog)
    json_file(output / "client/catalog.json", catalog)
    members = [path.relative_to(output).as_posix() for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = build_manifest(output, members, kind="gridlens_i1_demo_v1")
    write_once(output / "manifest.json", (manifest.model_dump_json(indent=2) + "\n").encode())
    write_report(output / "portable_parity.json", {"model_id": models["timexer"]["model_id"], "runtime": "onnx_cpu_fp32", "dataset_manifest_id": freeze["manifest_id"], "cases": parity, "max_abs_drift_mw": catalog["browser_parity_max_mw"], "locked_test_evaluated": False, "purpose": "portable deployment parity, not an inference optimisation claim"})
    archive_members = [output / path for path in members] + [output / "manifest.json", output / "portable_parity.json"]
    archive = archive_files(output, archive_members, output / "GridLens_I1_Demo.zip")
    write_report(output / "archive_inventory.json", archive)
    destination = ROOT / "apps/web/public/inference"
    shutil.copytree(output / "client", destination, dirs_exist_ok=True)
    return {"catalog_id": catalog["catalog_id"], "cases": len(cases), "max_abs_drift_mw": catalog["browser_parity_max_mw"], "archive_sha256": archive["sha256"], "archive_bytes": archive["size_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports/local/i0")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/i1")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/neso")
    args = parser.parse_args()
    print(json.dumps(prepare_demo(args.baseline.resolve(), args.output.resolve(), args.sources.resolve())))


if __name__ == "__main__":
    main()
