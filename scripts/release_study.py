"""I4 bounded multi-seed study. Never fits or evaluates the locked test role."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gridlens.artifacts import build_manifest, sha256
from gridlens.benchmark import write_report
from gridlens.data.dataset import FEATURE_ORDER
from gridlens.data.store import content_id
from gridlens.models.baselines import Baseline
from scripts.reproduce_baseline import ROOT, extended_metrics, frozen_data, verify_pins
from scripts.train_neural import factory, predict, tensors

OUT = ROOT / "artifacts/release"


def train(config, seed, training, validation, scaling, freeze, plan):
    key = content_id(
        {
            "config": config,
            "seed": seed,
            "plan_id": content_id(plan),
            "dataset_manifest_id": freeze["manifest_id"],
            "trainer_sha256": sha256(Path(__file__)),
        }
    )
    folder = OUT / "fits" / key
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "result.json"
    if cached.exists():
        return json.loads(cached.read_text())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = factory(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x, c, y = training
    vx, vc, vy = validation
    y = (y - scaling.means[0]) / scaling.scales[0]
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    stale = 0
    previous_seconds = 0.0
    start_epoch = 0
    checkpoint_path = folder / "checkpoint.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=True)
        if state["run_id"] != key:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_state, best, stale = (
            state["history"],
            state["best_state"],
            state["best_rmse"],
            state["stale"],
        )
        torch.set_rng_state(state["rng"])
        previous_seconds = state.get("elapsed_fit_seconds", 0.0)
        start_epoch = len(history)
    start = time.perf_counter()
    for epoch in range(start_epoch, plan["maximum_epochs"]) if stale < plan["patience"] else ():
        if previous_seconds + time.perf_counter() - start > plan["per_fit_seconds"]:
            raise RuntimeError("Declared per-fit budget exhausted; no partial promotion")
        model.train()
        losses = []
        for indices in torch.randperm(len(x)).split(32):
            optimizer.zero_grad()
            target = y[indices]
            output = model(x[indices], c[indices], target if config["kind"] != "timexer" else None)
            loss = F.mse_loss(output, target)
            if not bool(loss.isfinite()):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        forecasts = predict(model, vx, vc) * scaling.scales[0] + scaling.means[0]
        rmse = float((forecasts - vy).square().mean().sqrt())
        history.append(
            {
                "epoch": epoch + 1,
                "normalised_training_mse": float(np.mean(losses)),
                "validation_rmse_mw": rmse,
            }
        )
        if rmse < best:
            best = rmse
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_state": best_state,
            "best_rmse": best,
            "stale": stale,
            "rng": torch.get_rng_state(),
            "run_id": key,
            "elapsed_fit_seconds": previous_seconds + time.perf_counter() - start,
        }
        torch.save(checkpoint, folder / "checkpoint.pt")
        print(json.dumps({"config": config, "seed": seed, **history[-1]}), flush=True)
        if stale >= plan["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    prediction = (predict(model, vx, vc) * scaling.scales[0] + scaling.means[0]).numpy()
    weights = folder / "weights.pt"
    torch.save(model.state_dict(), weights)
    identity = content_id({"run_id": key, "weights_sha256": sha256(weights)})
    metadata = {
        "config": config,
        "seed": seed,
        "identity": identity,
        "run_id": key,
        "dataset_manifest_id": freeze["manifest_id"],
        "scaling": {"means": scaling.means, "scales": scaling.scales},
        "feature_order": FEATURE_ORDER,
        "gap_slots": 48,
        "target_slots": 48,
        "training_role": "point_train",
        "selection_role": "point_validation",
        "study_plan_id": content_id(plan),
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = build_manifest(folder, ["metadata.json", "weights.pt"], kind="neural_point_model")
    (folder / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    row = {
        "kind": config["kind"],
        "config": config,
        "seed": seed,
        "model_id": identity,
        "bundle": str(folder.relative_to(ROOT)),
        "fit_seconds": previous_seconds + time.perf_counter() - start,
        "epochs": len(history),
        "best_epoch": min(history, key=lambda r: r["validation_rmse_mw"])["epoch"],
        "history": history,
        "metrics": extended_metrics(vy.numpy(), prediction),
        "validation_predictions_mw": prediction.tolist(),
    }
    cached.write_text(json.dumps(row, allow_nan=False) + "\n")
    return row


def main():
    plan = json.loads((ROOT / "configs/release_study_v1.json").read_text())
    verify_pins(ROOT, json.loads((ROOT / "configs/upgrade_experiment_v2.json").read_text()))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    datasets, freeze, scaling = frozen_data(
        ROOT, ROOT / "data/neso", json.loads((ROOT / "reports/dataset-freeze-v1.json").read_text())
    )
    training, validation = datasets["point_train"], datasets["point_validation"]
    train_tensors, val_tensors = tensors(training, scaling), tensors(validation, scaling)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for kind, alphas in [("last_day", [10]), ("weekly", [10]), ("ridge", plan["ridge_alphas"])]:
        for alpha in alphas:
            model = Baseline(kind, alpha=alpha)
            model.fit(training)
            folder = OUT / "baselines" / model.model_version
            model.save_bundle(folder)
            values = np.asarray(
                [model.predict(e.inputs).prediction_mw for e in validation.examples]
            )
            rows.append(
                {
                    "kind": kind,
                    "config": {"kind": kind, "alpha": alpha},
                    "seed": None,
                    "model_id": model.model_version,
                    "bundle": str(folder.relative_to(ROOT)),
                    "metrics": extended_metrics(val_tensors[2].numpy(), values),
                    "validation_predictions_mw": values.tolist(),
                }
            )
    for config in plan["neural_configs"]:
        for seed in plan["seeds"]:
            if time.perf_counter() - started > plan["aggregate_seconds"]:
                raise RuntimeError("Study budget exhausted before completion")
            rows.append(train(config, seed, train_tensors, val_tensors, scaling, freeze, plan))
            (OUT / "study-progress.json").write_text(
                json.dumps({"completed": len(rows), "rows": rows}, allow_nan=False)
            )
    groups = {}
    for row in rows:
        groups.setdefault(content_id(row["config"]), []).append(row)
    ranking = sorted(
        [
            {
                "config": values[0]["config"],
                "mean_rmse_mw": float(np.mean([r["metrics"]["rmse_mw"] for r in values])),
                "std_rmse_mw": float(np.std([r["metrics"]["rmse_mw"] for r in values], ddof=1))
                if len(values) > 1
                else 0,
                "seeds": [r["seed"] for r in values],
                "model_ids": [r["model_id"] for r in values],
            }
            for values in groups.values()
        ],
        key=lambda r: r["mean_rmse_mw"],
    )
    chosen = min(groups[content_id(ranking[0]["config"])], key=lambda r: r["metrics"]["rmse_mw"])
    neural = min([r for r in rows if r["kind"] == "timexer"], key=lambda r: r["metrics"]["rmse_mw"])
    ridge = min([r for r in rows if r["kind"] == "ridge"], key=lambda r: r["metrics"]["rmse_mw"])
    report = {
        "schema_version": "gridlens_release_study_v1",
        "plan_id": content_id(plan),
        "trainer_sha256": sha256(Path(__file__)),
        "dataset_manifest_id": freeze["manifest_id"],
        "target_dates": [str(e.inputs.target_start.date()) for e in validation.examples],
        "training_origins": len(training.examples),
        "ranking": ranking,
        "champion_model_id": chosen["model_id"],
        "deployment_choices": {
            "champion": chosen["model_id"],
            "timexer": neural["model_id"],
            "ridge": ridge["model_id"],
        },
        "rows": rows,
        "completed_seed_fits": len(plan["neural_configs"]) * len(plan["seeds"]),
        "elapsed_seconds": time.perf_counter() - started,
        "locked_test_evaluated": False,
        "selection_rule": plan["selection_rule"],
    }
    write_report(OUT / "study.json", report)
    (OUT / "point-model-freeze.json").write_text(
        json.dumps(
            {
                "frozen_at": __import__("datetime")
                .datetime.now(__import__("datetime").UTC)
                .isoformat(),
                "study_report_id": content_id(report),
                "champion_model_id": chosen["model_id"],
                "choices": report["deployment_choices"],
                "model_bundles": {r["model_id"]: r["bundle"] for r in rows},
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "champion": chosen["model_id"],
                "validation_rmse_mw": chosen["metrics"]["rmse_mw"],
                "completed_seed_fits": report["completed_seed_fits"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
