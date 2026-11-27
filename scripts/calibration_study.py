"""Frozen-point causal interval campaign. Does not evaluate 2024."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from gridlens.artifacts import sha256
from gridlens.benchmark import write_report
from gridlens.calibration.metrics import interval_metrics
from gridlens.calibration.ncc import NCCNetwork, ncc_losses
from gridlens.calibration.replay import (
    LEVELS,
    IntervalReplay,
    ReplayState,
    finite_quantile,
    issue_time,
    release_time,
)
from gridlens.data.store import content_id
from gridlens.models.baselines import Baseline
from gridlens.models.bundle import NeuralBundle
from scripts.reproduce_baseline import ROOT, frozen_data
from scripts.train_neural import predict, tensors

OUT = ROOT / "artifacts/release/calibration"


def point_values(model, dataset, scaling):
    if isinstance(model, Baseline):
        return np.asarray([model.predict(e.inputs).prediction_mw for e in dataset.examples])
    h, c, _ = tensors(dataset, scaling)
    return (predict(model.model, h, c) * model.scaling.scales[0] + model.scaling.means[0]).numpy()


def seed_state(model_id, cal_id, method, actual, point, dates, before):
    indices = [i for i, day in enumerate(dates) if release_time(day) <= before][-90:]
    scores = np.abs(actual[indices] - point[indices])
    widths = np.asarray(
        [[finite_quantile(scores[:, slot], level) for level in LEVELS] for slot in range(48)]
    )
    quantiles = np.repeat(widths[None], len(scores), axis=0)
    return ReplayState(
        model_id,
        cal_id,
        method,
        scores=scores.tolist(),
        quantiles=quantiles.tolist(),
        errors=(scores[:, :, None] > quantiles).astype(float).tolist(),
        feedback_watermark=dates[indices[-1]],
    )


def campaign(replay, rows, *, network=None, optimizer=None, loss_weights=None):
    actuals = {row["target_date"]: row["actual"] for row in rows}
    issued = []
    training_inputs = {}
    loss_history = []
    for row in rows:
        target = row["target_date"]
        clock = issue_time(target)
        for day in sorted(list(replay.state.pending)):
            if release_time(day) <= clock:
                replay.observe(day, actuals[day], clock)
                if optimizer is not None and day in training_inputs:
                    args, score = training_inputs.pop(day)
                    network.train()
                    optimizer.zero_grad()
                    q = network(*args)
                    losses = ncc_losses(q, score, args[2])
                    loss = sum(losses[name] * weight for name, weight in loss_weights.items())
                    if not bool(loss.isfinite()):
                        raise ValueError("Nonfinite NCC loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.step()
                    loss_history.append(
                        {name: float(value.detach()) for name, value in losses.items()}
                    )
        if optimizer is not None:
            args = tuple(value.clone() for value in replay.neural_inputs(row["context"]))
            score = torch.tensor(
                np.abs(np.asarray(row["actual"]) - np.asarray(row["point"])) / replay.scale,
                dtype=torch.float32,
            )
            # Labels are retained offline but optimisers see them only on maturity.
            training_inputs[target] = (args, score)
        tick = time.perf_counter()
        result = replay.issue(target, row["point"], row["context"])
        result["interval_ms"] = (time.perf_counter() - tick) * 1000
        issued.append(result)
    return issued, loss_history


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    plan = json.loads((ROOT / "configs/calibration_study_v1.json").read_text())
    freeze = json.loads((ROOT / "artifacts/release/point-model-freeze.json").read_text())
    study = json.loads((ROOT / "artifacts/release/study.json").read_text())
    model_id = freeze["champion_model_id"]
    row = next(r for r in study["rows"] if r["model_id"] == model_id)
    bundle = ROOT / row["bundle"]
    model = (
        Baseline.load_bundle(bundle)
        if row["kind"] in ("ridge", "last_day", "weekly")
        else NeuralBundle.load_bundle(bundle)
    )
    datasets, _, scaling = frozen_data(
        ROOT, ROOT / "data/neso", json.loads((ROOT / "reports/dataset-freeze-v1.json").read_text())
    )
    validation = datasets["point_validation"]
    vactual = np.asarray([e.target_mw for e in validation.examples])
    vpoint = np.asarray(row["validation_predictions_mw"])
    vdates = study["target_dates"]
    roles = {}
    for role in ("calibrator_train", "calibrator_validation", "calibration_initialisation"):
        dataset = datasets[role]
        points = point_values(model, dataset, scaling)
        roles[role] = [
            {
                "target_date": str(e.inputs.target_start.date()),
                "point": p.tolist(),
                "actual": list(e.target_mw),
                "context": [list(c) for c in feature.target_calendar],
            }
            for e, p, feature in zip(dataset.examples, points, dataset.features, strict=True)
        ]
    scale = float(
        np.std(
            [np.asarray(r["actual"]) - np.asarray(r["point"]) for r in roles["calibrator_train"]]
        )
    )
    scale = max(1.0, scale)
    cal_id = content_id(
        {
            "plan": plan,
            "model_id": model_id,
            "point_freeze": freeze,
            "source": sha256(Path(__file__)),
        }
    )
    initial = seed_state(
        model_id,
        cal_id,
        "ncc",
        vactual,
        vpoint,
        vdates,
        issue_time(roles["calibrator_train"][0]["target_date"]),
    )
    network = NCCNetwork(plan["hidden"])
    optimizer = torch.optim.Adam(network.parameters(), lr=plan["learning_rate"])
    history = []
    for epoch in range(plan["epochs"]):
        replay = IntervalReplay(initial, network=network, scale=scale, eta=plan["controller_eta"])
        _, losses = campaign(
            replay,
            roles["calibrator_train"],
            network=network,
            optimizer=optimizer,
            loss_weights=plan["loss_weights"],
        )
        history.append(
            {
                "epoch": epoch + 1,
                "feedback_updates": len(losses),
                "losses": {
                    name: float(np.mean([r[name] for r in losses])) for name in plan["loss_weights"]
                },
            }
        )
        print(json.dumps(history[-1]), flush=True)
    network.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    weights = OUT / "ncc.pt"
    torch.save(network.state_dict(), weights)
    comparisons = {}
    checkpoints = {}
    all_records = {}
    for method in plan["methods"]:
        state = copy.deepcopy(initial)
        state.method = method
        replay = IntervalReplay(
            state,
            network=network if method == "ncc" else None,
            scale=scale,
            eta=plan["controller_eta"],
        )
        campaign(replay, roles["calibrator_train"])
        # Fixed widths are refitted once at the pre-validation cut, then frozen.
        if method == "fixed":
            replay.fixed = None
            for day in sorted(list(replay.state.pending)):
                if release_time(day) <= issue_time(
                    roles["calibrator_validation"][0]["target_date"]
                ):
                    actual = next(
                        r["actual"] for r in roles["calibrator_train"] if r["target_date"] == day
                    )
                    replay.observe(
                        day, actual, issue_time(roles["calibrator_validation"][0]["target_date"])
                    )
        # Pending final training origins also mature in the validation replay.
        # Supply pending labels without reissuing previously generated dates.
        combined = {r["target_date"]: r["actual"] for rows in roles.values() for r in rows}
        firstclock = issue_time(roles["calibrator_validation"][0]["target_date"])
        for day in sorted(list(replay.state.pending)):
            if release_time(day) <= firstclock:
                replay.observe(day, combined[day], firstclock)
        # campaign needs access to preceding pending labels on later maturity.
        records, _ = campaign_with_history(replay, roles["calibrator_validation"], combined)
        actual = [r["actual"] for r in roles["calibrator_validation"]]
        point = [r["point"] for r in roles["calibrator_validation"]]
        metrics = interval_metrics(actual, point, [r["thresholds"] for r in records])
        metrics["coverage_gate_passed"] = all(
            value["absolute_coverage_error"] <= 0.03 for value in metrics["levels"].values()
        )
        metrics["max_repair_mw"] = max(
            float(r["diagnostics"]["repair_max"]) * scale for r in records
        )
        comparisons[method] = metrics
        all_records[method] = records
        campaign_with_history(replay, roles["calibration_initialisation"], combined)
        checkpoints[method] = replay.checkpoint()
    residuals = [name for name in ("fixed", "rolling") if comparisons[name]["coverage_gate_passed"]]
    baseline = min(
        residuals or ["fixed", "rolling"],
        key=lambda name: comparisons[name]["weighted_interval_score_mw"],
    )
    ncc_gain = (
        1
        - comparisons["ncc"]["weighted_interval_score_mw"]
        / comparisons[baseline]["weighted_interval_score_mw"]
    )
    selected = (
        "ncc" if comparisons["ncc"]["coverage_gate_passed"] and ncc_gain >= 0.05 else baseline
    )
    report = {
        "schema_version": "gridlens_calibration_study_v1",
        "plan_id": content_id(plan),
        "calibration_id": cal_id,
        "point_model_id": model_id,
        "scale_mw": scale,
        "ncc_weights_sha256": sha256(weights),
        "history": history,
        "comparisons": comparisons,
        "selected": selected,
        "ncc_wis_improvement_fraction": ncc_gain,
        "ncc_status": "selected" if selected == "ncc" else "experimental",
        "coverage_gate_passed": comparisons[selected]["coverage_gate_passed"],
        "locked_test_evaluated": False,
        "theorem_transfer": False,
        "validation_target_dates": [r["target_date"] for r in roles["calibrator_validation"]],
        "point_freeze": freeze,
        "source_hashes": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [Path(__file__), *sorted((ROOT / "src/gridlens/calibration").glob("*.py"))]
        },
    }
    write_report(OUT / "study.json", report)
    (OUT / "checkpoints.json").write_text(json.dumps(checkpoints, allow_nan=False) + "\n")
    (OUT / "replay-validation.json").write_text(
        json.dumps(
            {"rows": roles["calibrator_validation"], "intervals": all_records}, allow_nan=False
        )
        + "\n"
    )
    (OUT / "freeze.json").write_text(
        json.dumps(
            {
                "calibration_id": cal_id,
                "selected": selected,
                "weights_sha256": sha256(weights),
                "checkpoints_sha256": sha256(OUT / "checkpoints.json"),
                "study_report_id": content_id(report),
                "frozen_at": __import__("datetime")
                .datetime.now(__import__("datetime").UTC)
                .isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "coverage_gate": report["coverage_gate_passed"],
                "ncc_status": report["ncc_status"],
            }
        ),
        flush=True,
    )


def campaign_with_history(replay, rows, actuals):
    records = []
    for row in rows:
        clock = issue_time(row["target_date"])
        for day in sorted(list(replay.state.pending)):
            if release_time(day) <= clock:
                replay.observe(day, actuals[day], clock)
        tick = time.perf_counter()
        record = replay.issue(row["target_date"], row["point"], row["context"])
        record["interval_ms"] = (time.perf_counter() - tick) * 1000
        records.append(record)
    return records, []


if __name__ == "__main__":
    main()
