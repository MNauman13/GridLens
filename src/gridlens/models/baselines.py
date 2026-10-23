"""Eligible-lag persistence and a directly solved multi-output ridge model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from gridlens.artifacts import ArtifactManifest, build_manifest, verify_manifest
from gridlens.contracts import ForecastModel, InputWindow, PointForecast, TrainingDataset
from gridlens.data.dataset import calendar
from gridlens.data.store import content_id, write_once

Kind = Literal["last_day", "weekly", "ridge"]


def lag_features(inputs: InputWindow) -> tuple[float, ...]:
    # Target = issue+1; target-7 = issue-6, one day into the seven-day context.
    return inputs.demand_mw[-48:] + inputs.demand_mw[48:96] + calendar(inputs.target_start)[2:]


class Baseline(ForecastModel):
    def __init__(self, kind: Kind, *, alpha: float = 10.0) -> None:
        if kind not in ("last_day", "weekly", "ridge") or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Unknown baseline or non-positive ridge penalty")
        self.kind = kind
        self.alpha = alpha
        self.training_dataset_id: str | None = None
        self.x_mean: NDArray[np.float64] | None = None
        self.x_scale: NDArray[np.float64] | None = None
        self.y_mean: NDArray[np.float64] | None = None
        self.weights: NDArray[np.float64] | None = None

    def fit(self, dataset: TrainingDataset) -> None:
        self.training_dataset_id = dataset.manifest.dataset_id
        if self.kind != "ridge":
            return
        examples = list(dataset.iter_examples())
        if not examples:
            raise ValueError("Cannot fit an empty dataset")
        x = np.asarray([lag_features(e.inputs) for e in examples], dtype=np.float64)
        y = np.asarray([e.target_mw for e in examples], dtype=np.float64)
        self.x_mean = x.mean(axis=0)
        self.x_scale = np.maximum(x.std(axis=0), 1e-8)
        self.y_mean = y.mean(axis=0)
        z = (x - self.x_mean) / self.x_scale
        self.weights = np.linalg.solve(z.T @ z + self.alpha * np.eye(z.shape[1]), z.T @ (y - self.y_mean))

    @property
    def model_version(self) -> str:
        return f"{self.kind}_v1_" + content_id(self.parameters())

    def parameters(self) -> dict[str, Any]:
        return {"kind": self.kind, "alpha": self.alpha, "training_dataset_id": self.training_dataset_id, "lag_definition": "last_available_day_48_then_target_minus_7_days_48_then_target_week_and_year_calendar_4", **{name: None if (array := getattr(self, name)) is None else array.tolist() for name in ("x_mean", "x_scale", "y_mean", "weights")}}

    def predict(self, inputs: InputWindow) -> PointForecast:
        if self.kind == "last_day":
            values = inputs.demand_mw[-48:]
        elif self.kind == "weekly":
            values = inputs.demand_mw[48:96]
        else:
            if self.x_mean is None or self.x_scale is None or self.y_mean is None or self.weights is None:
                raise ValueError("Ridge has not been fitted")
            values = tuple(float(v) for v in ((np.asarray(lag_features(inputs)) - self.x_mean) / self.x_scale) @ self.weights + self.y_mean)
        return PointForecast(model_version=self.model_version, feature_snapshot_id=inputs.feature_snapshot_id, prediction_mw=values)

    def save_bundle(self, destination: Path) -> None:
        write_once(destination / "model.json", (json.dumps(self.parameters(), sort_keys=True, allow_nan=False) + "\n").encode())
        manifest = build_manifest(destination, ["model.json"], kind="gridlens_baseline_v1")
        write_once(destination / "manifest.json", (manifest.model_dump_json(indent=2) + "\n").encode())

    @classmethod
    def load_bundle(cls, source: Path) -> Baseline:
        manifest = ArtifactManifest.model_validate_json((source / "manifest.json").read_text())
        verify_manifest(source, manifest)
        if manifest.kind != "gridlens_baseline_v1" or [f.path for f in manifest.files] != ["model.json"]:
            raise ValueError("Incompatible baseline bundle")
        record = json.loads((source / "model.json").read_text())
        model = cls(record["kind"], alpha=record["alpha"])
        model.training_dataset_id = record["training_dataset_id"]
        if model.kind == "ridge":
            for name, shape in (("x_mean", (100,)), ("x_scale", (100,)), ("y_mean", (48,)), ("weights", (100, 48))):
                value = np.asarray(record[name], dtype=np.float64)
                if value.shape != shape or not np.isfinite(value).all() or (name == "x_scale" and (value <= 0).any()):
                    raise ValueError("Invalid fitted ridge parameters")
                setattr(model, name, value)
        return model
