"""Verified CPU ONNX execution with the same 48-slot feature/output contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from gridlens.artifacts import sha256
from gridlens.contracts import PointForecast
from gridlens.data.dataset import FeatureSnapshot
from gridlens.data.store import content_id
from gridlens.models.bundle import NeuralBundle


class PortableBundle:
    def __init__(
        self, reference: NeuralBundle, graph: Path, expected_sha256: str, threads: int = 1
    ) -> None:
        if threads < 1 or sha256(graph) != expected_sha256:
            raise ValueError("Invalid portable graph identity or thread count")
        self.metadata, self.scaling = reference.metadata, reference.scaling
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
        record = {
            "name": "onnx_cpu_fp32",
            "model_id": reference.metadata["identity"],
            "graph_sha256": expected_sha256,
            "threads": threads,
            "onnxruntime": ort.__version__,
            "source_sha256": sha256(Path(__file__)),
        }
        self.runtime = {**record, "runtime_id": content_id(record)}

    def predict(self, snapshot: FeatureSnapshot) -> PointForecast:
        raw = np.asarray(snapshot.history, dtype=np.float64)
        calendar = np.asarray(snapshot.target_calendar, dtype=np.float64)
        if (
            raw.shape != (336, 15)
            or calendar.shape != (48, 6)
            or not np.isfinite(raw).all()
            or not np.isfinite(calendar).all()
        ):
            raise ValueError("Finite complete portable inputs required")
        if tuple(raw[:, 0]) != snapshot.inputs.demand_mw:
            raise ValueError("Inconsistent feature snapshot")
        h = np.asarray([self.scaling.transform(snapshot.history)], dtype=np.float32)
        c = np.asarray([snapshot.target_calendar], dtype=np.float32)
        if (
            h.shape != (1, 336, 15)
            or c.shape != (1, 48, 6)
            or not np.isfinite(h).all()
            or not np.isfinite(c).all()
        ):
            raise ValueError("Finite complete portable inputs required")
        values = self.session.run(["prediction"], {"history": h, "calendar": c})[0]
        if values.shape != (1, 48) or not np.isfinite(values).all():
            raise ValueError("Portable output contract failed")
        prediction = values[0].astype(np.float64) * self.scaling.scales[0] + self.scaling.means[0]
        return PointForecast(
            model_version=self.metadata["identity"],
            feature_snapshot_id=snapshot.inputs.feature_snapshot_id,
            prediction_mw=tuple(prediction.tolist()),
        )

    def runtime_record(self) -> dict[str, Any]:
        return dict(self.runtime)
