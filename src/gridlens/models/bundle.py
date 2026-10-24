"""Verified neural prediction from the complete immutable feature snapshot."""
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gridlens.artifacts import ArtifactManifest, sha256, verify_manifest
from gridlens.contracts import PointForecast
from gridlens.data.dataset import FEATURE_ORDER, FeatureSnapshot, Scaling
from gridlens.data.store import content_id
from gridlens.models.numeric import NumericLSTM, NumericTransformer
from gridlens.models.timexer import GridLensTimeXer


def make_model(config: dict[str, Any]) -> nn.Module:
    options = {key: value for key, value in config.items() if key != "kind"}
    constructors: dict[str, Any] = {"transformer": NumericTransformer, "lstm": NumericLSTM, "timexer": GridLensTimeXer}
    if config.get("kind") not in constructors:
        raise ValueError("Unknown neural model kind")
    model: nn.Module = constructors[config["kind"]](**options)
    return model


class NeuralBundle:
    def __init__(self, model: nn.Module, metadata: dict[str, Any]) -> None:
        self.model = model.eval()
        self.metadata = metadata
        parameters = metadata["scaling"]
        self.scaling = Scaling(tuple(parameters["means"]), tuple(parameters["scales"]), ())
        if len(self.scaling.means) != 3 or len(self.scaling.scales) != 3 or any(s <= 0 or not math.isfinite(s) for s in self.scaling.scales) or not all(math.isfinite(m) for m in self.scaling.means):
            raise ValueError("Invalid training scaling")
        if tuple(metadata["feature_order"]) != FEATURE_ORDER or metadata["gap_slots"] != 48 or metadata["target_slots"] != 48:
            raise ValueError("Unsupported neural feature or timing contract")

    @classmethod
    def load_bundle(cls, folder: Path) -> "NeuralBundle":
        manifest = ArtifactManifest.model_validate_json((folder / "manifest.json").read_text())
        verify_manifest(folder, manifest)
        if manifest.kind != "neural_point_model" or {f.path for f in manifest.files} != {"metadata.json", "weights.pt"}:
            raise ValueError("Unexpected neural bundle members")
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["identity"] != content_id({"run_id": metadata["run_id"], "weights_sha256": sha256(folder / "weights.pt")}):
            raise ValueError("Model identity does not match its saved weights")
        model = make_model(metadata["config"])
        model.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        return cls(model, metadata)

    def predict(self, snapshot: FeatureSnapshot) -> PointForecast:
        # InputWindow alone lacks wind/solar and missingness; never silently fabricate them.
        if len(snapshot.history) != 336 or len(snapshot.target_calendar) != 48:
            raise ValueError("Incomplete feature snapshot")
        if tuple(row[0] for row in snapshot.history) != snapshot.inputs.demand_mw:
            raise ValueError("Snapshot demand differs from its input contract")
        history = torch.tensor([self.scaling.transform(snapshot.history)], dtype=torch.float32)
        calendar = torch.tensor([snapshot.target_calendar], dtype=torch.float32)
        with torch.no_grad():
            prediction = self.model(history, calendar)[0] * self.scaling.scales[0] + self.scaling.means[0]
        return PointForecast(model_version=self.metadata["identity"], feature_snapshot_id=snapshot.inputs.feature_snapshot_id, prediction_mw=tuple(prediction.tolist()))
