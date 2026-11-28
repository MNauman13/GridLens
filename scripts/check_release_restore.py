"""Restore a private release artifact into a disposable directory and verify it."""

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from gridlens.artifacts import safe_member
from gridlens.calibration.ncc import NCCNetwork
from gridlens.calibration.replay import IntervalReplay
from gridlens.contracts import InputWindow
from gridlens.data.dataset import FeatureSnapshot
from gridlens.models.bundle import NeuralBundle
from scripts.reproduce_baseline import ROOT


def verify(archive, expected_sha=None):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected_sha and digest != expected_sha:
        raise ValueError("Archive digest mismatch")
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as folder:
        restored = Path(folder)
        with zipfile.ZipFile(archive) as source:
            inventory = json.loads(source.read("ARCHIVE_INVENTORY.json"))
            expected = {row["path"]: row for row in inventory}
            if sum(row.file_size for row in source.infolist()) > 250_000_000:
                raise ValueError("Oversized restore archive")
            for row in source.infolist():
                destination = safe_member(restored, row.filename)
                if row.is_dir():
                    continue
                if ((row.external_attr >> 16) & 0xF000) == 0xA000:
                    raise ValueError("Symlink in restore archive")
                data = source.read(row)
                if row.filename not in ("ARCHIVE_INVENTORY.json", "RESTORE.md"):
                    item = expected.pop(row.filename, None)
                    if (
                        item is None
                        or item["bytes"] != len(data)
                        or item["sha256"] != hashlib.sha256(data).hexdigest()
                    ):
                        raise ValueError("Archive member digest mismatch")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            if expected:
                raise ValueError("Restore archive is incomplete")
        release = restored / "artifacts/release"
        study = json.loads((release / "study.json").read_text())
        frozen = json.loads((ROOT / "configs/final_release_freeze_v1.json").read_text())
        selected = next(row for row in study["rows"] if row["model_id"] == frozen["point_model_id"])
        model = NeuralBundle.load_bundle(restored / selected["bundle"])
        catalog = json.loads((release / "live-demo/client/catalog.json").read_text())
        entry = catalog["cases"][0]
        case = json.loads((release / "live-demo/client" / entry["asset"]).read_text())
        snapshot = FeatureSnapshot(
            InputWindow.model_validate(case["inputs"]),
            tuple(map(tuple, case["history"])),
            tuple(map(tuple, case["target_calendar"])),
        )
        actual = np.asarray(model.predict(snapshot).prediction_mw)
        original = np.asarray(
            selected["validation_predictions_mw"][study["target_dates"].index(entry["target_date"])]
        )
        drift = float(np.abs(actual - original).max())
        if drift > 0.5 or model.metadata["identity"] != frozen["point_model_id"]:
            raise ValueError("Restored model disagrees with frozen validation evidence")
        for filename, key in (
            ("ncc.pt", "weights_sha256"),
            ("checkpoints.json", "checkpoints_sha256"),
        ):
            if (
                hashlib.sha256((release / "calibration" / filename).read_bytes()).hexdigest()
                != frozen["calibration"][key]
            ):
                raise ValueError("Restored calibration differs from the committed freeze")
        network = NCCNetwork()
        network.load_state_dict(torch.load(release / "calibration/ncc.pt", weights_only=True))
        checkpoints = json.loads((release / "calibration/checkpoints.json").read_text())
        for method, record in checkpoints.items():
            state = IntervalReplay.restore(record, network=network if method == "ncc" else None)
            if state.checkpoint()["checkpoint_id"] != record["checkpoint_id"]:
                raise ValueError("Restored calibration state changed")
        return {
            "archive_sha256": digest,
            "verified_members": len(inventory),
            "restored_model_id": model.metadata["identity"],
            "validation_target_date": entry["target_date"],
            "max_drift_from_recorded_batch_validation_mw": drift,
            "restored_calibration_methods": sorted(checkpoints),
            "locked_test_re_evaluated": False,
            "passed": True,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archive", type=Path, default=ROOT / "reports/local/GridLens_Release_Artifacts.zip"
    )
    p.add_argument("--expected-sha")
    args = p.parse_args()
    result = verify(args.archive, args.expected_sha)
    (ROOT / "reports/local/release-restore.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
