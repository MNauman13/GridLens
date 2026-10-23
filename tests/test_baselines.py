"""Eligible lags, hand metrics, artifact integrity, and saved-forecast HTTP behavior."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from gridlens.contracts import BaselineForecast, DatasetManifest, TrainingExample
from gridlens.fixture import make_fixture
from gridlens.metrics import point_metrics
from gridlens.models.baselines import Baseline, lag_features

try:
    from fastapi.testclient import TestClient

    from gridlens.api import app
except ModuleNotFoundError:
    TestClient = None

ROOT = Path(__file__).resolve().parents[1]


class MiniDataset:
    def __init__(self):
        self.manifest = DatasetManifest(dataset_id="fixture", evidence_mode="retrospective_revised", truth_policy="fixture", feature_order=("ND",), source_sha256=("0" * 64,), split_start="2018-01-01", split_end="2018-01-08")
        self.examples = []
        for offset in range(8):
            inputs = make_fixture().model_copy(update={"demand_mw": tuple(float(i + offset) for i in range(336))})
            self.examples.append(TrainingExample(inputs=inputs, target_mw=tuple(2 * v + 1 for v in inputs.demand_mw[-48:]), truth_version="fixture"))

    def iter_examples(self):
        return iter(self.examples)


class BaselineTests(unittest.TestCase):
    def test_persistence_uses_exact_eligible_days(self):
        inputs = make_fixture().model_copy(update={"demand_mw": tuple(float(i) for i in range(336))})
        self.assertEqual(Baseline("last_day").predict(inputs).prediction_mw, tuple(float(i) for i in range(288, 336)))
        self.assertEqual(Baseline("weekly").predict(inputs).prediction_mw, tuple(float(i) for i in range(48, 96)))
        self.assertEqual(len(lag_features(inputs)), 100)

    def test_ridge_reloads_identical_outputs_and_rejects_corrupt_bundle(self):
        data = MiniDataset()
        model = Baseline("ridge", alpha=0.01)
        model.fit(data)
        predicted = model.predict(data.examples[0].inputs)
        self.assertEqual(len(predicted.prediction_mw), 48)
        np.testing.assert_allclose(predicted.prediction_mw, data.examples[0].target_mw, atol=0.001)
        scratch = ROOT / "reports/local"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            model.save_bundle(root)
            restored = Baseline.load_bundle(root)
            self.assertEqual(predicted, restored.predict(data.examples[0].inputs))
            (root / "model.json").write_text("{}")
            with self.assertRaises(ValueError):
                Baseline.load_bundle(root)

    def test_unfitted_ridge_and_invalid_penalties_fail(self):
        with self.assertRaises(ValueError):
            Baseline("ridge").predict(make_fixture())
        for alpha in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                Baseline("ridge", alpha=alpha)

    def test_hand_computed_point_and_peak_metrics(self):
        result = point_metrics([[1.0] * 48], [[3.0] * 48])
        self.assertEqual((result["mae_mw"], result["rmse_mw"], result["mean_daily_peak_absolute_error_mw"]), (2, 2, 2))
        actual, predicted = [0.0] * 48, [0.0] * 48
        actual[3], predicted[7] = 10, 8
        result = point_metrics([actual], [predicted])
        self.assertEqual(result["mean_peak_timing_absolute_error_slots"], 4)
        self.assertAlmostEqual(result["mae_mw"], 18 / 48)
        with self.assertRaises(ValueError):
            point_metrics([[1.0] * 47], [[1.0] * 48])

    def test_saved_artifact_rejects_edited_values_or_wrong_feature_reference(self):
        saved = json.loads((ROOT / "src/gridlens/resources/baseline_forecasts_v1.json").read_text())
        for artifact in saved.values():
            BaselineForecast.model_validate(artifact)
        changed = saved["ridge"]
        changed["point"]["prediction_mw"][0] += 1
        with self.assertRaises(ValidationError):
            BaselineForecast.model_validate(changed)


@unittest.skipIf(TestClient is None, "API extra required; full HTTP checks run in CI")
class ForecastApiTests(unittest.TestCase):
    def test_typed_48_point_response_and_invalid_model(self):
        with TestClient(app) as client:
            for model in ("last_day", "weekly", "ridge"):
                response = client.get(f"/v1/forecasts/baseline?model={model}")
                self.assertEqual(response.status_code, 200)
                artifact = BaselineForecast.model_validate(response.json())
                self.assertEqual(len(artifact.point.prediction_mw), 48)
                self.assertTrue(artifact.inputs.target_start.isoformat().startswith("2022-02-10"))
            self.assertEqual(client.get("/v1/forecasts/baseline?model=unknown").status_code, 422)

    def test_missing_artifact_returns_unavailable(self):
        with patch("gridlens.api.files") as mock:
            mock.return_value.joinpath.return_value.read_text.side_effect = OSError("not found")
            with TestClient(app) as client:
                self.assertEqual(client.get("/v1/forecasts/baseline").status_code, 503)
