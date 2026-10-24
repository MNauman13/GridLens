import json
import unittest
from importlib.resources import files
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from gridlens.api import app
from gridlens.saved_neural import SavedNeuralForecast


class NeuralApiTests(unittest.TestCase):
    def test_saved_models_have_finite_contract_predictions(self):
        with TestClient(app) as client:
            for model in ("transformer", "lstm", "timexer"):
                response = client.get(f"/v1/forecasts/neural?model={model}")
                self.assertEqual(response.status_code, 200)
                forecast = SavedNeuralForecast.model_validate(response.json())
                self.assertEqual(len(forecast.prediction_mw), 48)
                self.assertEqual(str(forecast.inputs.target_start.date()), "2022-02-10")
            self.assertEqual(client.get("/v1/forecasts/neural?model=unknown").status_code, 422)

    def test_changed_values_fail_integrity_and_unavailable_returns_503(self):
        saved = json.loads(files("gridlens").joinpath("resources/neural_forecasts_v1.json").read_text())["timexer"]
        saved["prediction_mw"][0] += 1
        with self.assertRaises(ValidationError):
            SavedNeuralForecast.model_validate(saved)
        with patch("gridlens.api.files") as resource:
            resource.return_value.joinpath.return_value.read_text.return_value = json.dumps({"timexer": saved})
            with TestClient(app) as client:
                self.assertEqual(client.get("/v1/forecasts/neural").status_code, 503)
        with patch("gridlens.api.files") as resource:
            resource.return_value.joinpath.return_value.read_text.side_effect = OSError("not found")
            with TestClient(app) as client:
                self.assertEqual(client.get("/v1/forecasts/neural").status_code, 503)
