"""Exercise real verified bundles when the pinned I1 artifact is installed."""
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from gridlens.api import app
from gridlens.live import AdmissionError, LiveService, owner_hash

ROOT = Path(__file__).resolve().parents[1] / "artifacts/i1"


class ReadinessTests(unittest.TestCase):
    def test_unconfigured_service_is_explicitly_unavailable(self):
        with patch.dict(os.environ, {"GRIDLENS_LIVE_ASSETS": ""}), TestClient(app) as client:
            self.assertEqual(client.get("/v1/health/live").status_code, 200)
            self.assertEqual(client.get("/v1/health/ready").status_code, 503)
            self.assertEqual(client.post("/v1/forecast-jobs", json={"model": "ridge", "target_date": "2022-01-01"}).status_code, 503)


@unittest.skipUnless((ROOT / "manifest.json").exists(), "Install pinned I1 artifact with node scripts/prepare_live_assets.mjs")
class LiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = LiveService(ROOT, Path(self.directory.name) / "jobs.sqlite3", capacity=2)

    def tearDown(self):
        self.service.close()
        self.directory.cleanup()

    def finish(self, identifier, owner):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = self.service.job(identifier, owner)
            if result["state"] in {"succeeded", "failed"}:
                return result
            time.sleep(0.01)
        self.fail("Reference worker exceeded deadline")

    def test_real_models_two_dates_new_executions_and_ownership(self):
        owner = owner_hash("one")
        identities = set()
        for kind in ("ridge", "timexer"):
            forecasts = []
            for day in ("2022-01-01", "2022-01-12"):
                identifier = self.service.submit(owner, kind, day)
                self.assertIsNone(self.service.job(identifier, owner_hash("two")))
                result = self.finish(identifier, owner)
                self.assertEqual(result["state"], "succeeded")
                payload = result["result"]
                self.assertFalse(payload["result_cache_hit"])
                self.assertEqual(len(payload["point"]["prediction_mw"]), 48)
                self.assertGreater(payload["predict_ms"], 0)
                identities.add(payload["execution_id"])
                forecasts.append(payload["point"]["prediction_mw"])
            self.assertNotEqual(*forecasts)
        self.assertEqual(len(identities), 4)
        with self.assertRaises(ValueError):
            self.service.submit(owner, "ridge", "2024-01-01")
        self.assertTrue(self.service.ready())

    def test_bounded_capacity_and_worker_failure(self):
        with patch.object(self.service._executor, "submit"):
            self.service.submit("one", "ridge", "2022-01-01")
            with self.assertRaises(AdmissionError):
                self.service.submit("one", "ridge", "2022-01-01")
            self.service.submit("two", "ridge", "2022-01-01")
            with self.assertRaises(AdmissionError):
                self.service.submit("three", "ridge", "2022-01-01")
        with self.service.connect() as db:
            db.execute("UPDATE live_jobs SET state='failed'")
        with patch.object(self.service._models["ridge"], "predict", side_effect=ValueError("test failure")):
            with self.assertLogs("gridlens.live", level="ERROR"):
                identifier = self.service.submit("one", "ridge", "2022-01-01")
                result = self.finish(identifier, "one")
            self.assertEqual(result["state"], "failed")
            self.assertIsNone(result["result"])

    def test_corrupt_weights_fail_before_worker_becomes_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            shutil.copytree(ROOT, bundle)
            weights = bundle / "server/timexer/weights.pt"
            weights.write_bytes(weights.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Corrupt or missing artifact"):
                LiveService(bundle, Path(directory) / "jobs.sqlite3")

    def test_restart_preserves_results_and_fails_interrupted_jobs(self):
        completed = self.service.submit("one", "ridge", "2022-01-01")
        self.assertEqual(self.finish(completed, "one")["state"], "succeeded")
        with patch.object(self.service._executor, "submit"):
            interrupted = self.service.submit("two", "ridge", "2022-01-01")
        database = self.service.database
        self.service.close()
        self.service = LiveService(ROOT, database)
        self.assertEqual(self.service.job(completed, "one")["state"], "succeeded")
        job = self.service.job(interrupted, "two")
        self.assertEqual(job["state"], "failed")
        self.assertEqual(job["error"], "worker_restarted")

    def test_api_cookie_isolation_and_validation(self):
        with patch.dict(os.environ, {"GRIDLENS_LIVE_ASSETS": ""}), TestClient(app) as client:
            app.state.live_service = self.service
            self.assertEqual(client.get("/v1/health/ready").status_code, 200)
            bad = client.post("/v1/forecast-jobs", json={"model": "ridge", "target_date": "2024-01-01"})
            self.assertEqual(bad.status_code, 422)
            accepted = client.post("/v1/forecast-jobs", json={"model": "ridge", "target_date": "2022-01-01"})
            self.assertEqual(accepted.status_code, 202)
            self.assertIn("HttpOnly", accepted.headers["set-cookie"])
            identifier = accepted.json()["job_id"]
            self.assertEqual(client.get(f"/v1/jobs/{identifier}").status_code, 200)
            client.cookies.clear()
            self.assertEqual(client.get(f"/v1/jobs/{identifier}").status_code, 404)
            app.state.live_service = None
