import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from gridlens.api import app
from gridlens.worker import DurableLiveService

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    (ROOT / "artifacts/i1/manifest.json").exists(), "Verified live artifacts required"
)
class WorkerTests(unittest.TestCase):
    def test_restore_compute_ownership_and_idempotent_api(self):
        with tempfile.TemporaryDirectory() as folder:
            service = DurableLiveService(
                ROOT / "artifacts/i1", Path(folder) / "queue.db", start_worker=False
            )
            try:
                ticket = service.submit("owner", "timexer", "2022-01-01", request_key="once")
                self.assertEqual(
                    service.submit("owner", "timexer", "2022-01-01", request_key="once"), ticket
                )
                self.assertTrue(service.run_once())
                result = service.job(ticket, "owner")
                self.assertEqual(result["state"], "succeeded")
                self.assertFalse(result["result"]["result_cache_hit"])
                self.assertEqual(len(result["result"]["point"]["prediction_mw"]), 48)
                self.assertIsNone(service.job(ticket, "stranger"))
                self.assertEqual(result["result"]["attempt"], 1)
            finally:
                service.close()
            with (
                patch.dict(
                    os.environ,
                    {
                        "GRIDLENS_LIVE_ASSETS": str(ROOT / "artifacts/i1"),
                        "GRIDLENS_JOB_DB": str(Path(folder) / "queue.db"),
                    },
                ),
                TestClient(app) as client,
            ):
                self.assertEqual(client.get("/v1/health/ready").status_code, 200)
                self.assertEqual(app.state.live_service.job(ticket, "owner")["state"], "succeeded")
                payload = {"model": "ridge", "target_date": "2022-01-01"}
                first = client.post(
                    "/v1/forecast-jobs", json=payload, headers={"idempotency-key": "api-once"}
                )
                self.assertEqual(first.status_code, 202)
                identifier = first.json()["job_id"]
                second = client.post(
                    "/v1/forecast-jobs", json=payload, headers={"idempotency-key": "api-once"}
                )
                self.assertEqual(second.json()["job_id"], identifier)
                for _ in range(100):
                    result = client.get("/v1/jobs/" + identifier).json()
                    if result["state"] == "succeeded":
                        break
                    time.sleep(0.02)
                self.assertEqual(result["state"], "succeeded")
                cookies = dict(client.cookies)
                client.cookies.clear()
                self.assertEqual(client.get("/v1/jobs/" + identifier).status_code, 404)
                client.cookies.update(cookies)
                self.assertEqual(client.delete("/v1/jobs/" + identifier).status_code, 200)
