import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from gridlens.contracts import InputWindow
from gridlens.data.dataset import FeatureSnapshot
from gridlens.inference.portable import PortableBundle
from gridlens.models.bundle import NeuralBundle
from gridlens.worker import DurableLiveService

ROOT = Path(__file__).resolve().parents[1] / "artifacts/i1"


@unittest.skipUnless((ROOT / "manifest.json").exists(), "Verified live artifact required")
class PortableTests(unittest.TestCase):
    def test_active_corpus_parity_invalid_inputs_and_runtime_identity(self):
        catalog = json.loads((ROOT / "client/catalog.json").read_text())
        entry = catalog["models"]["timexer"]
        reference = NeuralBundle.load_bundle(ROOT / entry["bundle"])
        candidate = PortableBundle(reference, ROOT / "client" / entry["asset"], entry["sha256"])
        self.assertEqual(candidate.runtime_record()["model_id"], reference.metadata["identity"])
        with self.assertRaises(ValueError):
            PortableBundle(reference, ROOT / "client" / entry["asset"], "0" * 64)
        for row in catalog["cases"]:
            case = json.loads((ROOT / "client" / row["asset"]).read_text())
            snapshot = FeatureSnapshot(
                InputWindow.model_validate(case["inputs"]),
                tuple(map(tuple, case["history"])),
                tuple(map(tuple, case["target_calendar"])),
            )
            got = candidate.predict(snapshot)
            want = reference.predict(snapshot)
            self.assertLessEqual(
                max(abs(a - b) for a, b in zip(got.prediction_mw, want.prediction_mw, strict=True)),
                0.5,
            )
        with self.assertRaises(ValueError):
            candidate.predict(replace(snapshot, target_calendar=()))
        with self.assertRaises(ValueError):
            candidate.predict(
                replace(snapshot, history=tuple((float("nan"), *r[1:]) for r in snapshot.history))
            )
        with tempfile.TemporaryDirectory() as folder:
            service = DurableLiveService(
                ROOT, Path(folder) / "jobs.db", enable_onnx=True, start_worker=False
            )
            try:
                self.assertIn("onnx", service.runtime_catalog["timexer"])
                ticket = service.submit(
                    "owner", "timexer", catalog["cases"][0]["target_date"], "onnx", "once"
                )
                service.run_once()
                result = service.job(ticket, "owner")
                self.assertEqual(result["state"], "succeeded")
                self.assertEqual(result["result"]["runtime"], "onnx_cpu_fp32")
                self.assertEqual(len(result["result"]["runtime_id"]), 64)
                with self.assertRaises(ValueError):
                    service.submit(
                        "another", "ridge", catalog["cases"][0]["target_date"], "onnx", "bad"
                    )
            finally:
                service.close()
