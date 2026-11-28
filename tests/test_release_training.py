import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gridlens.data.dataset import Scaling
from gridlens.models.bundle import NeuralBundle
from scripts import release_study


class ReleaseTrainingTests(unittest.TestCase):
    def test_interrupted_fit_resumes_optimizer_rng_and_best_weights(self):
        torch.set_num_threads(2)
        torch.manual_seed(10)
        inputs = (torch.randn(8, 336, 15), torch.randn(8, 48, 6), torch.rand(8, 48) * 10 + 100)
        scaling = Scaling((100.0, 0.0, 0.0), (10.0, 1.0, 1.0), ())
        config = {"kind": "timexer", "width": 8, "heads": 2, "patch": 48}
        plan = {"maximum_epochs": 2, "patience": 8, "per_fit_seconds": 60}
        freeze = {"manifest_id": "test-freeze"}
        with tempfile.TemporaryDirectory(dir=release_study.ROOT / "artifacts") as folder:
            base = Path(folder)
            with patch.object(release_study, "OUT", base / "straight"):
                expected = release_study.train(config, 17, inputs, inputs, scaling, freeze, plan)
            original_save = torch.save
            interrupted = False

            def save_then_interrupt(value, path, *args, **kwargs):
                nonlocal interrupted
                original_save(value, path, *args, **kwargs)
                if Path(path).name == "checkpoint.pt" and not interrupted:
                    interrupted = True
                    raise RuntimeError("simulated process interruption")

            with patch.object(release_study, "OUT", base / "resumed"):
                with (
                    patch.object(torch, "save", side_effect=save_then_interrupt),
                    self.assertRaisesRegex(RuntimeError, "simulated"),
                ):
                    release_study.train(config, 17, inputs, inputs, scaling, freeze, plan)
                actual = release_study.train(config, 17, inputs, inputs, scaling, freeze, plan)
                cached = release_study.train(config, 17, inputs, inputs, scaling, freeze, plan)
                self.assertEqual(cached, actual)
            a = NeuralBundle.load_bundle(release_study.ROOT / expected["bundle"])
            b = NeuralBundle.load_bundle(release_study.ROOT / actual["bundle"])
            for name, value in a.model.state_dict().items():
                torch.testing.assert_close(value, b.model.state_dict()[name], atol=0, rtol=0)
            self.assertEqual(actual["history"], expected["history"])
            self.assertEqual(
                json.dumps(actual["validation_predictions_mw"]),
                json.dumps(expected["validation_predictions_mw"]),
            )
