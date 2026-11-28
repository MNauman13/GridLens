import copy
import unittest
from datetime import timedelta

import numpy as np
import torch

from gridlens.calibration.ncc import NCCNetwork, adapt_thresholds, ncc_losses
from gridlens.calibration.replay import (
    IntervalReplay,
    ReplayState,
    finite_quantile,
    issue_time,
    release_time,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.seed = ReplayState(
            "model", "cal", "rolling", scores=[[float(i + 1)] * 48 for i in range(90)]
        )

    def test_finite_rank_and_pooled_fallback(self):
        self.assertEqual(finite_quantile(np.arange(1, 11), 0.8), 9)
        self.assertEqual(finite_quantile(np.arange(1, 11), 0.95), 10)
        for values in ([], [float("nan")]):
            with self.assertRaises(ValueError):
                finite_quantile(np.asarray(values), 0.9)
        short = IntervalReplay(ReplayState("m", "c", "rolling", scores=[[10.0] * 48]))
        result = short.issue("2023-01-01", [100.0] * 48)
        self.assertTrue(result["degraded"])
        self.assertEqual(result["thresholds"][0], [10.0, 10.0, 10.0])

    def test_clock_duplicate_checkpoint_and_isolation(self):
        replay = IntervalReplay(self.seed)
        before = replay.checkpoint()
        issued = replay.issue("2023-01-01", [100.0] * 48)
        state = replay.state_id
        with self.assertRaises(ValueError):
            replay.observe(
                "2023-01-01", [150.0] * 48, release_time("2023-01-01") - timedelta(microseconds=1)
            )
        self.assertEqual(state, replay.state_id)
        self.assertTrue(replay.observe("2023-01-01", [150.0] * 48, release_time("2023-01-01")))
        self.assertFalse(replay.observe("2023-01-01", [150.0] * 48, release_time("2023-01-01")))
        with self.assertRaises(ValueError):
            replay.observe("2023-01-01", [151.0] * 48, release_time("2023-01-01"))
        restored = IntervalReplay.restore(replay.checkpoint())
        self.assertEqual(
            restored.issue("2023-01-04", [80.0] * 48), replay.issue("2023-01-04", [80.0] * 48)
        )
        rewind = IntervalReplay.restore(before)
        self.assertEqual(rewind.issue("2023-01-01", [100.0] * 48), issued)
        self.assertEqual(len(self.seed.scores), 90)
        self.assertIsNone(self.seed.last_target)
        corrupt = copy.deepcopy(before)
        corrupt["state"]["model_id"] = "other"
        with self.assertRaises(ValueError):
            IntervalReplay.restore(corrupt)
        self.assertNotEqual(
            IntervalReplay(self.seed, eta=0.01).state_id,
            IntervalReplay(self.seed, eta=0.02).state_id,
        )

    def test_delayed_feedback_must_be_applied(self):
        replay = IntervalReplay(self.seed)
        replay.issue("2023-01-01", [100.0] * 48)
        replay.issue("2023-01-02", [100.0] * 48)
        with self.assertRaises(ValueError):
            replay.issue("2023-01-03", [100.0] * 48)
        self.assertLess(release_time("2023-01-01"), issue_time("2023-01-03"))
        replay.observe("2023-01-01", [101.0] * 48, issue_time("2023-01-03"))
        replay.issue("2023-01-03", [100.0] * 48)
        with self.assertRaises(ValueError):
            replay.issue("2023-01-03", [100.0] * 48)

    def test_network_loss_gradients_and_bounded_tta(self):
        torch.manual_seed(17)
        network = NCCNetwork()
        args = (
            torch.ones(48, 30, 1),
            torch.ones(48, 30, 3),
            torch.zeros(48, 30, 3),
            torch.zeros(48, 6),
            torch.arange(48),
        )
        q = network(*args)
        self.assertEqual(tuple(q.shape), (48, 3))
        self.assertTrue(bool((torch.diff(q, dim=-1) >= 0).all()))
        losses = ncc_losses(q, torch.ones(48), args[2])
        torch.autograd.backward(sum(losses.values()))
        self.assertTrue(
            all(bool(p.grad.isfinite().all()) for p in network.parameters() if p.grad is not None)
        )
        for p in network.parameters():
            p.requires_grad_(False)
        weights = copy.deepcopy(network.state_dict())
        raw = torch.tensor([[3.0, -1.0, 2.0]]).repeat(48, 1)
        fixed, metrics = adapt_thresholds(raw, max_steps=2)
        self.assertEqual(metrics["tta_steps"], 2)
        self.assertGreater(metrics["repair_max"], 0)
        self.assertTrue(bool((fixed >= 0).all()))
        self.assertTrue(bool((fixed[:, 1:] >= fixed[:, :-1]).all()))
        for k, v in network.state_dict().items():
            torch.testing.assert_close(v, weights[k], atol=0, rtol=0)
        q = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        losses = ncc_losses(q, torch.tensor([2.0]), torch.zeros(1, 30, 3))
        self.assertAlmostEqual(float(losses["quantile"].detach()), 0.8 + 0.05, places=6)
        self.assertEqual(float(losses["monotonicity"].detach()), 0)

    def test_fixed_widths_and_controller_state(self):
        self.seed.method = "fixed"
        replay = IntervalReplay(self.seed)
        first = replay.issue("2023-01-01", [100.0] * 48)
        replay.observe("2023-01-01", [10000.0] * 48, release_time("2023-01-01"))
        second = replay.issue("2023-01-04", [100.0] * 48)
        self.assertEqual(first["thresholds"], second["thresholds"])
        state = copy.deepcopy(self.seed)
        state.method = "ncc"
        state.quantiles = [[[1.0, 2.0, 3.0]] * 48 for _ in state.scores]
        state.errors = [[[0.0, 0.0, 0.0]] * 48 for _ in state.scores]
        replay = IntervalReplay(state, network=NCCNetwork())
        replay.issue("2023-01-01", [100.0] * 48, [[0.0] * 6] * 48)
        replay.observe("2023-01-01", [100000.0] * 48, release_time("2023-01-01"))
        self.assertTrue(np.isfinite(replay.state.offset).all())
