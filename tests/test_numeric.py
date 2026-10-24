import unittest

import torch

from gridlens.models.numeric import NumericLSTM, NumericTransformer
from gridlens.models.timexer import GridLensTimeXer, TimeXerReference


class NumericTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.history = torch.randn(2, 336, 15)
        self.calendar = torch.randn(2, 48, 6)

    def test_generated_forecasts_and_training_only_labels(self):
        for model in (NumericTransformer(width=8, heads=2), NumericLSTM(width=8), GridLensTimeXer(width=8, heads=2)):
            model.eval()
            with torch.no_grad():
                prediction = model(self.history, self.calendar)
            self.assertEqual(prediction.shape, (2, 48))
            self.assertTrue(bool(prediction.isfinite().all()))
            with self.assertRaises(ValueError):
                model(self.history, self.calendar, torch.randn(2, 48))

    def test_gap_and_future_calendar_causality(self):
        model = NumericTransformer(width=8, heads=2, dropout=0).eval()
        self.assertEqual(model.positions.encoding.shape[0], 432)
        changed = self.calendar.clone()
        changed[:, 1:] += 9
        with torch.no_grad():
            torch.testing.assert_close(model(self.history, self.calendar)[:, 0], model(self.history, changed)[:, 0])

    def test_reference_shapes_global_route_and_gradients(self):
        model = TimeXerReference(length=12, horizon=4, patch=3, width=8, heads=2, dropout=0)
        x = torch.randn(2, 12, 3, requires_grad=True)
        output = model(x)
        output.square().mean().backward()
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(bool(x.grad.isfinite().all()))
        self.assertGreater(model.global_token.grad.abs().sum().item(), 0)
        self.assertGreater(model.blocks[0].external_attention.q_proj.weight.grad.abs().sum().item(), 0)
        # Only the global token changes during the external-attention substep.
        block = model.blocks[0].eval()
        tokens = torch.randn(2, 5, 8)
        a, b = torch.randn(2, 2, 8), torch.randn(2, 2, 8)
        with torch.no_grad():
            torch.testing.assert_close(block(tokens, a)[:, :-1], block(tokens, b)[:, :-1])
        with self.assertRaises(ValueError):
            TimeXerReference(length=12, patch=5)
