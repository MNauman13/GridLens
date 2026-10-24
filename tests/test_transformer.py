import io
import unittest

import torch

from gridlens.models.transformer import TokenTransformer, warmup_rate


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.model = TokenTransformer(width=16, heads=2, hidden=32, layers=2, dropout=0).eval()

    def test_causal_decoder_and_source_padding(self):
        source = torch.tensor([[3, 4, 2, 0, 0]])
        target = torch.tensor([[1, 4, 3, 2]])
        original = self.model(source, target)
        changed = target.clone()
        changed[:, 2:] = 8
        torch.testing.assert_close(original[:, :2], self.model(source, changed)[:, :2])
        torch.testing.assert_close(original, self.model(source[:, :3], target))
        self.assertEqual(original.shape, (1, 4, 13))

    def test_gradients_and_reload(self):
        source, target = torch.tensor([[3, 4, 2]]), torch.tensor([[1, 4, 3]])
        result = self.model(source, target)
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in self.model.parameters()))
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        clone = TokenTransformer(width=16, heads=2, hidden=32, layers=2, dropout=0).eval()
        clone.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(result, clone(source, target), rtol=0, atol=0)

    def test_bounded_generation_and_schedule(self):
        generated = self.model.generate(torch.tensor([[3, 4, 2]]), max_tokens=5)
        self.assertLessEqual(generated.shape[1], 6)
        self.assertEqual(generated[0, 0], 1)
        self.assertLess(warmup_rate(1, 128, 32), warmup_rate(32, 128, 32))
        self.assertLess(warmup_rate(64, 128, 32), warmup_rate(32, 128, 32))
        with self.assertRaises(ValueError):
            self.model.train().generate(torch.tensor([[3, 2]]))
