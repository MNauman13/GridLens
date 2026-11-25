"""Independent inference equivalence, error semantics and replacement isolation."""

import unittest

import torch

from gridlens.inference.sdpa import SDPAMultiHeadAttention, adapt_model, runtime_record
from gridlens.models.attention import MultiHeadAttention, causal_mask, padding_mask
from gridlens.models.numeric import NumericTransformer
from gridlens.models.timexer import GridLensTimeXer
from gridlens.models.transformer import TokenTransformer


class SDPATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_self_cross_masks_and_optional_inspection(self):
        for queries, keys in ((5, 5), (3, 7)):
            manual = MultiHeadAttention(8, 2, dropout=0.8).double().eval()
            candidate, count = adapt_model(torch.nn.Sequential(manual).eval())
            adapter = candidate[0]
            self.assertEqual(count, 1)
            q, k, v = [torch.randn(2, n, 8, dtype=torch.float64) for n in (queries, keys, keys)]
            masks = [
                None,
                torch.zeros(queries, keys, dtype=torch.bool),
                padding_mask(torch.tensor([[False] * (keys - 1) + [True]]).expand(2, -1)),
            ]
            if queries == keys:
                masks.append(causal_mask(queries))
            with torch.no_grad():
                for blocked in masks:
                    expected, weights = manual(q, k, v, blocked=blocked)
                    actual, absent = adapter(q, k, v, blocked=blocked)
                    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
                    self.assertIsNone(absent)
                    inspected, inspected_weights = adapter(
                        q, k, v, blocked=blocked, need_weights=True
                    )
                    torch.testing.assert_close(inspected, expected, atol=0, rtol=0)
                    torch.testing.assert_close(inspected_weights, weights, atol=0, rtol=0)
                    torch.testing.assert_close(
                        adapter(q, k, v, blocked=blocked)[0], actual, atol=0, rtol=0
                    )

    def test_causal_padding_invariance(self):
        module = SDPAMultiHeadAttention(MultiHeadAttention(8, 2).eval())
        values = torch.randn(2, 5, 8)
        changed = values.clone()
        changed[:, 3:] = 900
        with torch.no_grad():
            for mask in (
                causal_mask(5),
                padding_mask(torch.tensor([[False, False, False, True, True]]).expand(2, -1)),
            ):
                torch.testing.assert_close(
                    module(values, blocked=mask)[0][:, :3], module(changed, blocked=mask)[0][:, :3]
                )

    def test_invalid_inputs_masks_and_overflow_match_rejection(self):
        manual = MultiHeadAttention(8, 2).eval()
        candidate = SDPAMultiHeadAttention(manual)
        x = torch.ones(1, 3, 8)
        with torch.no_grad():
            for module in (manual, candidate):
                for mask in (
                    torch.ones(3, 3, dtype=torch.bool),
                    torch.zeros(3, 3),
                    torch.zeros(4, 4, dtype=torch.bool),
                ):
                    with self.assertRaises(ValueError):
                        module(x, blocked=mask)
                for invalid in (
                    x[:, :0],
                    torch.full_like(x, float("nan")),
                    torch.full_like(x, float("inf")),
                    torch.full_like(x, 1e30),
                ):
                    with self.assertRaises(ValueError):
                        module(invalid)
                with self.assertRaises(ValueError):
                    module(x, torch.ones(2, 4, 8))
                with self.assertRaises(ValueError):
                    module(x[:, :, :7])

    def test_full_model_parity_weights_and_original_unchanged(self):
        for reference in (
            NumericTransformer(width=8, heads=2, layers=1),
            GridLensTimeXer(width=8, heads=2, patch=48),
        ):
            reference.eval()
            candidate, count = adapt_model(reference)
            self.assertIn(count, (3, 4))
            self.assertFalse(
                any(isinstance(m, SDPAMultiHeadAttention) for m in reference.modules())
            )
            self.assertEqual(reference.state_dict().keys(), candidate.state_dict().keys())
            for name, tensor in reference.state_dict().items():
                self.assertTrue(torch.equal(tensor, candidate.state_dict()[name]))
                self.assertNotEqual(tensor.data_ptr(), candidate.state_dict()[name].data_ptr())
            h, c = torch.randn(2, 336, 15), torch.randn(2, 48, 6)
            with torch.no_grad():
                torch.testing.assert_close(candidate(h, c), reference(h, c), atol=1e-5, rtol=1e-5)
            with self.assertRaises(ValueError):
                adapt_model(candidate)
            candidate.train()
            with self.assertRaisesRegex(ValueError, "inference-only"):
                candidate(h, c)

    def test_token_padding_causal_generation_and_rejections(self):
        reference = TokenTransformer(width=8, heads=2, hidden=16, layers=1, dropout=0.5).eval()
        candidate, _ = adapt_model(reference)
        source = torch.tensor([[3, 4, 2, 0], [5, 2, 0, 0]])
        target = torch.tensor([[1, 3, 4], [1, 5, 2]])
        with torch.no_grad():
            torch.testing.assert_close(
                candidate(source, target), reference(source, target), atol=1e-5, rtol=1e-5
            )
            self.assertTrue(
                torch.equal(
                    candidate.generate(source, max_tokens=8),
                    reference.generate(source, max_tokens=8),
                )
            )
        with self.assertRaises(ValueError):
            adapt_model(reference.train())
        with self.assertRaises(ValueError):
            adapt_model(torch.nn.Linear(2, 2).eval())
        self.assertEqual(runtime_record(), runtime_record())

    def test_range_certificate_safe_fallback_and_overflow(self):
        from gridlens.inference.sdpa import check_score_overflow

        q = torch.ones(1, 2, 80, 4)
        k = torch.ones_like(q)
        self.assertTrue(check_score_overflow(q, k))
        self.assertFalse(check_score_overflow(q[:, :, :2], k[:, :, :2]))
        # A failed sufficient bound must not reject finite, cancellation-free zeros.
        q = torch.zeros(1, 2, 80, 4)
        k = torch.zeros_like(q)
        q[..., 0] = 1e25
        k[..., 1] = 1e25
        self.assertFalse(check_score_overflow(q, k))
        with self.assertRaisesRegex(ValueError, "scores overflowed"):
            check_score_overflow(torch.full_like(q, 1e25), torch.full_like(k, 1e25))
