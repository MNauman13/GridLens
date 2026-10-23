"""Independent PyTorch oracles, invariance tests, numerical derivatives, and positions."""

import io
import math
import unittest

try:
    import torch
    from torch.nn import functional as F

    from gridlens.models.attention import (
        MultiHeadAttention,
        SinusoidalPositionalEncoding,
        causal_mask,
        padding_mask,
        scaled_dot_product_attention,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "ML extra required; critical attention tests run with CPU PyTorch in CI")
class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_hand_calculation(self):
        q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
        k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
        v = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]], dtype=torch.float64)
        output, weights = scaled_dot_product_attention(q, k, v)
        first = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
        torch.testing.assert_close(weights, torch.tensor([[[[first, 1 - first]]]], dtype=torch.float64))
        torch.testing.assert_close(output, torch.tensor([[[[10 * first, 20 * (1 - first)]]]], dtype=torch.float64))

    def test_scaled_attention_matches_independent_operator_and_gradients(self):
        tensors = [torch.randn(2, 3, length, width, dtype=torch.float64, requires_grad=True) for length, width in [(4, 5), (7, 5), (7, 6)]]
        blocked = torch.zeros(4, 7, dtype=torch.bool)
        blocked[:, -2:] = True
        output, weights = scaled_dot_product_attention(*tensors, blocked=blocked)
        references = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
        # SDPA's boolean True means KEEP, the opposite of our single public convention.
        expected = F.scaled_dot_product_attention(*references, attn_mask=~blocked, dropout_p=0.0)
        torch.testing.assert_close(output, expected, atol=1e-10, rtol=1e-9)
        output.square().sum().backward()
        expected.square().sum().backward()
        for actual, reference in zip(tensors, references, strict=True):
            torch.testing.assert_close(actual.grad, reference.grad, atol=1e-10, rtol=1e-9)
        self.assertTrue(torch.equal(weights[..., -2:], torch.zeros_like(weights[..., -2:])))

    def test_multihead_matches_reference_self_and_cross_attention(self):
        for query_length, key_length in ((4, 4), (3, 7)):
            manual = MultiHeadAttention(8, 2).double().eval()
            oracle = torch.nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True).double().eval()
            with torch.no_grad():
                oracle.in_proj_weight.copy_(torch.cat([manual.q_proj.weight, manual.k_proj.weight, manual.v_proj.weight]))
                oracle.in_proj_bias.copy_(torch.cat([manual.q_proj.bias, manual.k_proj.bias, manual.v_proj.bias]))
                oracle.out_proj.load_state_dict(manual.out_proj.state_dict())
            q, k, v = (torch.randn(2, length, 8, dtype=torch.float64) for length in (query_length, key_length, key_length))
            if query_length == key_length:
                k = v = q
            blocked = torch.zeros(query_length, key_length, dtype=torch.bool)
            blocked[:, -1] = True
            actual, weights = manual(q, k, v, blocked=blocked)
            expected, reference_weights = oracle(q, k, v, attn_mask=blocked, average_attn_weights=False)
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(weights, reference_weights, atol=1e-10, rtol=1e-9)

    def test_future_and_padding_token_changes_cannot_affect_allowed_outputs(self):
        model = MultiHeadAttention(8, 2).eval()
        x = torch.randn(2, 5, 8)
        changed = x.clone()
        changed[:, 3:] = 999
        original = model(x, blocked=causal_mask(5))[0]
        perturbed = model(changed, blocked=causal_mask(5))[0]
        torch.testing.assert_close(original[:, :3], perturbed[:, :3])
        pad = torch.tensor([[False, False, False, True, True]]).expand(2, -1)
        original, weights = model(x, blocked=padding_mask(pad))
        perturbed = model(changed, blocked=padding_mask(pad))[0]
        # Query rows whose own input changed are not claimed invariant.
        torch.testing.assert_close(original[:, :3], perturbed[:, :3])
        self.assertEqual(weights[..., 3:].count_nonzero(), 0)

    def test_fully_blocked_queries_and_bad_masks_fail_explicitly(self):
        q = torch.ones(1, 1, 3, 2)
        for mask in (torch.ones(3, 3, dtype=torch.bool), torch.zeros(3, 3), torch.zeros(4, 4, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                scaled_dot_product_attention(q, q, q, blocked=mask)
        for width, heads in ((7, 2), (0, 1), (8, 0)):
            with self.assertRaises(ValueError):
                MultiHeadAttention(width, heads)

    def test_numerical_gradient_check_and_eval_dropout(self):
        q, k, v = [torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        self.assertTrue(torch.autograd.gradcheck(lambda a,b,c: scaled_dot_product_attention(a,b,c)[0], (q,k,v)))
        model = MultiHeadAttention(8, 2, dropout=0.5).eval()
        x = torch.randn(2, 4, 8, requires_grad=True)
        self.assertTrue(torch.equal(model(x)[0], model(x)[0]))
        model.train()
        model(x)[0].square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_positions_known_values_odd_width_offsets_and_reload(self):
        module = SinusoidalPositionalEncoding(5, 10)
        encoded = module(torch.zeros(1, 3, 5))
        torch.testing.assert_close(encoded[0, 0], torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32))
        for column in range(5):
            pair = column // 2
            angle = 1 / 10000 ** (2 * pair / 5)
            self.assertAlmostEqual(encoded[0, 1, column].item(), math.sin(angle) if column % 2 == 0 else math.cos(angle), places=6)
        torch.testing.assert_close(module(torch.zeros(1, 2, 5), offset=1), encoded[:, 1:])
        self.assertEqual(list(module.parameters()), [])
        self.assertIn("encoding", dict(module.named_buffers()))
        stream = io.BytesIO()
        torch.save(module.state_dict(), stream)
        stream.seek(0)
        restored = SinusoidalPositionalEncoding(5, 10)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored(torch.zeros(1, 3, 5)), encoded, rtol=0, atol=0)
        self.assertEqual(module.double()(torch.zeros(1, 2, 5, dtype=torch.float64)).dtype, torch.float64)
        with self.assertRaises(ValueError):
            module(torch.zeros(1, 11, 5))

    def test_half_precision_accumulation_keeps_outputs_finite(self):
        q = torch.full((1, 1, 2, 8), 100, dtype=torch.float16)
        output, weights = scaled_dot_product_attention(q, q, q)
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(weights, torch.full_like(weights, .5))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable; CPU tests remain mandatory")
    def test_cuda_device_and_gradients(self):
        model = MultiHeadAttention(8, 2).cuda()
        position = SinusoidalPositionalEncoding(8).cuda()
        x = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
        output, _ = model(position(x), blocked=causal_mask(4, device=x.device))
        output.sum().backward()
        self.assertEqual(output.device.type, "cuda")
        self.assertTrue(torch.isfinite(x.grad).all())
