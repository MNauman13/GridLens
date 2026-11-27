"""Cache semantics: positions, full-prefix parity, isolation and lifecycle."""

import io
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch

from gridlens.inference.cached import CacheDiagnostics, CachedNumericTransformer, ForecastCancelled
from gridlens.models.numeric import NumericTransformer


class CachedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.reference = NumericTransformer(width=8, heads=2, layers=2, dropout=0.5).eval()
        self.h = torch.randn(4, 336, 15)
        self.c = torch.randn(4, 48, 6)

    def test_batches_backends_and_independent_weights(self):
        for mode in ("cross_kv", "full_kv"):
            for backend in ("manual", "sdpa"):
                model = CachedNumericTransformer(self.reference, mode, backend)
                self.assertEqual(model.state_dict().keys(), self.reference.state_dict().keys())
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value, self.reference.state_dict()[name], atol=0, rtol=0
                    )
                    self.assertNotEqual(
                        value.data_ptr(), self.reference.state_dict()[name].data_ptr()
                    )
                for batch in (1, 4):
                    with torch.no_grad():
                        expected = self.reference(self.h[:batch], self.c[:batch])
                    torch.testing.assert_close(
                        model(self.h[:batch], self.c[:batch]), expected, atol=1e-5, rtol=1e-5
                    )
                    torch.testing.assert_close(
                        model(self.h[:batch], self.c[:batch]), expected, atol=1e-5, rtol=1e-5
                    )

    def test_interleaved_sessions_cancel_close_and_output_mutation(self):
        model = CachedNumericTransformer(self.reference)
        a_stats, b_stats = CacheDiagnostics(), CacheDiagnostics()
        event = threading.Event()
        a = model.iter_forecast(self.h[:1], self.c[:1], diagnostics=a_stats, cancelled=event.is_set)
        b = model.iter_forecast(self.h[1:2], self.c[1:2], diagnostics=b_stats)
        result = []
        for _ in range(7):
            next(a).fill_(99999)
            result.append(next(b))
        self.assertTrue(torch.is_grad_enabled())
        event.set()
        with self.assertRaises(ForecastCancelled):
            next(a)
        self.assertTrue(a_stats.closed)
        result.extend(b)
        with torch.no_grad():
            expected = self.reference(self.h[1:2], self.c[1:2])
        torch.testing.assert_close(torch.cat(result, dim=1), expected, atol=1e-5, rtol=1e-5)
        self.assertTrue(b_stats.closed)
        iterator = model.iter_forecast(self.h[:1], self.c[:1], diagnostics=CacheDiagnostics())
        next(iterator)
        iterator.close()
        with self.assertRaises(StopIteration):
            next(iterator)
        with torch.no_grad():
            expected = self.reference(self.h[:1], self.c[:1])
        torch.testing.assert_close(model(self.h[:1], self.c[:1]), expected, atol=1e-5, rtol=1e-5)

    def test_snapshots_positions_and_unwritten_cache_poison(self):
        model = CachedNumericTransformer(self.reference, backend="sdpa")
        h, c = self.h[:1].clone(), self.c[:1].clone()
        with torch.no_grad():
            expected = self.reference(h, c)
        iterator = model.iter_forecast(h, c)
        first = next(iterator)
        h.fill_(float("nan"))
        c.fill_(float("nan"))
        torch.testing.assert_close(
            torch.cat([first, *iterator], dim=1), expected, atol=1e-5, rtol=1e-5
        )
        allocate = torch.empty

        def poisoned(*args, **kwargs):
            return allocate(*args, **kwargs).fill_(float("nan"))

        with patch("gridlens.inference.cached.torch.empty", side_effect=poisoned):
            torch.testing.assert_close(
                model(self.h[:1], self.c[:1]), expected, atol=1e-5, rtol=1e-5
            )
        seen = []
        handle = model.positions.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs.get("offset", 0)), with_kwargs=True
        )
        try:
            model(self.h[:1], self.c[:1])
        finally:
            handle.remove()
        self.assertEqual(seen, [0, *range(384, 432)])

    def test_parallel_calls_state_reload_and_memory_accounting(self):
        model = CachedNumericTransformer(self.reference)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        with ThreadPoolExecutor(max_workers=2) as executor:
            outputs = list(
                executor.map(lambda i: model(self.h[i : i + 1], self.c[i : i + 1]), (0, 1))
            )
        for i, output in enumerate(outputs):
            torch.testing.assert_close(
                output, model(self.h[i : i + 1], self.c[i : i + 1]), atol=0, rtol=0
            )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        reference = NumericTransformer(width=8, heads=2, layers=2, dropout=0.5).eval()
        reference.load_state_dict(torch.load(stream, weights_only=True))
        restored = CachedNumericTransformer(reference)
        stats = CacheDiagnostics()
        actual = restored(self.h, self.c, diagnostics=stats)
        torch.testing.assert_close(actual, model(self.h, self.c), atol=0, rtol=0)
        self.assertEqual(stats.produced_slots, 48)
        self.assertEqual(len(stats.step_ms), 48)
        self.assertEqual(stats.cache_used_bytes, stats.cache_allocated_bytes)
        self.assertEqual(stats.cache_allocated_bytes, 2 * 4 * 2 * (336 + 48) * 8 * 4)
        with self.assertRaises(ValueError):
            restored(self.h, self.c, diagnostics=stats)

    def test_invalid_inputs_training_targets_and_precancel(self):
        with self.assertRaises(ValueError):
            CachedNumericTransformer(self.reference.train())
        self.reference.eval()
        model = CachedNumericTransformer(self.reference)
        for h, c in (
            (self.h[:, :335], self.c),
            (self.h, self.c[:, :47]),
            (self.h * float("nan"), self.c),
        ):
            with self.assertRaises(ValueError):
                model(h, c)
        with self.assertRaises(ValueError):
            model(self.h, self.c, target=torch.zeros(4, 48))
        with self.assertRaises(ForecastCancelled):
            model(self.h, self.c, cancelled=lambda: True)
        model.train()
        with self.assertRaises(ValueError):
            model(self.h, self.c)


class FrozenCacheTests(unittest.TestCase):
    def test_verified_real_corpus_batch_one_and_four(self):
        from pathlib import Path

        from gridlens.inference.cached import adapt_cached_bundle
        from scripts.benchmark_cached import prepared
        from scripts.benchmark_sdpa import ROOT, load_inputs, load_model

        baseline = ROOT / "reports/local/i0-public"
        if not baseline.exists():
            baseline = ROOT / "reports/local/i0"
        if not (baseline / "baseline_reproduction.json").exists():
            self.skipTest("Install pinned baseline assets first")
        if not (ROOT / "artifacts/i1/manifest.json").exists():
            self.skipTest("Install pinned I1 assets first")
        _, snapshots, _ = load_inputs(ROOT / "artifacts/i1")
        reference, _ = load_model(Path(baseline), "transformer", "manual")
        for batch in (1, 4):
            inputs = prepared(reference, snapshots, batch)
            with torch.no_grad():
                expected = [reference.model(h, c) for h, c in inputs]
            for mode, backend in [
                ("cross_kv", "manual"),
                ("full_kv", "manual"),
                ("full_kv", "sdpa"),
            ]:
                candidate = adapt_cached_bundle(reference, mode, backend)
                for (h, c), want in zip(inputs, expected, strict=True):
                    got = candidate.model(h, c)
                    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
                    self.assertLessEqual(
                        float((got - want).abs().max()) * reference.scaling.scales[0], 0.5
                    )
