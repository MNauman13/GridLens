"""Request-local cross/self K/V reuse for the frozen numerical Transformer."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gridlens.artifacts import sha256
from gridlens.data.store import content_id
from gridlens.inference.sdpa import (
    SDPAMultiHeadAttention,
    adapt_model,
    check_score_overflow,
)
from gridlens.inference.sdpa import (
    runtime_record as sdpa_record,
)
from gridlens.models.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from gridlens.models.bundle import NeuralBundle
from gridlens.models.numeric import NumericTransformer, check_inputs

Mode = Literal["cross_kv", "full_kv"]
Backend = Literal["manual", "sdpa"]
Attention = MultiHeadAttention | SDPAMultiHeadAttention


class ForecastCancelled(RuntimeError):
    """The request terminated without publishing a complete forecast."""


@dataclass
class CacheDiagnostics:
    """One caller-owned diagnostic record; never stored on a shared model."""

    cache_allocated_bytes: int = 0
    cache_used_bytes: int = 0
    produced_slots: int = 0
    closed: bool = False
    stage_ms: dict[str, float] = field(default_factory=dict)
    step_ms: list[float] = field(default_factory=list)


def project(attention: Attention, layer: nn.Module, x: Tensor) -> Tensor:
    batch, length = x.shape[:2]
    return (
        cast(Tensor, layer(x))
        .reshape(batch, length, attention.heads, attention.d_head)
        .transpose(1, 2)
    )


def attend(attention: Attention, q: Tensor, k: Tensor, v: Tensor, backend: Backend) -> Tensor:
    if backend == "manual":
        output, _ = scaled_dot_product_attention(q, k, v)
    else:
        # Internal tensors have matched heads/width/device by construction. Keep
        # finite and overflow checks, including the immutable I2 range guard.
        if not all(bool(torch.isfinite(x).all()) for x in (q, k, v)):
            raise ValueError("Attention inputs must be finite")
        check_score_overflow(q, k)
        # A single LAST-position query may attend to every populated prefix key.
        # is_causal=True would incorrectly mask this non-square attention grid.
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Cached SDPA produced nonfinite attention output")
    merged = output.transpose(1, 2).contiguous().reshape(q.shape[0], q.shape[2], attention.d_model)
    return cast(Tensor, attention.out_proj(merged))


class CachedNumericTransformer(nn.Module):
    """Inference-only copy with the same parameter/state names as the reference."""

    def __init__(
        self, reference: NumericTransformer, mode: Mode = "full_kv", backend: Backend = "manual"
    ) -> None:
        super().__init__()
        if mode not in ("cross_kv", "full_kv") or backend not in ("manual", "sdpa"):
            raise ValueError("Unknown cache mode or attention backend")
        if any(m.training for m in reference.modules()):
            raise ValueError("Call eval() on the entire reference before adapting")
        model: Any = copy.deepcopy(reference) if backend == "manual" else adapt_model(reference)[0]
        self.history_projection = model.history_projection
        self.target_projection = model.target_projection
        self.positions = model.positions
        self.core = model.core
        self.output = model.output
        self.mode, self.backend = mode, backend
        self.eval()

    def iter_forecast(
        self,
        history: Tensor,
        calendar: Tensor,
        *,
        cancelled: Callable[[], bool] | None = None,
        diagnostics: CacheDiagnostics | None = None,
    ) -> Iterator[Tensor]:
        """Independent generator state; close/cancel never affects another caller.

        No no_grad context spans a yield (interleaved generators cannot alter
        each other's grad mode). Inputs and yielded values do not alias state.
        """
        if any(m.training for m in self.modules()):
            raise ValueError("Cached decoding is inference-only")
        if diagnostics is not None and (
            diagnostics.closed or diagnostics.produced_slots or diagnostics.stage_ms
        ):
            raise ValueError("Use a fresh diagnostic record per request")
        cross: list[tuple[Tensor, Tensor]] = []
        selves: list[tuple[Tensor, Tensor]] = []

        def stopped() -> None:
            if cancelled is not None and cancelled():
                raise ForecastCancelled("Forecast request cancelled")

        try:
            stopped()
            tick = time.perf_counter_ns() if diagnostics is not None else 0
            with torch.no_grad():
                check_inputs(history, calendar)
                if history.shape[0] < 1:
                    raise ValueError("A nonempty batch is required")
                h, c = history.detach().clone(), calendar.detach().clone()
                previous = h[:, -1:, 0]
            if diagnostics is not None:
                diagnostics.stage_ms["snapshot_validate"] = (time.perf_counter_ns() - tick) / 1e6
                tick = time.perf_counter_ns()
            with torch.no_grad():
                memory = self.core.encode(self.positions(self.history_projection(h)))
            if diagnostics is not None:
                diagnostics.stage_ms["encode"] = (time.perf_counter_ns() - tick) / 1e6
                tick = time.perf_counter_ns()
            with torch.no_grad():
                for block in self.core.decoders:
                    cross_attention = block.cross_attention
                    cross.append(
                        (
                            project(cross_attention, cross_attention.k_proj, memory),
                            project(cross_attention, cross_attention.v_proj, memory),
                        )
                    )
                    if self.mode == "full_kv":
                        attention = block.self_attention
                        shape = (h.shape[0], attention.heads, 48, attention.d_head)
                        selves.append(
                            (
                                torch.empty(shape, dtype=cross[-1][0].dtype, device=h.device),
                                torch.empty(shape, dtype=cross[-1][0].dtype, device=h.device),
                            )
                        )
            cross_bytes = sum(t.numel() * t.element_size() for pair in cross for t in pair)
            self_bytes = sum(t.numel() * t.element_size() for pair in selves for t in pair)
            if diagnostics is not None:
                diagnostics.cache_allocated_bytes = cross_bytes + self_bytes
                diagnostics.cache_used_bytes = cross_bytes
                diagnostics.stage_ms["cross_project_cache_allocate"] = (
                    time.perf_counter_ns() - tick
                ) / 1e6
            generated: list[Tensor] = []
            for slot in range(48):
                stopped()
                tick = time.perf_counter_ns() if diagnostics is not None else 0
                with torch.no_grad():
                    if self.mode == "full_kv":
                        x = self.positions(
                            self.target_projection(
                                torch.cat((previous[:, :, None], c[:, slot : slot + 1]), dim=-1)
                            ),
                            offset=384 + slot,
                        )
                    else:
                        shifted = torch.cat((previous, *generated), dim=1)
                        x = self.positions(
                            self.target_projection(
                                torch.cat((shifted[:, :, None], c[:, : slot + 1]), dim=-1)
                            ),
                            offset=384,
                        )
                    for index, block in enumerate(self.core.decoders):
                        attention = block.self_attention
                        if self.mode == "full_kv":
                            q = project(attention, attention.q_proj, x)
                            key, value = selves[index]
                            key[:, :, slot : slot + 1].copy_(
                                project(attention, attention.k_proj, x)
                            )
                            value[:, :, slot : slot + 1].copy_(
                                project(attention, attention.v_proj, x)
                            )
                            attended = attend(
                                attention,
                                q,
                                key[:, :, : slot + 1],
                                value[:, :, : slot + 1],
                                self.backend,
                            )
                        else:
                            attended, _ = attention(
                                x, blocked=causal_mask(slot + 1, device=x.device)[None, None]
                            )
                        x = block.norm1(x + block.dropout(attended))
                        attention = block.cross_attention
                        attended = attend(
                            attention,
                            project(attention, attention.q_proj, x),
                            *cross[index],
                            self.backend,
                        )
                        x = block.norm2(x + block.dropout(attended))
                        x = block.norm3(x + block.dropout(block.feed_forward(x)))
                    prediction = self.output(x[:, -1])
                    if self.mode == "full_kv":
                        previous = prediction
                    else:
                        generated.append(prediction)
                    if not bool(torch.isfinite(prediction).all()):
                        raise ValueError("Cached decoder produced nonfinite output")
                if diagnostics is not None:
                    diagnostics.produced_slots = slot + 1
                    diagnostics.cache_used_bytes = cross_bytes + self_bytes * (slot + 1) // 48
                    diagnostics.step_ms.append((time.perf_counter_ns() - tick) / 1e6)
                stopped()
                # A consumer may mutate its output without changing recurrence.
                yield prediction.clone()
            stopped()
        finally:
            cross.clear()
            selves.clear()
            if diagnostics is not None:
                diagnostics.closed = True

    @torch.no_grad()
    def forward(
        self,
        history: Tensor,
        calendar: Tensor,
        target: Tensor | None = None,
        *,
        cancelled: Callable[[], bool] | None = None,
        diagnostics: CacheDiagnostics | None = None,
    ) -> Tensor:
        if target is not None:
            raise ValueError("Cached inference never accepts teacher-forcing targets")
        return torch.cat(
            list(
                self.iter_forecast(history, calendar, cancelled=cancelled, diagnostics=diagnostics)
            ),
            dim=1,
        )


def adapt_cached_bundle(
    reference: NeuralBundle, mode: Mode = "full_kv", backend: Backend = "manual"
) -> NeuralBundle:
    if not isinstance(reference.model, NumericTransformer):
        raise ValueError("KV caching supports the numerical autoregressive Transformer only")
    return NeuralBundle(
        CachedNumericTransformer(reference.model, mode, backend), copy.deepcopy(reference.metadata)
    )


def cache_runtime_record(mode: Mode, backend: Backend) -> dict[str, Any]:
    if mode not in ("cross_kv", "full_kv") or backend not in ("manual", "sdpa"):
        raise ValueError("Unknown cache runtime")
    record = {
        "name": f"pytorch_{mode}_{backend}_v1",
        "cache_source_sha256": sha256(Path(__file__)),
        "torch_version": str(torch.__version__),
        "mode": mode,
        "attention_backend": backend,
        "sdpa_runtime_id": sdpa_record()["runtime_id"] if backend == "sdpa" else None,
        "cache_scope": "request-local; self capacity 48; cross capacity 336; no model buffers",
        "position_offset": 384,
        "dropout": 0,
        "result_cache": False,
        "diagnostics": "off for headline timings",
    }
    return {**record, "runtime_id": content_id(record)}
