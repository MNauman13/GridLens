"""Checked, inference-only SDPA adapter with identical projection/state names.

All public masks keep GridLens's True=BLOCKED convention. The adapter inverts
that convention at the PyTorch SDPA boundary. It preserves finite-score rejection using a sufficient range certificate on
normal inputs, with the exact reference score preflight as a fallback.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gridlens.artifacts import sha256
from gridlens.data.store import content_id
from gridlens.models.attention import MultiHeadAttention, scaled_dot_product_attention
from gridlens.models.bundle import NeuralBundle


def check_score_overflow(q: Tensor, k: Tensor) -> bool:
    """Prove finite scores conservatively, otherwise perform the exact preflight.

    For head widths <=1024, each dot product's absolute partial sums are bounded
    by d*max(abs(q))*max(abs(k)). A factor-four margin below the accumulator's
    finite maximum covers rounding comfortably at these widths. Failure to prove
    this bound is NOT a rejection: use the reference matmul/finite check instead.
    Small score grids use that check directly to avoid extra reduction overhead.
    Returns whether the sufficient range certificate was used (for diagnostics).
    """
    work = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    elements = q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2]
    if elements > 4096 and q.shape[-1] <= 1024:
        bound = float(q.abs().amax()) * float(k.abs().amax()) * q.shape[-1]
        if math.isfinite(bound) and bound <= torch.finfo(work).max * 0.25:
            return True
    scores = q.to(work) @ k.to(work).transpose(-2, -1) / math.sqrt(q.shape[-1])
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Attention scores overflowed")
    return False


class SDPAMultiHeadAttention(nn.Module):
    """Return no weights by default; explicit inspection uses the manual operator."""

    def __init__(self, reference: MultiHeadAttention) -> None:
        super().__init__()
        if reference.training:
            raise ValueError("SDPA adapter requires an eval-mode reference")
        self.d_model, self.heads, self.d_head = reference.d_model, reference.heads, reference.d_head
        self.q_proj, self.k_proj = reference.q_proj, reference.k_proj
        self.v_proj, self.out_proj = reference.v_proj, reference.out_proj
        self.eval()

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        *,
        blocked: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if self.training:
            raise ValueError("SDPA adapter is inference-only; use the manual model for training")
        key = query if key is None else key
        value = key if value is None else value
        if any(t.ndim != 3 or t.shape[-1] != self.d_model for t in (query, key, value)):
            raise ValueError("Expected [B,T,d_model] inputs")
        batch, queries = query.shape[:2]
        if key.shape[0] != batch or value.shape[:2] != key.shape[:2]:
            raise ValueError("Incompatible cross-attention batch/key/value lengths")
        q = self.q_proj(query).reshape(batch, queries, self.heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).reshape(batch, key.shape[1], self.heads, self.d_head).transpose(1, 2)
        v = (
            self.v_proj(value)
            .reshape(batch, value.shape[1], self.heads, self.d_head)
            .transpose(1, 2)
        )
        weights: Tensor | None = None
        if need_weights:
            attended, weights = scaled_dot_product_attention(q, k, v, blocked=blocked)
        else:
            if min(q.shape) < 1 or min(k.shape) < 1 or min(v.shape) < 1:
                raise ValueError("Attention dimensions must be nonempty")
            if (
                q.device != k.device
                or k.device != v.device
                or q.dtype != k.dtype
                or k.dtype != v.dtype
                or not q.is_floating_point()
            ):
                raise ValueError("Q, K, V must share a floating dtype and device")
            if not all(bool(torch.isfinite(t).all()) for t in (q, k, v)):
                raise ValueError("Attention inputs must be finite")
            check_score_overflow(q, k)
            score_shape = (*q.shape[:3], k.shape[2])
            allowed = None
            if blocked is not None:
                if blocked.dtype != torch.bool or blocked.device != q.device:
                    raise ValueError("Mask must be boolean and on the attention device")
                try:
                    mask = torch.broadcast_to(blocked, score_shape)
                except RuntimeError as exc:
                    raise ValueError("Mask is not broadcastable to [B,H,Tq,Tk]") from exc
                if bool(mask.all(dim=-1).any()):
                    raise ValueError("A query has no available keys")
                allowed = ~mask
            # Functional SDPA applies dropout regardless of eval(); pass zero explicitly.
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
            if not bool(torch.isfinite(attended).all()):
                raise ValueError("SDPA produced nonfinite attention output")
        merged = attended.transpose(1, 2).contiguous().reshape(batch, queries, self.d_model)
        return cast(Tensor, self.out_proj(merged)), weights


def adapt_model(reference: nn.Module) -> tuple[nn.Module, int]:
    """Deep-copy an eval model and replace attention modules without changing weights."""
    if any(m.training for m in reference.modules()):
        raise ValueError("Call eval() on the entire reference before adapting")
    if any(isinstance(m, SDPAMultiHeadAttention) for m in reference.modules()):
        raise ValueError("Model is already adapted")
    model = copy.deepcopy(reference)
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, MultiHeadAttention):
                setattr(parent, name, SDPAMultiHeadAttention(child))
                count += 1
    if not count:
        raise ValueError("Model contains no supported manual attention")
    return model.eval(), count


def runtime_record() -> dict[str, Any]:
    record = {
        "name": "pytorch_sdpa_checked_v2",
        "torch_version": str(torch.__version__),
        "adapter_sha256": sha256(Path(__file__)),
        "backend_policy": "pytorch_auto",
        "mask_convention": "True=blocked; inverted before SDPA",
        "dropout_p": 0.0,
        "score_overflow_preflight": "sufficient range certificate or exact fallback",
        "range_certificate": {
            "minimum_score_elements_exclusive": 4096,
            "maximum_head_width": 1024,
            "maximum_accumulator_fraction": 0.25,
        },
        "training_supported": False,
        "attention_weights": "not materialised by SDPA; explicit inspection uses manual operator",
    }
    return {**record, "runtime_id": content_id(record)}


def adapt_bundle(reference: NeuralBundle) -> tuple[NeuralBundle, int]:
    model, count = adapt_model(reference.model)
    return NeuralBundle(model, copy.deepcopy(reference.metadata)), count
