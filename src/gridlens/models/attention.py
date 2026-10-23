"""Paper attention operators, implemented without PyTorch attention/Transformer layers.

Shapes: Q [B,H,Tq,Dk], K [B,H,Tk,Dk], V [B,H,Tk,Dv].
Every boolean mask in this module uses True = BLOCKED.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def causal_mask(length: int, *, device: torch.device | None = None) -> Tensor:
    if length < 1:
        raise ValueError("Causal mask requires a positive length")
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


def padding_mask(is_padding: Tensor) -> Tensor:
    if is_padding.dtype != torch.bool or is_padding.ndim != 2:
        raise ValueError("Padding flags must be boolean [B,Tk]")
    return is_padding[:, None, None, :]


def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor, *, blocked: Tensor | None = None,
    dropout: float = 0.0, training: bool = False,
) -> tuple[Tensor, Tensor]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Attention inputs must have [B,H,T,D] shape")
    if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Incompatible batch, head, key-length, or key-width dimensions")
    if min(query.shape) < 1 or min(key.shape) < 1 or min(value.shape) < 1:
        raise ValueError("Attention dimensions must be nonempty")
    if query.device != key.device or key.device != value.device or query.dtype != key.dtype or key.dtype != value.dtype or not query.is_floating_point():
        raise ValueError("Q, K, V must share a floating dtype and device")
    if not 0 <= dropout < 1:
        raise ValueError("Dropout probability must be in [0,1)")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (query, key, value)):
        raise ValueError("Attention inputs must be finite")
    # Accumulate half/bfloat16 scores in float32, avoiding avoidable softmax overflow.
    work_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    scores = query.to(work_dtype) @ key.to(work_dtype).transpose(-2, -1) / math.sqrt(query.shape[-1])
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Attention scores overflowed")
    if blocked is not None:
        if blocked.dtype != torch.bool or blocked.device != query.device:
            raise ValueError("Mask must be boolean and on the attention device")
        try:
            mask = torch.broadcast_to(blocked, scores.shape)
        except RuntimeError as exc:
            raise ValueError("Mask is not broadcastable to [B,H,Tq,Tk]") from exc
        if bool(mask.all(dim=-1).any()):
            raise ValueError("A query has no available keys")
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout, training=training)
    output = (weights @ value.to(work_dtype)).to(value.dtype)
    return output, weights.to(value.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model < 1 or heads < 1 or d_model % heads:
            raise ValueError("Require a positive model width divisible by positive head count")
        if not 0 <= dropout < 1:
            raise ValueError("Invalid dropout")
        self.d_model, self.heads, self.d_head = d_model, heads, d_model // heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor | None = None, value: Tensor | None = None, *, blocked: Tensor | None = None) -> tuple[Tensor, Tensor]:
        key = query if key is None else key
        value = key if value is None else value
        if any(t.ndim != 3 or t.shape[-1] != self.d_model for t in (query, key, value)):
            raise ValueError("Expected [B,T,d_model] inputs")
        batch, queries = query.shape[:2]
        if key.shape[0] != batch or value.shape[:2] != key.shape[:2]:
            raise ValueError("Incompatible cross-attention batch/key/value lengths")
        q = self.q_proj(query).reshape(batch, queries, self.heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).reshape(batch, key.shape[1], self.heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).reshape(batch, value.shape[1], self.heads, self.d_head).transpose(1, 2)
        attended, weights = scaled_dot_product_attention(q, k, v, blocked=blocked, dropout=self.dropout, training=self.training)
        merged = attended.transpose(1, 2).contiguous().reshape(batch, queries, self.d_model)
        return cast(Tensor, self.out_proj(merged)), weights


class SinusoidalPositionalEncoding(nn.Module):
    encoding: Tensor

    def __init__(self, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        if d_model < 1 or max_length < 1:
            raise ValueError("Positive width and maximum length required")
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies[:d_model // 2])
        self.register_buffer("encoding", encoding, persistent=True)

    def forward(self, inputs: Tensor, *, offset: int = 0) -> Tensor:
        if inputs.ndim != 3 or not inputs.is_floating_point() or inputs.shape[-1] != self.encoding.shape[-1]:
            raise ValueError("Positions require floating [B,T,d_model] input")
        if offset < 0 or offset + inputs.shape[1] > self.encoding.shape[0]:
            raise ValueError("Requested positions exceed the registered buffer")
        if inputs.device != self.encoding.device:
            raise ValueError("Move the positional module to the input device first")
        return inputs + self.encoding[offset:offset + inputs.shape[1]].to(dtype=inputs.dtype)
