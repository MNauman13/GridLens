"""Independent single-target TimeXer path and explicit GridLens adapter.

Operator reference: thuml/TimeXer@7601190, models/TimeXer.py.
No reference implementation is vendored.
"""
from typing import cast

import torch
from torch import Tensor, nn

from gridlens.models.attention import MultiHeadAttention, SinusoidalPositionalEncoding
from gridlens.models.numeric import check_inputs


class TimeXerBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(width, heads, dropout)
        self.external_attention = MultiHeadAttention(width, heads, dropout)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(width) for _ in range(3))
        self.feed_forward = nn.Sequential(nn.Linear(width, width * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width * 4, width), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, external: Tensor | None) -> Tensor:
        attended, _ = self.self_attention(tokens)
        tokens = self.norm1(tokens + self.dropout(attended))
        global_token = tokens[:, -1:]
        if external is not None:
            attended, _ = self.external_attention(global_token, external)
            global_token = self.norm2(global_token + self.dropout(attended))
        tokens = torch.cat((tokens[:, :-1], global_token), dim=1)
        return cast(Tensor, self.norm3(tokens + self.feed_forward(tokens)))


class TimeXerReference(nn.Module):
    """Gap-free single-target, variable-token path; target is channel zero."""
    def __init__(self, length: int = 336, horizon: int = 48, patch: int = 24, width: int = 32, heads: int = 4, layers: int = 2, dropout: float = 0.1, external: bool = True, positions: bool = True) -> None:
        super().__init__()
        if length < 1 or patch < 1 or length % patch:
            raise ValueError("Patch length must divide context length")
        self.length, self.patch = length, patch
        self.use_external, self.use_positions = external, positions
        self.patch_projection = nn.Linear(patch, width, bias=False)
        self.variable_projection = nn.Linear(length, width)
        self.global_token = nn.Parameter(torch.randn(1, 1, width))
        self.positions = SinusoidalPositionalEncoding(width, length // patch)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TimeXerBlock(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width * (length // patch + 1), horizon)
        self.head_dropout = nn.Dropout(dropout)

    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 3 or history.shape[1] != self.length or history.shape[2] < 2:
            raise ValueError("Expected target plus external channels [B,T,C]")
        means = history.mean(dim=1, keepdim=True).detach()
        scales = (history.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        normalized = (history - means) / scales
        patches = self.patch_projection(normalized[:, :, 0].unfold(1, self.patch, self.patch))
        if self.use_positions:
            patches = self.positions(patches)
        tokens = self.dropout(torch.cat((patches, self.global_token.expand(history.shape[0], -1, -1)), dim=1))
        external = self.dropout(self.variable_projection(normalized[:, :, 1:].transpose(1, 2))) if self.use_external else None
        for block in self.blocks:
            tokens = block(tokens, external)
        # Match reference FlattenHead's channel-before-token flatten order.
        result = self.head_dropout(self.head(self.norm(tokens).transpose(1, 2).flatten(1)))
        return cast(Tensor, result * scales[:, 0, :1] + means[:, 0, :1])


class GridLensTimeXer(nn.Module):
    """Application changes: 96-step head, discard 48 gap slots, calendar residual."""
    def __init__(self, width: int = 32, heads: int = 4, patch: int = 24, external: bool = True, positions: bool = True, calendar: bool = True) -> None:
        super().__init__()
        self.reference = TimeXerReference(horizon=96, width=width, heads=heads, patch=patch, external=external, positions=positions)
        self.calendar_head = nn.Sequential(nn.Linear(6, width), nn.ReLU(), nn.Linear(width, 1))
        self.use_calendar = calendar

    def forward(self, history: Tensor, calendar: Tensor, target: Tensor | None = None) -> Tensor:
        check_inputs(history, calendar)
        if target is not None:
            raise ValueError("Direct TimeXer requires no teacher-forcing labels")
        result = self.reference(history)[:, 48:]
        if self.use_calendar:
            result = result + self.calendar_head(calendar).squeeze(-1)
        return cast(Tensor, result)
