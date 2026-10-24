"""Post-norm encoder-decoder built entirely from GridLens attention."""
import math
from typing import cast

import torch
from torch import Tensor, nn

from gridlens.models.attention import (
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    causal_mask,
)


class EncoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(width, heads, dropout)
        self.feed_forward = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, width))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, blocked: Tensor | None = None) -> Tensor:
        attended, _ = self.attention(x, blocked=blocked)
        x = self.norm1(x + self.dropout(attended))
        return cast(Tensor, self.norm2(x + self.dropout(self.feed_forward(x))))


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(width, heads, dropout)
        self.cross_attention = MultiHeadAttention(width, heads, dropout)
        self.feed_forward = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, width))
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(width) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, target_mask: Tensor, source_mask: Tensor | None = None) -> Tensor:
        attended, _ = self.self_attention(x, blocked=target_mask)
        x = self.norm1(x + self.dropout(attended))
        attended, _ = self.cross_attention(x, memory, blocked=source_mask)
        x = self.norm2(x + self.dropout(attended))
        return cast(Tensor, self.norm3(x + self.dropout(self.feed_forward(x))))


class TransformerCore(nn.Module):
    def __init__(self, width: int = 128, heads: int = 4, hidden: int = 512, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("At least one layer required")
        self.encoders = nn.ModuleList([EncoderBlock(width, heads, hidden, dropout) for _ in range(layers)])
        self.decoders = nn.ModuleList([DecoderBlock(width, heads, hidden, dropout) for _ in range(layers)])

    def encode(self, source: Tensor, blocked: Tensor | None = None) -> Tensor:
        for layer in self.encoders:
            source = layer(source, blocked)
        return source

    def decode(self, target: Tensor, memory: Tensor, source_mask: Tensor | None = None, target_padding: Tensor | None = None) -> Tensor:
        blocked = causal_mask(target.shape[1]).to(target.device)[None, None]
        if target_padding is not None:
            blocked = blocked | target_padding[:, None, None, :]
        for layer in self.decoders:
            target = layer(target, memory, blocked, source_mask)
        return target


class TokenTransformer(nn.Module):
    PAD, BOS, EOS = 0, 1, 2

    def __init__(self, vocabulary: int = 13, width: int = 128, heads: int = 4, hidden: int = 512, layers: int = 2, dropout: float = 0.1, tie_weights: bool = True) -> None:
        super().__init__()
        if vocabulary < 4:
            raise ValueError("Vocabulary must include data tokens")
        self.width = width
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=self.PAD)
        self.positions = SinusoidalPositionalEncoding(width)
        self.dropout = nn.Dropout(dropout)
        self.core = TransformerCore(width, heads, hidden, layers, dropout)
        self.output = nn.Linear(width, vocabulary, bias=False)
        if tie_weights:
            self.output.weight = self.embedding.weight

    def embed(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, self.dropout(self.positions(self.embedding(tokens) * math.sqrt(self.width))))

    def forward(self, source: Tensor, shifted_target: Tensor) -> Tensor:
        source_mask = (source == self.PAD)[:, None, None, :]
        memory = self.core.encode(self.embed(source), source_mask)
        decoded = self.core.decode(self.embed(shifted_target), memory, source_mask, shifted_target == self.PAD)
        return cast(Tensor, self.output(decoded))

    @torch.no_grad()
    def generate(self, source: Tensor, max_tokens: int = 32) -> Tensor:
        if self.training:
            raise ValueError("Call eval() before generation")
        if not 1 <= max_tokens < 512:
            raise ValueError("Generation bound must be between 1 and 511")
        source_mask = (source == self.PAD)[:, None, None, :]
        memory = self.core.encode(self.embed(source), source_mask)
        result = torch.full((source.shape[0], 1), self.BOS, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
        for _ in range(max_tokens):
            logits = self.output(self.core.decode(self.embed(result), memory, source_mask)[:, -1])
            logits[:, :2] = -torch.inf
            token = logits.argmax(-1).masked_fill(finished, self.PAD)
            result = torch.cat((result, token[:, None]), dim=1)
            finished |= token == self.EOS
            if bool(finished.all()):
                break
        return result


def warmup_rate(step: int, width: int, warmup: int = 4000) -> float:
    if min(step, width, warmup) < 1:
        raise ValueError("Positive schedule arguments required")
    return float(width ** -0.5 * min(step ** -0.5, step * warmup ** -1.5))
