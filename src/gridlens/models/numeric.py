"""Gap-aware numerical Transformer and recurrent comparison model."""
from typing import cast

import torch
from torch import Tensor, nn

from gridlens.models.attention import SinusoidalPositionalEncoding
from gridlens.models.transformer import TransformerCore


def check_inputs(history: Tensor, calendar: Tensor) -> None:
    if history.ndim != 3 or history.shape[1:] != (336, 15):
        raise ValueError("Expected history [B,336,15]")
    if calendar.shape != (history.shape[0], 48, 6):
        raise ValueError("Expected known target calendar [B,48,6]")
    if not bool(history.isfinite().all() and calendar.isfinite().all()):
        raise ValueError("Non-finite features")


class NumericTransformer(nn.Module):
    def __init__(self, width: int = 32, heads: int = 4, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.history_projection = nn.Linear(15, width)
        self.target_projection = nn.Linear(7, width)
        self.positions = SinusoidalPositionalEncoding(width, 432)
        self.core = TransformerCore(width, heads, width * 4, layers, dropout)
        self.output = nn.Linear(width, 1)

    def forward(self, history: Tensor, calendar: Tensor, target: Tensor | None = None) -> Tensor:
        check_inputs(history, calendar)
        memory = self.core.encode(self.positions(self.history_projection(history)))
        previous = history[:, -1:, 0]
        if target is not None:
            if not self.training or target.shape != (history.shape[0], 48):
                raise ValueError("Teacher forcing is training-only with [B,48] labels")
            shifted = torch.cat((previous, target[:, :-1]), dim=1)
            embedded = self.positions(self.target_projection(torch.cat((shifted[:, :, None], calendar), dim=-1)), offset=384)
            return cast(Tensor, self.output(self.core.decode(embedded, memory)).squeeze(-1))
        generated = []
        for slot in range(48):
            shifted = torch.cat((previous, *generated), dim=1)
            embedded = self.positions(self.target_projection(torch.cat((shifted[:, :, None], calendar[:, :slot + 1]), dim=-1)), offset=384)
            generated.append(self.output(self.core.decode(embedded, memory)[:, -1]))
        return torch.cat(generated, dim=1)


class NumericLSTM(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.encoder = nn.LSTM(15, width, batch_first=True)
        self.decoder = nn.LSTM(7, width, batch_first=True)
        self.output = nn.Linear(width, 1)

    def forward(self, history: Tensor, calendar: Tensor, target: Tensor | None = None) -> Tensor:
        check_inputs(history, calendar)
        _, state = self.encoder(history)
        if target is not None and (not self.training or target.shape != (history.shape[0], 48)):
            raise ValueError("Teacher forcing is training-only with [B,48] labels")
        previous = history[:, -1:, 0]
        outputs = []
        for slot in range(48):
            decoded, state = self.decoder(torch.cat((previous, calendar[:, slot]), dim=-1)[:, None], state)
            prediction = self.output(decoded[:, 0])
            outputs.append(prediction)
            previous = target[:, slot:slot + 1] if target is not None else prediction
        return torch.cat(outputs, dim=1)
