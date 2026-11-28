"""NCC score controller, independently implemented from arXiv:2412.18144v1.

Three residual/control GRUs, an auxiliary view, attention fusion and cumulative
thresholds. Shared weights across horizon slots are an application adaptation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gridlens.models.attention import MultiHeadAttention

ALPHAS = (0.2, 0.1, 0.05)


class NCCNetwork(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.score_encoder = nn.GRU(1, hidden, batch_first=True)
        self.quantile_encoder = nn.GRU(3, hidden, batch_first=True)
        self.error_encoder = nn.GRU(3, hidden, batch_first=True)
        self.slot_embedding = nn.Embedding(48, 4)
        self.view = nn.Sequential(nn.Linear(10, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.fusion = MultiHeadAttention(hidden, 2)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        # Positive initial widths avoid a dead all-zero ReLU controller.
        nn.init.constant_(self.head.get_submodule("2").bias, 0.5)  # type: ignore[arg-type]

    def forward(
        self, scores: Tensor, quantiles: Tensor, errors: Tensor, context: Tensor, slots: Tensor
    ) -> Tensor:
        _, s = self.score_encoder(scores)
        _, q = self.quantile_encoder(quantiles)
        _, e = self.error_encoder(errors)
        view = self.view(torch.cat((context, self.slot_embedding(slots)), dim=-1))[:, None]
        tokens = torch.stack((s[-1], q[-1], e[-1]), dim=1)
        mixed, _ = self.fusion(view, tokens)
        increments = F.relu(self.head(mixed[:, 0]))
        return torch.cumsum(increments, dim=-1)


def ncc_losses(
    thresholds: Tensor, scores: Tensor, past_errors: Tensor, temperature: float = 0.1
) -> dict[str, Tensor]:
    """Paper equations 1, 2, 3 and finite-difference monotonicity.

    Scores and thresholds use the same train-fitted scale. Efficiency uses
    nonconformity scores, consistent with the pinned reference implementation.
    """
    alphas = torch.tensor(ALPHAS, dtype=thresholds.dtype, device=thresholds.device)
    difference = scores[:, None] - thresholds
    quantile = torch.maximum((1 - alphas) * difference, -alphas * difference).sum(dim=-1).mean()
    miss = torch.sigmoid(difference / temperature).clamp(0.01, 0.99)
    # At equality choose coverage, following the paper's cov indicator.
    desired_miss = (past_errors.mean(dim=1) < alphas).to(thresholds.dtype)
    coverage = F.binary_cross_entropy(miss, desired_miss)
    efficiency = ((thresholds - scores[:, None]).square() * (1 - miss)).mean()
    monotonic = F.relu(torch.diff(thresholds, dim=-1) / torch.diff(alphas)).mean()
    return {
        "quantile": quantile,
        "coverage": coverage,
        "efficiency": efficiency,
        "monotonicity": monotonic,
    }


def adapt_thresholds(
    raw: Tensor, max_steps: int = 32, rate: float = 0.02
) -> tuple[Tensor, dict[str, float | int | bool]]:
    """Bounded per-forecast h adaptation; neural weights are never changed."""
    h = torch.zeros_like(raw, requires_grad=True)
    optimizer = torch.optim.SGD([h], lr=rate)
    steps = 0
    with torch.enable_grad():
        for _ in range(max_steps):
            adjusted = raw.detach() + h
            crossing = F.relu(adjusted[:, :-1] - adjusted[:, 1:])
            if not bool(crossing.any()) and bool((adjusted >= 0).all()):
                break
            loss = crossing.sum() + F.relu(-adjusted).sum() + 0.001 * h.abs().sum()
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()
            steps += 1
    candidate = raw.detach() + h.detach()
    converged = bool((candidate[:, 1:] >= candidate[:, :-1]).all() and (candidate >= 0).all())
    # Explicit application safeguard, not the paper's unmodified guarantee.
    repaired = torch.cummax(candidate.clamp_min(0), dim=-1).values
    return repaired, {
        "tta_steps": steps,
        "converged": converged,
        "repair_max": float((repaired - candidate).abs().max()),
        "raw_crossings": int((raw[:, 1:] < raw[:, :-1]).sum()),
    }
