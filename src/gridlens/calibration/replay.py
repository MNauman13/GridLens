"""Serializable, session-local intervals with matured, idempotent feedback."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import numpy as np

from gridlens.data.store import content_id

LEVELS = (0.8, 0.9, 0.95)


def issue_time(target: str) -> datetime:
    return datetime.combine(date.fromisoformat(target) - timedelta(days=1), time(9, 30), tzinfo=UTC)


def release_time(target: str) -> datetime:
    return datetime.combine(date.fromisoformat(target) + timedelta(days=1), time(9), tzinfo=UTC)


def finite_quantile(values: np.ndarray, level: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or not 0 < level < 1:
        raise ValueError("Finite nonempty scores and a valid level required")
    rank = min(len(values), math.ceil((len(values) + 1) * level))
    return float(np.sort(values)[rank - 1])


@dataclass
class ReplayState:
    model_id: str
    calibration_id: str
    method: str
    scores: list[list[float]] = field(default_factory=list)
    quantiles: list[list[list[float]]] = field(default_factory=list)
    errors: list[list[list[float]]] = field(default_factory=list)
    offset: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0] for _ in range(48)])
    receipts: dict[str, str] = field(default_factory=dict)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_target: str | None = None
    feedback_watermark: str | None = None


class IntervalReplay:
    def __init__(
        self,
        state: ReplayState,
        *,
        network: Any = None,
        scale: float = 1000.0,
        window: int = 90,
        minimum: int = 60,
        fixed: list[list[float]] | None = None,
        lookback: int = 30,
        eta: float = 0.01,
    ) -> None:
        if state.method not in ("fixed", "rolling", "ncc"):
            raise ValueError("Unknown interval method")
        if (
            window < minimum
            or minimum < 1
            or not math.isfinite(scale)
            or scale <= 0
            or lookback < 1
            or lookback > window
            or not math.isfinite(eta)
            or eta < 0
        ):
            raise ValueError("Invalid calibration configuration")
        self.state = copy.deepcopy(state)
        self.network = network
        self.scale = scale
        self.window, self.minimum, self.lookback, self.eta = window, minimum, lookback, eta
        self.fixed = copy.deepcopy(fixed)

    def residual_thresholds(self) -> tuple[np.ndarray, bool]:
        scores = np.asarray(self.state.scores[-self.window :], dtype=float)
        if scores.ndim != 2 or scores.shape[1] != 48 or not len(scores):
            raise ValueError("No eligible calibration feedback")
        degraded = len(scores) < self.minimum
        if degraded:
            pooled = [finite_quantile(scores, level) for level in LEVELS]
            return np.tile(pooled, (48, 1)), True
        return np.asarray(
            [[finite_quantile(scores[:, slot], level) for level in LEVELS] for slot in range(48)]
        ), False

    def neural_inputs(self, context: list[list[float]]) -> tuple[Any, ...]:
        import torch

        if np.asarray(context).shape != (48, 6) or not np.isfinite(context).all():
            raise ValueError("Finite 48-slot calendar context required")
        count = min(self.lookback, len(self.state.scores))
        if count < 1:
            raise ValueError("NCC requires eligible historical scores")
        scores = np.asarray(self.state.scores[-count:])[:, :, None] / self.scale
        quantiles = np.asarray(self.state.quantiles[-count:]) / self.scale
        errors = np.asarray(self.state.errors[-count:])
        if len(quantiles) != count or len(errors) != count:
            raise ValueError("Incomplete controller history")
        if count < self.lookback:
            scores = np.concatenate((np.repeat(scores[:1], self.lookback - count, axis=0), scores))
            quantiles = np.concatenate(
                (np.repeat(quantiles[:1], self.lookback - count, axis=0), quantiles)
            )
            errors = np.concatenate((np.repeat(errors[:1], self.lookback - count, axis=0), errors))
        return (
            torch.tensor(scores.transpose(1, 0, 2), dtype=torch.float32),
            torch.tensor(quantiles.transpose(1, 0, 2), dtype=torch.float32),
            torch.tensor(errors.transpose(1, 0, 2), dtype=torch.float32),
            torch.tensor(context, dtype=torch.float32),
            torch.arange(48),
        )

    def observe(self, target: str, actual: list[float], now: datetime) -> bool:
        if now.tzinfo is None or now.utcoffset() != timedelta(0):
            raise ValueError("UTC feedback clock required")
        if release_time(target) > now:
            raise ValueError("Future feedback is not eligible")
        values = np.asarray(actual, dtype=float)
        if values.shape != (48,) or not np.isfinite(values).all():
            raise ValueError("Invalid feedback curve")
        digest = content_id(values.tolist())
        if target in self.state.receipts:
            if self.state.receipts[target] != digest:
                raise ValueError("Conflicting duplicate feedback")
            return False
        if self.state.feedback_watermark is not None and target < self.state.feedback_watermark:
            raise ValueError("Out-of-order feedback")
        event = self.state.pending.get(target)
        if event is None:
            raise ValueError("No issued forecast for this feedback")
        scores = np.abs(values - np.asarray(event["point"]))
        thresholds = np.asarray(event["thresholds"])
        errors = (scores[:, None] > thresholds).astype(float)
        self.state.scores.append(scores.tolist())
        self.state.quantiles.append(thresholds.tolist())
        self.state.errors.append(errors.tolist())
        self.state.scores = self.state.scores[-self.window :]
        self.state.quantiles = self.state.quantiles[-self.window :]
        self.state.errors = self.state.errors[-self.window :]
        if self.state.method == "ncc":
            mean = np.mean(self.state.errors[-self.lookback :], axis=0)
            self.state.offset = (
                np.asarray(self.state.offset) + self.eta * (mean - np.asarray([0.2, 0.1, 0.05]))
            ).tolist()
        self.state.receipts[target] = digest
        self.state.feedback_watermark = target
        del self.state.pending[target]
        return True

    def issue(
        self, target: str, point: list[float], context: list[list[float]] | None = None
    ) -> dict[str, Any]:
        values = np.asarray(point, dtype=float)
        if values.shape != (48,) or not np.isfinite(values).all():
            raise ValueError("Invalid point forecast")
        if self.state.last_target is not None and target <= self.state.last_target:
            raise ValueError("Forecast clock must advance; restore a checkpoint to rewind")
        if any(release_time(day) <= issue_time(target) for day in self.state.pending):
            raise ValueError("Apply all matured feedback before issuing")
        if self.state.feedback_watermark is not None and release_time(
            self.state.feedback_watermark
        ) > issue_time(target):
            raise ValueError(
                "Feedback state is ahead of the issue clock; restore an earlier checkpoint"
            )
        state_before = self.state_id
        thresholds, degraded = self.residual_thresholds()
        diagnostics = {"tta_steps": 0, "converged": True, "repair_max": 0.0, "raw_crossings": 0}
        if self.state.method == "fixed":
            if self.fixed is None:
                self.fixed = thresholds.tolist()
            thresholds = np.asarray(self.fixed)
        elif self.state.method == "ncc":
            if self.network is None or context is None:
                raise ValueError("NCC weights and eligible context required")
            import torch

            from gridlens.calibration.ncc import adapt_thresholds

            self.network.eval()
            with torch.no_grad():
                raw = self.network(*self.neural_inputs(context)) + torch.tensor(
                    self.state.offset, dtype=torch.float32
                )
            corrected, diagnostics = adapt_thresholds(raw)
            thresholds = corrected.numpy() * self.scale
        if (
            thresholds.shape != (48, 3)
            or not np.isfinite(thresholds).all()
            or (thresholds < 0).any()
            or (np.diff(thresholds, axis=1) < 0).any()
        ):
            raise ValueError("Invalid interval thresholds")
        record = {
            "target_date": target,
            "issued_at": issue_time(target).isoformat(),
            "point": values.tolist(),
            "thresholds": thresholds.tolist(),
            "lower": (values[:, None] - thresholds).tolist(),
            "upper": (values[:, None] + thresholds).tolist(),
            "levels": list(LEVELS),
            "method": self.state.method,
            "model_id": self.state.model_id,
            "calibration_id": self.state.calibration_id,
            "state_before": state_before,
            "feedback_watermark": self.state.feedback_watermark,
            "degraded": degraded,
            "diagnostics": diagnostics,
        }
        self.state.pending[target] = {"point": record["point"], "thresholds": record["thresholds"]}
        self.state.last_target = target
        record["state_after"] = self.state_id
        return record

    @property
    def state_id(self) -> str:
        return str(self.checkpoint()["checkpoint_id"])

    def checkpoint(self) -> dict[str, Any]:
        result = {
            "state": asdict(self.state),
            "fixed": self.fixed,
            "scale": self.scale,
            "window": self.window,
            "minimum": self.minimum,
            "lookback": self.lookback,
            "eta": self.eta,
        }
        return {**result, "checkpoint_id": content_id(result)}

    @classmethod
    def restore(cls, record: dict[str, Any], *, network: Any = None) -> IntervalReplay:
        if record["checkpoint_id"] != content_id(
            {k: v for k, v in record.items() if k != "checkpoint_id"}
        ):
            raise ValueError("Corrupt calibration checkpoint")
        return cls(
            ReplayState(**record["state"]),
            network=network,
            **{k: record[k] for k in ("fixed", "scale", "window", "minimum", "lookback", "eta")},
        )
