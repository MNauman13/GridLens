"""Marginal interval metrics, with daily and slot slices kept explicit."""

from __future__ import annotations

from typing import Any

import numpy as np

from gridlens.calibration.replay import LEVELS


def interval_metrics(actual: Any, point: Any, thresholds: Any) -> dict[str, Any]:
    y, p, q = (np.asarray(value, dtype=float) for value in (actual, point, thresholds))
    if (
        y.ndim != 2
        or y.shape[1] != 48
        or p.shape != y.shape
        or q.shape != (*y.shape, 3)
        or len(y) < 1
    ):
        raise ValueError("Aligned nonempty day/slot/level arrays required")
    if (
        any(not np.isfinite(value).all() for value in (y, p, q))
        or (q < 0).any()
        or (np.diff(q, axis=-1) < 0).any()
    ):
        raise ValueError("Finite nested nonnegative widths required")
    residual = np.abs(y - p)
    alpha = 1 - np.asarray(LEVELS)
    covered = residual[:, :, None] <= q
    scores = 2 * q + 2 / alpha * np.maximum(residual[:, :, None] - q, 0)
    wis = (0.5 * residual + (scores * alpha / 2).sum(axis=-1)) / 3.5
    levels = {}
    for i, level in enumerate(LEVELS):
        daily = covered[:, :, i].mean(axis=1)
        levels[str(level)] = {
            "coverage": float(covered[:, :, i].mean()),
            "mean_width_mw": float((2 * q[:, :, i]).mean()),
            "mean_interval_score_mw": float(scores[:, :, i].mean()),
            "daily_coverage": daily.tolist(),
            "slot_coverage": covered[:, :, i].mean(axis=0).tolist(),
            "rolling_30day_coverage": [
                float(daily[max(0, j - 29) : j + 1].mean()) for j in range(len(y))
            ],
            "absolute_coverage_error": abs(float(covered[:, :, i].mean()) - level),
        }
    return {
        "origins": len(y),
        "levels": levels,
        "weighted_interval_score_mw": float(wis.mean()),
        "daily_weighted_interval_score_mw": wis.mean(axis=1).tolist(),
        "crossings": int((np.diff(q, axis=-1) < 0).sum()),
        "unit": "MW",
        "scope": "marginal per half-hour; no joint daily coverage guarantee",
    }
