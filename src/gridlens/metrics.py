"""MW point errors and daily peak errors with explicit, hand-testable definitions."""

import math
from collections.abc import Sequence
from typing import Any


def point_metrics(actual: Sequence[Sequence[float]], predicted: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not actual or len(actual) != len(predicted) or any(len(a) != 48 or len(p) != 48 for a, p in zip(actual, predicted, strict=True)):
        raise ValueError("Require aligned nonempty 48-slot daily curves")
    if any(not math.isfinite(v) for curve in (*actual, *predicted) for v in curve):
        raise ValueError("Metrics require finite values")
    errors = [[p - a for a, p in zip(observed, forecast, strict=True)] for observed, forecast in zip(actual, predicted, strict=True)]
    n = len(actual) * 48
    return {
        "origins": len(actual), "points": n,
        "mae_mw": math.fsum(abs(e) for row in errors for e in row) / n,
        "rmse_mw": math.sqrt(math.fsum(e * e for row in errors for e in row) / n),
        "mean_daily_peak_absolute_error_mw": math.fsum(abs(max(a) - max(p)) for a, p in zip(actual, predicted, strict=True)) / len(actual),
        "mean_peak_timing_absolute_error_slots": math.fsum(abs(max(range(48), key=lambda i: a[i]) - max(range(48), key=lambda i: p[i])) for a, p in zip(actual, predicted, strict=True)) / len(actual),
        "slot_mae_mw": [math.fsum(abs(row[slot]) for row in errors) / len(actual) for slot in range(48)],
        "slot_rmse_mw": [math.sqrt(math.fsum(row[slot] ** 2 for row in errors) / len(actual)) for slot in range(48)],
        "peak_tie_policy": "earliest_slot", "unit": "MW",
    }
