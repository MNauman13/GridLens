"""Explicit CSV schema, actual-only filtering, duplicate handling, and quality audit."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal

from gridlens.data.settlement import (
    HALF_HOUR,
    day_bounds,
    parse_date,
    periods_in_day,
    settlement_interval,
)

TRANSFORM_VERSION = "neso_interval_v2"
REQUIRED = ("SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND")
NUMERIC_FIELDS = ("ND", "TSD", "EMBEDDED_WIND_GENERATION", "EMBEDDED_SOLAR_GENERATION")
INDICATOR = "FORECAST_ACTUAL_INDICATOR"


@dataclass(frozen=True)
class Source:
    resource_id: str
    kind: Literal["historic_outturn", "mixed_actual_forecast"]
    year: int | None = None


@dataclass(frozen=True)
class Observation:
    settlement_date: date
    settlement_period: int
    interval_start: datetime
    interval_end: datetime
    # Tuple order is fixed by NUMERIC_FIELDS. None is a gap, never an invented zero.
    values: tuple[float | None, ...]


@dataclass(frozen=True)
class ParsedSnapshot:
    rows: tuple[Observation, ...]
    audit: dict[str, Any]


class QuarantineError(ValueError):
    def __init__(self, reason: str, details: list[str] | None = None) -> None:
        super().__init__(reason)
        self.details = (details or [])[:100]


def number(value: str) -> float | None:
    if value.strip() == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite numeric value")
    return parsed


def parse_csv(content: bytes, source: Source) -> ParsedSnapshot:
    """Fail closed on ambiguous labels and incompatible types; preserve genuine gaps."""
    try:
        return _parse_csv(content, source)
    except (UnicodeError, csv.Error) as exc:
        raise QuarantineError("Invalid UTF-8 CSV", [str(exc)]) from exc


def _parse_csv(content: bytes, source: Source) -> ParsedSnapshot:
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
    fields = reader.fieldnames or []
    missing = sorted(set(REQUIRED) - set(fields))
    if source.kind == "mixed_actual_forecast" and INDICATOR not in fields:
        missing.append(INDICATOR)
    if missing or len(fields) != len(set(fields)):
        raise QuarantineError("Missing or duplicate CSV columns", missing)
    rows: dict[datetime, Observation] = {}
    original_rows: dict[datetime, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    errors: list[str] = []
    for line, raw in enumerate(reader, 2):
        counts["input_rows"] += 1
        try:
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError("CSV row width differs from header")
            kind = raw.get(INDICATOR, "A").strip()
            if kind not in ("A", "F"):
                raise ValueError(f"Unknown actual/forecast indicator: {kind!r}")
            day = parse_date(raw["SETTLEMENT_DATE"].strip(), resource_year=source.year)
            if source.year is not None and day.year != source.year:
                raise ValueError("Settlement date outside resource year")
            period_text = raw["SETTLEMENT_PERIOD"].strip()
            if not period_text.isascii() or not period_text.isdigit():
                raise ValueError("Settlement period must be an integer ordinal")
            period = int(period_text)
            start, end = settlement_interval(day, period)
            if kind == "F":
                counts["forecast_rows_excluded"] += 1
                continue
            values = tuple(number(raw.get(field, "")) for field in NUMERIC_FIELDS)
            if values[0] is not None and values[0] < 0:
                raise ValueError("Negative National Demand")
            observation = Observation(day, period, start, end, values)
            if start in rows:
                if rows[start] != observation or original_rows[start] != raw:
                    raise ValueError(f"Conflicting duplicate UTC interval: {start.isoformat()}")
                counts["identical_duplicates_collapsed"] += 1
                continue
            rows[start] = observation
            original_rows[start] = raw
            counts["actual_rows_unique"] += 1
            if values[0] == 0:
                counts["zero_nd_flagged"] += 1
            if any(value is not None and value < 0 for value in values[2:]):
                counts["negative_embedded_generation_flagged"] += 1
        except (ValueError, OverflowError) as exc:
            counts["invalid_rows"] += 1
            if len(errors) < 100:
                errors.append(f"line {line}: {exc}")
    if errors:
        raise QuarantineError(f"{counts['invalid_rows']} invalid CSV rows", errors)
    if not rows:
        raise QuarantineError("Snapshot contains no actual observations")
    ordered = tuple(rows[key] for key in sorted(rows))
    years = [source.year] if source.year is not None else sorted({r.settlement_date.year for r in ordered})
    year_reports = []
    for year in years:
        selected = [row for row in ordered if row.settlement_date.year == year]
        begin, finish = day_bounds(date(year, 1, 1))[0], day_bounds(date(year + 1, 1, 1))[0]
        expected = (finish - begin) // HALF_HOUR
        missing_fields = {field: sum(row.values[index] is None for row in selected) for index, field in enumerate(NUMERIC_FIELDS)}
        missing_slots = expected - len(selected)
        days = {row.settlement_date for row in selected}
        transition_days = []
        day = date(year, 1, 1)
        while day.year == year:
            expected_periods = periods_in_day(day)
            if expected_periods != 48:
                present = sum(row.settlement_date == day for row in selected)
                transition_days.append({"date": str(day), "expected_periods": expected_periods, "present_periods": present})
            day += timedelta(days=1)
        year_reports.append({
            "settlement_year": year, "expected_intervals": expected,
            "utc_start": begin.isoformat(), "utc_end_exclusive": finish.isoformat(),
            "present_intervals": len(selected), "missing_intervals": missing_slots,
            "days_present": len(days), "missing_values_by_field": missing_fields,
            "usable_nd_intervals": len(selected) - missing_fields["ND"],
            "missing_nd_slots": missing_slots + missing_fields["ND"],
            "transition_days": transition_days,
            "field_ranges": {field: _range([r.values[index] for r in selected]) for index, field in enumerate(NUMERIC_FIELDS)},
        })
    audit = {
        "transform_version": TRANSFORM_VERSION, "raw_sha256": hashlib.sha256(content).hexdigest(),
        "resource_id": source.resource_id, "source_kind": source.kind,
        "columns": fields, "additional_columns": sorted(set(fields) - set(REQUIRED) - {INDICATOR} - set(NUMERIC_FIELDS)),
        "absent_optional_fields": sorted(set(NUMERIC_FIELDS) - set(fields)),
        "counts": {key: counts[key] for key in ("input_rows", "forecast_rows_excluded", "identical_duplicates_collapsed", "actual_rows_unique", "invalid_rows", "zero_nd_flagged", "negative_embedded_generation_flagged")},
        "years": year_reports, "conflicting_duplicates": 0,
        "evidence_mode": "retrospective_revised" if source.kind == "historic_outturn" else "captured_availability",
        "historical_revision_availability_known": False,
        "actual_filter": "indicator_A_only" if INDICATOR in fields else "declared_historic_outturn_resource",
    }
    return ParsedSnapshot(ordered, audit)


def _range(values: list[float | None]) -> dict[str, float | None]:
    available = [value for value in values if value is not None]
    return {"min": min(available, default=None), "max": max(available, default=None)}
