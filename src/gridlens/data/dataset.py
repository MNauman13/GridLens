"""Causal feature snapshots, explicit truth ownership, and frozen temporal datasets."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

from gridlens.contracts import DatasetManifest, EvidenceMode, InputWindow, TrainingExample
from gridlens.data.store import Store, content_id, utc_text, write_once
from gridlens.timing import boundaries

STEP = timedelta(minutes=30)
VALUE_FIELDS = ("ND", "EMBEDDED_WIND_GENERATION", "EMBEDDED_SOLAR_GENERATION")
CALENDAR_FIELDS = ("slot_sin", "slot_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos")
FEATURE_ORDER = VALUE_FIELDS + tuple(f"{f}_missing" for f in VALUE_FIELDS) + tuple(f"{f}_age_slots" for f in VALUE_FIELDS) + CALENDAR_FIELDS


@dataclass(frozen=True)
class Version:
    start: datetime
    values: tuple[float | None, float | None, float | None]
    first_seen_at: datetime
    snapshot_id: str


class Series:
    def __init__(self, versions: list[Version]) -> None:
        self.index: dict[datetime, list[Version]] = {}
        for row in versions:
            utc_text(row.start)
            utc_text(row.first_seen_at)
            self.index.setdefault(row.start, []).append(row)
        for rows in self.index.values():
            rows.sort(key=lambda row: (row.first_seen_at, row.snapshot_id))

    def at(self, stamp: datetime, mode: EvidenceMode, issued_at: datetime) -> Version | None:
        available = self.index.get(stamp, [])
        if mode == "captured_availability":
            available = [row for row in available if row.first_seen_at <= issued_at]
        elif mode != "retrospective_revised":
            raise ValueError("Unknown evidence mode")
        return available[-1] if available else None


def load_series(store: Store, audits: list[dict[str, Any]]) -> Series:
    """Only explicitly frozen, validated snapshot IDs can enter this dataset."""
    if not audits or any(a["status"] != "valid" for a in audits):
        raise ValueError("Require explicitly frozen valid snapshots")
    rows = []
    with store.connect() as connection:
        for audit in audits:
            saved = connection.execute("SELECT raw_sha256,status FROM snapshots WHERE snapshot_id=?", (audit["snapshot_id"],)).fetchone()
            if saved != (audit["raw_sha256"], "valid"):
                raise ValueError("Frozen snapshot is missing or incompatible")
            for row in connection.execute("""SELECT o.interval_start,o.nd_mw,o.wind_mw,o.solar_mw,s.first_seen_at,o.snapshot_id
                FROM observations o JOIN snapshots s USING(snapshot_id) WHERE o.snapshot_id=?""", (audit["snapshot_id"],)):
                rows.append(Version(datetime.fromisoformat(row[0]), (row[1], row[2], row[3]), datetime.fromisoformat(row[4]), row[5]))
    return Series(rows)


@lru_cache(maxsize=40000)
def cyclic(numerator: int, denominator: int) -> tuple[float, float]:
    """Fixed decimal arithmetic avoids OS libm differences in hashed features."""
    with localcontext() as context:
        context.prec = 40
        pi = Decimal("3.1415926535897932384626433832795028841971693993751")
        angle = 2 * pi * Decimal(numerator % denominator) / Decimal(denominator)
        if angle > pi:
            angle -= 2 * pi
        sine_term = sine = angle
        cosine_term = cosine = Decimal(1)
        for index in range(1, 40):
            sine_term *= -angle * angle / Decimal((2 * index) * (2 * index + 1))
            cosine_term *= -angle * angle / Decimal((2 * index - 1) * (2 * index))
            sine += sine_term
            cosine += cosine_term
            if abs(sine_term) < Decimal("1e-38") and abs(cosine_term) < Decimal("1e-38"):
                break
        first, second = (value.quantize(Decimal("1e-12")) for value in (sine, cosine))
        return (0.0 if first == 0 else float(first), 0.0 if second == 0 else float(second))


def calendar(stamp: datetime) -> tuple[float, ...]:
    """UTC cycles with integer phases and canonical fixed-precision arithmetic."""
    slot = stamp.hour * 2 + stamp.minute // 30
    year_start = datetime(stamp.year, 1, 1, tzinfo=UTC)
    next_year = datetime(stamp.year + 1, 1, 1, tzinfo=UTC)
    return cyclic(slot, 48) + cyclic(stamp.weekday(), 7) + cyclic((stamp-year_start)//STEP, (next_year-year_start)//STEP)


class DroppedOrigin(ValueError):
    pass


@dataclass(frozen=True)
class FeatureSnapshot:
    inputs: InputWindow
    history: tuple[tuple[float, ...], ...]
    target_calendar: tuple[tuple[float, ...], ...]


def feature_snapshot(series: Series, issue_date: date, mode: EvidenceMode) -> FeatureSnapshot:
    limits = boundaries(issue_date)
    previous: list[float | None] = [None] * 3
    age = [3] * 3
    history = []
    # Two earlier observations may seed a strictly causal fill at the context boundary.
    for index in range(-2, 336):
        stamp = limits["context_start"] + index * STEP
        if stamp + STEP > limits["context_end_exclusive"]:
            raise AssertionError("Context query crossed its cutoff")
        row = series.at(stamp, mode, limits["issued_at"])
        raw = row.values if row is not None else (None, None, None)
        values = []
        flags = []
        for field, value in enumerate(raw):
            age[field] = 0 if value is not None else age[field] + 1
            if value is not None:
                if not math.isfinite(value):
                    raise DroppedOrigin("non_finite_context")
                previous[field] = value
            if index >= 0 and (previous[field] is None or age[field] > 2):
                raise DroppedOrigin(f"context_gap:{VALUE_FIELDS[field]}")
            last = previous[field]
            values.append(last if last is not None else 0.0)
            flags.append(float(value is None))
        if index >= 0:
            history.append(tuple(values + flags + [float(a) for a in age]) + calendar(stamp))
    target_calendar = tuple(calendar(limits["target_start"] + slot * STEP) for slot in range(48))
    identity = content_id({"contract": "gb_nd_day_ahead_v1", "issue": str(issue_date), "mode": mode, "order": FEATURE_ORDER, "history": history, "target_calendar": target_calendar, "fill_limit": 2})
    inputs = InputWindow.model_validate({**limits, "evidence_mode": mode, "feature_snapshot_id": identity, "demand_mw": tuple(row[0] for row in history)})
    return FeatureSnapshot(inputs, tuple(history), target_calendar)


def training_example(features: FeatureSnapshot, truth: Series, truth_version: str) -> TrainingExample:
    labels = []
    for slot in range(48):
        stamp = features.inputs.target_start + slot * STEP
        row = truth.at(stamp, "retrospective_revised", features.inputs.issued_at)
        if row is None or row.values[0] is None:
            raise DroppedOrigin("missing_target_nd")
        labels.append(row.values[0])
    return TrainingExample(inputs=features.inputs, target_mw=tuple(labels), truth_version=truth_version)


@dataclass(frozen=True)
class FrozenDataset:
    manifest: DatasetManifest
    examples: tuple[TrainingExample, ...]
    features: tuple[FeatureSnapshot, ...]
    role: str

    def iter_examples(self) -> Iterator[TrainingExample]:
        return iter(self.examples)


@dataclass(frozen=True)
class Scaling:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    training_origin_ids: tuple[str, ...]

    @classmethod
    def fit(cls, training: FrozenDataset) -> Scaling:
        if training.role != "point_train":
            raise ValueError("Scaling can only fit the point_train split")
        if not training.examples:
            raise ValueError("Cannot fit scaling without training examples")
        # Each training-context slot gets equal weight, including overlapping windows.
        counts = len(training.features) * 336
        means = tuple(math.fsum(row[column] for f in training.features for row in f.history) / counts for column in range(3))
        scales = tuple(max(math.sqrt(math.fsum((row[column] - means[column]) ** 2 for f in training.features for row in f.history) / counts), 1e-8) for column in range(3))
        return cls(means, scales, tuple(example.inputs.issued_at.date().isoformat() for example in training.examples))

    def transform(self, history: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple((row[i] - self.means[i]) / self.scales[i] for i in range(3)) + row[3:] for row in history)


def build_datasets(series: Series, splits: list[dict[str, str]], source_hashes: tuple[str, ...], *, mode: EvidenceMode = "retrospective_revised", truth: Series | None = None) -> tuple[dict[str, FrozenDataset], dict[str, Any], Scaling]:
    ownership: set[date] = set()
    datasets = {}
    membership: dict[str, Any] = {}
    truth_id = content_id({"frozen_source_sha256": source_hashes, "policy": "frozen_actuals_no_label_fill"})
    for split in splits:
        role, start, end = split["role"], date.fromisoformat(split["start"]), date.fromisoformat(split["end"])
        if start > end or role in datasets:
            raise ValueError("Invalid split range or duplicate role")
        features, examples, dropped = [], [], []
        target_day = start
        while target_day <= end:
            if target_day in ownership:
                raise ValueError("Target date is owned by more than one split")
            ownership.add(target_day)
            try:
                feature = feature_snapshot(series, target_day - timedelta(days=1), mode)
                example = training_example(feature, truth or series, truth_id)
                features.append(feature)
                examples.append(example)
            except DroppedOrigin as exc:
                dropped.append({"target_date": str(target_day), "reason": str(exc)})
            target_day += timedelta(days=1)
        record = {"role": role, "start": str(start), "end": str(end), "target_dates": [str(e.inputs.target_start.date()) for e in examples], "feature_ids": [f.inputs.feature_snapshot_id for f in features], "label_hashes": [content_id(e.target_mw) for e in examples], "dropped_origins": dropped, "drop_counts": dict(Counter(d["reason"] for d in dropped))}
        dataset_id = content_id({"record": record, "source_sha256": source_hashes, "mode": mode, "features": FEATURE_ORDER, "truth": truth_id})
        manifest = DatasetManifest(dataset_id=dataset_id, evidence_mode=mode, truth_policy=truth_id, feature_order=FEATURE_ORDER, source_sha256=source_hashes, split_start=start, split_end=end)
        datasets[role] = FrozenDataset(manifest, tuple(examples), tuple(features), role)
        membership[role] = {key: value for key, value in record.items() if key not in ("feature_ids", "label_hashes")}
        membership[role].update({"dataset_id": dataset_id, "feature_ids_sha256": content_id(record["feature_ids"]), "label_hashes_sha256": content_id(record["label_hashes"])})
    if "point_train" not in datasets:
        raise ValueError("A point_train split is required to fit scaling")
    scaling = Scaling.fit(datasets["point_train"])
    scaling_record = {"means": scaling.means, "scales": scaling.scales, "training_origin_ids": scaling.training_origin_ids, "weighting": "equal_weight_per_training_context_slot"}
    report = {"schema_version": "gridlens_dataset_freeze_v1", "source_sha256": source_hashes, "feature_order": FEATURE_ORDER, "evidence_mode": mode, "truth_version": truth_id, "fill_limit_slots": 2, "calendar_encoding": "fixed_decimal_cyclic_v1", "scaling": scaling_record, "scaler_id": content_id(scaling_record), "splits": membership, "locked_test_evaluated": False}
    report["manifest_id"] = content_id(report)
    return datasets, report, scaling


def write_freeze(path: Path, report: dict[str, Any]) -> None:
    write_once(path, (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
