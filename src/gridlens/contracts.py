"""Shared immutable boundaries for datasets, point models, and calibration."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EvidenceMode = Literal["retrospective_revised", "captured_availability"]
CONTRACT_VERSION = "gb_nd_day_ahead_v1"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CoverageLevel(float, Enum):
    P80 = 0.8
    P90 = 0.9
    P95 = 0.95


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class InputWindow(StrictRecord):
    contract_version: Literal["gb_nd_day_ahead_v1"] = "gb_nd_day_ahead_v1"
    issued_at: datetime
    context_start: datetime
    context_end_exclusive: datetime
    target_start: datetime
    target_end_exclusive: datetime
    evidence_mode: EvidenceMode
    feature_snapshot_id: str = Field(min_length=1)
    demand_mw: tuple[float, ...] = Field(min_length=336, max_length=336)

    @field_validator("issued_at", "context_start", "context_end_exclusive", "target_start", "target_end_exclusive")
    @classmethod
    def utc_only(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("Explicit UTC timestamps are required")
        return value

    @model_validator(mode="after")
    def check_boundaries(self) -> InputWindow:
        from gridlens.timing import boundaries

        expected = boundaries(self.issued_at.date())
        for field in ["issued_at", "context_start", "context_end_exclusive", "target_start", "target_end_exclusive"]:
            if getattr(self, field) != expected[field]:
                raise ValueError(f"Incompatible forecast boundary: {field}")
        return self


class PointForecast(StrictRecord):
    contract_version: Literal["gb_nd_day_ahead_v1"] = "gb_nd_day_ahead_v1"
    model_version: str = Field(min_length=1)
    feature_snapshot_id: str = Field(min_length=1)
    prediction_mw: tuple[float, ...] = Field(min_length=48, max_length=48)


class PredictionBand(StrictRecord):
    level: CoverageLevel
    lower_mw: tuple[float, ...] = Field(min_length=48, max_length=48)
    upper_mw: tuple[float, ...] = Field(min_length=48, max_length=48)

    @model_validator(mode="after")
    def ordered(self) -> PredictionBand:
        if any(lower > upper for lower, upper in zip(self.lower_mw, self.upper_mw, strict=True)):
            raise ValueError("Lower bounds cannot exceed upper bounds")
        return self


class IntervalForecast(StrictRecord):
    point: PointForecast
    calibrator_version: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    scope: Literal["marginal_per_output_slot"] = "marginal_per_output_slot"
    bands: tuple[PredictionBand, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def nested(self) -> IntervalForecast:
        if tuple(band.level for band in self.bands) != (0.8, 0.9, 0.95):
            raise ValueError("Bands must be ordered 80%, 90%, 95%")
        for inner, outer in zip(self.bands, self.bands[1:], strict=False):
            if any(outer_l > inner_l or outer_u < inner_u for inner_l, inner_u, outer_l, outer_u in zip(inner.lower_mw, inner.upper_mw, outer.lower_mw, outer.upper_mw, strict=True)):
                raise ValueError("Wider-level bands must contain narrower-level bands")
        return self


class DatasetManifest(StrictRecord):
    dataset_id: str = Field(min_length=1)
    contract_version: Literal["gb_nd_day_ahead_v1"] = "gb_nd_day_ahead_v1"
    evidence_mode: EvidenceMode
    truth_policy: str = Field(min_length=1)
    feature_order: tuple[str, ...] = Field(min_length=1)
    source_sha256: tuple[Sha256, ...] = Field(min_length=1)
    split_start: date
    split_end: date

    @model_validator(mode="after")
    def valid_range(self) -> DatasetManifest:
        if self.split_start > self.split_end:
            raise ValueError("Dataset split is inverted")
        if len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("Feature names must be unique")
        return self


class MaturedFeedback(StrictRecord):
    forecast_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    slot: int = Field(ge=0, lt=48)
    target_end: datetime
    first_seen_at: datetime
    actual_mw: float
    truth_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def release_after_target(self) -> MaturedFeedback:
        for stamp in (self.target_end, self.first_seen_at):
            offset = stamp.utcoffset()
            if offset is None or offset.total_seconds() != 0:
                raise ValueError("Feedback timestamps must be UTC")
        if self.first_seen_at < self.target_end:
            raise ValueError("Actual cannot mature before its target interval ends")
        return self


class TrainingExample(StrictRecord):
    inputs: InputWindow
    target_mw: tuple[float, ...] = Field(min_length=48, max_length=48)
    truth_version: str = Field(min_length=1)


class BaselineForecast(StrictRecord):
    schema_version: Literal["gridlens_baseline_forecast_v1"] = "gridlens_baseline_forecast_v1"
    forecast_id: Sha256
    inputs: InputWindow
    point: PointForecast
    model_kind: Literal["last_day", "weekly", "ridge"]
    observed_mw: tuple[float, ...] = Field(min_length=48, max_length=48)
    dataset_id: Sha256
    source_sha256: tuple[Sha256, ...] = Field(min_length=1)
    truth_version: str = Field(min_length=1)
    evaluation_role: Literal["point_validation"] = "point_validation"
    validation_report_id: Sha256

    @model_validator(mode="after")
    def aligned_and_identified(self) -> BaselineForecast:
        from gridlens.data.store import content_id

        if self.point.feature_snapshot_id != self.inputs.feature_snapshot_id:
            raise ValueError("Prediction and inputs reference different feature snapshots")
        if self.forecast_id != content_id(self.model_dump(mode="json", exclude={"forecast_id"})):
            raise ValueError("Forecast artifact content ID is invalid")
        return self


class TrainingDataset(Protocol):
    @property
    def manifest(self) -> DatasetManifest: ...
    def iter_examples(self) -> Iterator[TrainingExample]: ...


class ForecastModel(Protocol):
    def fit(self, dataset: TrainingDataset) -> None: ...
    def predict(self, inputs: InputWindow) -> PointForecast: ...
    def save_bundle(self, destination: Path) -> None: ...

    @classmethod
    def load_bundle(cls, source: Path) -> Self: ...


class Calibrator(Protocol):
    def predict(self, point: PointForecast, *, issued_at: datetime) -> IntervalForecast: ...
    def observe(self, feedback: MaturedFeedback, *, as_of: datetime) -> None: ...
    def save_state(self, destination: Path) -> None: ...
