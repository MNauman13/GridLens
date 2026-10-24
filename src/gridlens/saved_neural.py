"""Integrity-checked saved validation forecasts, independent of Torch at serving."""
from typing import Literal, Self

from pydantic import Field, model_validator

from gridlens.contracts import InputWindow, StrictRecord
from gridlens.data.store import content_id


class SavedNeuralForecast(StrictRecord):
    architecture: dict[str, str | bool | int | float]
    inputs: InputWindow
    prediction_mw: tuple[float, ...] = Field(min_length=48, max_length=48)
    observed_mw: tuple[float, ...] = Field(min_length=48, max_length=48)
    model_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_role: Literal["point_validation"]
    evidence_mode: Literal["retrospective_revised"]
    validation_report_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    forecast_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def check_identity(self) -> Self:
        if content_id(self.model_dump(mode="json", exclude={"forecast_id"})) != self.forecast_id:
            raise ValueError("Forecast identity mismatch")
        return self
