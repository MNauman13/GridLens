"""Validated fixtures and immutable saved baseline forecasts."""

import json
from datetime import date
from importlib.resources import files
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from gridlens.contracts import BaselineForecast
from gridlens.fixture import fixture_summary
from gridlens.saved_neural import SavedNeuralForecast

app = FastAPI(title="GridLens", version="0.1.0")


@app.get("/v1/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/v1/fixture")
def fixture(issue_date: date = date(2024, 2, 10)) -> dict[str, Any]:
    return fixture_summary(issue_date)


@app.get("/v1/forecasts/baseline", response_model=BaselineForecast)
def baseline_forecast(model: Literal["last_day", "weekly", "ridge"] = "ridge") -> BaselineForecast:
    try:
        saved = json.loads(files("gridlens").joinpath("resources/baseline_forecasts_v1.json").read_text())
        return BaselineForecast.model_validate(saved[model])
    except (OSError, ValueError, KeyError, ValidationError) as exc:
        raise HTTPException(status_code=503, detail="Saved forecast is unavailable or failed integrity validation") from exc


@app.get("/v1/forecasts/neural", response_model=SavedNeuralForecast)
def neural_forecast(model: Literal["transformer", "lstm", "timexer"] = "timexer") -> SavedNeuralForecast:
    try:
        saved = json.loads(files("gridlens").joinpath("resources/neural_forecasts_v1.json").read_text())
        return SavedNeuralForecast.model_validate(saved[model])
    except (OSError, ValueError, KeyError, ValidationError) as exc:
        raise HTTPException(status_code=503, detail="Saved neural forecast is unavailable or failed integrity validation") from exc
