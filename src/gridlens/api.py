"""Validated fixtures and immutable saved baseline forecasts."""

import json
import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import ValidationError

from gridlens.contracts import BaselineForecast, StrictRecord
from gridlens.fixture import fixture_summary
from gridlens.saved_neural import SavedNeuralForecast


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.live_service = None
    root = os.environ.get("GRIDLENS_LIVE_ASSETS")
    if root:
        try:
            from gridlens.live import LiveService

            application.state.live_service = LiveService(Path(root), Path(os.environ.get("GRIDLENS_JOB_DB", "reports/local/i1/jobs.sqlite3")))
        except Exception:
            logging.getLogger(__name__).exception("Live inference artifacts are not ready")
    yield
    if application.state.live_service is not None:
        application.state.live_service.close()


app = FastAPI(title="GridLens", version="0.2.0", lifespan=lifespan)


@app.get("/v1/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


class LiveRequest(StrictRecord):
    model: Literal["ridge", "timexer"]
    target_date: date


def reference_service(request: Request) -> Any:
    service = getattr(request.app.state, "live_service", None)
    if service is None or not service.ready():
        raise HTTPException(status_code=503, detail="Reference inference is not configured or its verified artifacts are unavailable")
    return service


@app.get("/v1/health/ready")
def ready(request: Request) -> dict[str, Any]:
    service = reference_service(request)
    return {"status": "ready", "models": list(service.catalog["models"]), "execution_location": "python_server"}


@app.get("/v1/live/catalog")
def live_catalog(request: Request) -> dict[str, Any]:
    service = reference_service(request)
    return {**service.catalog, "execution_location": "python_server", "capacity": service.capacity}


@app.post("/v1/forecast-jobs", status_code=202)
def submit_live_job(payload: LiveRequest, request: Request, response: Response) -> dict[str, Any]:
    from gridlens.live import AdmissionError, owner_hash

    service = reference_service(request)
    token = request.cookies.get("gridlens_session", "")
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        token = secrets.token_hex(32)
    owner = owner_hash(token)
    try:
        identifier = service.submit(owner, payload.model, payload.target_date.isoformat())
    except AdmissionError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    response.set_cookie("gridlens_session", token, httponly=True, samesite="strict", secure=os.environ.get("GRIDLENS_COOKIE_SECURE") == "1", max_age=86400)
    return {"job_id": identifier, "state": "accepted", "execution_location": "python_server"}


@app.get("/v1/jobs/{identifier}")
def live_job(identifier: UUID, request: Request) -> dict[str, Any]:
    from gridlens.live import owner_hash

    service = reference_service(request)
    token = request.cookies.get("gridlens_session", "")
    record = service.job(str(identifier), owner_hash(token)) if token else None
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found for this session")
    return dict(record)


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
