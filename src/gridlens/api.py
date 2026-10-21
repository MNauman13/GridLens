"""Phase 1 fixture API. Serving trained forecasts is a later milestone."""

from datetime import date
from typing import Any

from fastapi import FastAPI

from gridlens.fixture import fixture_summary

app = FastAPI(title="GridLens", version="0.1.0")


@app.get("/v1/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/v1/fixture")
def fixture(issue_date: date = date(2024, 2, 10)) -> dict[str, Any]:
    return fixture_summary(issue_date)
