"""Deterministic synthetic contract fixture; never a measured forecasting result."""

from datetime import date
from typing import Any

from gridlens.contracts import InputWindow
from gridlens.timing import boundaries


def make_fixture(issue_date: date = date(2024, 2, 10)) -> InputWindow:
    times = boundaries(issue_date)
    return InputWindow(
        issued_at=times["issued_at"],
        context_start=times["context_start"],
        context_end_exclusive=times["context_end_exclusive"],
        target_start=times["target_start"],
        target_end_exclusive=times["target_end_exclusive"],
        evidence_mode="retrospective_revised",
        feature_snapshot_id="synthetic-fixture-v1",
        demand_mw=tuple(20000.0 + (index % 48) * 100.0 for index in range(336)),
    )


def fixture_summary(issue_date: date = date(2024, 2, 10)) -> dict[str, Any]:
    inputs = make_fixture(issue_date)
    return {
        "fixture_kind": "synthetic_contract_smoke_test",
        "real_demand_data": False,
        "trained_model": False,
        "contract_version": inputs.contract_version,
        "issued_at": inputs.issued_at.isoformat(),
        "context_start": inputs.context_start.isoformat(),
        "context_end_exclusive": inputs.context_end_exclusive.isoformat(),
        "target_start": inputs.target_start.isoformat(),
        "target_end_exclusive": inputs.target_end_exclusive.isoformat(),
        "input_count": len(inputs.demand_mw),
        "gap_intervals": 48,
        "output_count": 48,
    }
