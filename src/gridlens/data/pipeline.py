"""Pinned source retrieval and deterministic, explicitly scoped data-audit bundles."""

from __future__ import annotations

import json
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any

from gridlens.data.neso import Source
from gridlens.data.store import Store, canonical_json, content_id, write_once
from gridlens.data.transport import download


def catalog() -> dict[str, Any]:
    return json.loads(files("gridlens").joinpath("resources/neso_resources_v1.json").read_text())  # type: ignore[no-any-return]


def source_by_name(name: str) -> dict[str, Any]:
    matches = [source for source in catalog()["sources"] if source["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown source {name!r}; use 2017..2024 or daily")
    return matches[0]  # type: ignore[no-any-return]


def fetch_source(store: Store, name: str) -> dict[str, Any]:
    spec = source_by_name(name)
    response = download(spec["csv_url"])
    return store.ingest(response, Source(spec["resource_id"], spec["kind"], spec["year"]))


def freeze_audit(store: Store, audits: list[dict[str, Any]], *, requested: list[str]) -> dict[str, Any]:
    """Missing/quarantined resources block the full gate; partial daily data is separate."""
    if len(requested) != len(set(requested)) or len(audits) != len(requested):
        raise ValueError("Require exactly one result for every distinct requested source")
    expected_ids = {source_by_name(name)["resource_id"] for name in requested}
    if {audit["resource_id"] for audit in audits} != expected_ids:
        raise ValueError("Audit sources do not match requested resources")
    annual = [a for a in audits if a.get("source_kind") == "historic_outturn"]
    complete = len(annual) == 8 and all(a["status"] == "valid" and all(y["missing_nd_slots"] == 0 and all(t["expected_periods"] == t["present_periods"] for t in y["transition_days"]) for y in a["years"]) for a in annual)
    report: dict[str, Any] = {
        "schema_version": "neso_audit_v1", "timezone": "Europe/London", "tzdata_version": version("tzdata"),
        "attribution": catalog()["attribution"], "license_url": catalog()["license_url"],
        "requested_sources": sorted(requested), "full_2017_2024_nd_coverage_gate": complete,
        "resources": sorted(audits, key=lambda audit: audit["resource_id"]),
        "limitations": [
            "Annual files are revised historical outturn; past revision publication times are unknown.",
            "Daily update coverage is partial and is not part of the 2017-2024 annual completeness gate.",
            "Only actual records enter the observation index; missing ND values remain gaps.",
            "This audit does not freeze feature windows, split membership, scaling, or model outcomes.",
        ],
    }
    report["audit_id"] = content_id(report)
    write_once(store.root / "reports" / f"{report['audit_id']}.json", (canonical_json(report) + "\n").encode())
    return report


def run_ingestion(root: Path, names: list[str]) -> dict[str, Any]:
    store = Store(root)
    audits = []
    for name in names:
        audits.append(fetch_source(store, name))
    return freeze_audit(store, audits, requested=names)
