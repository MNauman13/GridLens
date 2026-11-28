"""Issue a real baseline forecast only from an eligible captured vintage."""

import argparse
import hashlib
import json
import zipfile
from datetime import UTC, date, datetime, time
from pathlib import Path

from gridlens.data.dataset import Series, Version, feature_snapshot
from gridlens.data.neso import Source, parse_csv
from gridlens.data.store import content_id
from gridlens.models.baselines import Baseline
from scripts.reproduce_baseline import ROOT


def forecast(raw, receipt, issue_date, model):
    if hashlib.sha256(raw).hexdigest() != receipt["raw_sha256"] or receipt[
        "capture_id"
    ] != content_id({k: v for k, v in receipt.items() if k != "capture_id"}):
        raise ValueError("Invalid capture bytes or receipt identity")
    first_seen = datetime.fromisoformat(receipt["first_seen_at"])
    if first_seen.utcoffset() is None or first_seen.utcoffset().total_seconds() != 0:
        raise ValueError("UTC capture required")
    issued = datetime.combine(issue_date, time(9, 30), tzinfo=UTC)
    if first_seen > issued:
        raise ValueError("No eligible vintage before this issue clock")
    if receipt.get("status", "valid") != "valid":
        raise ValueError("Quarantined vintage cannot enter a forecast")
    parsed = parse_csv(raw, Source("177f6fa4-ae49-4182-81ea-0c6b35f26ca6", "mixed_actual_forecast"))
    series = Series(
        [
            Version(
                row.interval_start,
                (row.values[0], row.values[2], row.values[3]),
                first_seen,
                receipt["capture_id"],
            )
            for row in parsed.rows
        ]
    )
    snapshot = feature_snapshot(series, issue_date, "captured_availability")
    point = model.predict(snapshot.inputs)
    record = {
        "evidence_mode": "captured_availability",
        "source_capture_id": receipt["capture_id"],
        "capture_first_seen_at": first_seen.isoformat(),
        "executed_at": datetime.now(UTC).isoformat(),
        "logical_issue_at": issued.isoformat(),
        "inputs": snapshot.inputs.model_dump(mode="json"),
        "point": point.model_dump(mode="json"),
        "model_policy": "Operational ridge baseline. The offline TimeXer calibrator is not applied to a different model.",
        "intervals": None,
        "served_from": "fresh_inference",
        "result_cache_hit": False,
    }
    return {**record, "forecast_id": content_id(record)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture-archive", type=Path, required=True)
    p.add_argument("--issue-date", type=date.fromisoformat, default=datetime.now(UTC).date())
    args = p.parse_args()
    output = ROOT / "reports/local/prospective-forecast.json"
    try:
        with zipfile.ZipFile(args.capture_archive) as archive:
            members = archive.infolist()
            if sum(row.file_size for row in members) > 25_000_000:
                raise ValueError("Oversized capture artifact")
            receipts = [
                row
                for row in members
                if row.filename.endswith("/capture.json") or row.filename == "capture.json"
            ]
            raws = [
                row
                for row in members
                if row.filename.endswith("/raw.csv") or row.filename == "raw.csv"
            ]
            if len(receipts) != 1 or len(raws) != 1:
                raise ValueError("Expected exactly one raw capture and receipt")
            raw = archive.read(raws[0])
            receipt = json.loads(archive.read(receipts[0]))
        baseline = ROOT / "reports/local/i0-public"
        report = json.loads((baseline / "baseline_reproduction.json").read_text())
        row = next(r for r in report["results"] if r["kind"] == "ridge")
        model = Baseline.load_bundle(baseline / row["bundle"])
        result = forecast(raw, receipt, args.issue_date, model)
    except ValueError as exc:
        result = {
            "state": "skipped",
            "reason": str(exc),
            "issued": False,
            "executed_at": datetime.now(UTC).isoformat(),
            "note": "No saved or invented forecast substituted for an unavailable captured window",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("inputs", "point")}))


if __name__ == "__main__":
    main()
