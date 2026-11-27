"""Capture a real vintage without pretending missing recent actuals exist."""

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from gridlens.data.neso import Source, parse_csv
from gridlens.data.store import content_id

URL = "https://api.neso.energy/dataset/7a12172a-939c-404c-b581-a6128b74f588/resource/177f6fa4-ae49-4182-81ea-0c6b35f26ca6/download/demanddataupdate.csv"


def main():
    start = datetime.now(UTC)
    request = urllib.request.Request(URL, headers={"User-Agent": "GridLens-capture/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(20_000_001)
        if len(raw) > 20_000_000:
            raise ValueError("Capture size limit exceeded")
        headers = {
            key: response.headers.get(key) for key in ("ETag", "Last-Modified", "Content-Type")
        }
    received = datetime.now(UTC)
    digest = hashlib.sha256(raw).hexdigest()
    parsed = parse_csv(raw, Source("177f6fa4-ae49-4182-81ea-0c6b35f26ca6", "mixed_actual_forecast"))
    valid = [row for row in parsed.rows if row.values[0] is not None]
    newest = max((row.interval_end for row in valid), default=None)
    report = {
        "schema_version": "gridlens_prospective_capture_v1",
        "requested_at": start.isoformat(),
        "first_seen_at": received.isoformat(),
        "source_url": URL,
        "raw_sha256": digest,
        "response_headers": headers,
        "accepted_actual_rows": len(valid),
        "latest_actual_end": newest.isoformat() if newest else None,
        "actual_age_hours": (received - newest).total_seconds() / 3600 if newest else None,
        "audit": parsed.audit,
        "evidence_mode": "captured_vintage",
        "forecast_issued": False,
        "note": "Capture only. Actual-data publication delays can make the seven-day day-ahead input contract unavailable. No forecast or missing actuals are fabricated.",
    }
    report["capture_id"] = content_id(report)
    root = Path("reports/local/prospective") / report["capture_id"]
    root.mkdir(parents=True, exist_ok=False)
    (root / "raw.csv").write_bytes(raw)
    (root / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "audit"}))


if __name__ == "__main__":
    main()
