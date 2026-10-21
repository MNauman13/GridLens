"""Audit original HTTP captures produced outside Python; never relabel them as urllib runs.

Capture directory: <name>.csv plus <name>.json HTTP receipts for pinned source names.
The normal entry point downloads directly: gridlens ingest-neso.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from gridlens.data.neso import Source
from gridlens.data.pipeline import freeze_audit, source_by_name
from gridlens.data.store import Store
from gridlens.data.transport import Response, final_download_url, receipt_url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/neso"))
    parser.add_argument("--report", type=Path, default=Path("reports/neso-audit-v1.json"))
    args = parser.parse_args()
    store = Store(args.root)
    audits = []
    receipts = []
    names = [str(year) for year in range(2017, 2025)] + ["daily"]
    for name in names:
        spec = source_by_name(name)
        receipt = json.loads((args.capture_directory / f"{name}.json").read_text())
        raw = (args.capture_directory / f"{name}.csv").read_bytes()
        if receipt["resource_id"] != spec["resource_id"] or receipt["requested_url"] != spec["csv_url"] or receipt["status"] != 200 or receipt["bytes"] != len(raw):
            raise ValueError(f"Capture identity mismatch for {name}")
        final_download_url(receipt["requested_url"], receipt["final_url"])
        response = Response(raw, receipt["requested_url"], receipt["final_url"], receipt["status"], receipt["headers"], datetime.fromisoformat(receipt["retrieved_at"].replace("Z", "+00:00")))
        audit = store.ingest(response, Source(spec["resource_id"], spec["kind"], spec["year"]))
        audits.append(audit)
        receipts.append({"name": name, "resource_id": spec["resource_id"], "raw_sha256": audit["raw_sha256"], "retrieved_at": receipt["retrieved_at"], "byte_size": len(raw), "requested_url": receipt["requested_url"], "final_url_without_signed_query": receipt_url(receipt["final_url"]), "headers": receipt["headers"]})
        print(json.dumps({"source": name, "status": audit["status"], "counts": audit.get("counts"), "reason": audit.get("reason")}))
    result = freeze_audit(store, audits, requested=names)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    args.report.with_name("neso-capture-receipts-v1.json").write_text(json.dumps({"transport": "Node fetch; original response bytes, not a reconstructed API preview", "captures": receipts}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_id": result["audit_id"], "complete": result["full_2017_2024_nd_coverage_gate"]}))
    if any(audit["status"] != "valid" for audit in audits):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
