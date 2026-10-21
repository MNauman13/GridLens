"""Compare new annual captures with the frozen audit, and verify stored bytes/index counts."""

import argparse
import hashlib
import json
from pathlib import Path

from gridlens.data.store import Store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/neso"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("reports/neso-audit-v1.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    frozen = json.loads(args.frozen.read_text())
    if not report["full_2017_2024_nd_coverage_gate"]:
        raise SystemExit("Annual ND coverage gate failed")
    expected = {r["resource_id"]: r["raw_sha256"] for r in frozen["resources"] if r["source_kind"] == "historic_outturn"}
    store = Store(args.root)
    total = 0
    for resource in report["resources"]:
        if resource["status"] != "valid":
            raise SystemExit("Quarantined resource in live audit")
        sha = resource["raw_sha256"]
        if hashlib.sha256((args.root / "raw" / f"{sha}.csv").read_bytes()).hexdigest() != sha:
            raise SystemExit("Stored raw bytes do not match their hash")
        if resource["source_kind"] == "historic_outturn" and expected.get(resource["resource_id"]) != sha:
            raise SystemExit("Annual provider bytes changed; review a new frozen audit before training")
        with store.connect() as connection:
            count = connection.execute("SELECT count(*) FROM observations WHERE snapshot_id=?", (resource["snapshot_id"],)).fetchone()[0]
        if count != resource["counts"]["actual_rows_unique"]:
            raise SystemExit("Observation index does not match the source audit")
        total += count
    print(json.dumps({"raw_hashes_verified": len(report["resources"]), "indexed_actual_observations": total, "annual_raw_bytes_match_frozen_audit": True, "full_2017_2024_nd_coverage_gate": True}))


if __name__ == "__main__":
    main()
