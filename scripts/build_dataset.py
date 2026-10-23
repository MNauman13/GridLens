"""Materialise split membership and training-only scaling from the frozen annual audit."""

import argparse
import json
from pathlib import Path

from gridlens.data.dataset import build_datasets, load_series, write_freeze
from gridlens.data.store import Store

ROOT = Path(__file__).resolve().parents[1]


def prepare(root: Path):
    audit = json.loads((ROOT / "reports/neso-audit-v1.json").read_text())
    sources = sorted((r for r in audit["resources"] if r["source_kind"] == "historic_outturn"), key=lambda r: r["resource_id"])
    plan = json.loads((ROOT / "configs/experiment_manifest_v1.json").read_text())
    return build_datasets(load_series(Store(root), sources), plan["splits"], tuple(r["raw_sha256"] for r in sources))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/neso")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/dataset-freeze-v1.json")
    args = parser.parse_args()
    _, report, _ = prepare(args.root)
    write_freeze(args.report, report)
    print(json.dumps({"manifest_id": report["manifest_id"], "splits": {role: {"accepted": len(record["target_dates"]), "dropped": len(record["dropped_origins"])} for role, record in report["splits"].items()}, "locked_test_evaluated": False}))


if __name__ == "__main__":
    main()
