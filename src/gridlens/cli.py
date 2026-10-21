"""Small installed entry point used to verify a clean checkout."""

import argparse
import json
from datetime import date
from pathlib import Path

from gridlens.fixture import fixture_summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="gridlens")
    subcommands = parser.add_subparsers(dest="command", required=True)
    fixture = subcommands.add_parser("fixture", help="Validate deterministic synthetic inputs")
    fixture.add_argument("--issue-date", type=date.fromisoformat, default=date(2024, 2, 10))
    ingest = subcommands.add_parser("ingest-neso", help="Fetch, preserve, validate and audit pinned NESO CSVs")
    ingest.add_argument("--sources", nargs="+", default=[str(year) for year in range(2017, 2025)])
    ingest.add_argument("--root", type=Path, default=Path("data/neso"))
    ingest.add_argument("--report", type=Path, help="Also export a readable frozen audit")
    args = parser.parse_args()
    if args.command == "fixture":
        print(json.dumps(fixture_summary(args.issue_date), sort_keys=True, indent=2))
    elif args.command == "ingest-neso":
        from gridlens.data.pipeline import run_ingestion

        result = run_ingestion(args.root, args.sources)
        rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        if any(resource["status"] != "valid" for resource in result["resources"]):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
