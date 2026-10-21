"""Small installed entry point used to verify a clean checkout."""

import argparse
import json
from datetime import date

from gridlens.fixture import fixture_summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="gridlens")
    subcommands = parser.add_subparsers(dest="command", required=True)
    fixture = subcommands.add_parser("fixture", help="Validate deterministic synthetic inputs")
    fixture.add_argument("--issue-date", type=date.fromisoformat, default=date(2024, 2, 10))
    args = parser.parse_args()
    if args.command == "fixture":
        print(json.dumps(fixture_summary(args.issue_date), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
