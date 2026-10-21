"""Export teaching fixtures from the same Python conversion used for ingestion."""

import argparse
import json
from datetime import date
from pathlib import Path

from gridlens.data.settlement import LONDON, periods_in_day, settlement_interval

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = {}
    for name, day in [("ordinary", date(2024, 2, 29)), ("spring", date(2024, 3, 31)), ("autumn", date(2024, 10, 27))]:
        periods = []
        for period in range(1, periods_in_day(day) + 1):
            start, end = settlement_interval(day, period)
            periods.append({"period": period, "utc_start": start.isoformat(), "utc_end": end.isoformat(), "local_start": start.astimezone(LONDON).isoformat(), "local_end": end.astimezone(LONDON).isoformat()})
        result[name] = {"date": str(day), "periods": periods}
    path = ROOT / "apps/web/data/settlement-demo.json"
    rendered = json.dumps(result, indent=2) + "\n"
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Settlement demo fixtures are stale")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
