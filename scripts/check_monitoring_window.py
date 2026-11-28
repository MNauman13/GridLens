"""Assess actual public probes. A schedule alone never passes availability."""

import argparse
import json
from datetime import datetime
from pathlib import Path


def assess(records):
    valid = sorted(records, key=lambda row: row["started_at"])
    if not valid:
        return {"status": "pending", "probes": 0, "reason": "No observed public probes"}
    target = next(
        (row.get("release_freeze_id") for row in reversed(valid) if row.get("release_freeze_id")),
        None,
    )
    if target:
        prior_other = max(
            (
                i
                for i, row in enumerate(valid)
                if row.get("release_freeze_id") and row.get("release_freeze_id") != target
            ),
            default=-1,
        )
        first = next(
            i
            for i, row in enumerate(valid)
            if i > prior_other and row.get("release_freeze_id") == target
        )
        # Keep failed probes without metadata after the release starts.
        valid = valid[first:]
    stamps = [datetime.fromisoformat(row["started_at"]) for row in valid]
    if any(stamp.utcoffset() is None for stamp in stamps):
        raise ValueError("Timezone-aware probe records required")
    unique = len(set(stamps)) == len(stamps)
    days = (stamps[-1] - stamps[0]).total_seconds() / 86400
    gaps = [(b - a).total_seconds() / 3600 for a, b in zip(stamps, stamps[1:], strict=False)]
    failures = sum(not row.get("passed", False) for row in valid)
    passed = (
        unique and days >= 7 and len(valid) >= 29 and failures == 0 and max(gaps, default=0) <= 12
    )
    return {
        "status": "passed" if passed else "pending",
        "release_freeze_id": target,
        "probes": len(valid),
        "elapsed_days": days,
        "failed_probes": failures,
        "maximum_gap_hours": max(gaps, default=0),
        "duplicate_timestamps": not unique,
        "required_days": 7,
        "required_probes": 29,
        "note": "One configured six-hour probe location, not multi-region availability",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    records = [json.loads(path.read_text()) for path in args.folder.rglob("public-smoke.json")]
    print(json.dumps(assess(records), indent=2))


if __name__ == "__main__":
    main()
