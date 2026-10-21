"""Pure UTC timing; no data retrieval or future measurements."""

from datetime import UTC, date, datetime, time, timedelta


def boundaries(issue_date: date) -> dict[str, datetime]:
    if type(issue_date) is not date:
        raise TypeError("Expected an issue date")
    cutoff = datetime.combine(issue_date, time(), tzinfo=UTC)
    return {
        "issued_at": cutoff + timedelta(hours=9, minutes=30),
        "context_start": cutoff - timedelta(days=7),
        "context_end_exclusive": cutoff,
        "target_start": cutoff + timedelta(days=1),
        "target_end_exclusive": cutoff + timedelta(days=2),
    }
