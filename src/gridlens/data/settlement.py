"""Map a British settlement ordinal to a unique half-hour UTC interval."""

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
HALF_HOUR = timedelta(minutes=30)
MONTHS = {name: index for index, name in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1
)}


def parse_date(value: str) -> date:
    """Accept the two explicit provider formats, without host-locale dependence."""
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        return date.fromisoformat(value)
    parts = value.split("-")
    if len(parts) == 3 and len(parts[0]) == 2 and len(parts[2]) == 4:
        try:
            return date(int(parts[2]), MONTHS[parts[1].upper()], int(parts[0]))
        except KeyError as exc:
            raise ValueError("Unknown settlement month") from exc
    raise ValueError("Expected YYYY-MM-DD or DD-MON-YYYY")


def day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time(), LONDON).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), time(), LONDON).astimezone(UTC)
    return start, end


def periods_in_day(day: date) -> int:
    start, end = day_bounds(day)
    return (end - start) // HALF_HOUR


def settlement_interval(day: date, period: int) -> tuple[datetime, datetime]:
    if not 1 <= period <= periods_in_day(day):
        raise ValueError(f"Invalid settlement period {period} on {day}")
    start = day_bounds(day)[0] + (period - 1) * HALF_HOUR
    return start, start + HALF_HOUR
