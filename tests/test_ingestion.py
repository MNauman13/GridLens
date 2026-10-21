"""Adversarial fixtures for settlement time, label eligibility, and raw provenance."""

from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, timedelta

from gridlens.data.neso import QuarantineError, Source, parse_csv
from gridlens.data.settlement import HALF_HOUR, parse_date, periods_in_day, settlement_interval

ANNUAL = Source("fixture-2024", "historic_outturn", 2024)
DAILY = Source("fixture-daily", "mixed_actual_forecast")
HEADER = "SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND"


def csv_bytes(*rows: str, header: str = HEADER) -> bytes:
    return (header + "\n" + "\n".join(rows) + "\n").encode()


class SettlementTests(unittest.TestCase):
    def test_ordinary_leap_spring_and_autumn_lengths(self) -> None:
        for day, length in [(date(2024, 1, 1), 48), (date(2024, 2, 29), 48), (date(2024, 3, 31), 46), (date(2024, 10, 27), 50)]:
            self.assertEqual(periods_in_day(day), length)
            with self.assertRaises(ValueError):
                settlement_interval(day, length + 1)
            with self.assertRaises(ValueError):
                settlement_interval(day, 0)

    def test_spring_and_autumn_match_published_conversion_table(self) -> None:
        # Elexon: spring P3 = 01:00 GMT / 02:00 BST; autumn P5 = 01:00 GMT.
        self.assertEqual(settlement_interval(date(2024, 3, 31), 3)[0], datetime(2024, 3, 31, 1, tzinfo=UTC))
        self.assertEqual(settlement_interval(date(2024, 10, 27), 1)[0], datetime(2024, 10, 26, 23, tzinfo=UTC))
        self.assertEqual(settlement_interval(date(2024, 10, 27), 5)[0], datetime(2024, 10, 27, 1, tzinfo=UTC))
        self.assertEqual(settlement_interval(date(2024, 7, 1), 1)[0], datetime(2024, 6, 30, 23, tzinfo=UTC))

    def test_all_intervals_2017_to_2024_are_unique_and_contiguous(self) -> None:
        day, finish = date(2017, 1, 1), date(2025, 1, 1)
        previous = None
        count = 0
        while day < finish:
            for period in range(1, periods_in_day(day) + 1):
                start, end = settlement_interval(day, period)
                if previous is not None:
                    self.assertEqual(start, previous)
                self.assertEqual(end - start, HALF_HOUR)
                previous = end
                count += 1
            day += timedelta(days=1)
        self.assertEqual(count, 140256)

    def test_explicit_date_formats_are_locale_independent(self) -> None:
        self.assertEqual(parse_date("29-FEB-2024"), date(2024, 2, 29))
        self.assertEqual(parse_date("2024-02-29"), date(2024, 2, 29))
        for text in ("01/02/2024", "29-FEB-2023", "2024-13-01", "1-JAN-2024"):
            with self.assertRaises(ValueError):
                parse_date(text)


class CsvTests(unittest.TestCase):
    def test_required_and_duplicate_columns_fail_closed(self) -> None:
        for header in ("SETTLEMENT_DATE,SETTLEMENT_PERIOD,TSD", HEADER + ",ND"):
            with self.assertRaises(QuarantineError):
                parse_csv(csv_bytes("2024-01-01,1,24000", header=header), ANNUAL)

    def test_daily_requires_indicator_and_forecasts_never_become_actuals(self) -> None:
        with self.assertRaises(QuarantineError):
            parse_csv(csv_bytes("2024-01-01,1,24000"), DAILY)
        parsed = parse_csv(csv_bytes("2024-01-01,1,24000,A", "2024-01-01,2,999999,F", header=HEADER + ",FORECAST_ACTUAL_INDICATOR"), DAILY)
        self.assertEqual(len(parsed.rows), 1)
        self.assertEqual(parsed.audit["counts"]["forecast_rows_excluded"], 1)
        for indicator in ("", "X", "Actual"):
            with self.assertRaises(QuarantineError):
                parse_csv(csv_bytes(f"2024-01-01,1,24000,{indicator}", header=HEADER + ",FORECAST_ACTUAL_INDICATOR"), DAILY)

    def test_annual_indicator_is_respected_if_present(self) -> None:
        parsed = parse_csv(csv_bytes("2024-01-01,1,24000,A", "2024-01-01,2,25000,F", header=HEADER + ",FORECAST_ACTUAL_INDICATOR"), ANNUAL)
        self.assertEqual(len(parsed.rows), 1)

    def test_identical_duplicates_collapse_but_conflicts_quarantine(self) -> None:
        parsed = parse_csv(csv_bytes("2024-01-01,1,24000", "2024-01-01,1,24000"), ANNUAL)
        self.assertEqual(len(parsed.rows), 1)
        self.assertEqual(parsed.audit["counts"]["identical_duplicates_collapsed"], 1)
        with self.assertRaises(QuarantineError):
            parse_csv(csv_bytes("2024-01-01,1,24000", "2024-01-01,1,24001"), ANNUAL)

    def test_missing_is_a_gap_extra_columns_are_logged(self) -> None:
        parsed = parse_csv(csv_bytes("2024-01-01,1,,x", "2024-01-01,2,24000,x", header=HEADER + ",NEW_COLUMN"), ANNUAL)
        year = parsed.audit["years"][0]
        self.assertEqual(year["missing_intervals"], 17566)
        self.assertEqual(year["missing_values_by_field"]["ND"], 1)
        self.assertEqual(year["missing_nd_slots"], 17567)
        self.assertIn("NEW_COLUMN", parsed.audit["additional_columns"])
        self.assertIsNone(parsed.rows[0].values[0])

    def test_incompatible_rows_quarantine_instead_of_silent_drops(self) -> None:
        for row in ("2024-01-01,1,NaN", "2024-01-01,1,inf", "2024-01-01,1,-1", "2024-01-01,1,words", "2024-01-01,1.0,100", "2024-03-31,47,100", "2023-12-31,1,100", "2024-01-01,1,100,extra", "2024-01-01,1"):
            with self.subTest(row=row), self.assertRaises(QuarantineError):
                parse_csv(csv_bytes(row), ANNUAL)


if __name__ == "__main__":
    unittest.main()
