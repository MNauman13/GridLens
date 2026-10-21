"""Adversarial fixtures for settlement time, label eligibility, and raw provenance."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from email.message import Message
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

from gridlens.data.neso import QuarantineError, Source, parse_csv
from gridlens.data.settlement import HALF_HOUR, parse_date, periods_in_day, settlement_interval
from gridlens.data.store import Store
from gridlens.data.transport import (
    DownloadError,
    Response,
    download,
    final_download_url,
    receipt_url,
)

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
        self.assertEqual(parse_date("01-Jan-23", resource_year=2023), date(2023, 1, 1))
        with self.assertRaises(ValueError):
            parse_date("01-Jan-23")
        with self.assertRaises(ValueError):
            parse_date("01-Jan-24", resource_year=2023)
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


URL = "https://api.neso.energy/fixture.csv"
STAMP = datetime(2026, 10, 21, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


def response(content: bytes, stamp: datetime = STAMP) -> Response:
    return Response(content, URL, URL, 200, {"content-type": "text/csv"}, stamp)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = ROOT / "reports/local"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.store = Store(Path(self.temporary.name))

    def test_same_bytes_are_idempotent_but_both_retrievals_are_recorded(self) -> None:
        raw = csv_bytes("2024-01-01,1,24000")
        first = self.store.ingest(response(raw), ANNUAL)
        second = self.store.ingest(response(raw, STAMP + timedelta(hours=1)), ANNUAL)
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.store.root / "raw").glob("*.csv"))), 1)
        with self.store.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM captures").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT count(*) FROM observations").fetchone()[0], 1)
        versions = self.store.versions(datetime(2024, 1, 1, tzinfo=UTC))
        self.assertEqual(len(versions), 1)
        self.assertTrue(versions[0]["first_seen_at"].startswith("2026-10-21T00:00:00"))

    def test_revision_keeps_prior_value_and_capture_cutoff(self) -> None:
        self.store.ingest(response(csv_bytes("2024-01-01,1,24000")), ANNUAL)
        self.store.ingest(response(csv_bytes("2024-01-01,1,24001"), STAMP + timedelta(hours=1)), ANNUAL)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        self.assertEqual([v["nd_mw"] for v in self.store.versions(start)], [24000, 24001])
        self.assertEqual([v["nd_mw"] for v in self.store.versions(start, as_of=STAMP)], [24000])
        self.assertEqual(self.store.versions(start, as_of=STAMP - timedelta(seconds=1)), [])

    def test_invalid_csv_is_preserved_and_quarantined_without_partial_rows(self) -> None:
        raw = csv_bytes("2024-01-01,1,24000", "2024-01-01,2,NaN")
        audit = self.store.ingest(response(raw), ANNUAL)
        self.assertEqual(audit["status"], "quarantined")
        self.assertEqual((self.store.root / "raw" / f"{audit['raw_sha256']}.csv").read_bytes(), raw)
        with self.store.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM observations").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM captures").fetchone()[0], 1)

    def test_raw_corruption_is_detected_on_repeat_import(self) -> None:
        raw = csv_bytes("2024-01-01,1,24000")
        audit = self.store.ingest(response(raw), ANNUAL)
        (self.store.root / "raw" / f"{audit['raw_sha256']}.csv").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self.store.ingest(response(raw), ANNUAL)

    def test_naive_capture_time_rejected_before_writes(self) -> None:
        with self.assertRaises(ValueError):
            self.store.ingest(response(csv_bytes("2024-01-01,1,24000"), datetime(2026, 1, 1)), ANNUAL)

    def test_failed_index_transaction_keeps_raw_receipt_but_no_partial_snapshot(self) -> None:
        with self.store.connect() as connection:
            connection.execute("CREATE TRIGGER fail_index BEFORE INSERT ON observations BEGIN SELECT RAISE(ABORT, 'injected write failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.ingest(response(csv_bytes("2024-01-01,1,24000")), ANNUAL)
        with self.store.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM snapshots").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM observations").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM captures").fetchone()[0], 1)
            connection.execute("DROP TRIGGER fail_index")
        self.assertEqual(self.store.ingest(response(csv_bytes("2024-01-01,1,24000")), ANNUAL)["status"], "valid")


class TransportTests(unittest.TestCase):
    def test_retry_after_is_respected_and_timeout_is_forwarded(self) -> None:
        headers = Message()
        headers["Retry-After"] = "7"
        request = Mock(side_effect=[HTTPError(URL, 429, "busy", headers, None), response(b"csv")])
        sleep = Mock()
        self.assertEqual(download(URL, request=request, sleep=sleep, jitter=lambda: 0).content, b"csv")
        sleep.assert_called_once_with(7.0)
        request.assert_called_with(URL, 20)

    def test_http_date_retry_after_and_large_delays(self) -> None:
        headers = Message()
        headers["Retry-After"] = "Wed, 21 Oct 2026 00:00:09 GMT"
        request = Mock(side_effect=[HTTPError(URL, 503, "busy", headers, None), response(b"csv")])
        sleep = Mock()
        download(URL, request=request, sleep=sleep, jitter=lambda: 0, now=lambda: STAMP)
        sleep.assert_called_once_with(9.0)
        headers.replace_header("Retry-After", "120")
        sleep = Mock()
        with self.assertRaises(DownloadError):
            download(URL, request=Mock(side_effect=HTTPError(URL, 429, "busy", headers, None)), sleep=sleep, jitter=lambda: 0)
        sleep.assert_not_called()

    def test_observed_provider_redirect_is_resource_bound_and_receipts_omit_query(self) -> None:
        requested = "https://api.neso.energy/dataset/example/resource/abc/download/demand.csv"
        final = "https://83025b28472d6aa2bf5ae59f3724aa78.eu.r2.cloudflarestorage.com/dx-national-grid/national-grid/resources/abc/demand.csv?X-Amz-Signature=temporary"
        self.assertEqual(final_download_url(requested, final), final)
        self.assertNotIn("?", receipt_url(final))
        for bad in (final.replace("/abc/", "/different/"), final.replace("eu.r2.cloudflarestorage.com", "example.com")):
            with self.assertRaises(DownloadError):
                final_download_url(requested, bad)

    def test_permanent_http_errors_are_not_retried(self) -> None:
        request = Mock(side_effect=HTTPError(URL, 404, "missing", Message(), None))
        sleep = Mock()
        with self.assertRaises(DownloadError):
            download(URL, request=request, sleep=sleep)
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_network_failures_have_a_bounded_retry_budget(self) -> None:
        request = Mock(side_effect=URLError("offline"))
        sleep = Mock()
        with self.assertRaises(DownloadError):
            download(URL, request=request, sleep=sleep, jitter=lambda: 0)
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_untrusted_hosts_and_unbounded_options_rejected(self) -> None:
        for url in ("http://api.neso.energy/data", "https://example.com/data", "https://user@api.neso.energy/data"):
            with self.assertRaises(DownloadError):
                download(url)
        with self.assertRaises(ValueError):
            download(URL, attempts=99)


if __name__ == "__main__":
    unittest.main()
