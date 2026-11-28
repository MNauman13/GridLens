import unittest
from datetime import UTC, datetime, timedelta

from scripts.check_monitoring_window import assess


class MonitoringTests(unittest.TestCase):
    def test_real_duration_required_even_with_many_probe_records(self):
        first = datetime(2026, 1, 1, tzinfo=UTC)
        records = [
            {"started_at": (first + timedelta(hours=6 * i)).isoformat(), "passed": True}
            for i in range(29)
        ]
        self.assertEqual(assess(records)["status"], "passed")
        self.assertEqual(assess(records[:28])["status"], "pending")
        self.assertEqual(assess(records[:1] * 29)["status"], "pending")
        records[12]["passed"] = False
        self.assertEqual(assess(records)["status"], "pending")
        self.assertEqual(assess([])["status"], "pending")

    def test_release_changes_reset_window_and_metadata_free_failures_still_count(self):
        first = datetime(2026, 1, 1, tzinfo=UTC)
        records = [
            {
                "started_at": (first + timedelta(hours=6 * i)).isoformat(),
                "passed": True,
                "release_freeze_id": "old" if i < 14 else "new",
            }
            for i in range(29)
        ]
        result = assess(records)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["probes"], 15)
        records[20].pop("release_freeze_id")
        records[20]["passed"] = False
        self.assertEqual(assess(records)["failed_probes"], 1)
