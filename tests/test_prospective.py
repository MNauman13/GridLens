import hashlib
import unittest
from datetime import UTC, date, datetime, timedelta

from gridlens.data.store import content_id
from gridlens.models.baselines import Baseline
from scripts.forecast_prospective import forecast


class ProspectiveTests(unittest.TestCase):
    def test_only_eligible_captured_actuals_can_form_an_issue(self):
        lines = [
            "SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND,EMBEDDED_WIND_GENERATION,EMBEDDED_SOLAR_GENERATION,FORECAST_ACTUAL_INDICATOR"
        ]
        start = datetime(2024, 2, 1, tzinfo=UTC)
        for index in range(9 * 48):
            stamp = start + timedelta(minutes=30 * index)
            lines.append(
                f"{stamp:%Y-%m-%d},{stamp.hour * 2 + stamp.minute // 30 + 1},{20000 + index},1000,500,A"
            )
        raw = ("\n".join(lines) + "\n").encode()
        receipt = {
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "first_seen_at": "2024-02-10T08:45:00+00:00",
        }
        receipt["capture_id"] = content_id(receipt)
        result = forecast(raw, receipt, date(2024, 2, 10), Baseline("weekly"))
        self.assertEqual(result["evidence_mode"], "captured_availability")
        self.assertEqual(result["inputs"]["context_end_exclusive"], "2024-02-10T00:00:00Z")
        self.assertEqual(len(result["point"]["prediction_mw"]), 48)
        self.assertIsNone(result["intervals"])
        late = {"raw_sha256": receipt["raw_sha256"], "first_seen_at": "2024-02-10T09:31:00+00:00"}
        late["capture_id"] = content_id(late)
        with self.assertRaisesRegex(ValueError, "No eligible vintage"):
            forecast(raw, late, date(2024, 2, 10), Baseline("weekly"))
        with self.assertRaisesRegex(ValueError, "capture bytes"):
            forecast(raw + b"changed", receipt, date(2024, 2, 10), Baseline("weekly"))
