"""Independent calendar examples guard the specification against leakage."""
import json
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scripts.forecast_contract import load_contract, origin_boundaries


class ForecastContractTests(unittest.TestCase):
    def test_worked_origin_and_actual_lead_times(self):
        result = origin_boundaries(date(2024, 2, 10), load_contract())
        self.assertEqual(result['issued_at'], datetime(2024, 2, 10, 9, 30, tzinfo=UTC))
        self.assertEqual(result['context_start'], datetime(2024, 2, 3, tzinfo=UTC))
        self.assertEqual(result['last_observed_interval_start'], datetime(2024, 2, 9, 23, 30, tzinfo=UTC))
        self.assertEqual(result['target_start'], datetime(2024, 2, 11, tzinfo=UTC))
        self.assertEqual(result['target_end_exclusive'], datetime(2024, 2, 12, tzinfo=UTC))
        self.assertEqual(result['points'][0]['lead_minutes'], 870)
        self.assertEqual(result['points'][-1]['lead_minutes'], 2280)

    def test_baselines_cannot_read_issue_day_actuals(self):
        result = origin_boundaries(date(2024, 2, 10), load_contract())
        for point in result['points']:
            for name in ['weekly_baseline_start', 'last_available_day_baseline_start']:
                self.assertGreaterEqual(point[name], result['context_start'])
                self.assertLess(point[name], result['context_end_exclusive'])
        self.assertEqual(result['points'][0]['weekly_baseline_start'].date(), date(2024, 2, 4))
        self.assertEqual(result['points'][0]['last_available_day_baseline_start'].date(), date(2024, 2, 9))

    def test_utc_targets_stay_48_points_across_clock_changes(self):
        for issue in [date(2024, 3, 30), date(2024, 10, 26), date(2024, 2, 28), date(2024, 12, 31)]:
            with self.subTest(issue=issue):
                result = origin_boundaries(issue, load_contract())
                self.assertEqual(len(result['points']), 48)
                self.assertEqual(result['gap_intervals'], 48)
                self.assertEqual(result['context_end_exclusive'] - result['context_start'], timedelta(days=7))
                self.assertEqual(result['target_end_exclusive'] - result['target_start'], timedelta(days=1))
                for left, right in zip(result['points'], result['points'][1:], strict=False):
                    self.assertEqual(left['interval_end'], right['interval_start'])

    def test_first_training_target_needs_2017_warmup(self):
        result = origin_boundaries(date(2017, 12, 31), load_contract())
        self.assertEqual(result['context_start'].date(), date(2017, 12, 24))
        self.assertEqual(result['target_start'].date(), date(2018, 1, 1))

    def test_non_date_origin_is_rejected(self):
        with self.assertRaises(TypeError):
            origin_boundaries(datetime(2024, 2, 10), load_contract())

    def test_target_timing_and_scope_drift_are_rejected(self):
        cases = [('gap_intervals', 0), ('target', 'NESO_TSD'), ('storage_timezone', 'Europe/London'),
                 ('coverage_scope', 'simultaneous'), ('horizon_intervals', True),
                 ('nominal_coverage_levels', [0.9, 0.8, 0.95])]
        for key, value in cases:
            with self.subTest(key=key):
                data = load_contract()
                data[key] = value
                with patch.object(Path, 'read_text', return_value=json.dumps(data)):
                    with self.assertRaises(ValueError):
                        load_contract()
                with self.assertRaises(ValueError):
                    origin_boundaries(date(2024, 2, 10), data)

    def test_unknown_contract_fields_are_rejected(self):
        data = load_contract()
        data['secret_target_change'] = True
        with patch.object(Path, 'read_text', return_value=json.dumps(data)):
            with self.assertRaises(ValueError):
                load_contract()


if __name__ == '__main__':
    unittest.main()
