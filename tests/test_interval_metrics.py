import unittest

import numpy as np

from gridlens.calibration.metrics import interval_metrics


class IntervalMetricsTests(unittest.TestCase):
    def test_hand_calculated_scores_and_nested_contract(self):
        y = np.full((1, 48), 10.0)
        p = np.full((1, 48), 8.0)
        q = np.tile([1.0, 2.0, 3.0], (1, 48, 1))
        result = interval_metrics(y, p, q)
        self.assertEqual(result["levels"]["0.8"]["coverage"], 0)
        self.assertEqual(result["levels"]["0.9"]["coverage"], 1)
        self.assertAlmostEqual(result["levels"]["0.8"]["mean_interval_score_mw"], 12.0)
        self.assertEqual(result["crossings"], 0)
        self.assertAlmostEqual(
            result["weighted_interval_score_mw"], (1.0 + 0.1 * 12 + 0.05 * 4 + 0.025 * 6) / 3.5
        )
        with self.assertRaises(ValueError):
            interval_metrics(y, p, q[:, :, ::-1])
        with self.assertRaises(ValueError):
            interval_metrics([], [], [])
