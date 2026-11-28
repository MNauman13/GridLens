import copy
import unittest
from datetime import date, timedelta

import torch

from gridlens.calibration.ncc import NCCNetwork
from gridlens.calibration.replay import IntervalReplay, ReplayState
from scripts.calibration_study import campaign, campaign_with_history


class CampaignTests(unittest.TestCase):
    def test_training_and_replay_delay_labels_and_restore_exactly(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        network = NCCNetwork()
        state = ReplayState(
            "m",
            "c",
            "ncc",
            scores=[[2.0] * 48] * 90,
            quantiles=[[[1.0, 2.0, 3.0]] * 48] * 90,
            errors=[[[1.0, 0.0, 0.0]] * 48] * 90,
        )
        replay = IntervalReplay(state, network=network)
        rows = [
            {
                "target_date": str(date(2023, 1, 1) + timedelta(days=i)),
                "point": [100.0] * 48,
                "actual": [101.0 + i] * 48,
                "context": [[0.0] * 6] * 48,
            }
            for i in range(6)
        ]
        before = copy.deepcopy(network.state_dict())
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        records, losses = campaign(
            replay,
            rows,
            network=network,
            optimizer=optimizer,
            loss_weights={
                "quantile": 1.0,
                "coverage": 0.1,
                "efficiency": 0.01,
                "monotonicity": 1.0,
            },
        )
        self.assertEqual(len(losses), 4)
        self.assertIsNone(records[0]["feedback_watermark"])
        self.assertEqual(records[2]["feedback_watermark"], "2023-01-01")
        self.assertTrue(
            any(not torch.equal(value, before[key]) for key, value in network.state_dict().items())
        )
        all_actual = {row["target_date"]: row["actual"] for row in rows}
        future = [
            {
                "target_date": "2023-01-08",
                "point": [100.0] * 48,
                "actual": [108.0] * 48,
                "context": [[0.0] * 6] * 48,
            }
        ]
        all_actual["2023-01-08"] = future[0]["actual"]
        restored = IntervalReplay.restore(replay.checkpoint(), network=network)
        a, _ = campaign_with_history(replay, future, all_actual)
        b, _ = campaign_with_history(restored, future, all_actual)
        a[0].pop("interval_ms")
        b[0].pop("interval_ms")
        self.assertEqual(a, b)
