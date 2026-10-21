"""Check that corrupt artifacts and invalid forecasting boundaries are rejected."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from gridlens.artifacts import build_manifest, manifest_id, safe_member, verify_manifest
from gridlens.contracts import IntervalForecast, MaturedFeedback, PointForecast, PredictionBand
from gridlens.fixture import fixture_summary, make_fixture
from scripts.forecast_contract import load_contract, origin_boundaries

ROOT = Path(__file__).resolve().parents[1]


class FoundationTests(unittest.TestCase):
    def test_fixture_is_deterministic_and_matches_phase0_contract(self) -> None:
        actual = make_fixture()
        expected = origin_boundaries(date(2024, 2, 10), load_contract())
        for key in ("issued_at", "context_start", "context_end_exclusive", "target_start", "target_end_exclusive"):
            self.assertEqual(getattr(actual, key), expected[key])
        self.assertEqual(fixture_summary(), fixture_summary())
        self.assertFalse(fixture_summary()["real_demand_data"])
        self.assertEqual(len(actual.demand_mw), 336)

    def test_input_window_rejects_future_context_and_naive_times(self) -> None:
        original = make_fixture().model_dump()
        for replacement in [datetime(2024, 2, 10), datetime(2024, 2, 10, 1, tzinfo=UTC)]:
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValidationError):
                    type(make_fixture())(**{**original, "context_end_exclusive": replacement})

    def test_point_output_rejects_wrong_length_and_non_finite_values(self) -> None:
        for values in [(1.0,) * 47, (1.0,) * 49, (float("nan"),) * 48, (float("inf"),) * 48]:
            with self.subTest(length=len(values)):
                with self.assertRaises(ValidationError):
                    PointForecast(model_version="test", feature_snapshot_id="fixture", prediction_mw=values)

    def test_prediction_intervals_reject_inversion_and_crossing(self) -> None:
        point = PointForecast(model_version="test", feature_snapshot_id="fixture", prediction_mw=(10.0,) * 48)
        with self.assertRaises(ValidationError):
            PredictionBand(level=0.9, lower_mw=(20.0,) * 48, upper_mw=(10.0,) * 48)
        bands = tuple(PredictionBand(level=level, lower_mw=(lower,) * 48, upper_mw=(upper,) * 48) for level, lower, upper in [(0.8, 0.0, 20.0), (0.9, 5.0, 15.0), (0.95, -5.0, 25.0)])
        with self.assertRaises(ValidationError):
            IntervalForecast(point=point, calibrator_version="fixture", state_id="initial", bands=bands)

    def test_unsupported_coverage_level_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            PredictionBand(level=0.7, lower_mw=(0.0,) * 48, upper_mw=(20.0,) * 48)

    def test_feedback_cannot_mature_before_interval_completion(self) -> None:
        end = datetime(2024, 2, 11, 0, 30, tzinfo=UTC)
        with self.assertRaises(ValidationError):
            MaturedFeedback(forecast_id="fixture", model_version="fixture", slot=0, target_end=end, first_seen_at=end - timedelta(minutes=30), actual_mw=1.0, truth_version="v1")

    def test_content_hash_is_stable_and_corruption_is_detected(self) -> None:
        # Keep scratch inside the workspace on hosts whose system temp is restricted.
        scratch = ROOT / "reports" / "local"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            artifact = root / "weights.json"
            artifact.write_text(json.dumps({"weight": 1}), encoding="utf-8")
            first = build_manifest(root, ["weights.json"], kind="test")
            second = build_manifest(root, ["weights.json"], kind="test")
            self.assertEqual(manifest_id(first), manifest_id(second))
            verify_manifest(root, first)
            artifact.write_text(json.dumps({"weight": 2}), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_manifest(root, first)

    def test_bundle_paths_reject_traversal_and_foreign_absolute_paths(self) -> None:
        for member in ["../outside", "nested/../../outside", "/absolute", "C:/outside", "nested\\..\\outside", "."]:
            with self.subTest(member=member):
                with self.assertRaises(ValueError):
                    safe_member(ROOT, member)


if __name__ == "__main__":
    unittest.main()
