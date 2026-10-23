"""Perturbation tests prove which information can affect a feature snapshot."""

import unittest
from datetime import UTC, date, datetime, timedelta

from gridlens.data.dataset import (
    FEATURE_ORDER,
    DroppedOrigin,
    Series,
    Version,
    build_datasets,
    cyclic,
    feature_snapshot,
    training_example,
)


def fixture() -> list[Version]:
    start = datetime(2017, 12, 20, tzinfo=UTC)
    return [Version(start + timedelta(minutes=30 * i), (20000.0 + i, 3000.0 + i, float(i % 48)), start + timedelta(minutes=30 * i + 60), "fixture") for i in range(35 * 48)]


SPLITS = [{"role": "point_train", "start": "2018-01-01", "end": "2018-01-03"}, {"role": "point_validation", "start": "2018-01-04", "end": "2018-01-06"}, {"role": "locked_test", "start": "2018-01-07", "end": "2018-01-09"}]
HASHES = ("0" * 64,)


class DatasetTests(unittest.TestCase):
    def test_canonical_calendar_quadrants_and_periodicity(self) -> None:
        for phase, expected in [(0, (0.0, 1.0)), (1, (1.0, 0.0)), (2, (0.0, -1.0)), (3, (-1.0, 0.0))]:
            self.assertEqual(cyclic(phase, 4), expected)
            self.assertEqual(cyclic(phase + 4, 4), expected)


    def test_exact_context_gap_target_and_declared_feature_order(self) -> None:
        rows = fixture()
        feature = feature_snapshot(Series(rows), date(2018, 1, 5), "retrospective_revised")
        example = training_example(feature, Series(rows), "frozen")
        self.assertEqual(feature.inputs.context_start, datetime(2017, 12, 29, tzinfo=UTC))
        self.assertEqual(feature.inputs.context_end_exclusive, datetime(2018, 1, 5, tzinfo=UTC))
        self.assertEqual(feature.inputs.target_start, datetime(2018, 1, 6, tzinfo=UTC))
        self.assertEqual(len(feature.history), 336)
        self.assertEqual(len(feature.history[0]), len(FEATURE_ORDER))
        self.assertEqual(len(feature.target_calendar), 48)
        expected = next(row for row in rows if row.start == datetime(2018, 1, 6, tzinfo=UTC))
        self.assertEqual(example.target_mw[0], expected.values[0])

    def test_future_and_issue_day_values_cannot_change_feature_identity(self) -> None:
        original = fixture()
        cutoff = datetime(2018, 1, 5, tzinfo=UTC)
        changed = [Version(r.start, (999999.0,) * 3 if r.start >= cutoff else r.values, r.first_seen_at, r.snapshot_id) for r in original]
        self.assertEqual(feature_snapshot(Series(original), cutoff.date(), "retrospective_revised"), feature_snapshot(Series(changed), cutoff.date(), "retrospective_revised"))

    def test_later_captured_revision_is_ineligible_before_issue(self) -> None:
        original = fixture()
        base = feature_snapshot(Series(original), date(2018, 1, 5), "captured_availability")
        old = next(r for r in original if r.start == base.inputs.context_start)
        revised = Version(old.start, (99999.0, 9999.0, 999.0), base.inputs.issued_at + timedelta(seconds=1), "revision")
        self.assertEqual(base, feature_snapshot(Series(original + [revised]), date(2018, 1, 5), "captured_availability"))
        self.assertNotEqual(base.inputs.demand_mw, feature_snapshot(Series(original + [revised]), date(2018, 1, 5), "retrospective_revised").inputs.demand_mw)

    def test_causal_fill_limit_flags_and_age(self) -> None:
        original = fixture()
        missing = datetime(2018, 1, 1, tzinfo=UTC)
        changed = [Version(r.start, (None, None, None) if missing <= r.start < missing + timedelta(hours=1) else r.values, r.first_seen_at, r.snapshot_id) for r in original]
        feature = feature_snapshot(Series(changed), date(2018, 1, 5), "retrospective_revised")
        index = int((missing - feature.inputs.context_start) / timedelta(minutes=30))
        self.assertEqual(feature.history[index][:3], feature.history[index - 1][:3])
        self.assertEqual(feature.history[index][3:9], (1, 1, 1, 1, 1, 1))
        self.assertEqual(feature.history[index + 1][6:9], (2, 2, 2))
        changed = [Version(r.start, (None, None, None) if missing <= r.start < missing + timedelta(minutes=90) else r.values, r.first_seen_at, r.snapshot_id) for r in original]
        with self.assertRaises(DroppedOrigin):
            feature_snapshot(Series(changed), date(2018, 1, 5), "retrospective_revised")

    def test_missing_targets_are_never_filled(self) -> None:
        rows = fixture()
        f = feature_snapshot(Series(rows), date(2018, 1, 5), "retrospective_revised")
        truth = Series([r for r in rows if r.start != f.inputs.target_start])
        with self.assertRaisesRegex(DroppedOrigin, "missing_target"):
            training_example(f, truth, "frozen")

    def test_validation_and_test_perturbation_cannot_fit_the_scaler(self) -> None:
        rows = fixture()
        datasets, report, scaler = build_datasets(Series(rows), SPLITS, HASHES)
        modified = [Version(r.start, (900000.0,) * 3 if r.start >= datetime(2018, 1, 4, tzinfo=UTC) else r.values, r.first_seen_at, r.snapshot_id) for r in rows]
        _, _, changed_scaler = build_datasets(Series(modified), SPLITS, HASHES)
        self.assertEqual(scaler, changed_scaler)
        self.assertEqual(report["scaler_id"], build_datasets(Series(rows), SPLITS, HASHES)[1]["scaler_id"])
        self.assertEqual(report, build_datasets(Series(rows), SPLITS, HASHES)[1])
        self.assertEqual(len(datasets["point_train"].examples), 3)

    def test_overlapping_label_ownership_rejected_even_when_origins_drop(self) -> None:
        with self.assertRaisesRegex(ValueError, "owned"):
            build_datasets(Series(fixture()), SPLITS + [{"role": "other", "start": "2018-01-09", "end": "2018-01-10"}], HASHES)

    def test_missing_captured_history_is_reported_as_dropped(self) -> None:
        rows = fixture()
        late = [Version(r.start, r.values, datetime(2026, 1, 1, tzinfo=UTC), r.snapshot_id) for r in rows]
        with self.assertRaisesRegex(ValueError, "without training"):
            build_datasets(Series(late), SPLITS, HASHES, mode="captured_availability")
