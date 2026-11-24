"""Baseline evidence must not hide warmups, change dates or archive unsafe files."""

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from gridlens.benchmark import (
    archive_files,
    latency_summary,
    quantile,
    select_origins,
    timed_calls,
    validate_development_dates,
    write_report,
)
from gridlens.data.store import content_id
from scripts.reproduce_baseline import verify_pins

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def test_quantiles_have_declared_interpolation_and_reject_bad_samples(self):
        self.assertEqual(quantile([4, 1, 2, 3], .5), 2.5)
        self.assertAlmostEqual(quantile([4, 1, 2, 3], .95), 3.85)
        self.assertEqual(quantile([2], 1), 2)
        for samples in ([], [float("nan")], [float("inf")], [-1]):
            with self.assertRaises(ValueError):
                quantile(samples, .5)
        with self.assertRaises(ValueError):
            quantile([1], 1.1)
        self.assertEqual(latency_summary([1, 3])["calls"], 2)

    def test_warmups_are_excluded_and_cases_cycle_across_repetitions(self):
        seen = []
        clock = [0, 1_000_000, 2_000_000, 5_000_000, 6_000_000, 11_000_000, 12_000_000, 19_000_000]
        with patch("gridlens.benchmark.time.perf_counter_ns", side_effect=clock):
            report, rows = timed_calls(seen.append, cases=3, warmup=2, repetitions=2, calls=2)
        self.assertEqual(seen, [0, 1, 0, 1, 2, 0])
        self.assertEqual([row["elapsed_ms"] for row in rows], [1, 3, 5, 7])
        self.assertEqual(report["calls"], 4)
        self.assertEqual(report["p50_ms"], 4)
        self.assertEqual(report["warmup_calls_excluded"], 2)
        self.assertEqual(len(report["repetitions"]), 2)
        with self.assertRaises(ValueError):
            timed_calls(seen.append, cases=0, warmup=0, repetitions=1, calls=1)

    def test_corpus_is_frozen_validation_membership_not_an_arbitrary_date_range(self):
        dates = ["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01"]
        self.assertEqual(select_origins(dates, 3), [dates[0], dates[1], dates[3]])
        for requested in ([], [dates[0], dates[0]], ["2024-01-01"]):
            with self.assertRaises(ValueError):
                validate_development_dates(requested, dates)
        with self.assertRaises(ValueError):
            select_origins(dates, 5)

    def test_clock_resolution_cannot_produce_infinite_capacity(self):
        with patch("gridlens.benchmark.time.perf_counter_ns", return_value=1):
            report, _ = timed_calls(lambda _: None, cases=1, warmup=0, repetitions=1, calls=1)
        self.assertIsNone(report["sequential_capacity_forecasts_per_second"])

    def test_archive_contains_only_explicit_bytes_and_rejects_escape_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            root.mkdir()
            weights = root / "weights.bin"
            weights.write_bytes(b"small fixture, not trained model evidence")
            (root / "secret-not-requested.txt").write_text("not included")
            archive = root / "models.zip"
            report = archive_files(root, [weights], archive)
            with zipfile.ZipFile(archive) as saved:
                self.assertEqual(saved.namelist(), ["weights.bin"])
                self.assertEqual(saved.read("weights.bin"), weights.read_bytes())
            self.assertEqual(len(report["files"]), 1)
            outside = root.parent / "outside.bin"
            outside.write_bytes(b"outside")
            for members in ([outside], [weights, weights], []):
                with self.assertRaises(ValueError):
                    archive_files(root, members, root / "bad.zip")
            with self.assertRaises(ValueError):
                archive_files(root, [archive], archive)

    def test_report_is_content_addressed_immutable_and_rejects_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            write_report(path, {"metric": 1})
            record = json.loads(path.read_text())
            self.assertEqual(record["report_id"], content_id({"metric": 1}))
            write_report(path, {"metric": 1})
            with self.assertRaises(ValueError):
                write_report(path, {"metric": 2})
            with self.assertRaises(ValueError):
                write_report(path.parent / "nan.json", {"metric": float("nan")})

    def test_plan_detects_changed_references_or_test_unlock(self):
        plan = json.loads((ROOT / "configs/upgrade_experiment_v2.json").read_text())
        verify_pins(ROOT, plan)
        for change in ({"locked_test_evaluated": True}, {"allowed_inference_role": "locked_test"}, {"seed": 42}, {"calls_per_repetition": -1}, {"repetitions": True}):
            with self.assertRaises(ValueError):
                verify_pins(ROOT, {**plan, **change})
        tampered = copy.deepcopy(plan)
        tampered["reference_sha256"]["reports/dataset-freeze-v1.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Pinned reference changed"):
            verify_pins(ROOT, tampered)
