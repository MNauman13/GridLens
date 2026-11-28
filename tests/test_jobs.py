import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gridlens.jobs import JobStore
from gridlens.live import AdmissionError


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = 1_795_820_000.125
        self.path = Path(self.temp.name) / "queue.db"
        self.store = JobStore(self.path, clock=lambda: self.now, lease_seconds=5, max_attempts=2)

    def test_idempotency_ownership_coalescing_and_fresh_recompute(self):
        a, c = self.store.submit("a", {"date": "2022-01-01"}, "key")
        self.assertFalse(c)
        self.assertEqual(self.store.submit("a", {"date": "2022-01-01"}, "key"), (a, True))
        with self.assertRaises(ValueError):
            self.store.submit("a", {"date": "other"}, "key")
        b, c = self.store.submit("b", {"date": "2022-01-01"}, "key")
        self.assertTrue(c)
        self.assertIsNone(self.store.job(a, "b"))
        task = self.store.claim()
        self.assertIsNone(self.store.claim())
        self.assertTrue(self.store.complete(task["id"], task["fence"], {"value": 48}))
        self.assertEqual(self.store.job(a, "a")["result"], self.store.job(b, "b")["result"])
        new, _ = self.store.submit("a", {"date": "2022-01-01"}, "new")
        self.assertNotEqual(new, a)
        self.assertIsNotNone(self.store.claim())

    def test_expired_lease_recovery_fencing_and_heartbeat(self):
        ticket, _ = self.store.submit("a", {"x": 1}, "key")
        old = self.store.claim()
        self.now += 6
        self.assertFalse(self.store.heartbeat(old["id"], old["fence"]))
        self.assertFalse(self.store.complete(old["id"], old["fence"], {"stale": True}))
        restarted = JobStore(self.path, clock=lambda: self.now, lease_seconds=5, max_attempts=2)
        new = restarted.claim()
        self.assertEqual(new["id"], old["id"])
        self.assertGreater(new["fence"], old["fence"])
        self.assertFalse(self.store.complete(old["id"], old["fence"], {}))
        self.assertTrue(restarted.heartbeat(new["id"], new["fence"]))
        self.assertTrue(restarted.complete(new["id"], new["fence"], {"fresh": True}))
        self.assertEqual(self.store.job(ticket, "a")["result"], {"fresh": True})

    def test_last_subscriber_cancellation_fences_running_work(self):
        a, _ = self.store.submit("a", {"x": 1}, "a")
        b, _ = self.store.submit("b", {"x": 1}, "b")
        task = self.store.claim()
        self.assertFalse(self.store.cancel(a, "other"))
        self.assertTrue(self.store.cancel(a, "a"))
        self.assertEqual(self.store.job(b, "b")["state"], "running")
        self.store.cancel(b, "b")
        self.assertFalse(self.store.complete(task["id"], task["fence"], {"x": 1}))
        self.assertEqual(self.store.job(a, "a")["state"], "cancelled")

    def test_retry_backoff_exhaustion_and_retention(self):
        ticket, _ = self.store.submit("a", {"x": 1}, "a")
        task = self.store.claim()
        self.store.fail(task["id"], task["fence"])
        self.assertIsNone(self.store.claim())
        self.now += 3
        task = self.store.claim()
        self.store.fail(task["id"], task["fence"])
        self.assertEqual(self.store.job(ticket, "a")["state"], "failed")
        self.now += 86401
        self.assertEqual(self.store.prune(), 1)
        self.assertIsNone(self.store.job(ticket, "a"))
        ticket, _ = self.store.submit("a", {"x": 1}, "new")
        self.store.claim()
        self.now += 6
        self.store.claim()
        self.now += 6
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.store.job(ticket, "a")["error"], "attempts_exhausted")

    def test_concurrent_claims_capacity_and_owner_limit(self):
        store = JobStore(self.path, capacity=2, clock=lambda: self.now)
        store.submit("a", {"x": 1}, "a")
        store.submit("b", {"x": 2}, "b")
        with self.assertRaises(AdmissionError):
            store.submit("c", {"x": 3}, "c")
        with self.assertRaises(AdmissionError):
            store.submit("a", {"x": 4}, "d")
        with ThreadPoolExecutor(4) as pool:
            tasks = list(pool.map(lambda _: store.claim(), range(4)))
        self.assertEqual(len([t for t in tasks if t]), 2)
        self.assertEqual(len({t["id"] for t in tasks if t}), 2)

    def test_actual_process_death_after_claim_is_recovered(self):
        import json
        import subprocess
        import sys

        ticket, _ = self.store.submit("crash-owner", {"x": 9}, "once")
        script = "from gridlens.jobs import JobStore; import sys,os,json; store=JobStore(sys.argv[1],clock=lambda:float(sys.argv[2]),lease_seconds=5,max_attempts=2); task=store.claim(); os.write(1,json.dumps(task).encode()); os._exit(17)"
        child = subprocess.run(
            [sys.executable, "-c", script, str(self.path), str(self.now)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(child.returncode, 17, child.stderr)
        old = json.loads(child.stdout)
        self.now += 6.0
        new = self.store.claim()
        self.assertEqual(new["id"], old["id"])
        self.assertFalse(self.store.complete(old["id"], old["fence"], {"stale": True}))
        self.assertTrue(self.store.complete(new["id"], new["fence"], {"recovered": True}))
        self.assertEqual(self.store.job(ticket, "crash-owner")["result"], {"recovered": True})


class PostgresQueueTests(QueueTests):
    def setUp(self):
        import os

        database = os.environ.get("GRIDLENS_TEST_POSTGRES")
        if not database:
            self.skipTest("Set GRIDLENS_TEST_POSTGRES to a dedicated test database")
        self.now = 1_795_820_000.125
        self.path = database
        self.store = JobStore(database, clock=lambda: self.now, lease_seconds=5, max_attempts=2)
        with self.store.connect() as db:
            db.execute("DELETE FROM queue_tickets")
            db.execute("DELETE FROM queue_tasks")

    def test_legacy_float32_columns_are_widened_and_preserve_second_precision(self):
        with self.store.connect() as db:
            for table, columns in (
                ("queue_tasks", ("created", "available", "lease_until")),
                ("queue_tickets", ("created",)),
            ):
                for column in columns:
                    db.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE REAL")
        migrated = JobStore(self.path, clock=lambda: self.now, lease_seconds=5)
        ticket, _ = migrated.submit("migration", {"epoch": True}, "once")
        task = migrated.claim()
        with migrated.connect() as db:
            row = db.execute(
                "SELECT created,lease_until FROM queue_tasks WHERE id=?", (task["id"],)
            ).fetchone()
            self.assertAlmostEqual(row["created"], self.now, places=6)
            self.assertAlmostEqual(row["lease_until"], self.now + 5, places=6)
        self.assertTrue(migrated.complete(task["id"], task["fence"], {"precise": True}))
        self.assertEqual(migrated.job(ticket, "migration")["state"], "succeeded")
