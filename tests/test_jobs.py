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
        self.now = 1000.0
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


class PostgresQueueTests(QueueTests):
    def setUp(self):
        import os

        database = os.environ.get("GRIDLENS_TEST_POSTGRES")
        if not database:
            self.skipTest("Set GRIDLENS_TEST_POSTGRES to a dedicated test database")
        self.now = 1000.0
        self.path = database
        self.store = JobStore(database, clock=lambda: self.now, lease_seconds=5, max_attempts=2)
        with self.store.connect() as db:
            db.execute("DELETE FROM queue_tickets")
            db.execute("DELETE FROM queue_tasks")
