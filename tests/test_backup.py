import tempfile
import unittest
from pathlib import Path

from gridlens.jobs import JobStore
from scripts.backup_jobs import backup, restore


class BackupTests(unittest.TestCase):
    def test_consistent_backup_restore_and_corruption_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = JobStore(root / "original.db")
            ticket, _ = store.submit("owner", {"date": "2022-01-01"}, "once")
            task = store.claim()
            store.complete(task["id"], task["fence"], {"value": 48})
            record = backup(root / "original.db", root / "backup.sqlite3")
            self.assertEqual(record["task_counts"], {"succeeded": 1})
            store.cancel(ticket, "owner")
            restore(root / "backup.sqlite3", root / "restored.db")
            recovered = JobStore(root / "restored.db")
            self.assertEqual(recovered.job(ticket, "owner")["state"], "succeeded")
            self.assertEqual(recovered.job(ticket, "owner")["result"], {"value": 48})
            with self.assertRaises(ValueError):
                restore(root / "backup.sqlite3", root / "restored.db")
            (root / "backup.sqlite3").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                restore(root / "backup.sqlite3", root / "corrupt.db")
            self.assertFalse((root / "corrupt.db").exists())
