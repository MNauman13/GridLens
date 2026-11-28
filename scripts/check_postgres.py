"""Run the real queue contract against an ephemeral UTF-8 PostgreSQL server.

uv run --extra api --extra ml --with embedded-postgres python -m scripts.check_postgres
Uses TCP only, including on hosts that prohibit Unix-domain sockets.
"""

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import embedded_postgres as pg


def main():
    with tempfile.TemporaryDirectory() as folder:
        data = Path(folder) / "postgres"
        pg.initdb(["-A", "trust", "-U", "gridlens", "--no-locale", "--encoding=UTF8"], pgdata=data)
        pg.pg_ctl(
            [
                "-l",
                str(Path(folder) / "server.log"),
                "-o",
                "-h 127.0.0.1 -p 55432 -k ''",
                "-w",
                "start",
            ],
            pgdata=data,
        )
        try:
            os.environ["GRIDLENS_TEST_POSTGRES"] = "postgresql://gridlens@127.0.0.1:55432/postgres"
            suite = unittest.defaultTestLoader.loadTestsFromName(
                "tests.test_jobs.PostgresQueueTests"
            )
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            if result.wasSuccessful():
                from gridlens.jobs import JobStore

                store = JobStore(os.environ["GRIDLENS_TEST_POSTGRES"])
                ticket, _ = store.submit("restore-owner", {"restore": True}, "backup-once")
                task = store.claim()
                store.complete(task["id"], task["fence"], {"restored": True})
                binaries = pg.pg_ctl.__globals__["POSTGRES_BIN_PATH"]
                archive = Path(folder) / "backup.dump"
                connection = ["-h", "127.0.0.1", "-p", "55432", "-U", "gridlens"]
                subprocess.run(
                    [
                        str(binaries / "pg_dump"),
                        *connection,
                        "-d",
                        "postgres",
                        "-Fc",
                        "-f",
                        str(archive),
                    ],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    [str(binaries / "createdb"), *connection, "restore_check"],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    [
                        str(binaries / "pg_restore"),
                        *connection,
                        "-d",
                        "restore_check",
                        str(archive),
                    ],
                    check=True,
                    capture_output=True,
                )
                restored = JobStore("postgresql://gridlens@127.0.0.1:55432/restore_check")
                if restored.job(ticket, "restore-owner") != store.job(ticket, "restore-owner"):
                    raise ValueError("PostgreSQL restore differs from backup")
                record = {
                    "queue_tests": result.testsRun,
                    "restore_result_identical": True,
                    "dump_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "dump_bytes": archive.stat().st_size,
                }
                Path("reports/local/postgres-recovery.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                print(json.dumps(record))
        finally:
            pg.pg_ctl(["-m", "fast", "stop"], pgdata=data)
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
