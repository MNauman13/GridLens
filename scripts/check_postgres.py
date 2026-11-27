"""Run the real queue contract against an ephemeral UTF-8 PostgreSQL server.

uv run --extra api --extra ml --with embedded-postgres python -m scripts.check_postgres
Uses TCP only, including on hosts that prohibit Unix-domain sockets.
"""

import os
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
        finally:
            pg.pg_ctl(["-m", "fast", "stop"], pgdata=data)
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
