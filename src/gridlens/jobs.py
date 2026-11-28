"""Durable bounded work queue with owner tickets, leases and fenced commits.

SQLite uses BEGIN IMMEDIATE. PostgreSQL uses one short advisory transaction lock
for admission/claim transitions. Compute runs outside transactions. A cancelled
or expired lease can never publish through its old fencing token.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gridlens.data.store import content_id
from gridlens.live import AdmissionError


class JobStore:
    def __init__(
        self,
        database: str | Path,
        *,
        capacity: int = 8,
        lease_seconds: float = 10,
        max_attempts: int = 3,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.database = str(database)
        self.postgres = self.database.startswith(("postgres://", "postgresql://"))
        if capacity < 1 or lease_seconds <= 0 or max_attempts < 1:
            raise ValueError("Invalid queue limits")
        self.capacity, self.lease_seconds, self.max_attempts = capacity, lease_seconds, max_attempts
        self.clock = clock
        if not self.postgres:
            Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS queue_tasks (id TEXT PRIMARY KEY,work_key TEXT NOT NULL,payload TEXT NOT NULL,state TEXT NOT NULL,created DOUBLE PRECISION NOT NULL,available DOUBLE PRECISION NOT NULL,lease_until DOUBLE PRECISION NOT NULL DEFAULT 0,fence INTEGER NOT NULL DEFAULT 0,attempts INTEGER NOT NULL DEFAULT 0,result TEXT,error TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS queue_tickets (id TEXT PRIMARY KEY,task_id TEXT NOT NULL REFERENCES queue_tasks(id),owner TEXT NOT NULL,request_key TEXT NOT NULL,created DOUBLE PRECISION NOT NULL,cancelled INTEGER NOT NULL DEFAULT 0,UNIQUE(owner,request_key))"
            )
            if self.postgres:
                # REAL is float32 in PostgreSQL and loses seconds at Unix epochs.
                # Widen older installations under the same transaction lock.
                for table, columns in (
                    ("queue_tasks", ("created", "available", "lease_until")),
                    ("queue_tickets", ("created",)),
                ):
                    for column in columns:
                        definition = db.execute(
                            "SELECT data_type FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=? AND column_name=?",
                            (table, column),
                        ).fetchone()
                        if definition is not None and definition["data_type"] == "real":
                            db.execute(
                                f"ALTER TABLE {table} ALTER COLUMN {column} TYPE DOUBLE PRECISION"
                            )
            db.execute("CREATE INDEX IF NOT EXISTS queue_ready ON queue_tasks(state,available)")
            db.execute("CREATE INDEX IF NOT EXISTS queue_owner ON queue_tickets(owner,created)")

    @contextmanager
    def connect(self) -> Iterator[Any]:
        if self.postgres:
            import psycopg
            from psycopg.rows import dict_row

            with psycopg.connect(self.database, row_factory=dict_row) as pg_connection:
                # Database-backed serialisation applies across API and worker processes.
                pg_connection.execute("SELECT pg_advisory_xact_lock(74192831)")

                class Adapter:
                    def execute(self, sql: str, args: tuple[Any, ...] = ()) -> Any:
                        return pg_connection.execute(sql.replace("?", "%s"), args)

                yield Adapter()
        else:
            connection = sqlite3.connect(self.database, timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def now(self, db: Any) -> float:
        if self.clock:
            return self.clock()
        if self.postgres:
            return float(
                db.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp()) AS now").fetchone()["now"]
            )
        return time.time()

    def submit(self, owner: str, payload: dict[str, Any], request_key: str) -> tuple[str, bool]:
        if not owner or not 1 <= len(request_key) <= 128:
            raise ValueError("Owner and bounded idempotency key required")
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        key = content_id(payload)
        with self.connect() as db:
            now = self.now(db)
            existing = db.execute(
                "SELECT t.id,q.work_key FROM queue_tickets t JOIN queue_tasks q ON q.id=t.task_id WHERE owner=? AND request_key=?",
                (owner, request_key),
            ).fetchone()
            if existing:
                if existing["work_key"] != key:
                    raise ValueError("Idempotency key already used for different work")
                return str(existing["id"]), True
            active = db.execute(
                "SELECT COUNT(*) AS n FROM queue_tickets t JOIN queue_tasks q ON q.id=t.task_id WHERE owner=? AND t.cancelled=0 AND q.state IN ('queued','running')",
                (owner,),
            ).fetchone()["n"]
            recent = db.execute(
                "SELECT COUNT(*) AS n FROM queue_tickets WHERE owner=? AND created>?",
                (owner, now - 60),
            ).fetchone()["n"]
            total = db.execute(
                "SELECT COUNT(*) AS n FROM queue_tickets WHERE created>?", (now - 60,)
            ).fetchone()["n"]
            if active >= 1 or recent >= 60 or total >= 240:
                raise AdmissionError("Session or global admission limit reached")
            task = db.execute(
                "SELECT id FROM queue_tasks WHERE work_key=? AND state IN ('queued','running') ORDER BY created LIMIT 1",
                (key,),
            ).fetchone()
            coalesced = task is not None
            if task:
                task_id = task["id"]
            else:
                count = db.execute(
                    "SELECT COUNT(*) AS n FROM queue_tasks WHERE state IN ('queued','running')"
                ).fetchone()["n"]
                if count >= self.capacity:
                    raise AdmissionError("Compute queue capacity reached")
                task_id = str(uuid.uuid4())
                db.execute(
                    "INSERT INTO queue_tasks(id,work_key,payload,state,created,available) VALUES(?,?,?,?,?,?)",
                    (task_id, key, encoded, "queued", now, now),
                )
            ticket = str(uuid.uuid4())
            db.execute(
                "INSERT INTO queue_tickets(id,task_id,owner,request_key,created) VALUES(?,?,?,?,?)",
                (ticket, task_id, owner, request_key, now),
            )
            return ticket, coalesced

    def claim(self) -> dict[str, Any] | None:
        with self.connect() as db:
            now = self.now(db)
            db.execute(
                "UPDATE queue_tasks SET state='failed',error='attempts_exhausted' WHERE state='running' AND lease_until<=? AND attempts>=?",
                (now, self.max_attempts),
            )
            row = db.execute(
                "SELECT * FROM queue_tasks WHERE (state='queued' AND available<=?) OR (state='running' AND lease_until<=? AND attempts<?) ORDER BY created,id LIMIT 1",
                (now, now, self.max_attempts),
            ).fetchone()
            if row is None:
                return None
            fence = row["fence"] + 1
            db.execute(
                "UPDATE queue_tasks SET state='running',fence=?,attempts=attempts+1,lease_until=? WHERE id=?",
                (fence, now + self.lease_seconds, row["id"]),
            )
            return {
                "id": row["id"],
                "fence": fence,
                "payload": json.loads(row["payload"]),
                "queue_ms": max(0, (now - row["created"]) * 1000),
                "attempt": row["attempts"] + 1,
            }

    def heartbeat(self, task_id: str, fence: int) -> bool:
        with self.connect() as db:
            now = self.now(db)
            return bool(
                db.execute(
                    "UPDATE queue_tasks SET lease_until=? WHERE id=? AND fence=? AND state='running' AND lease_until>?",
                    (now + self.lease_seconds, task_id, fence, now),
                ).rowcount
            )

    def complete(self, task_id: str, fence: int, result: dict[str, Any]) -> bool:
        encoded = json.dumps(result, allow_nan=False)
        with self.connect() as db:
            now = self.now(db)
            return bool(
                db.execute(
                    "UPDATE queue_tasks SET state='succeeded',result=?,error=NULL WHERE id=? AND fence=? AND state='running' AND lease_until>?",
                    (encoded, task_id, fence, now),
                ).rowcount
            )

    def fail(self, task_id: str, fence: int, *, retryable: bool = True) -> bool:
        with self.connect() as db:
            now = self.now(db)
            row = db.execute(
                "SELECT attempts FROM queue_tasks WHERE id=? AND fence=? AND state='running' AND lease_until>?",
                (task_id, fence, now),
            ).fetchone()
            if row is None:
                return False
            retry = retryable and row["attempts"] < self.max_attempts
            db.execute(
                "UPDATE queue_tasks SET state=?,error=?,available=? WHERE id=? AND fence=?",
                (
                    "queued" if retry else "failed",
                    "retry_pending" if retry else "inference_failed",
                    now + min(2 ** row["attempts"], 30),
                    task_id,
                    fence,
                ),
            )
            return True

    def job(self, ticket: str, owner: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT t.id,t.cancelled,q.state,q.result,q.error,q.attempts FROM queue_tickets t JOIN queue_tasks q ON q.id=t.task_id WHERE t.id=? AND t.owner=?",
                (ticket, owner),
            ).fetchone()
            if row is None:
                return None
            cancelled = bool(row["cancelled"])
            return {
                "job_id": row["id"],
                "state": "cancelled" if cancelled else row["state"],
                "error": None if cancelled else row["error"],
                "result": json.loads(row["result"]) if row["result"] and not cancelled else None,
                "attempts": row["attempts"],
            }

    def cancel(self, ticket: str, owner: str) -> bool:
        with self.connect() as db:
            row = db.execute(
                "SELECT task_id FROM queue_tickets WHERE id=? AND owner=?", (ticket, owner)
            ).fetchone()
            if row is None:
                return False
            db.execute("UPDATE queue_tickets SET cancelled=1 WHERE id=?", (ticket,))
            active = db.execute(
                "SELECT COUNT(*) AS n FROM queue_tickets WHERE task_id=? AND cancelled=0",
                (row["task_id"],),
            ).fetchone()["n"]
            if active == 0:
                db.execute(
                    "UPDATE queue_tasks SET state='cancelled',fence=fence+1 WHERE id=? AND state IN ('queued','running')",
                    (row["task_id"],),
                )
            return True

    def prune(self, retention_seconds: float = 86400) -> int:
        if retention_seconds < 60:
            raise ValueError("Retention must be at least one minute")
        with self.connect() as db:
            cutoff = self.now(db) - retention_seconds
            terminal = "SELECT id FROM queue_tasks WHERE state IN ('succeeded','failed','cancelled') AND created<?"
            db.execute(f"DELETE FROM queue_tickets WHERE task_id IN ({terminal})", (cutoff,))
            return int(
                db.execute(f"DELETE FROM queue_tasks WHERE id IN ({terminal})", (cutoff,)).rowcount
            )

    def stats(self) -> dict[str, int]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT state,COUNT(*) AS n FROM queue_tasks GROUP BY state"
            ).fetchall()
            return {row["state"]: row["n"] for row in rows}
