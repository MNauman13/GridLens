"""Content-addressed bytes and a transactional SQLite observation-version index."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gridlens.data.neso import TRANSFORM_VERSION, QuarantineError, Source, parse_csv
from gridlens.data.transport import Response, receipt_url

SCHEMA = """
CREATE TABLE IF NOT EXISTS captures (
  capture_id TEXT PRIMARY KEY, resource_id TEXT NOT NULL, raw_sha256 TEXT NOT NULL,
  retrieved_at TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS captures_content ON captures(resource_id, raw_sha256, retrieved_at);
CREATE TABLE IF NOT EXISTS snapshots (
  snapshot_id TEXT PRIMARY KEY, resource_id TEXT NOT NULL, raw_sha256 TEXT NOT NULL,
  transform_version TEXT NOT NULL, first_seen_at TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('valid','quarantined')), audit_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
  snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id), interval_start TEXT NOT NULL,
  interval_end TEXT NOT NULL, settlement_date TEXT NOT NULL, settlement_period INTEGER NOT NULL,
  nd_mw REAL, tsd_mw REAL, wind_mw REAL, solar_mw REAL,
  value_kind TEXT NOT NULL CHECK(value_kind='actual'),
  PRIMARY KEY(snapshot_id, interval_start)
);
CREATE INDEX IF NOT EXISTS observation_time ON observations(interval_start, snapshot_id);
"""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_id(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def utc_text(value: datetime) -> str:
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("Explicit UTC capture time required")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def write_once(destination: Path, content: bytes) -> None:
    """Publish complete bytes atomically; never overwrite a content object."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=destination.parent, prefix=".pending-")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "nt":
                # Windows rename refuses an existing destination; POSIX rename replaces it.
                os.rename(temporary, destination)
            else:
                os.link(temporary, destination)
        except FileExistsError:
            pass
        if destination.read_bytes() != content:
            raise ValueError(f"Existing immutable object is corrupt: {destination.name}")
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.database = root / "observations.sqlite3"
        with self.connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Unsupported ingestion database schema")
            connection.executescript(SCHEMA)
            connection.execute("PRAGMA user_version = 1")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def ingest(self, response: Response, source: Source) -> dict[str, Any]:
        stamp = utc_text(response.retrieved_at)
        sha = hashlib.sha256(response.content).hexdigest()
        write_once(self.root / "raw" / f"{sha}.csv", response.content)
        snapshot_id = content_id({"resource_id": source.resource_id, "raw_sha256": sha, "transform_version": TRANSFORM_VERSION, "kind": source.kind, "year": source.year})
        metadata = {"requested_url": receipt_url(response.requested_url), "final_url": receipt_url(response.final_url), "status": response.status, "headers": dict(response.headers), "byte_size": len(response.content)}
        # Raw bytes and the retrieval receipt commit before validation starts.
        with self.connect() as connection:
            connection.execute("INSERT INTO captures VALUES (?,?,?,?,?)", (uuid.uuid4().hex, source.resource_id, sha, stamp, canonical_json(metadata)))
            existing = connection.execute("SELECT audit_json FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if existing:
            return json.loads(existing[0])  # type: ignore[no-any-return]
        parsed = None
        try:
            parsed = parse_csv(response.content, source)
            status = "valid"
            audit = {**parsed.audit, "status": status, "snapshot_id": snapshot_id}
        except QuarantineError as exc:
            status = "quarantined"
            audit = {"status": status, "snapshot_id": snapshot_id, "raw_sha256": sha, "resource_id": source.resource_id, "reason": str(exc), "details": exc.details, "transform_version": TRANSFORM_VERSION}
        write_once(self.root / "audits" / f"{snapshot_id}.json", (canonical_json(audit) + "\n").encode())
        with self.connect() as connection:
            # A concurrent ingestion can win; its entire row set is already committed.
            inserted = connection.execute("INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?,?,?)", (snapshot_id, source.resource_id, sha, TRANSFORM_VERSION, stamp, status, canonical_json(audit))).rowcount
            if inserted and parsed is not None:
                connection.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)", [
                    (snapshot_id, row.interval_start.isoformat(), row.interval_end.isoformat(), str(row.settlement_date), row.settlement_period, *row.values, "actual") for row in parsed.rows
                ])
        return audit

    def versions(self, interval_start: datetime, *, as_of: datetime | None = None) -> list[dict[str, Any]]:
        """Retain all revisions; optional captured-time cutoff never rewrites history."""
        sql = """SELECT o.snapshot_id, o.nd_mw, s.first_seen_at, s.raw_sha256
                 FROM observations o JOIN snapshots s USING(snapshot_id)
                 WHERE o.interval_start=?"""
        utc_text(interval_start)
        parameters = [interval_start.astimezone(UTC).isoformat()]
        if as_of is not None:
            sql += " AND s.first_seen_at<=?"
            parameters.append(utc_text(as_of))
        sql += " ORDER BY s.first_seen_at, o.snapshot_id"
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(sql, parameters)]
