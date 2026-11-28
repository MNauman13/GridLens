"""Consistent SQLite job backup and verified restoration to a new path."""

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from gridlens.data.store import write_once


def backup(source: Path, destination: Path) -> dict:
    if not source.is_file():
        raise ValueError("Source database does not exist")
    if destination.exists():
        raise ValueError("Choose a new backup path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with (
            sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as original,
            sqlite3.connect(temporary) as snapshot,
        ):
            original.backup(snapshot)
            if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Database integrity check failed")
            counts = dict(
                snapshot.execute("SELECT state,COUNT(*) FROM queue_tasks GROUP BY state").fetchall()
            )
        raw = temporary.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        write_once(destination, raw)
        record = {
            "schema_version": "gridlens_job_backup_v1",
            "created_at": datetime.now(UTC).isoformat(),
            "file": destination.name,
            "sha256": digest,
            "bytes": len(raw),
            "task_counts": counts,
        }
        write_once(
            destination.with_suffix(destination.suffix + ".json"),
            (json.dumps(record, indent=2) + "\n").encode(),
        )
        return record
    finally:
        temporary.unlink(missing_ok=True)


def restore(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError("Restore requires a new target database path")
    record = json.loads(archive.with_suffix(archive.suffix + ".json").read_text())
    raw = archive.read_bytes()
    if record["sha256"] != hashlib.sha256(raw).hexdigest() or record["bytes"] != len(raw):
        raise ValueError("Backup digest/size mismatch")
    with sqlite3.connect(archive.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Backup database is corrupt")
    write_once(destination, raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=["backup", "restore"])
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    if args.operation == "backup":
        print(json.dumps(backup(args.source, args.destination)))
    else:
        restore(args.source, args.destination)
        print("Verified backup restored to a new database")


if __name__ == "__main__":
    main()
