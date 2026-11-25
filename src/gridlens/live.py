"""Bounded single-worker reference inference; durable SQLite records for I1."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gridlens.artifacts import ArtifactManifest, safe_member, sha256, verify_manifest
from gridlens.contracts import InputWindow
from gridlens.data.dataset import FeatureSnapshot
from gridlens.data.store import content_id

LOG = logging.getLogger(__name__)


class AdmissionError(ValueError):
    """The caller has exceeded a declared public compute limit."""


def owner_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class LiveService:
    """One resident model worker. Multi-process leases belong to I6, not this preview."""

    def __init__(self, root: Path, database: Path, *, capacity: int = 8) -> None:
        if capacity < 1:
            raise ValueError("Positive worker capacity required")
        manifest = ArtifactManifest.model_validate_json((root / "manifest.json").read_text())
        if manifest.kind != "gridlens_i1_demo_v1":
            raise ValueError("Unexpected live artifact kind")
        verify_manifest(root, manifest)
        catalog = json.loads((root / "client/catalog.json").read_text())
        if catalog["catalog_id"] != content_id({k: v for k, v in catalog.items() if k != "catalog_id"}):
            raise ValueError("Live catalog integrity mismatch")
        if catalog["evaluation_role"] != "point_validation" or catalog["evidence_mode"] != "retrospective_revised":
            raise ValueError("I1 exposes approved validation replay only")
        if set(catalog["models"]) != {"ridge", "timexer"}:
            raise ValueError("Unexpected live model catalog")
        self.catalog: dict[str, Any] = catalog
        self.root, self.database, self.capacity = root, database, capacity
        self._cases: dict[str, dict[str, Any]] = {}
        self._models: dict[str, Any] = {}
        import torch

        from gridlens.models.baselines import Baseline
        from gridlens.models.bundle import NeuralBundle

        torch.set_num_threads(2)

        for kind, row in catalog["models"].items():
            folder = safe_member(root, row["bundle"])
            model: Any = Baseline.load_bundle(folder) if kind == "ridge" else NeuralBundle.load_bundle(folder)
            identity = model.model_version if kind == "ridge" else model.metadata["identity"]
            if identity != row["model_id"]:
                raise ValueError("Live model identity mismatch")
            self._models[kind] = model
        for row in catalog["cases"]:
            if not row["target_date"].startswith("2022-") or row["target_date"] in self._cases:
                raise ValueError("Duplicate or non-development demo origin")
            path = safe_member(root / "client", row["asset"])
            if sha256(path) != row["sha256"]:
                raise ValueError("Live feature digest mismatch")
            case = json.loads(path.read_text())
            inputs = InputWindow.model_validate(case["inputs"])
            if inputs.target_start.date().isoformat() != row["target_date"] or inputs.feature_snapshot_id != row["feature_snapshot_id"]:
                raise ValueError("Live origin and feature identity disagree")
            if len(case["history"]) != 336 or any(len(item) != 15 for item in case["history"]) or len(case["target_calendar"]) != 48 or any(len(item) != 6 for item in case["target_calendar"]):
                raise ValueError("Incomplete live feature snapshot")
            if tuple(item[0] for item in case["history"]) != inputs.demand_mw:
                raise ValueError("Live demand history disagrees with input contract")
            if len(case["observed_mw"]) != 48 or not all(math.isfinite(value) for row_values in (*case["history"], *case["target_calendar"], case["observed_mw"]) for value in row_values):
                raise ValueError("Nonfinite or incomplete live observations/features")
            self._cases[row["target_date"]] = case
        if len(self._cases) < 2:
            raise ValueError("At least two real development origins required")
        database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS live_jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, model TEXT NOT NULL,
                target_date TEXT NOT NULL, state TEXT NOT NULL,
                created_ns INTEGER NOT NULL, result_json TEXT, error TEXT)""")
            connection.execute("CREATE INDEX IF NOT EXISTS live_job_owner_time ON live_jobs(owner,created_ns)")
            connection.execute("UPDATE live_jobs SET state='failed',error='worker_restarted' WHERE state IN ('queued','running')")
            connection.execute("DELETE FROM live_jobs WHERE created_ns < ?", (time.time_ns() - 86_400_000_000_000,))
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gridlens-reference")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def ready(self) -> bool:
        try:
            with self.connect() as connection:
                connection.execute("SELECT 1 FROM live_jobs LIMIT 1").fetchone()
            return not self._closed
        except sqlite3.Error:
            return False

    def submit(self, owner: str, kind: str, target_date: str) -> str:
        if self._closed:
            raise RuntimeError("Reference worker is stopped")
        if kind not in self._models or target_date not in self._cases:
            raise ValueError("Unknown model or target date; locked-test dates are not available")
        identifier = str(uuid.uuid4())
        now = time.time_ns()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pending = connection.execute("SELECT COUNT(*) FROM live_jobs WHERE state IN ('queued','running')").fetchone()[0]
            active = connection.execute("SELECT COUNT(*) FROM live_jobs WHERE owner=? AND state IN ('queued','running')", (owner,)).fetchone()[0]
            recent = connection.execute("SELECT COUNT(*) FROM live_jobs WHERE owner=? AND created_ns>?", (owner, now - 60_000_000_000)).fetchone()[0]
            global_recent = connection.execute("SELECT COUNT(*) FROM live_jobs WHERE created_ns>?", (now - 60_000_000_000,)).fetchone()[0]
            if pending >= self.capacity or active >= 1 or recent >= 60 or global_recent >= 240:
                raise AdmissionError("Reference compute capacity reached; try again shortly")
            connection.execute("INSERT INTO live_jobs VALUES (?,?,?,?,?,?,NULL,NULL)", (identifier, owner, kind, target_date, "queued", now))
        try:
            self._executor.submit(self._execute, identifier, kind, target_date, now)
        except RuntimeError:
            with self.connect() as connection:
                connection.execute("UPDATE live_jobs SET state='failed',error='worker_unavailable' WHERE id=?", (identifier,))
            raise
        return identifier

    def _execute(self, identifier: str, kind: str, target_date: str, created_ns: int) -> None:
        try:
            with self.connect() as connection:
                connection.execute("UPDATE live_jobs SET state='running' WHERE id=? AND state='queued'", (identifier,))
            queue_ms = max(0.0, (time.time_ns() - created_ns) / 1e6)
            case = self._cases[target_date]
            inputs = InputWindow.model_validate(case["inputs"])
            snapshot = FeatureSnapshot(inputs, tuple(tuple(item) for item in case["history"]), tuple(tuple(item) for item in case["target_calendar"]))
            started = time.perf_counter_ns()
            point = self._models[kind].predict(inputs if kind == "ridge" else snapshot)
            duration = (time.perf_counter_ns() - started) / 1e6
            payload = {
                "execution_id": identifier, "served_from": "fresh_inference", "execution_location": "python_server",
                "executed_at": datetime.now(UTC).isoformat(), "model": kind, "model_id": point.model_version,
                "runtime": self.catalog["models"][kind]["reference_runtime"], "point": point.model_dump(mode="json"),
                "inputs": inputs.model_dump(mode="json"), "observed_mw": case["observed_mw"],
                "dataset_manifest_id": self.catalog["dataset_manifest_id"], "evidence_mode": "retrospective_revised",
                "evaluation_role": "point_validation", "queue_ms": queue_ms, "predict_ms": duration,
                "result_cache_hit": False,
            }
            payload["result_id"] = content_id({"point": payload["point"], "runtime": payload["runtime"]})
            with self.connect() as connection:
                connection.execute("UPDATE live_jobs SET state='succeeded',result_json=? WHERE id=? AND state='running'", (json.dumps(payload, allow_nan=False), identifier))
        except Exception:
            LOG.exception("Reference inference failed for execution %s", identifier)
            with self.connect() as connection:
                connection.execute("UPDATE live_jobs SET state='failed',error='inference_failed' WHERE id=?", (identifier,))

    def job(self, identifier: str, owner: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM live_jobs WHERE id=? AND owner=?", (identifier, owner)).fetchone()
        if row is None:
            return None
        return {"job_id": row["id"], "state": row["state"], "error": row["error"], "result": json.loads(row["result_json"]) if row["result_json"] else None}

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=True)
