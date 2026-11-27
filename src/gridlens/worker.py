"""Resident inference worker shared by local SQLite and PostgreSQL deployment."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from gridlens.data.store import content_id
from gridlens.jobs import JobStore
from gridlens.live import LiveService

LOG = logging.getLogger(__name__)


class DurableLiveService:
    def __init__(
        self,
        root: Path,
        database: str | Path,
        *,
        enable_sdpa: bool = False,
        start_worker: bool = True,
    ) -> None:
        self.engine = LiveService(
            root, Path("reports/local/i6/engine.sqlite3"), enable_sdpa=enable_sdpa
        )
        self.store = JobStore(database)
        self.catalog = self.engine.catalog
        self.runtime_catalog = self.engine.runtime_catalog
        self.capacity = self.store.capacity
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self.work, name="gridlens-durable-worker", daemon=True
        )
        self.store.prune()
        if start_worker:
            self.thread.start()

    def ready(self) -> bool:
        try:
            self.store.stats()
            return not self.stop.is_set() and self.engine.ready()
        except Exception:
            LOG.exception("Readiness database check failed")
            return False

    def submit(
        self,
        owner: str,
        kind: str,
        target_date: str,
        runtime: str = "manual",
        request_key: str = "",
    ) -> str:
        if kind not in self.engine._models or target_date not in self.engine._cases:
            raise ValueError("Unknown model or approved validation date")
        if runtime not in self.runtime_catalog[kind]:
            raise ValueError("Runtime unavailable")
        model = self.catalog["models"][kind]["model_id"]
        payload = {
            "kind": kind,
            "target_date": target_date,
            "runtime": runtime,
            "model_id": model,
            "catalog_id": self.catalog["catalog_id"],
        }
        ticket, _ = self.store.submit(owner, payload, request_key)
        return ticket

    def job(self, identifier: str, owner: str) -> dict[str, Any] | None:
        return self.store.job(identifier, owner)

    def cancel(self, identifier: str, owner: str) -> bool:
        return self.store.cancel(identifier, owner)

    def run_once(self) -> bool:
        task = self.store.claim()
        if task is None:
            return False
        done = threading.Event()

        def heartbeat() -> None:
            while not done.wait(self.store.lease_seconds / 3):
                try:
                    if not self.store.heartbeat(task["id"], task["fence"]):
                        return
                except Exception:
                    LOG.exception("Lease heartbeat failed")

        pulse = threading.Thread(target=heartbeat, daemon=True)
        pulse.start()
        try:
            row = task["payload"]
            # Never execute queued work under silently changed model/catalog identities.
            if (
                row["catalog_id"] != self.catalog["catalog_id"]
                or row["model_id"] != self.catalog["models"][row["kind"]]["model_id"]
            ):
                self.store.fail(task["id"], task["fence"], retryable=False)
                return True
            result = self.engine.compute(
                task["id"], row["kind"], row["target_date"], task["queue_ms"], row["runtime"]
            )
            result["attempt"] = task["attempt"]
            result["fence"] = task["fence"]
            result["result_id"] = content_id(
                {
                    "point": result["point"],
                    "runtime": result["runtime"],
                    "runtime_id": result["runtime_id"],
                }
            )
            self.store.complete(task["id"], task["fence"], result)
        except Exception:
            LOG.exception("Durable inference task failed")
            self.store.fail(task["id"], task["fence"])
        finally:
            done.set()
            pulse.join()
        return True

    def work(self) -> None:
        next_prune = time.monotonic() + 3600
        while not self.stop.is_set():
            try:
                if not self.run_once():
                    self.stop.wait(0.05)
                if time.monotonic() >= next_prune:
                    self.store.prune()
                    next_prune = time.monotonic() + 3600
            except Exception:
                LOG.exception("Worker poll failed")
                self.stop.wait(0.5)

    def close(self) -> None:
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=15)
        self.engine.close()
