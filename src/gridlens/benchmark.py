"""CPU benchmark records with explicit scope and no hidden result caching."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from gridlens.data.store import content_id, write_once


def quantile(values: Sequence[float], probability: float) -> float:
    """Linear interpolation between ordered observations, including endpoints."""
    if not values or not 0 <= probability <= 1 or any(
        not math.isfinite(value) or value < 0 for value in values
    ):
        raise ValueError("Require finite nonnegative samples and probability in [0,1]")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def latency_summary(values: Sequence[float]) -> dict[str, int | float]:
    return {
        "calls": len(values),
        "p50_ms": quantile(values, 0.5),
        "p95_ms": quantile(values, 0.95),
        "p99_ms": quantile(values, 0.99),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def environment_record(root: Path) -> dict[str, Any]:
    """Record observable capacity, separating host hardware and container limits."""
    packages: dict[str, str | None] = {}
    for name in ("torch", "numpy", "pydantic", "fastapi", "tzdata"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        memory: int | None = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        memory = None
    cpu = platform.processor() or None
    if cpu is None and (info := read_optional(Path("/proc/cpuinfo"))):
        cpu = next((s.partition(":")[2].strip() for s in info.splitlines() if s.startswith("model name")), None)
    try:
        affinity: int | None = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unavailable", None
    return {
        "python": platform.python_version(), "system": platform.system(),
        "machine": platform.machine(), "cpu_model": cpu,
        "host_logical_cpus": os.cpu_count(), "affinity_cpus": affinity,
        "host_physical_memory_bytes": memory,
        "cgroup_v2_memory_max": read_optional(Path("/sys/fs/cgroup/memory.max")),
        "cgroup_v2_cpu_max": read_optional(Path("/sys/fs/cgroup/cpu.max")),
        "packages": packages, "source_commit": commit, "working_tree_dirty": dirty,
        "device": "cpu", "gpu_metrics": None,
        "gpu_metrics_reason": "CPU-only baseline; no GPU performance claim",
    }


def memory_record() -> dict[str, Any]:
    status = read_optional(Path("/proc/self/status"))
    fields = {} if status is None else dict(
        row.split(":", 1) for row in status.splitlines() if ":" in row
    )
    return {
        "rss_bytes": int(fields["VmRSS"].split()[0]) * 1024 if "VmRSS" in fields else None,
        "process_lifetime_peak_rss_bytes": int(fields["VmHWM"].split()[0]) * 1024 if "VmHWM" in fields else None,
        "peak_scope": "process lifetime, includes dataset construction/training; not isolated model memory",
    }


def timed_calls(
    call: Callable[[int], object], *, cases: int, warmup: int, repetitions: int, calls: int,
) -> tuple[dict[str, Any], list[dict[str, int | float]]]:
    """Sequential batch-one calls; no warmup observations enter the summary."""
    if min(cases, repetitions, calls) < 1 or warmup < 0:
        raise ValueError("Positive case/repetition/call counts and nonnegative warmup required")
    for index in range(warmup):
        call(index % cases)
    rows: list[dict[str, int | float]] = []
    per_repeat: list[dict[str, int | float]] = []
    for repeat in range(repetitions):
        observed = []
        for index in range(calls):
            case = (repeat * calls + index) % cases
            start = time.perf_counter_ns()
            call(case)
            duration = (time.perf_counter_ns() - start) / 1_000_000
            observed.append(duration)
            rows.append({"repeat": repeat, "case_index": case, "elapsed_ms": duration})
        per_repeat.append(latency_summary(observed))
    return {
        **latency_summary([float(row["elapsed_ms"]) for row in rows]),
        "warmup_calls_excluded": warmup, "repetitions": per_repeat,
        "batch_size": 1, "prediction_result_cache": "disabled",
        "sequential_capacity_forecasts_per_second": (
            1000 / statistics.mean(float(row["elapsed_ms"]) for row in rows)
            if any(float(row["elapsed_ms"]) > 0 for row in rows) else None
        ),
        "capacity_scope": "sequential model calls, not concurrent HTTP service throughput",
    }, rows


def validate_development_dates(dates: Sequence[str], allowed: Sequence[str]) -> None:
    if not dates or len(set(dates)) != len(dates) or not set(dates).issubset(allowed):
        raise ValueError("Benchmark origins must be distinct members of frozen point_validation")


def select_origins(dates: Sequence[str], count: int = 32) -> list[str]:
    if count < 2 or len(dates) < count:
        raise ValueError("Not enough origins for benchmark corpus")
    result = [dates[index * (len(dates) - 1) // (count - 1)] for index in range(count)]
    validate_development_dates(result, dates)
    return result


def archive_files(root: Path, members: Sequence[Path], destination: Path) -> dict[str, Any]:
    """Archive only explicit files; never follow symlinks or include unlisted data."""
    import io
    import zipfile

    from gridlens.artifacts import sha256

    base = root.resolve()
    if not members:
        raise ValueError("Archive requires an explicit nonempty file list")
    names: set[str] = set()
    buffer = io.BytesIO()
    records = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            path = member.resolve()
            if member.is_symlink() or not path.is_relative_to(base) or not path.is_file():
                raise ValueError("Archive members must be regular files inside the artifact root")
            name = path.relative_to(base).as_posix()
            if name in names or path == destination.resolve():
                raise ValueError("Duplicate or self-referencing archive member")
            names.add(name)
            raw = path.read_bytes()
            archive.writestr(name, raw)
            records.append({"path": name, "sha256": sha256(path), "size_bytes": len(raw)})
    write_once(destination, buffer.getvalue())
    return {"sha256": sha256(destination), "size_bytes": destination.stat().st_size, "files": records}


def write_report(path: Path, value: dict[str, Any]) -> None:
    payload = {**value, "report_id": content_id(value)}
    write_once(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
