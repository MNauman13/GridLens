"""Offered-load campaign. Counts rejections/timeouts instead of dropping them."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from gridlens.benchmark import latency_summary, write_report
from gridlens.data.store import content_id


async def run(url, duration, output):
    clients = [httpx.AsyncClient(base_url=url, timeout=6, trust_env=False) for _ in range(5)]
    cases = (await clients[0].get("/v1/live/catalog")).json()["cases"]
    events = []
    reads = []
    started = time.perf_counter()
    wall = datetime.now(UTC).isoformat()

    async def request(index):
        begin = time.perf_counter()
        client = clients[index % 5]
        event = {
            "index": index,
            "offered_at_ms": (begin - started) * 1000,
            "status": None,
            "state": None,
        }
        try:
            response = await client.post(
                "/v1/forecast-jobs",
                json={"model": "timexer", "target_date": cases[index % len(cases)]["target_date"]},
                headers={"idempotency-key": f"{wall}-{index}"},
            )
            event["status"] = response.status_code
            if response.status_code == 202:
                identifier = response.json()["job_id"]
                while time.perf_counter() - begin < 5:
                    tick = time.perf_counter()
                    reply = await client.get("/v1/jobs/" + identifier)
                    reads.append(
                        {
                            "elapsed_ms": (time.perf_counter() - tick) * 1000,
                            "status": reply.status_code,
                        }
                    )
                    record = reply.json()
                    if record.get("state") in ("succeeded", "failed", "cancelled"):
                        event["state"] = record["state"]
                        break
                    await asyncio.sleep(0.025)
                if event["state"] is None:
                    event["state"] = "deadline_exceeded"
        except Exception as exc:
            event["state"] = type(exc).__name__
        event["elapsed_ms"] = (time.perf_counter() - begin) * 1000
        events.append(event)

    tasks = []
    for index in range(int(duration * 2)):
        deadline = started + index / 2
        await asyncio.sleep(max(0, deadline - time.perf_counter()))
        tasks.append(asyncio.create_task(request(index)))
    await asyncio.gather(*tasks)
    for client in clients:
        await client.aclose()
    admitted = [r for r in events if r["status"] == 202]
    completed = [r for r in admitted if r["state"] == "succeeded"]
    successful_read = [r["elapsed_ms"] for r in reads if r["status"] == 200]
    latency = latency_summary([r["elapsed_ms"] for r in completed]) if completed else None
    result = {
        "schema_version": "gridlens_service_load_v1",
        "started_at": wall,
        "duration_seconds": duration,
        "offered_rate_per_second": 2,
        "clients": 5,
        "url": url,
        "offered": len(events),
        "admitted": len(admitted),
        "completed": len(completed),
        "rejected_429": sum(r["status"] == 429 for r in events),
        "other_http_failures": sum(r["status"] not in (202, 429) for r in events),
        "completion_within_five_seconds": sum(
            r["state"] == "succeeded" and r["elapsed_ms"] <= 5000 for r in admitted
        )
        / max(1, len(admitted)),
        "post_to_result": latency,
        "reads": latency_summary(successful_read) if successful_read else None,
        "failed_reads": sum(r["status"] != 200 for r in reads),
        "raw_events": events,
        "raw_reads": reads,
        "campaign_id": content_id({"wall": wall, "duration": duration, "url": url}),
        "host_note": "Shared development CPU with model-study workload; not a public multi-region availability measurement",
    }
    result["gate_passed"] = bool(
        duration >= 900
        and len(admitted) > 0
        and result["completion_within_five_seconds"] >= 0.99
        and latency
        and latency["p95_ms"] <= 1000
        and result["reads"]["p95_ms"] <= 300
        and not result["other_http_failures"]
        and not result["failed_reads"]
    )
    write_report(output, result)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("raw_")}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8091")
    p.add_argument("--duration", type=int, default=900)
    p.add_argument("--output", type=Path, default=Path("reports/local/service-load.json"))
    args = p.parse_args()
    env = {
        **os.environ,
        "GRIDLENS_LIVE_ASSETS": "artifacts/i1",
        "GRIDLENS_JOB_DB": "reports/local/i6/load-jobs.sqlite3",
    }
    log = open("reports/local/load-api.log", "w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "gridlens.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8091",
        ],
        env=env,
        stdout=log,
        stderr=log,
    )
    try:
        for _ in range(100):
            try:
                if (
                    httpx.get(args.url + "/v1/health/ready", trust_env=False, timeout=1).status_code
                    == 200
                ):
                    break
            except httpx.HTTPError:
                pass
            if process.poll() is not None:
                raise RuntimeError("Load-test backend failed to start")
            time.sleep(0.1)
        asyncio.run(run(args.url, args.duration, args.output))
    finally:
        process.terminate()
        process.wait(timeout=15)
        log.close()


if __name__ == "__main__":
    main()
