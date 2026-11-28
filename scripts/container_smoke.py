"""Real Compose compute, restart and ownership check using the standard library."""

import http.cookiejar
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def main():
    jar = http.cookiejar.CookieJar()
    client = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar)
    )
    base = "http://127.0.0.1:8000"

    def call(path, payload=None):
        request = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode() if payload else None,
            headers={"Content-Type": "application/json", "Idempotency-Key": "container-once"},
        )
        with client.open(request, timeout=10) as reply:
            return json.loads(reply.read())

    if call("/v1/health/ready")["status"] != "ready":
        raise ValueError("Container not ready")
    accepted = call("/v1/forecast-jobs", {"model": "timexer", "target_date": "2022-01-01"})
    for _ in range(100):
        result = call("/v1/jobs/" + accepted["job_id"])
        if result["state"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    if (
        result["state"] != "succeeded"
        or len(result["result"]["point"]["prediction_mw"]) != 48
        or result["result"]["result_cache_hit"]
    ):
        raise ValueError("Container fresh inference failed")
    stranger = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        stranger.open(base + "/v1/jobs/" + accepted["job_id"])
        raise ValueError("Cross-session result exposure")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    subprocess.run(["docker", "compose", "restart", "api"], check=True)
    for _ in range(100):
        try:
            if call("/v1/health/ready")["status"] == "ready":
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    restored = call("/v1/jobs/" + accepted["job_id"])
    if restored != result:
        raise ValueError("Persisted result changed after container restart")
    duplicate = call("/v1/forecast-jobs", {"model": "timexer", "target_date": "2022-01-01"})
    if duplicate["job_id"] != accepted["job_id"]:
        raise ValueError("Idempotency was lost after restart")
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "fresh_compute": True,
        "owner_isolation": True,
        "restart_result_identical": True,
        "idempotency_after_restart": True,
        "model_id": result["result"]["model_id"],
        "runtime": result["result"]["runtime"],
    }
    destination = Path("reports/local/container-smoke.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
