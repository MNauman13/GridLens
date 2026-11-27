"""One real public compute probe. TLS verification stays enabled."""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def probe(url):
    started = time.perf_counter()
    record = {
        "started_at": datetime.now(UTC).isoformat(),
        "url": url,
        "checks": [],
        "passed": False,
    }

    def call(path, payload=None):
        request = urllib.request.Request(
            url.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload else None,
            headers={"Content-Type": "application/json", "User-Agent": "GridLens-release-smoke/1"},
        )
        tick = time.perf_counter()
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(5_000_001)
            if len(body) > 5_000_000:
                raise ValueError("Unexpected response size")
            record["checks"].append(
                {
                    "path": path,
                    "status": response.status,
                    "elapsed_ms": (time.perf_counter() - tick) * 1000,
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
            return (
                json.loads(body)
                if response.headers.get("content-type", "").startswith("application/json")
                else body
            )

    try:
        call("/live")
        call("/experiments")
        catalog = call("/api/live/catalog")
        day = catalog["cases"][0]["target_date"]
        for model in ("ridge", "timexer"):
            result = call("/api/compute", {"model": model, "target_date": day})
            if (
                result["execution_location"] != "web_server"
                or result["result_cache_hit"]
                or len(result["prediction_mw"]) != 48
                or result["model_id"] != catalog["models"][model]["model_id"]
            ):
                raise ValueError("Fresh public compute contract failed")
        record["passed"] = True
    except Exception as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)[:400]
    record["total_ms"] = (time.perf_counter() - started) * 1000
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="https://gridlens-mnaumansiddiqui06-6472s-projects.vercel.app")
    p.add_argument("--output", type=Path, default=Path("reports/local/public-smoke.json"))
    args = p.parse_args()
    if not args.url.startswith("https://"):
        raise ValueError("Public HTTPS URL required")
    result = probe(args.url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
