"""Compute a fresh, allowlisted 48-slot forecast using verified cached weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gridlens.inference.cached import CacheDiagnostics, adapt_cached_bundle, cache_runtime_record
from scripts.benchmark_sdpa import ROOT, load_inputs, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports/local/i0-public")
    parser.add_argument("--assets", type=Path, default=ROOT / "artifacts/i1")
    parser.add_argument(
        "--variant", choices=("cross_kv", "full_kv", "full_kv_sdpa"), default="full_kv"
    )
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    catalog, snapshots, _ = load_inputs(args.assets)
    dates = [row["target_date"] for row in catalog["cases"]]
    if args.target_date not in dates:
        parser.error("Use an allowlisted validation date; locked-test dates are unavailable")
    reference, _ = load_model(args.baseline, "transformer", "manual")
    mode = "cross_kv" if args.variant == "cross_kv" else "full_kv"
    backend = "sdpa" if args.variant == "full_kv_sdpa" else "manual"
    bundle = adapt_cached_bundle(reference, mode, backend)
    snapshot = snapshots[dates.index(args.target_date)]
    point = bundle.predict(snapshot)
    result = {
        "point": point.model_dump(mode="json"),
        "runtime": cache_runtime_record(mode, backend),
        "target_date": args.target_date,
        "served_from": "fresh_inference",
        "result_cache_hit": False,
    }
    if args.diagnostics:
        from dataclasses import asdict

        import torch

        stats = CacheDiagnostics()
        h = torch.tensor([bundle.scaling.transform(snapshot.history)], dtype=torch.float32)
        c = torch.tensor([snapshot.target_calendar], dtype=torch.float32)
        bundle.model(h, c, diagnostics=stats)
        result["separate_diagnostic_execution"] = asdict(stats)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
