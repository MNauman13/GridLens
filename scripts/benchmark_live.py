"""Measure real I1 worker service time; excludes HTTP/network/browser rendering."""
from __future__ import annotations

import argparse
import json
import platform
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from gridlens.live import LiveService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/i1'))
    parser.add_argument('--output', type=Path, default=Path('reports/local/i1/live_worker.json'))
    args = parser.parse_args()
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory() as directory:
        service = LiveService(args.root, Path(directory) / 'jobs.sqlite3')
        load_ms = (time.perf_counter_ns() - started) / 1e6
        samples = []
        try:
            for kind in ('ridge', 'timexer'):
                for entry in service.catalog['cases']:
                    for repetition in range(3):
                        tick = time.perf_counter_ns()
                        owner = f'{kind}:{entry["target_date"]}'
                        identifier = service.submit(owner, kind, entry['target_date'])
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            result = service.job(identifier, owner)
                            if result and result['state'] in {'succeeded', 'failed'}:
                                break
                            time.sleep(0.001)
                        else:
                            raise RuntimeError('Worker deadline exceeded')
                        if result['state'] != 'succeeded':
                            raise RuntimeError('Inference did not succeed')
                        payload = result['result']
                        samples.append({'model': kind, 'target_date': entry['target_date'], 'repetition': repetition, 'queue_ms': payload['queue_ms'], 'predict_ms': payload['predict_ms'], 'submit_to_persisted_ms': (time.perf_counter_ns()-tick)/1e6, 'result_cache_hit': payload['result_cache_hit']})
        finally:
            service.close()
    summaries = {}
    for kind in ('ridge', 'timexer'):
        selected = [row for row in samples if row['model'] == kind]
        summaries[kind] = {metric: {'p50': float(np.percentile([r[metric] for r in selected], 50)), 'p95': float(np.percentile([r[metric] for r in selected], 95))} for metric in ('queue_ms', 'predict_ms', 'submit_to_persisted_ms')}
    report = {'scope': 'local single-worker CPU; three runs per validation origin; first predictions included; excludes HTTP/network/browser', 'catalog_id': service.catalog['catalog_id'], 'platform': platform.platform(), 'python': platform.python_version(), 'torch': torch.__version__, 'threads': torch.get_num_threads(), 'verified_startup_ms': load_ms, 'process_peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'successes': len(samples), 'failures': 0, 'summary': summaries, 'samples': samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}, indent=2))


if __name__ == '__main__':
    main()
