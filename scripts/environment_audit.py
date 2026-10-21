"""Record reproducibility facts without collecting host names or user paths."""
import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def version(command):
    if not shutil.which(command[0]):
        return {'status': 'not_found'}
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {'status': 'available' if run.returncode == 0 else 'error', 'value': run.stdout.strip()[:200]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'status': 'error', 'error_type': type(error).__name__}


def audit():
    packages = {}
    for name in ['torch', 'numpy', 'pandas', 'pytest', 'pydantic', 'fastapi', 'tzdata']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        ZoneInfo('Europe/London')
        timezone_status = 'available'
    except ZoneInfoNotFoundError:
        timezone_status = 'missing_install_tzdata_before_source_time_conversion'
    memory_kib = ctypes.c_ulonglong()
    memory = None
    if platform.system() == 'Windows':
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(memory_kib)):
            memory = memory_kib.value * 1024
    gpu = version(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
    return {
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'system': platform.system(),
        'architecture': platform.machine(), 'logical_cpu_count': os.cpu_count(),
        'cpu_identifier': os.environ.get('PROCESSOR_IDENTIFIER') if platform.system() == 'Windows' else platform.processor(),
        'physical_memory_bytes': memory,
        'nvidia_probe': gpu,
        'gpu_training_ready': False,
        'gpu_note': 'GPU runtime and tensor execution must be verified in Phase 1; device visibility alone is insufficient.',
        'packages': packages, 'timezone_database': timezone_status,
        'node': version(['node', '--version']), 'uv': version(['uv', '--version']),
        'git': version(['git', '--version']),
        'runtime_policy': 'Python 3.12 target; resolve and lock ML dependencies in Phase 1; CPU first.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    content = json.dumps(audit(), indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')
    print(content)


if __name__ == '__main__':
    main()
