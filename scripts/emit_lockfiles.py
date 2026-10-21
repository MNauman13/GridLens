"""Expose generated, non-secret dependency locks for review after bootstrap CI.

Only these two allowlisted files are emitted. This never reads environment files.
Remove the bootstrap mode once reviewed lockfiles have been committed.
"""
import base64
import json
import zlib
from pathlib import Path

for filename in ['uv.lock', 'package-lock.json']:
    payload = base64.b64encode(zlib.compress(Path(filename).read_bytes(), level=9)).decode()
    chunks = [payload[index:index + 6000] for index in range(0, len(payload), 6000)]
    for index, chunk in enumerate(chunks):
        print('GRIDLENS_LOCK ' + json.dumps({'path': filename, 'index': index, 'count': len(chunks), 'data': chunk}))
