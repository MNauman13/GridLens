"""Bounded HTTP retrieval with injectable transport, sleep, and retry jitter."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunsplit
from urllib.request import Request, urlopen

MAX_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Response:
    content: bytes
    requested_url: str
    final_url: str
    status: int
    headers: Mapping[str, str]
    retrieved_at: datetime


class DownloadError(RuntimeError):
    pass


def official_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "api.neso.energy" or parsed.port not in (None, 443) or parsed.username or parsed.password:
        raise DownloadError("Expected an HTTPS api.neso.energy URL")
    return url


def final_download_url(requested: str, final: str) -> str:
    """NESO currently redirects CSV downloads to this resource-specific public bucket."""
    official_url(requested)
    parsed = urlparse(final)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise DownloadError("Unexpected CSV redirect")
    if parsed.hostname == "api.neso.energy" and parsed.path == urlparse(requested).path:
        return final
    parts = urlparse(requested).path.split("/")
    if "resource" in parts:
        resource_id = parts[parts.index("resource") + 1]
        expected = f"/dx-national-grid/national-grid/resources/{resource_id}/{parts[-1]}"
        if parsed.hostname == "83025b28472d6aa2bf5ae59f3724aa78.eu.r2.cloudflarestorage.com" and parsed.path == expected:
            return final
    raise DownloadError("Unexpected CSV redirect host or resource path")


def receipt_url(url: str) -> str:
    """Keep origin/path provenance without persisting expiring signed query credentials."""
    parsed = urlparse(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def request_once(url: str, timeout: float) -> Response:
    request = Request(official_url(url), headers={"User-Agent": "GridLens/0.1 (research data audit)", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=timeout) as result:
        final_url = final_download_url(url, result.geturl())
        body = result.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise DownloadError("Response exceeds 16 MiB limit")
        if result.status != 200:
            raise DownloadError(f"Unexpected HTTP status {result.status}")
        headers = {key.lower(): value for key, value in result.headers.items() if key.lower() in ("content-type", "etag", "last-modified", "content-length")}
        declared = headers.get("content-length")
        if declared is not None and int(declared) != len(body):
            raise DownloadError("Truncated or inconsistent Content-Length")
        return Response(body, url, final_url, result.status, headers, datetime.now(UTC))


def download(
    url: str, *, attempts: int = 3, timeout: float = 20,
    request: Callable[[str, float], Response] = request_once,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Response:
    official_url(url)
    if not 1 <= attempts <= 5 or not 0 < timeout <= 60:
        raise ValueError("Require 1..5 attempts and a timeout in (0, 60]")
    for attempt in range(attempts):
        try:
            return request(url, timeout)
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise DownloadError(f"HTTP {exc.code}; retrieval stopped") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = 2.0 ** attempt + jitter()
            if retry_after is not None:
                try:
                    minimum = float(retry_after)
                except ValueError:
                    try:
                        minimum = (parsedate_to_datetime(retry_after) - now()).total_seconds()
                    except (ValueError, TypeError):
                        minimum = 0
                delay = max(delay, minimum)
            if delay > 60:
                raise DownloadError("Retry-After exceeds retry budget; defer retrieval") from exc
            sleep(delay)
        except (URLError, TimeoutError, ConnectionError) as exc:
            if attempt == attempts - 1:
                raise DownloadError("Transport retry budget exhausted") from exc
            sleep(2.0 ** attempt + jitter())
    raise AssertionError("Unreachable retry state")
