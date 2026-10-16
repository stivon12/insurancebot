"""Shared HTTP for catalogue lookups: per-host concurrency + minimum spacing, retries, and a disk cache.

Why: identification was bound by Open Library's rate limit (~1-3 req/s), not by the model. Google Books takes
much more parallelism, so each host gets its own limiter instead of one global semaphore. Catalogue metadata
(titles, ISBNs, physical dimensions) does not change between runs, so successful JSON responses are cached on
disk. Prices are NOT cached here: every price must carry a fresh retrieval date.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import httpx

from .common import note_failure

CACHE_DIR = Path(os.getenv("LCA_CACHE_DIR", Path.home() / ".cache" / "library-claim-agent" / "http"))
# host -> (max concurrent requests, minimum seconds between request starts)
HOST_LIMITS = {
    "www.googleapis.com": (16, 0.0),
    "openlibrary.org": (3, 0.34),  # ~3 req/s with an identifying User-Agent (Open Library API guidance)
}
DEFAULT_LIMIT = (8, 0.0)


class _Host:
    def __init__(self, conc: int, gap: float):
        self.sem = asyncio.Semaphore(conc)
        self.gap = gap
        self.lock = asyncio.Lock()
        self.next_start = 0.0

    async def slot(self):
        if self.gap:
            async with self.lock:
                wait = self.next_start - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self.next_start = time.monotonic() + self.gap


_hosts: dict[tuple[int, str], _Host] = {}
STATS: dict[str, dict[str, float]] = {}  # host -> requests, cache hits, status counts, seconds spent waiting/requesting


def _stat(host: str, key: str, inc: float = 1) -> None:
    STATS.setdefault(host, {})
    STATS[host][key] = STATS[host].get(key, 0) + inc


def _host(name: str) -> _Host:
    # Limiters are bound to the running event loop (asyncio primitives cannot cross loops).
    key = (id(asyncio.get_running_loop()), name)
    if key not in _hosts:
        _hosts[key] = _Host(*HOST_LIMITS.get(name, DEFAULT_LIMIT))
    return _hosts[key]


def _cache_path(url: str, params: dict) -> Path:
    clean = {k: v for k, v in sorted(params.items()) if k != "key"}  # never key the cache (or files) on an API key
    h = hashlib.sha1((url + json.dumps(clean, sort_keys=True)).encode()).hexdigest()
    return CACHE_DIR / h[:2] / f"{h}.json"


async def get_json(client: httpx.AsyncClient, url: str, params: dict, metrics, stage: str, cache: bool = True,
                   timeout: float = 10, attempts: int = 3, retry_timeouts: bool = True) -> dict | None:
    p = _cache_path(url, params)
    hostname = httpx.URL(url).host
    if cache and p.exists():
        try:
            _stat(hostname, "cache_hits")
            return json.loads(p.read_text())
        except ValueError:
            pass
    host = _host(hostname)
    t_queue = time.monotonic()
    async with host.sem:
        _stat(hostname, "queue_s", time.monotonic() - t_queue)
        for attempt in range(attempts):
            await host.slot()
            t_req = time.monotonic()
            try:
                r = await client.get(url, params=params, timeout=timeout)
                metrics.add_http(stage)
                _stat(hostname, "requests")
                _stat(hostname, f"status_{r.status_code}")
                _stat(hostname, "request_s", time.monotonic() - t_req)
            except httpx.TimeoutException:
                _stat(hostname, "timeouts")
                _stat(hostname, "request_s", time.monotonic() - t_req)
                if not retry_timeouts:
                    note_failure(f"{hostname}: timeout")
                    return None  # a slow upstream will not be faster on a second try; don't triple the wait
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            except httpx.HTTPError:
                _stat(hostname, "errors")
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code != 200:
                note_failure(f"{hostname}: HTTP {r.status_code}")
                return None
            try:
                data = r.json()
            except ValueError:
                note_failure(f"{hostname}: reply not JSON")
                return None
            if cache:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(data))
            return data
    note_failure(f"{hostname}: failed after {attempts} tries")
    return None
