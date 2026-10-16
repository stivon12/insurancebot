"""Shared: config, locales, Gemini calls, memo, metrics, progress, HTTP with retries."""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import hashlib
import httpx
import json
import os
import time
from contextlib import contextmanager
from dotenv import load_dotenv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
load_dotenv(ROOT / ".env")
# Locale is a setting, never hardcoded into a stage. Keys are ISO country codes.
LOCALES: dict[str, dict[str, str]] = {
    "US": {"currency": "USD", "ebay_marketplace": "EBAY_US", "name": "United States"},
    "GB": {"currency": "GBP", "ebay_marketplace": "EBAY_GB", "name": "United Kingdom"},
    "DE": {"currency": "EUR", "ebay_marketplace": "EBAY_DE", "name": "Germany"},
    "AU": {"currency": "AUD", "ebay_marketplace": "EBAY_AU", "name": "Australia"},
    "CA": {"currency": "CAD", "ebay_marketplace": "EBAY_CA", "name": "Canada"},
}
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.8-flash")
LITE_MODEL = os.getenv("LITE_MODEL", "gemini-3.1-flash-lite")
# USD per 1M tokens (input, output), from ai.google.dev/gemini-api/docs/pricing on 2026-10-15.
# Used only to report cost per sweep; edit if pricing changes.
MODEL_PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}
def country() -> str:
    return os.getenv("CLAIM_COUNTRY", "US").upper()
def locale(code: str | None = None) -> dict[str, str]:
    code = (code or country()).upper()
    if code not in LOCALES:
        raise ValueError(f"Unsupported country {code}; add it to LOCALES")
    return {"country": code, **LOCALES[code]}
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())
def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)
def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
class Metrics:
    """Latency, call counts, tokens and estimated USD cost per stage."""

    def __init__(self) -> None:
        self.stages: dict[str, dict[str, float]] = {}

    def _stage(self, name: str) -> dict[str, float]:
        return self.stages.setdefault(
            name, {"latency_s": 0.0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "http_calls": 0}
        )

    @contextmanager
    def time(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stage(name)["latency_s"] += round(time.perf_counter() - start, 3)

    def add_llm(self, stage: str, model: str, usage: Any) -> None:
        s = self._stage(stage)
        tin = getattr(usage, "prompt_token_count", 0) or 0
        tout = (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        pin, pout = MODEL_PRICES.get(model, (0.0, 0.0))
        s["llm_calls"] += 1
        s["input_tokens"] += tin
        s["output_tokens"] += tout
        s["cost_usd"] = round(s["cost_usd"] + (tin * pin + tout * pout) / 1e6, 6)

    def add_http(self, stage: str, n: int = 1) -> None:
        self._stage(stage)["http_calls"] += n

    def summary(self) -> dict:
        total = sum(s["cost_usd"] for s in self.stages.values())
        return {"stages": self.stages, "total_cost_usd": round(total, 4)}
_client = None
def gemini():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or None)
    return _client
# Bound concurrent model calls so a sweep does not trip per-minute rate limits.
_llm_sem = asyncio.Semaphore(int(os.getenv("LLM_CONCURRENCY", "32")))
async def gemini_json(
    *,
    model: str,
    parts: list,
    schema: dict,
    metrics: Metrics,
    stage: str,
    system: str | None = None,
    retries: int = 2,
) -> dict | list | None:
    """Call Gemini with a JSON schema; return parsed JSON, or None after retries fail."""
    from google.genai import types

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=schema,
        system_instruction=system,
        temperature=0.0,
        # Gemini 3.x: lowest thinking level the model accepts (gemini-3.8-flash rejects MINIMAL with a 400);
        # 2.5 uses a token budget.
        thinking_config=(
            types.ThinkingConfig(thinking_budget=0)
            if "2.5" in model
            else types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
        ),
        max_output_tokens=8192,
    )
    for attempt in range(retries + 1):
        try:
            async with _llm_sem:
                resp = await gemini().aio.models.generate_content(model=model, contents=parts, config=config)
            metrics.add_llm(stage, model, resp.usage_metadata)
            return json.loads(resp.text)
        except (json.JSONDecodeError, ValueError):
            continue
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
    note_failure(f"{stage}: model reply not valid JSON")
    return None
def image_part(jpeg_bytes: bytes):
    from google.genai import types

    return types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
# ---------- per-sweep memo: lets work done DURING the sweep be reused when the packet is built ----------
_inflight: dict[str, asyncio.Future] = {}
# Lookups that failed (timeout, quota, 5xx) inside a memoised computation. A result built on a failed lookup looks
# like "no match" or "no price" but is not an answer, so it is used for this pass and NOT saved: the next pass retries.
_failures: contextvars.ContextVar[list | None] = contextvars.ContextVar("lca_failures", default=None)
def note_failure(what: str) -> None:
    f = _failures.get()
    if f is not None:
        f.append(what)
def current_failures() -> list[str]:
    return list(_failures.get() or [])
def memo_path(run_dir: Path, kind: str, key: Any, suffix: str = ".json") -> Path:
    import hashlib

    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:20]
    return Path(run_dir) / "cache" / kind / f"{h}{suffix}"
async def memo(run_dir: Path, kind: str, key: Any, fn) -> Any:
    """Return the saved result for (kind, key) in runs/<id>/cache/, else await fn() and save it."""
    p = memo_path(run_dir, kind, key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except ValueError:
            pass
    k = str(p)
    if k in _inflight:
        return await asyncio.shield(_inflight[k])
    fut = asyncio.get_running_loop().create_future()
    _inflight[k] = fut
    parent, failed = _failures.get(), []
    token = _failures.set(failed)
    try:
        r = await fn()
        if r is not None and not failed:
            write_json(p, r)
        fut.set_result(r)
        return r
    except BaseException as e:
        fut.set_exception(e)
        fut.exception()  # mark retrieved when nobody else is waiting
        raise
    finally:
        _failures.reset(token)
        if parent is not None:
            parent.extend(failed)
        _inflight.pop(k, None)


CACHE_DIR = Path(os.getenv("LCA_CACHE_DIR", Path.home() / ".cache" / "library-claim-agent" / "http"))
# host -> (max concurrent requests, minimum seconds between request starts)
HOST_LIMITS = {
    "www.googleapis.com": (16, 0.0),
    "openlibrary.org": (3, 0.34),
    "lx2.loc.gov": (2, 0.5),  # Library of Congress SRU: no published limit, so stay gentle  # ~3 req/s with an identifying User-Agent (Open Library API guidance)
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
                   timeout: float = 10, attempts: int = 3, retry_timeouts: bool = True, as_text: bool = False) -> dict | None:
    """JSON from url (or {"text": body} with as_text=True, e.g. XML), cached on disk unless cache=False."""
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
                data = {"text": r.text} if as_text else r.json()
            except ValueError:
                note_failure(f"{hostname}: reply not JSON")
                return None
            if cache:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(data))
            return data
    note_failure(f"{hostname}: failed after {attempts} tries")
    return None
