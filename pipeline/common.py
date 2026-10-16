"""Shared helpers: paths, locale settings, JSON IO, per-stage metrics and a Gemini JSON helper."""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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


# Weights are rough shares of post-sweep time, used only to move one overall progress bar smoothly.
PROGRESS_STAGES = [
    ("keyframes", "Selecting sharp frames", 4),
    ("spines", "Counting book spines", 25),
    ("identify", "Reading titles", 30),
    ("items", "Listing other contents", 10),
    ("room", "Measuring the room", 10),
    ("pricing", "Looking up local prices", 16),
    ("assemble", "Building the packet", 5),
]


class Progress:
    """Per-stage done/total counts, pushed to `emit` (an async callable taking a snapshot dict) at most 4 times a
    second, and always on a stage start or finish. Call from the event loop only (not from worker threads)."""

    def __init__(self, emit=None) -> None:
        self.emit = emit
        self.stages = {k: {"stage": k, "label": label, "weight": w, "done": 0, "total": 0, "state": "waiting"} for k, label, w in PROGRESS_STAGES}
        self._last = 0.0
        self._tasks: set = set()

    def snapshot(self) -> dict:
        tot = sum(s["weight"] for s in self.stages.values())
        got = 0.0
        for s in self.stages.values():
            if s["state"] == "done":
                got += s["weight"]
            elif s["state"] == "running" and s["total"]:
                got += s["weight"] * min(1.0, s["done"] / s["total"])
        return {"overall": round(got / tot, 3), "stages": [{k: v for k, v in s.items() if k != "weight"} for s in self.stages.values()]}

    def _push(self, force: bool = False) -> None:
        now = time.monotonic()
        if not self.emit or (not force and now - self._last < 0.25):
            return
        self._last = now
        t = asyncio.get_running_loop().create_task(self.emit(self.snapshot()))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def start(self, stage: str, total: int = 0) -> None:
        self.stages[stage].update(state="running", total=total, done=0)
        self._push(force=True)

    def tick(self, stage: str, n: int = 1) -> None:
        self.stages[stage]["done"] += n
        self._push()

    def finish(self, stage: str, note: str = "") -> None:
        s = self.stages[stage]
        s.update(state="done", done=max(s["done"], s["total"]))
        if note:
            s["note"] = note
        self._push(force=True)

    async def flush(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


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
    """Return the saved result for (kind, key) in runs/<id>/cache/, else await fn() and save it.

    Only for work that depends on nothing but its key (one frame's spine list, one spine crop's reading, one
    book's price). Concurrent callers of the same key share one computation, so the background worker and the
    post-sweep pipeline never pay twice. A result of None, or one computed while a lookup failed (note_failure),
    is not saved (it is retried next time)."""
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
