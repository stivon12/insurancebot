"""Process the sweep while it is still being recorded, so little is left to do after Finish.

Every few seconds the worker looks at the frames saved so far and starts, in the background:
  - spine lists for new shelf keyframes (one model call per frame, memoised per frame),
  - spine reading + catalogue lookup for each book found so far (memoised per spine crop),
  - prices for each identified book (memoised per book and country, with the real retrieval time),
  - non-book item batches that are complete (memoised per batch),
  - room floor/ceiling marks and Depth Anything 3, once the room turn is over.

It writes nothing the packet depends on directly: the post-sweep pipeline (pipeline/run.py) runs unchanged and
simply finds these results in runs/<id>/cache/. Anything the worker had not reached is computed then. Work for a
frame that later stops being a keyframe is wasted, never wrong.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import httpx

from . import identify, items, keyframes, pricing, room, spines
from .common import Metrics, memo_path, read_json, read_jsonl

log = logging.getLogger("worker")

STEP_S = 3.0
SETTLE_MS = 2500  # leave the newest frames alone: their keyframe window (up to 1.5 s) is still filling
ROOM_PHASES = ("room_turn", "room_tilt")
FRAME_CONCURRENCY = 12  # one frame takes ~20 s (two read passes per shelf row); keyframes arrive ~1/s
BOOK_CONCURRENCY = 8


class SweepWorker:
    def __init__(self, run_dir: Path, country: str, emit=None):
        self.run_dir = Path(run_dir)
        self.country = country
        self.emit = emit  # async callable(dict) for running counts
        self.metrics = Metrics()  # shared with the post-sweep pipeline so cost is counted once, in one place
        self.task: asyncio.Task | None = None
        self.jobs: set[asyncio.Task] = set()
        self.started: set[tuple] = set()
        self.counts = {"keyframes": 0, "spine_frames": 0, "books": 0, "identified": 0, "priced": 0, "item_frames": 0, "room": "waiting"}
        self.frame_sem = asyncio.Semaphore(FRAME_CONCURRENCY)
        self.book_sem = asyncio.Semaphore(BOOK_CONCURRENCY)
        self.client: httpx.AsyncClient | None = None
        self.ebay = self.fx = None
        self.items_running = False
        self.stopped = False

    def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop looking for new work; let started jobs finish (the pipeline would await the same results)."""
        self.stopped = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def close(self) -> None:
        await self.stop()
        if self.jobs:
            await asyncio.gather(*self.jobs, return_exceptions=True)
        if self.client:
            await self.client.aclose()

    def _spawn(self, key: tuple, coro) -> None:
        if key in self.started:
            coro.close()
            return
        self.started.add(key)
        t = asyncio.create_task(self._guard(key, coro))
        self.jobs.add(t)
        t.add_done_callback(self.jobs.discard)

    async def _guard(self, key: tuple, coro) -> None:
        try:
            await coro
        except Exception as e:  # never break the sweep; the pipeline retries anything that failed here
            log.warning("worker job %s failed: %s", key[0], e)
            self.started.discard(key)

    async def _loop(self) -> None:
        self.client = httpx.AsyncClient(headers={"User-Agent": identify.UA}, follow_redirects=True, timeout=12)
        self.ebay, self.fx = pricing.Ebay(self.client, self.metrics), pricing.FX(self.client)
        last_emit = 0.0
        while not self.stopped:
            await asyncio.sleep(STEP_S)
            try:
                await self.step()
            except Exception as e:
                log.warning("worker step failed: %s", e)
            if self.emit and time.monotonic() - last_emit > 2:
                last_emit = time.monotonic()
                await self.emit({"type": "processed", **self.counts})

    async def step(self) -> None:
        frames = read_jsonl(self.run_dir / "frames.jsonl")
        if not frames:
            return
        newest = max(f["t_ms"] for f in frames)
        settled = [f for f in frames if f["t_ms"] < newest - SETTLE_MS]
        kfs = await asyncio.to_thread(keyframes.select, settled, self.run_dir)
        self.counts["keyframes"] = sum(len(v) for v in kfs.values())

        # Spine lists per shelf keyframe.
        shelf = spines.frames_for(kfs)
        for kf in shelf:
            self._spawn(("spines", kf["file"]), self._spine_frame(kf))
        done = [kf for kf in shelf if memo_path(self.run_dir, "spine_frames", {"file": kf["file"], "prompt": spines.PROMPT_VERSION}).exists()]
        self.counts["spine_frames"] = len(done)

        # Books seen so far -> read, identify, price.
        if done:
            per_frame = [await spines._frame(self.run_dir, kf, self.metrics) for kf in done]  # all memo hits
            obs = [o for pf in per_frame for o in pf["observations"]]
            segs = await asyncio.to_thread(spines.chain_segments, self.run_dir, done)
            books = spines.assign_shelves(spines.merge(obs, segs))
            self.counts["books"] = len(books)
            for b in books:
                best = b["best"]
                self._spawn(("book", best["file"], tuple(round(v) for v in best["box_px"])), self._book(b))

        # Non-book items: only complete batches, one pass at a time (each batch sees the list so far).
        if not self.items_running:
            self.items_running = True
            self._spawn(("items", self.counts["keyframes"]), self._items(kfs))

        # Room: as soon as the sweep has moved past the room turn.
        latest_phase = max(frames, key=lambda f: f["t_ms"]).get("phase")
        if any(kfs.get(p) for p in ROOM_PHASES) and latest_phase not in ROOM_PHASES:
            self._spawn(("room",), self._room(kfs))

    async def _spine_frame(self, kf: dict) -> None:
        async with self.frame_sem:
            await spines._frame(self.run_dir, kf, self.metrics)

    async def _book(self, b: dict) -> None:
        async with self.book_sem:
            _, v = await identify._one(self.run_dir, self.client, b, self.metrics)
        if v.get("status") != "identified":
            return
        self.counts["identified"] += 1
        m = v["match"]
        inp = {
            "id": b["id"], "title": m["title"], "author": ", ".join(m.get("authors", [])), "isbn": m.get("isbn", ""),
            "isbns": [i for i in [m.get("isbn")] + [a.get("isbn") for a in v.get("alternates", [])] if i],
            "publisher": m.get("publisher", ""), "published": m.get("published", ""), "read": v.get("read", {}),
            "override_first_edition": None,
        }
        await pricing.memo_book_price(self.run_dir, self.ebay, self.fx, self.client, inp, self.country, self.metrics)
        self.counts["priced"] += 1

    async def _items(self, kfs: dict) -> None:
        try:
            await items.run(self.run_dir, kfs, self.metrics, complete_batches_only=True)
            state = read_json(self.run_dir / "cache" / "items_state.json", None) or {}
            self.counts["item_frames"] = len(state.get("done", []))
        finally:
            self.items_running = False

    async def _room(self, kfs: dict) -> None:
        self.counts["room"] = "running"
        await room.prewarm(self.run_dir, kfs, self.metrics)
        self.counts["room"] = "done"
