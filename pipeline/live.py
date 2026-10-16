"""Per-sweep engine: every frame and every agent log is processed as it arrives; Finish only assembles the packet."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from . import assemble, identify, items, keyframes, pricing, report, room, scale, spine_boxes, spines
from .common import Metrics, locale, read_json, read_jsonl, write_json

log = logging.getLogger("live")
SECOND_MARKET_BOOKS = 10
MERGE_EVERY_S = 2.0
FINISH_WAIT_S = 60


class Heading:
    """Phone heading (degrees, clockwise) from the gyro rate about gravity, integrated as samples arrive."""

    def __init__(self) -> None:
        self.up = None
        self.yaw = 0.0
        self.last_t = None
        self.track: list[tuple[float, float]] = []

    def add(self, samples: list[dict]) -> None:
        for s in samples:
            a = np.array([s["ax"], s["ay"], s["az"]], float)
            self.up = a if self.up is None else 0.9 * self.up + 0.1 * a
            u = self.up / max(np.linalg.norm(self.up), 1e-6)
            rate = float(np.dot(np.radians([s["rb"], s["rg"], s["ra"]]), u))  # counter-clockwise about up
            if self.last_t is not None:
                self.yaw -= math.degrees(rate * min(0.2, (s["t_ms"] - self.last_t) / 1000))
            self.last_t = s["t_ms"]
            self.track.append((s["t_ms"], self.yaw))
        if len(self.track) > 20000:
            self.track = self.track[-20000:]

    def at(self, t_ms: float) -> float | None:
        if not self.track:
            return None
        i = int(np.searchsorted([t for t, _ in self.track], t_ms))
        return self.track[min(i, len(self.track) - 1)][1]


class SweepEngine:
    def __init__(self, run_dir: Path, country: str, emit=None):
        self.run_dir = Path(run_dir)
        self.country = country
        self.emit = emit  # async callable(dict): inventory updates for the phone
        self.metrics = Metrics()
        self.heading = Heading()
        self.imu: list[dict] = []
        self.room_frames: dict[str, dict] = {}
        self.frames: list[dict] = []
        self.shelf_frames: list[dict] = []
        self.frame_obs: dict[str, list[dict]] = {}
        self.books: list[dict] = []
        self.idn: dict[str, dict] = {}  # spine read key -> verdict
        self.items: list[dict] = []
        self.corners: list[float] = []
        self.layout: dict | None = None
        self.gaps: list[str] = []
        self.jobs: set[asyncio.Task] = set()
        self.started: set = set()
        self.frame_sem, self.book_sem, self.rf_lock = asyncio.Semaphore(6), asyncio.Semaphore(8), asyncio.Lock()
        self.client = httpx.AsyncClient(headers={"User-Agent": identify.UA}, follow_redirects=True, timeout=12)
        self.ebay, self.fx = pricing.Ebay(self.client, self.metrics), pricing.FX(self.client)
        self.item_searches = 0
        self.dirty = False
        self.merger = asyncio.create_task(self._merge_loop())

    # ---------- input ----------
    def on_imu(self, samples: list[dict]) -> None:
        self.heading.add(samples)
        self.imu = (self.imu + samples)[-3000:]

    def on_frame(self, row: dict) -> None:
        row["heading"] = self.heading.at(row["t_ms"])
        self.frames.append(row)
        if row.get("phase") == "room_turn" and room.available():
            self._spawn(("room", row["file"]), self._room_frame(row, room.up_at(self.imu, row["t_ms"]), row["heading"]))
        if row.get("phase") == "shelves" and not row.get("blur_bad") and self._new_view(row):
            self.shelf_frames.append(row)
            self._spawn(("spines", row["file"]), self._shelf_frame(row))

    def _new_view(self, row: dict) -> bool:
        if not self.shelf_frames:
            return True
        fa, _, (gh, gw) = spines.frame_features(self.run_dir, self.shelf_frames[-1]["file"])
        H = spines.homography(fa, spines.frame_features(self.run_dir, row["file"])[0], min_inliers=60)
        return H is None or not keyframes.same_view(H, gw, gh)

    def _spawn(self, key, coro) -> None:
        if key in self.started:
            coro.close()
            return
        self.started.add(key)
        t = asyncio.create_task(self._guard(key, coro))
        self.jobs.add(t)
        t.add_done_callback(self.jobs.discard)

    async def _guard(self, key, coro) -> None:
        try:
            await coro
        except Exception as e:
            log.warning("job %s failed: %s", key[0], e)
            self.started.discard(key)

    # ---------- books ----------
    async def _shelf_frame(self, row: dict) -> None:
        async with self.frame_sem:
            pf = await spines._frame(self.run_dir, row, self.metrics)
        obs = pf["observations"]
        if spine_boxes.available():
            async with self.rf_lock:  # one detector run at a time keeps the laptop responsive
                img = cv2.cvtColor(cv2.imread(str(self.run_dir / row["file"])), cv2.COLOR_BGR2RGB)
                rf = await asyncio.to_thread(spine_boxes.detect, img)
            base = {"frame_i": row["i"], "file": row["file"]}
            obs = [{**base, **o} for o in spine_boxes.combine(obs, rf)]
        self.frame_obs[row["file"]] = obs
        self.dirty = True

    async def _merge_loop(self) -> None:
        while True:
            await asyncio.sleep(MERGE_EVERY_S)
            if self.dirty:
                self.dirty = False
                try:
                    await self._merge_books()
                except Exception as e:
                    log.warning("book merge failed: %s", e)

    async def _merge_books(self) -> None:
        frames = [f for f in self.shelf_frames if f["file"] in self.frame_obs]
        obs = [o for f in frames for o in self.frame_obs[f["file"]]]
        segs = await asyncio.to_thread(spines.chain_segments, self.run_dir, frames)
        self.books = spines.assign_shelves(spines.merge(obs, segs))
        for b in self.books:
            self._spawn(("book", _key(b["best"])), self._book(b))
        await self._push()

    async def _book(self, b: dict) -> None:
        async with self.book_sem:
            _, v = await identify._one(self.run_dir, self.client, b, self.metrics)
        self.idn[_key(b["best"])] = v
        if v.get("status") == "identified":
            inp = _book_input(b["id"], v, {})
            await pricing.memo_book_price(self.run_dir, self.ebay, self.fx, self.client, inp, self.country, self.metrics)
            if sum(1 for x in self.idn.values() if x.get("status") == "identified") <= SECOND_MARKET_BOOKS:
                await pricing.memo_book_price(self.run_dir, self.ebay, self.fx, self.client, inp, second_market(self.country), self.metrics)
        await self._push()

    async def _room_frame(self, row: dict, up, heading) -> None:
        if up is None or heading is None:
            return
        async with self.rf_lock:  # one heavy model at a time
            boxes = None
            if spine_boxes.available():
                img = cv2.cvtColor(cv2.imread(str(self.run_dir / row["file"])), cv2.COLOR_BGR2RGB)
                boxes = (await asyncio.to_thread(spine_boxes.detect, img)).tolist()
            self.room_frames[row["file"]] = await room.measure_frame(self.run_dir, row, up, heading, boxes)

    # ---------- items and room, from the live agent ----------
    def latest_frame(self) -> dict | None:
        return self.frames[-1] if self.frames else None

    async def log_item(self, args: dict) -> dict:
        frame = self.latest_frame()
        if not frame:
            return {"logged": False, "reason": "no camera frame yet"}
        claim = f"{args.get('category', '')}: {args.get('description', '')}".strip(": ")
        res = await items.check(self.run_dir, frame, claim, self.metrics)
        it, how = items.accept(self.items, res, frame, frame.get("heading"))
        if it is None:
            return {"logged": False, "reason": how}
        if args.get("wall"):
            it.setdefault("wall", args["wall"])
        if how == "new":
            it["_search_ok"] = self.item_searches < pricing.item_search_cap()
            self.item_searches += it["_search_ok"]
            self._spawn(("item_price", it["id"]), pricing.memo_item_price(self.run_dir, self.ebay, it, self.country, {}))
        await self._push()
        return {"logged": True, "id": it["id"], "status": "already logged: same item seen again" if how == "same" else "new"}

    def mark_corner(self) -> dict:
        h = self.heading.at(self.latest_frame()["t_ms"]) if self.frames else None
        if h is None:
            return {"recorded": False, "reason": "no motion data"}
        if all(items._turn(h, c) > 20 for c in self.corners):
            self.corners.append(round(h % 360, 1))
        return {"recorded": True, "corners_so_far": len(self.corners)}

    def log_room(self, args: dict) -> dict:
        self.layout = args
        return {"recorded": True}

    def inventory(self) -> dict:
        ident = sum(1 for v in self.idn.values() if v.get("status") == "identified")
        by_cat: dict[str, int] = {}
        for it in self.items:
            by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
        return {"books_seen": len(self.books), "books_identified": ident, "items": by_cat,
                "item_list": [f"{it['id']} {it['category']}: {it['description']}" for it in self.items][-30:],
                "corners_marked": len(self.corners), "room_layout": bool(self.layout)}

    async def _push(self) -> None:
        if self.emit:
            inv = [{"kind": "books", "label": f"{len(self.books)} books on shelves, {sum(1 for v in self.idn.values() if v.get('status') == 'identified')} identified"}] if self.books else []
            inv += [{"kind": it["category"], "label": it["description"]} for it in self.items]
            await self.emit({"type": "inventory", "items": inv})

    # ---------- finish ----------
    async def finish(self, progress=None) -> dict:
        t0 = time.perf_counter()
        self.merger.cancel()
        if self.jobs:
            await asyncio.wait(set(self.jobs), timeout=FINISH_WAIT_S)
        await self._merge_books()
        room_out = room.measure(self.run_dir, list(self.room_frames.values()), self.corners, self.layout)
        packet = await build_packet(self.run_dir, self.country, self.books, self.items, self.frames, self.metrics,
                                    room_out, self.client, self.ebay, self.fx, t0)
        return packet

    async def close(self) -> None:
        self.merger.cancel()
        for t in list(self.jobs):
            t.cancel()
        await self.client.aclose()


def _key(obs: dict) -> tuple:
    return (obs["file"], tuple(round(v) for v in obs["box_px"]))


def _book_input(bid: str, idn: dict, flags: dict) -> dict:
    m = idn["match"]
    return {"id": bid, "title": m["title"], "author": ", ".join(m.get("authors", [])), "isbn": m.get("isbn", ""),
            "isbns": idn.get("edition_isbns") or identify.edition_isbns(m, idn.get("alternates")),
            "publisher": m.get("publisher", ""), "published": m.get("published", ""), "read": idn.get("read", {}),
            "override_first_edition": flags.get("first_edition") or flags.get("signed")}


def second_market(country: str) -> str:
    s = os.getenv("SECOND_COUNTRY", "GB").upper()
    return s if s != country else ("US" if country != "US" else "GB")


async def build_packet(run_dir, country, books, item_list, frames, metrics, room_out, client, ebay, fx, t0) -> dict:
    """Assemble from what the sweep already produced: reads, prices and room work are cache hits."""
    sweep = {**(read_json(run_dir / "sweep.json", {}) or {}), "country": country, "currency": locale(country)["currency"]}
    idn = await identify.run(run_dir, books, metrics)
    books, idn = identify.merge_same_book(books, idn)
    kept, possible = [], []
    for b in books:
        ok, why = spines.countable(b, idn.get(b["id"], {}))
        if ok:
            kept.append(b)
        else:
            possible.append({"id": b["id"], "frame_ref": b["best"]["file"], "box_px": [round(v) for v in b["best"]["box_px"]], "reason": why})
    books = kept
    measurements = await asyncio.to_thread(scale.run, run_dir, books, idn, metrics)
    ov = assemble.resolve_overrides(read_json(run_dir / "overrides.json", []) or [], books, idn, item_list,
                                    {f["file"]: f["t_ms"] for f in frames})
    inputs = [_book_input(b["id"], idn[b["id"]], ov["books"].get(b["id"], {})) for b in books
              if idn.get(b["id"], {}).get("status") == "identified" and b["id"] not in ov["excluded_books"]]
    second = second_market(country)
    sample = inputs[:SECOND_MARKET_BOOKS]
    prices, p2 = await asyncio.gather(
        pricing.run(run_dir, inputs, item_list, {"items": ov["items"]}, metrics, country),
        pricing.run(run_dir, sample, [], {}, metrics, second),
    )
    item_list = await asyncio.to_thread(items.add_dimensions, run_dir, item_list, room_out)
    comparison = None
    if sample:
        fig = lambda p, k: (p.get(k) or {}).get("amount")  # noqa: E731
        comparison = {"primary_country": country, "primary_currency": locale(country)["currency"], "second_country": second,
                      "second_currency": locale(second)["currency"],
                      "books": [{"id": b["id"], "title": b["title"],
                                 "primary": {"new": fig(prices["books"][b["id"]], "replacement_cost"), "used": fig(prices["books"][b["id"]], "used_value")},
                                 "second": {"new": fig(p2["books"][b["id"]], "replacement_cost"), "used": fig(p2["books"][b["id"]], "used_value")},
                                 "second_sources": {k: (p2["books"][b["id"]].get(k) or {}).get("url", "") for k in ("replacement_cost", "used_value")}}
                                for b in sample]}
    metrics_out = metrics.summary() | {"finish_to_packet_s": round(time.perf_counter() - t0, 1)}
    packet = await asyncio.to_thread(assemble.build, run_dir, sweep=sweep, books=books, identified=idn, measurements=measurements,
                                     room=room_out, items=item_list, prices=prices, ov=ov, metrics=metrics_out,
                                     locale_comparison=comparison, possible_books=possible)
    await asyncio.to_thread(report.run, run_dir, packet)
    write_json(run_dir / "metrics.json", metrics_out)
    return packet


async def replay(run_dir: Path, country: str | None = None) -> dict:
    """Re-run a recorded sweep through the engine (books and room; items come from the saved list)."""
    sweep = read_json(run_dir / "sweep.json", {}) or {}
    eng = SweepEngine(run_dir, (country or sweep.get("country") or "US").upper())
    imu = read_jsonl(run_dir / "imu.jsonl")
    eng.on_imu(imu)
    eng.imu = imu
    for row in read_jsonl(run_dir / "frames.jsonl"):
        eng.on_frame(row)
    eng.items = read_json(run_dir / "items.json", []) or []
    try:
        return await eng.finish()
    finally:
        await eng.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--country", default=None)
    a = ap.parse_args()
    p = asyncio.run(replay(a.run_dir, a.country))
    t, m = p["totals"], p["pipeline_metrics"]
    print(f"books={t['book_count']} identified={t['books_identified']} items={len(p['items'])} floor={p['room'].get('floor_area_m2')} "
          f"review={len(p['review_queue'])} finish_to_packet={m.get('finish_to_packet_s')}s cost=${m['total_cost_usd']}")


if __name__ == "__main__":
    main()
