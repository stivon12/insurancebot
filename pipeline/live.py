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

from . import books, items, packet, pricing, room
from .common import Metrics, locale, read_json, read_jsonl, write_json

log = logging.getLogger("live")
SECOND_MARKET_BOOKS = 10
MERGE_EVERY_S = 2.0
FINISH_WAIT_S = 60
SHARP_MIN = 150  # frame blur score (phone, variance of the Laplacian): below this spines are not readable
MIN_SPINE_BOXES = 3  # detector boxes before a frame is worth a spine read
DOUBT_GAP_S, MAX_DOUBTS = 60, 3


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
        self.doubts: list[dict] = []  # flag_view from the agent
        self.pending_doubts: list[str] = []  # notes for the agent: shelves seen but nothing readable
        self.doubts_sent, self.last_doubt = 0, -1e9
        self.jobs: set[asyncio.Task] = set()
        self.started: set = set()
        self.frame_sem, self.book_sem, self.rf_lock = asyncio.Semaphore(6), asyncio.Semaphore(8), asyncio.Lock()
        self.client = httpx.AsyncClient(headers={"User-Agent": books.UA}, follow_redirects=True, timeout=12)
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
        if not row.get("blur_bad"):
            self._spawn(("frame", row["file"]), self._frame(row, room.up_at(self.imu, row["t_ms"]), row["heading"]))

    def _new_view(self, row: dict) -> bool:
        if not self.shelf_frames:
            return True
        fa, _, (gh, gw) = books.frame_features(self.run_dir, self.shelf_frames[-1]["file"])
        H = books.homography(fa, books.frame_features(self.run_dir, row["file"])[0], min_inliers=60)
        return H is None or not books.same_view(H, gw, gh)

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

    # ---------- every frame: spine boxes and room depth, then a spine read if books are in view ----------
    async def _frame(self, row: dict, up, heading) -> None:
        rf = None
        async with self.rf_lock:  # one heavy model at a time keeps the laptop responsive
            img = cv2.cvtColor(cv2.imread(str(self.run_dir / row["file"])), cv2.COLOR_BGR2RGB)
            if books.detector_available():
                rf = await asyncio.to_thread(books.detect, img)
            if room.available() and up is not None and heading is not None:
                self.room_frames[row["file"]] = await room.measure_frame(self.run_dir, row, up, heading, None if rf is None else rf.tolist())
        if row.get("blur", 0) < SHARP_MIN or (rf is not None and len(rf) < MIN_SPINE_BOXES):
            return
        if not self._new_view(row) and row["blur"] < 1.2 * self.shelf_frames[-1].get("blur", 0):
            return  # same view, not sharper
        self.shelf_frames.append(row)
        async with self.frame_sem:
            pf = await books._frame(self.run_dir, row, self.metrics)
        obs = pf["observations"]
        if rf is not None:
            base = {"frame_i": row["i"], "file": row["file"]}
            obs = [{**base, **o} for o in books.combine(obs, rf)]
        if not any(o.get("legible") for o in obs):  # blurred or far: these spines can't be matched to a sharp view, so not counted
            self.shelf_frames.remove(row)
            self._doubt("[doubt] books are in view but no spine could be read: ask them once to hold the camera closer to that shelf.")
            return
        self.frame_obs[row["file"]] = obs
        self.dirty = True

    def _doubt(self, text: str) -> None:
        if self.doubts_sent < MAX_DOUBTS and time.monotonic() - self.last_doubt > DOUBT_GAP_S:
            self.doubts_sent, self.last_doubt = self.doubts_sent + 1, time.monotonic()
            self.pending_doubts.append(text)

    def pop_doubt(self) -> str | None:
        return self.pending_doubts.pop(0) if self.pending_doubts else None

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
        frames = sorted((f for f in self.shelf_frames if f["file"] in self.frame_obs), key=lambda f: f["i"])
        obs = [o for f in frames for o in self.frame_obs[f["file"]]]
        segs = await asyncio.to_thread(books.chain_segments, self.run_dir, frames)
        self.books = books.assign_shelves(books.merge(obs, segs))
        for b in self.books:
            self._spawn(("book", _key(b["best"])), self._book(b))
        await self._push()

    async def _book(self, b: dict) -> None:
        async with self.book_sem:
            _, v = await books._one(self.run_dir, self.client, b, self.metrics)
        self.idn[_key(b["best"])] = v
        if v.get("status") == "identified":
            inp = _book_input(b["id"], v, {})
            await pricing.memo_book_price(self.run_dir, self.ebay, self.fx, self.client, inp, self.country, self.metrics)
            if sum(1 for x in self.idn.values() if x.get("status") == "identified") <= SECOND_MARKET_BOOKS:
                await pricing.memo_book_price(self.run_dir, self.ebay, self.fx, self.client, inp, second_market(self.country), self.metrics)
        await self._push()

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
        if args.get("location"):
            it.setdefault("location", args["location"])
        if how == "new":
            it["_search_ok"] = self.item_searches < pricing.item_search_cap()
            self.item_searches += it["_search_ok"]
            self._spawn(("item_price", it["id"]), pricing.memo_item_price(self.run_dir, self.ebay, it, self.country, {}))
        await self._push()
        return {"logged": True, "id": it["id"], "status": "already logged: same item seen again" if how == "same" else "new"}

    def flag_view(self, args: dict, t_ms: float, frame_ref: str) -> dict:
        what = str(args.get("what", ""))[:200]
        if args.get("resolved"):
            open_ = [d for d in self.doubts if not d["resolved"]]
            for d in [d for d in open_ if d["what"] == what] or open_[-1:]:
                d["resolved"] = True
        else:
            self.doubts.append({"what": what, "resolved": False, "t_ms": t_ms, "frame_ref": frame_ref})
        write_json(self.run_dir / "doubts.json", self.doubts)
        return {"recorded": True, "open_doubts": sum(not d["resolved"] for d in self.doubts)}

    def inventory(self) -> dict:
        ident = sum(1 for v in self.idn.values() if v.get("status") == "identified")
        by_cat: dict[str, int] = {}
        for it in self.items:
            by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
        return {"books_seen": len(self.books), "books_identified": ident, "items": by_cat,
                "item_list": [f"{it['id']} {it['category']}: {it['description']}" for it in self.items][-30:],
                "open_doubts": [d["what"] for d in self.doubts if not d["resolved"]],
                "titles_identified": [v["match"]["title"] for v in self.idn.values() if v.get("status") == "identified"][-20:]}

    async def _push(self) -> None:
        if self.emit:
            titles = [v["match"]["title"] for v in self.idn.values() if v.get("status") == "identified"]
            inv = [{"kind": "books", "label": f"{len(self.books)} books on shelves, {len(titles)} identified"}] if self.books else []
            inv += [{"kind": "book", "label": t} for t in titles[-12:]]
            inv += [{"kind": it["category"], "label": it["description"]} for it in self.items]
            await self.emit({"type": "inventory", "items": inv})

    # ---------- finish ----------
    async def finish(self) -> dict:
        t0 = time.perf_counter()
        self.merger.cancel()
        if self.jobs:
            await asyncio.wait(set(self.jobs), timeout=FINISH_WAIT_S)
        await self._merge_books()
        room_out = room.measure(self.run_dir, list(self.room_frames.values()), walked=True)
        return await build_packet(self.run_dir, self.country, self.books, self.items, self.frames, self.metrics,
                                    room_out, self.client, self.ebay, self.fx, t0)

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
            "isbns": idn.get("edition_isbns") or books.edition_isbns(m, idn.get("alternates")),
            "publisher": m.get("publisher", ""), "published": m.get("published", ""), "read": idn.get("read", {}),
            "override_first_edition": flags.get("first_edition") or flags.get("signed")}


def second_market(country: str) -> str:
    s = os.getenv("SECOND_COUNTRY", "GB").upper()
    return s if s != country else ("US" if country != "US" else "GB")


async def build_packet(run_dir, country, bks, item_list, frames, metrics, room_out, client, ebay, fx, t0) -> dict:
    """Assemble from what the sweep already produced: reads, prices and room work are cache hits."""
    sweep = {**(read_json(run_dir / "sweep.json", {}) or {}), "country": country, "currency": locale(country)["currency"]}
    idn = await books.identify_all(run_dir, bks, metrics)
    bks, idn = books.merge_same_book(bks, idn)
    kept, possible = [], []
    for b in bks:
        ok, why = books.countable(b, idn.get(b["id"], {}))
        if ok:
            kept.append(b)
        else:
            possible.append({"id": b["id"], "frame_ref": b["best"]["file"], "box_px": [round(v) for v in b["best"]["box_px"]], "reason": why})
    bks = kept
    measurements = await asyncio.to_thread(books.measure_all, run_dir, bks, idn, metrics)
    ov = packet.resolve_overrides(read_json(run_dir / "overrides.json", []) or [], bks, idn, item_list,
                                    {f["file"]: f["t_ms"] for f in frames})
    inputs = [_book_input(b["id"], idn[b["id"]], ov["books"].get(b["id"], {})) for b in bks
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
    pk = await asyncio.to_thread(packet.build, run_dir, sweep=sweep, books=bks, identified=idn, measurements=measurements,
                                     room=room_out, items=item_list, prices=prices, ov=ov, metrics=metrics_out,
                                     locale_comparison=comparison, possible_books=possible)
    pk["review_queue"] += [{"ref_id": "doubt", "reason": f"agent could not settle: {d['what']} ({d['frame_ref']})"}
                           for d in read_json(run_dir / "doubts.json", []) or [] if not d["resolved"]]
    write_json(run_dir / "claim_packet.json", pk)
    await asyncio.to_thread(packet.write_report, run_dir, pk)
    write_json(run_dir / "metrics.json", metrics_out)
    return pk


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
    while eng.jobs:  # a replay delivers every frame at once: let the queue drain, as it would during a real sweep
        await asyncio.wait(set(eng.jobs))
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
