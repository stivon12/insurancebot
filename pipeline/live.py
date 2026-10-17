"""Per-sweep engine: every frame and every agent log is processed as it arrives; Finish only assembles the packet."""

from __future__ import annotations

import argparse
import asyncio
import re
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
ROOM_STAGE = "room tracking"
IDENTIFY_BY_S, PRICE_BY_S = 150, 240  # after Finish: what isn't done by then goes to review, the packet stays under 5 min
SHARP_MIN = 150  # frame blur score (phone, variance of the Laplacian): below this spines are not readable
MIN_SPINE_BOXES = 3  # detector boxes before a frame is worth a spine read
DOUBT_GAP_S, MAX_DOUBTS = 60, 3
BOOK_SURE = 0.75  # an identified book below this confidence gets a 'closer look' question
COVER_BINS, COVER_GAP_DEG = 12, 90  # heading coverage in 30-degree bins; ask about an unseen sector this wide or wider
COVER_AFTER_MS, COVER_EVERY_MS = 30000, 20000  # counted from the start of the tour, not of the session
TOUR_START_LATEST_MS = 60000
SCAN_TURN_DEG, SCAN_EVERY_MS = 15, 3000  # item scan on a new view: the phone turned this far, or this long since the last
ART_ANSWER = {"is_print": re.compile(r"\b(print|poster|copy|reproduction|printed)\b", re.I),
              "is_original": re.compile(r"\b(original|painted|real one|real painting|hand ?made)\b", re.I)}


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
                self.yaw -= math.degrees(rate * min(0.2, max(0.0, (s["t_ms"] - self.last_t) / 1000)))
            self.last_t = s["t_ms"]
            self.track.append((s["t_ms"], self.yaw))
        if len(self.track) > 20000:
            self.track = self.track[-20000:]

    def at(self, t_ms: float) -> float | None:
        if not self.track or t_ms - self.track[-1][0] > 300:  # no motion data up to this frame yet: unknown, not stale
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
        self.tracker = room.Tracker()
        self.track_q: asyncio.Queue = asyncio.Queue()  # tracking frames in arrival order, for the tracker
        self.has_track_frames = False
        self.ori = room.Orientation()
        self.last_cover_check = -1e9
        self.tour_start_ms: float | None = None  # set when the locale is answered: no doubts during the opening chat
        self.rot_source: str | None = None  # 'phone orientation' or 'gyro', fixed for the whole sweep (never mixed)
        self.tracker_task = asyncio.create_task(self._track_loop())
        self.frames: list[dict] = []
        self.shelf_frames: list[dict] = []
        self.frame_obs: dict[str, list[dict]] = {}
        self.books: list[dict] = []
        self.idn: dict[str, dict] = {}  # spine read key -> verdict
        self.items: list[dict] = []
        self.doubts: list[dict] = []  # flag_view from the agent
        self.pending_doubts: list[str] = []  # notes for the agent: shelves seen but nothing readable
        self.doubts_sent: dict[str, int] = {}
        self.last_doubt: dict[str, float] = {}
        self.jobs: set[asyncio.Task] = set()
        self.started: set = set()
        self.frame_sem, self.book_sem, self.rf_lock = asyncio.Semaphore(6), asyncio.Semaphore(8), asyncio.Lock()
        self.client = httpx.AsyncClient(headers={"User-Agent": books.UA}, follow_redirects=True, timeout=12)
        self.ebay, self.fx = pricing.Ebay(self.client, self.metrics), pricing.FX(self.client)
        self.item_searches = 0
        self.pending_items: list[dict] = []  # unsure scans waiting for a second sighting
        self.item_sem = asyncio.Semaphore(2)
        self.last_scan: tuple[float, float | None] | None = None  # (t_ms, heading)
        self.art_question: dict | None = None  # the picture the agent was asked about
        self.dirty = False
        self.merger = asyncio.create_task(self._merge_loop())

    # ---------- input ----------
    def on_imu(self, samples: list[dict]) -> None:
        self.heading.add(samples)
        self.imu = (self.imu + samples)[-30000:]  # ~8 min at 60 Hz: the tracker may lag the phone

    def on_ori(self, rows: list[dict]) -> None:
        self.ori.add(rows, self.imu)

    def heading_at(self, t_ms: float) -> float | None:
        """Compass-like heading for item de-duplication: from the phone's orientation when available."""
        R = self.ori.at(t_ms)
        return room.yaw_deg(R) % 360 if R is not None else self.heading.at(t_ms)

    def _rotation(self, row: dict) -> np.ndarray | None:
        """One rotation source for the whole sweep: the phone's orientation once it is confirmed against gravity,
        else (no orientation stream, or it disagreed) the gyro. Frames while it is still being decided wait."""
        if self.rot_source is None:
            if self.ori.usable:
                self.rot_source = "phone orientation"
            elif self.ori.mode == "off" or (not self.ori.t and row["t_ms"] > 3000):
                self.rot_source = "gyro"
            else:
                return None
        if self.rot_source == "phone orientation":
            return self.ori.at(row["t_ms"], row.get("orient", 0))
        up, heading = room.up_at(self.imu, row["t_ms"], 60), self.heading.at(row["t_ms"])
        return None if up is None or heading is None else room.orientation(up, heading)

    def start_tour(self, t_ms: float) -> None:
        if self.tour_start_ms is None:
            self.tour_start_ms = t_ms

    def on_frame(self, row: dict) -> None:
        if row["t_ms"] > TOUR_START_LATEST_MS:  # the locale was never set: don't hold doubts back for ever
            self.start_tour(row["t_ms"])
        row["heading"] = self.heading_at(row["t_ms"])
        self.frames.append(row)
        if not self.has_track_frames:  # a phone that sends no tracking frames (old runs): track on the full frames
            self.track_q.put_nowait(row)
        if not row.get("blur_bad"):
            self._spawn(("frame", row["file"]), self._frame(row))
            if row.get("blur", 0) >= SHARP_MIN and self._scan_due(row):
                self.last_scan = (row["t_ms"], row["heading"])
                self._spawn(("scan", row["file"]), self._scan(row))

    def _scan_due(self, row: dict) -> bool:
        if self.last_scan is None or row["t_ms"] - self.last_scan[0] >= SCAN_EVERY_MS:
            return True
        h0, h = self.last_scan[1], row.get("heading")
        return h0 is not None and h is not None and items._turn(h0, h) >= SCAN_TURN_DEG

    async def _scan(self, row: dict) -> None:
        async with self.item_sem:
            with self.metrics.time(items.STAGE):
                found = await items.scan(self.run_dir, row, self.metrics)
        for res in found:
            it, how = items.accept(self.items, res, row, row.get("heading"), self.pending_items)
            if how == "pending" and res.get("confidence", 0) >= 0.5:  # seen once and not clearly: ask for another look
                self._doubt(f"[doubt] something that may be a {res.get('description', 'valuable item')} was seen but not "
                            "clearly: ask them once to show it again, a little closer.", "item")
            if how == "new":
                self._new_item(it)
                if it["category"] == "art" and self.art_question is None:
                    self.art_question = it
                    self._doubt(f"[doubt] a picture is in view ({it['description']}): ask once whether it is an original or a print.", "art")
        if found:
            await self._push()

    async def _timed(self, stage: str, coro):
        with self.metrics.time(stage):
            return await coro

    def _new_item(self, it: dict) -> None:
        it["_search_ok"] = self.item_searches < pricing.item_search_cap()
        self.item_searches += it["_search_ok"]
        self._spawn(("item_price", it["id"]), self._timed(pricing.STAGE, pricing.memo_item_price(self.run_dir, self.ebay, it, self.country, {})))

    def answer_art(self, text: str) -> dict | None:
        """The policyholder's answer to 'original or print?', read from their words (no tool call needed)."""
        if not self.art_question:
            return None
        for action, rx in ART_ANSWER.items():
            if rx.search(text):
                it, self.art_question = self.art_question, None
                return {"target_type": "item", "action": action, "target": it["description"]}
        return None

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
    def on_track(self, row: dict) -> None:
        self.has_track_frames = True
        self.track_q.put_nowait(row)

    async def _track_loop(self) -> None:
        """Tracking frames in arrival order: optical flow and PnP on each; MoGe only on the frames that become keyframes."""
        while True:
            row = await self.track_q.get()
            if row is None:
                return
            try:
                img = await asyncio.to_thread(cv2.imread, str(self.run_dir / row["file"]))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                R = self._rotation(row)
                if R is None or not room.available():
                    continue
                with self.metrics.time(ROOM_STAGE):
                    if not await asyncio.to_thread(self.tracker.track, room.track_gray(img), R):
                        continue
                async with self.rf_lock:  # one heavy model at a time
                    with self.metrics.time(ROOM_STAGE):
                        fp = {**await asyncio.to_thread(room.frame_points, img), "file": row["file"]}
                breaks = self.tracker.breaks
                await asyncio.to_thread(self.tracker.add, fp, R)
                if self.tracker.breaks > breaks:
                    self._doubt("[doubt] the room map lost track of where the camera is: ask them once to slow down, turn "
                                "slowly, and keep the wall beside them in view.", "room")
                self._coverage_doubt(row["t_ms"], R)
            except Exception as e:
                log.warning("tracker step failed: %s", e)

    async def _frame(self, row: dict) -> None:
        rf = None
        async with self.rf_lock:  # one heavy model at a time keeps the laptop responsive
            img = cv2.cvtColor(cv2.imread(str(self.run_dir / row["file"])), cv2.COLOR_BGR2RGB)
            if books.detector_available():
                rf = await asyncio.to_thread(books.detect, img)
        if row.get("blur", 0) < SHARP_MIN or (rf is not None and len(rf) < MIN_SPINE_BOXES):
            return
        if not self._new_view(row) and row["blur"] < 1.2 * self.shelf_frames[-1].get("blur", 0):
            return  # same view, not sharper
        self.shelf_frames.append(row)
        async with self.frame_sem:
            with self.metrics.time(books.SPINES_STAGE):
                pf = await books._frame(self.run_dir, row, self.metrics)
        obs = pf["observations"]
        if rf is not None:
            base = {"frame_i": row["i"], "file": row["file"]}
            obs = [{**base, **o} for o in books.combine(obs, rf)]
        if not any(o.get("legible") for o in obs):  # blurred or far: these spines can't be matched to a sharp view, so not counted
            self.shelf_frames.remove(row)
            if pf["observations"]:  # only when Gemini itself saw spines here; detector boxes alone are often clothes or doors
                self._doubt("[doubt] books are in view but no spine could be read: ask them once to hold the camera closer to that shelf.")
            return
        self.frame_obs[row["file"]] = obs
        self.dirty = True

    def coverage(self) -> tuple[int, float | None]:
        """(degrees of heading the camera has faced, centre of the largest unseen sector) from the placed keyframes."""
        seen = np.zeros(COVER_BINS, bool)
        for f in self.tracker.placed:
            seen[int((room.yaw_deg(f["R"]) % 360) // (360 / COVER_BINS))] = True
        if seen.all() or not seen.any():
            return int(seen.sum() * 360 / COVER_BINS), None
        k = np.nonzero(seen)[0][0]
        runs, cur = [], []
        for j in range(k, k + COVER_BINS):  # walk round from a seen bin: unseen runs never wrap mid-way
            if not seen[j % COVER_BINS]:
                cur.append(j % COVER_BINS)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        gap = max(runs, key=len)
        centre = (gap[len(gap) // 2] + 0.5) * 360 / COVER_BINS
        return int(seen.sum() * 360 / COVER_BINS), centre if len(gap) * 360 / COVER_BINS >= COVER_GAP_DEG else None

    def _coverage_doubt(self, t_ms: float, R: np.ndarray) -> None:
        if self.tour_start_ms is None or t_ms - self.tour_start_ms < COVER_AFTER_MS or t_ms - self.last_cover_check < COVER_EVERY_MS:
            return
        self.last_cover_check = t_ms
        deg, gap = self.coverage()
        if gap is None:
            return
        where = "of the room you haven't shown yet"
        if self.rot_source == "phone orientation":  # heading grows anticlockwise: positive means to the left
            d = (gap - room.yaw_deg(R) + 180) % 360 - 180
            where = "behind them" if abs(d) > 135 else "to their left" if d > 0 else "to their right"
        self._doubt(f"[doubt] only {deg} degrees of the room have been filmed; the part {where} hasn't been seen: ask them "
                    "once to show it.", "coverage")

    def _doubt(self, text: str, kind: str = "books") -> None:
        if self.tour_start_ms is None:  # still on the opening question: a note now would talk over it
            return
        n, last = self.doubts_sent.get(kind, 0), self.last_doubt.get(kind, -1e9)
        if n < MAX_DOUBTS and time.monotonic() - last > DOUBT_GAP_S:
            self.doubts_sent[kind], self.last_doubt[kind] = n + 1, time.monotonic()
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
            with self.metrics.time(books.IDENTIFY_STAGE):
                _, v = await books._one(self.run_dir, self.client, b, self.metrics)
        self.idn[_key(b["best"])] = v
        text = ((v.get("read") or {}).get("verbatim") or "").strip()[:60]
        if v.get("status") != "identified" and (v.get("read") or {}).get("legible") and text:
            self._doubt(f"[doubt] a book spine reading '{text}' could not be matched to a title: ask them once to hold the "
                        "camera closer to that spine.", "book")
        elif v.get("status") == "identified" and v.get("id_confidence", 1) < BOOK_SURE:
            self._doubt(f"[doubt] not sure the book is '{v['match']['title'][:60]}': ask them once to hold the camera closer "
                        "to that spine.", "book")
        if v.get("status") == "identified":
            inp = _book_input(b["id"], v, {})
            with self.metrics.time(pricing.STAGE):
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
            self._new_item(it)
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
            inv += [{"kind": it["category"].replace("_", " "), "label": f"{it['description']}" + (f" ({it['brand_model']})" if it.get("brand_model") else "")}
                    for it in self.items]  # confirmed items only: unsure scans wait in pending_items
            await self.emit({"type": "inventory", "items": inv})

    # ---------- finish ----------
    async def finish(self) -> dict:
        t0 = time.perf_counter()
        self.merger.cancel()
        with self.metrics.phase("wait for live jobs"):
            if self.jobs:
                await asyncio.wait(set(self.jobs), timeout=FINISH_WAIT_S)
            await self._merge_books()
            self.track_q.put_nowait(None)
            await asyncio.wait({self.tracker_task}, timeout=FINISH_WAIT_S)
        with self.metrics.phase("room outline"):
            room_out = await asyncio.to_thread(room.measure, self.run_dir, self.tracker)
        deg, gap = self.coverage()
        room_out["headings_filmed_deg"] = deg
        if self.tracker.placed and deg < 270:
            room_out.setdefault("review", []).append(f"only {deg} degrees of the room were filmed: parts of the room may be missing")
        room_out["rotation_source"] = self.rot_source
        room_out["orientation_vs_gravity_deg"] = self.ori.agreement_deg
        write_json(self.run_dir / "room.json", room_out)
        return await build_packet(self.run_dir, self.country, self.books, self.items, self.frames, self.metrics,
                                    room_out, self.client, self.ebay, self.fx, t0)

    async def close(self) -> None:
        self.merger.cancel()
        self.tracker_task.cancel()
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
    with metrics.phase("identify remaining books"):
        idn = await books.identify_all(run_dir, bks, metrics, deadline=t0 + IDENTIFY_BY_S)
    bks, idn = books.merge_same_book(bks, books.series_only(bks, idn))
    kept, possible = [], []
    for b in bks:
        ok, why = books.countable(b, idn.get(b["id"], {}))
        if ok:
            kept.append(b)
        else:
            possible.append({"id": b["id"], "frame_ref": b["best"]["file"], "box_px": [round(v) for v in b["best"]["box_px"]], "reason": why})
    bks = kept
    with metrics.phase("spine sizes"):
        measurements = await asyncio.to_thread(books.measure_all, run_dir, bks, idn, metrics)
    heights: dict[str, float] = {}
    for b in bks:  # shelved wall: each shelf's occupied run x its tallest spine (+5 cm clearance)
        hcm = (measurements["books"].get(b["id"]) or {}).get("spine_height_cm")
        if hcm:
            heights[b["shelf"]] = max(heights.get(b["shelf"], 0), hcm)
    shelved = sum(run_m * (heights[sh] + 5) / 100 for sh, run_m in measurements.get("shelves", {}).items() if sh in heights)
    if shelved:
        room_out = {**room_out, "shelved_wall_area_m2": round(shelved, 2), "shelved_wall_area_ft2": round(shelved * room.SQFT_PER_M2, 1)}
    with metrics.phase("merge duplicate items"):
        item_list = await items.consolidate(run_dir, item_list, metrics)
    ov = packet.resolve_overrides(read_json(run_dir / "overrides.json", []) or [], bks, idn, item_list,
                                    {f["file"]: f["t_ms"] for f in frames})
    inputs = [_book_input(b["id"], idn[b["id"]], ov["books"].get(b["id"], {})) for b in bks
              if idn.get(b["id"], {}).get("status") == "identified" and b["id"] not in ov["excluded_books"]]
    second = second_market(country)
    sample = inputs[:SECOND_MARKET_BOOKS]
    with metrics.phase("pricing"):
        prices, p2 = await asyncio.gather(
            pricing.run(run_dir, inputs, item_list, {"items": ov["items"]}, metrics, country, deadline=t0 + PRICE_BY_S),
            pricing.run(run_dir, sample, [], {}, metrics, second, deadline=t0 + PRICE_BY_S),
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
    metrics_out = metrics.summary() | {"finish_to_packet_s": round(time.perf_counter() - t0, 1)}  # before the files are written
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
    eng.start_tour(0.0)
    imu = read_jsonl(run_dir / "imu.jsonl")
    eng.on_imu(imu)
    eng.imu = imu
    eng.on_ori(read_jsonl(run_dir / "ori.jsonl"))
    rows = [("f", r) for r in read_jsonl(run_dir / "frames.jsonl")] + [("t", r) for r in read_jsonl(run_dir / "track.jsonl")]
    eng.has_track_frames = any(k == "t" for k, _ in rows)
    for kind, row in sorted(rows, key=lambda x: x[1]["t_ms"]):  # as the phone sent them
        eng.on_frame(row) if kind == "f" else eng.on_track(row)
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
