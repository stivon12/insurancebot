"""Run every post-sweep stage and write claim_packet.json + report.html + metrics.json.

Usage: uv run python -m pipeline.run runs/<sweep_id> [--country GB]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Awaitable, Callable

from . import assemble, identify, items, keyframes, pricing, report, room, scale, spine_boxes, spines
from .common import Metrics, Progress, country as default_country, locale, read_json, read_jsonl, write_json

Emit = Callable[[dict], Awaitable[None]] | None
COMPARISON_BOOKS = 10


def book_inputs(books: list[dict], identified: dict, ov: dict) -> list[dict]:
    out = []
    for b in books:
        idn = identified.get(b["id"], {})
        if idn.get("status") != "identified" or b["id"] in ov["excluded_books"]:
            continue
        m = idn["match"]
        flags = ov["books"].get(b["id"], {})
        out.append(
            {
                "id": b["id"],
                "title": m["title"],
                "author": ", ".join(m.get("authors", [])),
                "isbn": m.get("isbn", ""),
                "isbns": [i for i in [m.get("isbn")] + [a.get("isbn") for a in idn.get("alternates", [])] if i],
                "publisher": m.get("publisher", ""),
                "published": m.get("published", ""),
                "read": idn.get("read", {}),
                "override_first_edition": flags.get("first_edition") or flags.get("signed"),
            }
        )
    return out


async def process(run_dir: Path, emit: Emit = None, country: str | None = None, metrics: Metrics | None = None) -> dict:
    """emit receives progress snapshots: {overall: 0..1, stages: [{stage, label, done, total, state}]}.
    metrics: pass the live worker's Metrics so the cost of work done during the sweep is reported too."""
    t0 = time.perf_counter()
    pg = Progress(emit)
    metrics = metrics or Metrics()
    sweep = read_json(run_dir / "sweep.json", {}) or {}
    country = (country or sweep.get("country") or default_country()).upper()
    sweep = {**sweep, "country": country, "currency": locale(country)["currency"]}

    pg.start("keyframes")
    with metrics.time("keyframes"):
        kfs = await asyncio.to_thread(keyframes.run, run_dir)  # CPU work off the event loop (the server shares it)
    pg.finish("keyframes", f"{sum(len(v) for v in kfs.values())} keyframes")

    async def book_chain():
        pg.start("spines", len(spines.frames_for(kfs)))
        bks = await spines.run(run_dir, kfs, metrics, tick=lambda: pg.tick("spines"))
        pg.finish("spines", f"{len(bks)} spines")
        pg.start("identify", len(bks))
        idn = await identify.run(run_dir, bks, metrics, tick=lambda: pg.tick("identify"))
        n_read = len(bks)
        bks, idn = identify.merge_same_book(bks, idn)
        write_json(run_dir / "identity_merge.json", {"spines": n_read, "books_after_merge": len(bks),
                                                    "merged": [n for b in bks for n in b.get("review_notes", []) if n.startswith("merged")]})
        # Tighten boxes for measurement only after reading, so text is still read from Gemini's own crops.
        await asyncio.to_thread(spine_boxes.refine, run_dir, bks, metrics)
        pg.finish("identify", f"{sum(1 for v in idn.values() if v.get('status') == 'identified')} identified")
        return bks, idn

    async def item_chain():
        pg.start("items", len(items.frames_for(kfs)))
        out = await items.run(run_dir, kfs, metrics, tick=lambda n: pg.tick("items", n))
        pg.finish("items", f"{len(out)} items")
        return out

    (books, identified), item_list = await asyncio.gather(book_chain(), item_chain())
    measurements = await asyncio.to_thread(scale.run, run_dir, books, identified, metrics)

    pg.start("room")
    room_out = await room.run(run_dir, kfs, books, identified, metrics)
    item_list = await asyncio.to_thread(items.add_dimensions, run_dir, item_list, room_out)
    pg.finish("room", f"{room_out['floor_area_m2']} m² floor" if room_out.get("floor_area_m2") else "not measured (review)")

    frames_t = {f["file"]: f["t_ms"] for f in read_jsonl(run_dir / "frames.jsonl")}
    ov = assemble.resolve_overrides(read_json(run_dir / "overrides.json", []) or [], books, identified, item_list, frames_t)

    inputs = book_inputs(books, identified, ov)
    second = os.getenv("SECOND_COUNTRY", "GB").upper()
    n_second = min(COMPARISON_BOOKS, len(inputs)) if second != country else 0
    pg.start("pricing", len(inputs) + len(item_list) + n_second)
    prices = await pricing.run(run_dir, inputs, item_list, {"items": ov["items"]}, metrics, country, tick=lambda: pg.tick("pricing"))

    comparison = None
    if second != country:
        sample = [b for b in inputs if prices["books"].get(b["id"], {}).get("replacement_cost")][:COMPARISON_BOOKS]
        if len(sample) < COMPARISON_BOOKS:
            sample += [b for b in inputs if b not in sample][: COMPARISON_BOOKS - len(sample)]
        p2 = await pricing.run(run_dir, sample, [], {}, metrics, second, tick=lambda: pg.tick("pricing"))

        def fig(p, k):
            return (p.get(k) or {}).get("amount")

        comparison = {
            "primary_country": country,
            "primary_currency": locale(country)["currency"],
            "second_country": second,
            "second_currency": locale(second)["currency"],
            "books": [
                {
                    "id": b["id"],
                    "title": b["title"],
                    "primary": {"new": fig(prices["books"][b["id"]], "replacement_cost"), "used": fig(prices["books"][b["id"]], "used_value")},
                    "second": {"new": fig(p2["books"][b["id"]], "replacement_cost"), "used": fig(p2["books"][b["id"]], "used_value")},
                    "second_sources": {k: (p2["books"][b["id"]].get(k) or {}).get("url", "") for k in ("replacement_cost", "used_value")},
                }
                for b in sample
            ],
        }

    pg.finish("pricing")
    pg.start("assemble")
    metrics_out = metrics.summary() | {"wall_clock_s": round(time.perf_counter() - t0, 1)}
    packet = await asyncio.to_thread(
        assemble.build,
        run_dir,
        sweep=sweep,
        books=books,
        identified=identified,
        measurements=measurements,
        room=room_out,
        items=item_list,
        prices=prices,
        ov=ov,
        metrics=metrics_out,
        locale_comparison=comparison,
    )
    await asyncio.to_thread(report.run, run_dir, packet)
    write_json(run_dir / "metrics.json", metrics_out)
    pg.finish("assemble")
    await pg.flush()
    return packet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--country", default=None)
    args = ap.parse_args()

    seen: dict[str, str] = {}

    async def printer(snap: dict) -> None:
        for s in snap["stages"]:
            if s["state"] != "waiting" and seen.get(s["stage"]) != s["state"]:
                seen[s["stage"]] = s["state"]
                extra = f" ({s['note']})" if s.get("note") else (f" ({s['total']})" if s["total"] else "")
                print(f"[pipeline {snap['overall']:4.0%}] {s['label']}: {s['state']}{extra}", flush=True)

    packet = asyncio.run(process(args.run_dir, printer, args.country))
    t = packet["totals"]
    print(f"books={t['book_count']} identified={t['books_identified']} unidentified={t['books_unidentified']} "
          f"review={len(packet['review_queue'])} wall_clock={packet['pipeline_metrics']['wall_clock_s']}s "
          f"cost=${packet['pipeline_metrics']['total_cost_usd']}")
    print(f"report: {args.run_dir / 'report.html'}")


if __name__ == "__main__":
    main()
