"""Re-run catalogue matching on SAVED spine reads of a library_dataset_eval run (no spine re-reading; the only
model calls are the small query-rewrite / pick calls for hard cases).

  uv run python scripts/rescore_titles.py runs/libds_titles

Prints title accuracy against the crop's annotations and lists every confident identification that matches no
title annotated anywhere in the same original photo, with the spine text, so each can be checked by eye.
"""
import asyncio
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.eval import CONFIDENT, match_titles  # noqa: E402
from pipeline import identify  # noqa: E402
from pipeline.common import Metrics  # noqa: E402

CSV = Path.home() / "datasets/library-dataset/data/annotations/library_books_annotations.csv"


async def relookup(c, read, m):
    return await identify.resolve(c, read, m)


async def main(run: Path):
    split, orig = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(CSV)):
        if r["not_book"] != "True" and r["title"].strip():
            split[r["image_name"]].append({"title": r["title"].strip()})
            orig[r["image_name"][:15]].append({"title": r["title"].strip()})
    m = Metrics()
    tot_t = tot_c = tot_i = n = 0
    unconfirmed = []
    async with httpx.AsyncClient(headers={"User-Agent": identify.UA}, follow_redirects=True) as c:
        for d in sorted(run.iterdir()):
            if not (d / "identified.json").exists():
                continue
            old = json.load(open(d / "identified.json"))
            img = d.name + ".jpg"
            reads = [v.get("read") or {} for v in old.values()]
            res = await asyncio.gather(*(relookup(c, r, m) for r in reads))
            n += len(res)
            pred = [{"title": v["match"]["title"], "id_confidence": v["id_confidence"], "read": r.get("verbatim", "")}
                    for v, r in zip(res, reads) if v["status"] == "identified"]
            tot_t += len(split[img])
            tot_i += len(pred)
            tot_c += len(match_titles(pred, split[img]))
            mo = {id(p) for p, _ in match_titles(pred, orig[img[:15]])}
            unconfirmed += [(img, p["read"][:50].replace("\n", " "), p["title"][:50]) for p in pred if id(p) not in mo and p["id_confidence"] >= CONFIDENT]
    methods = {}
    print(f"cost ${m.summary()['total_cost_usd']} | catalogue requests {m.stages.get('identify', {}).get('http_calls', 0)}")
    print(f"identified {tot_i}/{n} | title accuracy vs crop annotations {tot_c}/{tot_t} = {tot_c / tot_t:.1%} | "
          f"confident but not in the photo's annotations: {len(unconfirmed)}")
    for img, read, title in unconfirmed:
        print(f"  {img}: {read!r} -> {title}")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
