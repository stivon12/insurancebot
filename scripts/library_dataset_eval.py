"""Score the spine-counting and identification stages on the Library Dataset (CVC Barcelona, arXiv 2407.19812).

Each shelf photo becomes a one-frame "sweep" that runs through pipeline.spines and pipeline.identify
exactly as in the app (Gemini + Google Books / Open Library), then is compared with the dataset's
per-book annotations (library_books_annotations.csv: one row per book per image).

  uv run python scripts/library_dataset_eval.py --images ~/datasets/library-dataset/data/images \
      --csv ~/datasets/library-dataset/data/annotations/library_books_annotations.csv --n 20 --out runs/libds

Metrics (same definitions as eval/eval.py):
  count error per image and overall; title accuracy = matched / annotated books with a title;
  confidently wrong = identified with id_confidence >= 0.7 but matching no annotated title in that image.
Images are a random sample with a fixed seed; every sampled image counts, none are dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import shutil
import time
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.eval import CONFIDENT, match_titles  # noqa: E402
from pipeline import identify, spines  # noqa: E402
from pipeline.common import Metrics, write_json  # noqa: E402


MASKS: dict[str, int] = defaultdict(int)  # every annotated mask per image, incl. not_book: upper bound on visible books


def load_truth(csv_path: Path) -> dict[str, list[dict]]:
    """Books per split image. The split images are overlapping crops of one photo and a book seems to be annotated in
    only one crop, so 'books' undercounts what is visible; 'books + not_book masks' bounds it from above."""
    by_img: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            MASKS[r["image_name"]] += 1
            if r.get("not_book", "False") == "True":
                continue
            by_img[r["image_name"]].append({"title": (r["title"] or "").strip(), "author": r["author"], "isbn": r["isbn"]})
    return by_img


def make_run(img: Path, root: Path) -> Path:
    d = root / img.stem
    (d / "frames").mkdir(parents=True, exist_ok=True)
    shutil.copy(img, d / "frames" / "f_000001.jpg")
    import cv2

    h, w = cv2.imread(str(img)).shape[:2]
    (d / "frames.jsonl").write_text(json.dumps({"i": 1, "file": "frames/f_000001.jpg", "t_ms": 0, "w": w, "h": h, "phase": "shelves"}) + "\n")
    return d


async def one(img: Path, truth: list[dict], root: Path, metrics: Metrics, count_only: bool = False) -> dict:
    d = make_run(img, root)
    kf = {"shelves": [{"i": 1, "file": "frames/f_000001.jpg", "t_ms": 0}]}
    books = await spines.run(d, kf, metrics)
    idn = {b["id"]: {} for b in books} if count_only else await identify.run(d, books, metrics)
    pred = [
        {"id": b["id"], "title": (idn[b["id"]].get("match") or {}).get("title", ""), "id_confidence": idn[b["id"]].get("id_confidence", 0.0)}
        for b in books
        if idn[b["id"]].get("status") == "identified"
    ]
    titled = [t for t in truth if t["title"]]
    pairs = match_titles(pred, titled)
    matched = {id(p) for p, _ in pairs}
    wrong = [p for p in pred if id(p) not in matched and p["id_confidence"] >= CONFIDENT]
    return {
        "image": img.name,
        "true_count": len(truth),
        "all_masks": MASKS[img.name],
        "pred_count": len(books),
        "titled": len(titled),
        "identified": len(pred),
        "correct": len(pairs),
        "confidently_wrong": len(wrong),
        "wrong_titles": [p["title"] for p in wrong],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("runs/libds"))
    ap.add_argument("--count-only", action="store_true", help="skip spine reading/identification (cheap count check)")
    ap.add_argument("--parallel", type=int, default=3, help="photos processed at once")
    ap.add_argument("--max-usd", type=float, default=2.0, help="stop before starting an image once spend reaches this")
    a = ap.parse_args()
    truth = load_truth(a.csv)
    imgs = sorted(p for p in a.images.rglob("*.jpg") if p.name in truth)
    if not imgs:
        sys.exit(f"no annotated images found under {a.images}")
    random.Random(a.seed).shuffle(imgs)
    imgs = imgs[: a.n]
    metrics = Metrics()
    rows = []
    sem = asyncio.Semaphore(a.parallel)
    stopped = False

    async def worker(img):
        nonlocal stopped
        async with sem:
            # Spend is checked before each photo starts; with N in flight the cap can be exceeded by < N photos.
            if stopped or metrics.summary()["total_cost_usd"] >= a.max_usd:
                stopped = True
                return
            r = await one(img, truth[img.name], a.out, metrics, a.count_only)
            rows.append(r)
            print(f"{r['image']}: count {r['pred_count']}/{r['true_count']}  titles {r['correct']}/{r['titled']}  confidently wrong {r['confidently_wrong']}", flush=True)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in imgs))
    if stopped:
        print(f"stopped early: spend reached the ${a.max_usd} cap after {len(rows)} images", flush=True)
    T = sum(r["true_count"] for r in rows)
    Pc = sum(r["pred_count"] for r in rows)
    titled = sum(r["titled"] for r in rows)
    ident = sum(r["identified"] for r in rows)
    summary = {
        "images": len(rows),
        "books_true": T,
        "books_pred": Pc,
        "count_error_overall": round(abs(Pc - T) / T, 4) if T else None,
        "books_all_masks": sum(r["all_masks"] for r in rows),
        "photos_within_books_to_all_masks": sum(r["true_count"] <= r["pred_count"] <= r["all_masks"] for r in rows),
        "count_error_per_image_mean": round(sum(abs(r["pred_count"] - r["true_count"]) / r["true_count"] for r in rows if r["true_count"]) / len(rows), 4),
        "title_accuracy": round(sum(r["correct"] for r in rows) / titled, 4) if titled else None,
        "confidently_wrong_rate": round(sum(r["confidently_wrong"] for r in rows) / ident, 4) if ident else None,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "cost": metrics.summary(),
    }
    write_json(a.out / "summary.json", {"summary": summary, "per_image": rows})
    print(json.dumps({k: v for k, v in summary.items() if k != "cost"}, indent=1))
    print(f"cost ${summary['cost']['total_cost_usd']}  ->  {a.out / 'summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
