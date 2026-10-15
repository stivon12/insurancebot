"""Score a claim packet against hand-collected ground truth, one row per pass bar in the brief.

Usage:
  uv run python -m eval.eval --packet runs/<id>/claim_packet.json --truth ground_truth/

ground_truth/ (templates in ground_truth_template/):
  books.csv   every book on the shelves: title, author, legible (yes/no), height_cm, thickness_cm,
              price_new, price_used, price_source   (dims for >=20 rows, prices for >=15 rows)
  items.csv   every non-book item: name, category
  room.json   tape measurements: length_m, width_m, height_m, floor_area_m2, wall_area_m2

Books are matched by title (fuzzy, one-to-one), so no shelf naming convention is needed.
All books count - accuracy is never reported on a subset of "good" shelves.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rapidfuzz import fuzz

TITLE_MATCH = 85
CONFIDENT = 0.7


def _norm(s: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in s or "").split())


def _read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def match_titles(pred: list[dict], truth: list[dict]) -> list[tuple[dict, dict]]:
    """Greedy one-to-one title matching, best scores first."""
    pairs = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            s = fuzz.token_set_ratio(_norm(p["title"]), _norm(t["title"]))
            if s >= TITLE_MATCH:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    used_p, used_t, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            out.append((pred[i], truth[j]))
    return out


def pct_err(pred, true) -> float | None:
    if pred is None or not true:
        return None
    return abs(pred - true) / true


def evaluate(packet: dict, truth_dir: Path) -> list[dict]:
    books_t = _read_csv(truth_dir / "books.csv")
    items_t = _read_csv(truth_dir / "items.csv")
    room_t = json.loads((truth_dir / "room.json").read_text()) if (truth_dir / "room.json").exists() else {}
    rows = []

    def row(measure, value, bar, passed, detail=""):
        rows.append({"measure": measure, "result": value, "pass_bar": bar, "pass": passed, "detail": detail})

    # 1. Book count
    true_n, pred_n = len(books_t), packet["totals"]["book_count"]
    e = pct_err(pred_n, true_n)
    row("Book count", f"{pred_n} vs {true_n} true ({e:.1%} off)" if e is not None else "no truth", "within 5%", e is not None and e <= 0.05)

    # 2. Title identification
    identified = [b for b in packet["books"] if b["status"] in ("identified", "needs_appraisal") and b["title"]]
    legible = [t for t in books_t if t.get("legible", "").lower() in ("yes", "y", "true", "1")]
    pairs = match_titles(identified, legible)
    rate = len(pairs) / len(legible) if legible else 0
    all_pairs = match_titles(identified, books_t)
    matched_ids = {id(p) for p, _ in all_pairs}
    wrong_conf = [b for b in identified if id(b) not in matched_ids and b["id_confidence"] >= CONFIDENT]
    wrong_rate = len(wrong_conf) / len(identified) if identified else 0
    row(
        "Title identification",
        f"{rate:.0%} of {len(legible)} legible correct; {wrong_rate:.1%} confidently wrong ({len(wrong_conf)})",
        ">=70% correct, <=3% confidently wrong",
        rate >= 0.7 and wrong_rate <= 0.03,
        "; ".join(f"{b['id']}: '{b['title']}'" for b in wrong_conf[:10]),
    )

    # 3. Spine dimensions (hand-measured sample)
    measured = [t for t in books_t if _num(t.get("height_cm")) and _num(t.get("thickness_cm"))]
    dim_pairs = match_titles(identified, measured)
    errs_h = [pct_err(p["spine_height_cm"], _num(t["height_cm"])) for p, t in dim_pairs]
    errs_t = [pct_err(p["spine_thickness_cm"], _num(t["thickness_cm"])) for p, t in dim_pairs]
    ok = [h is not None and th is not None and h <= 0.15 and th <= 0.15 for h, th in zip(errs_h, errs_t)]
    frac = sum(ok) / len(measured) if measured else 0
    mh = sum(e for e in errs_h if e is not None) / max(1, len([e for e in errs_h if e is not None]))
    mt = sum(e for e in errs_t if e is not None) / max(1, len([e for e in errs_t if e is not None]))
    row(
        "Spine dimensions",
        f"{sum(ok)}/{len(measured)} within 15% (matched {len(dim_pairs)}); mean error height {mh:.1%}, thickness {mt:.1%}",
        "within 15% on 20 hand-measured",
        len(measured) > 0 and frac >= 0.8,
        "unmatched measured spines count as misses",
    )

    # 4. Book prices (same source type)
    priced_t = [t for t in books_t if _num(t.get("price_new"))]
    price_pairs = match_titles(identified, priced_t)
    p_ok = [
        (p["replacement_cost"]["amount"] is not None and pct_err(p["replacement_cost"]["amount"], _num(t["price_new"])) <= 0.25)
        for p, t in price_pairs
    ]
    row(
        "Book prices",
        f"{sum(p_ok)}/{len(priced_t)} new prices within 25% (matched {len(price_pairs)})",
        "within 25% on 15 hand-checked",
        len(priced_t) > 0 and sum(p_ok) / len(priced_t) >= 0.8,
    )

    # 5. Non-book items: found with the correct category
    remaining = [i["category"] for i in packet["items"]]
    found = 0
    for t in items_t:
        if t["category"] in remaining:
            remaining.remove(t["category"])
            found += 1
    recall = found / len(items_t) if items_t else 0
    row("Non-book items", f"{found}/{len(items_t)} found with correct category ({recall:.0%})", ">=80%", recall >= 0.8)

    # 6-7. Room areas
    r = packet["room"]
    for key, label, bar in (("floor_area_m2", "Floor area", 0.10), ("wall_area_m2", "Wall area", 0.15)):
        e = pct_err(r.get(key), room_t.get(key))
        row(label, f"{r.get(key)} vs {room_t.get(key)} m2 ({e:.1%} off)" if e is not None else f"{r.get(key)} vs {room_t.get(key)}", f"within {bar:.0%}", e is not None and e <= bar)

    # 8. Time to packet
    wall = (packet.get("pipeline_metrics") or {}).get("wall_clock_s")
    row("Time to packet", f"{wall} s", "< 300 s", wall is not None and wall < 300)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    args = ap.parse_args()
    rows = evaluate(json.loads(args.packet.read_text()), args.truth)
    print("| Measure | Result | Pass bar | Pass |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r['measure']} | {r['result']} | {r['pass_bar']} | {'PASS' if r['pass'] else 'FAIL'} |")
    out = args.packet.parent / "eval_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
