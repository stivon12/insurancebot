"""Stage 8: build claim_packet.json from the stage outputs. Totals are computed here, in code.

Also resolves the policyholder's voice overrides (captured live) onto line items:
an override applies to what was in view when it was spoken, narrowed by its text.
"""

from __future__ import annotations

from pathlib import Path

from rapidfuzz import fuzz

from .common import read_json, write_json

SQFT_PER_M2 = 10.7639
LOW_ID_CONF = 0.7


def resolve_overrides(overrides: list[dict], books: list[dict], identified: dict, items: list[dict], frames_t: dict[str, float]) -> dict:
    """Map spoken overrides to ids. Unresolvable overrides go to the review queue, never silently dropped."""
    out = {"excluded_books": {}, "excluded_items": {}, "books": {}, "items": {}, "unresolved": []}

    def near(files: list[str], t: float, window: float) -> bool:
        return any(abs(frames_t.get(f, -1e12) - t) <= window for f in files)

    for ov in overrides:
        t, action, text = ov.get("t_ms", 0), ov.get("action"), (ov.get("target") or "") + " " + (ov.get("note") or "")
        if ov.get("target_type") in ("shelf", "book"):
            cands = [b for b in books if near([o["file"] for o in b["observations"]], t, 2500)]
            if ov.get("target_type") == "book" and text.strip():
                scored = [(fuzz.partial_ratio(text.lower(), (identified.get(b["id"], {}).get("match") or {}).get("title", "").lower()), b) for b in cands]
                scored = [s for s in scored if s[0] >= 70]
                cands = [max(scored, key=lambda s: s[0])[1]] if scored else []
            if not cands:
                out["unresolved"].append(ov)
                continue
            for b in cands:
                if action == "exclude":
                    out["excluded_books"][b["id"]] = ov
                elif action in ("first_edition", "signed"):
                    out["books"].setdefault(b["id"], {})[action] = True
        elif ov.get("target_type") == "item":
            cands = [i for i in items if near(i["sightings"], t, 3000)]
            if text.strip() and cands:
                cands = [max(cands, key=lambda i: fuzz.partial_ratio(text.lower(), f"{i['category']} {i['description']}".lower()))]
            if not cands:
                out["unresolved"].append(ov)
                continue
            for i in cands:
                if action == "exclude":
                    out["excluded_items"][i["id"]] = ov
                elif action in ("is_print", "is_original"):
                    out["items"].setdefault(i["id"], {})["is_print"] = action == "is_print"
        else:
            out["unresolved"].append(ov)
    return out


def _price_line(p: dict | None, used: bool = False) -> dict:
    if not p:
        base = {"amount": None, "source": "", "url": "", "retrieved_at": ""}
        return base | ({"condition_assumed": ""} if used else {"converted": False})
    line = {"amount": p["amount"], "currency": p["currency"], "source": p["source"], "url": p["url"], "retrieved_at": p["retrieved_at"]}
    line["evidence_urls"] = p.get("evidence_urls", [])
    if used:
        line["condition_assumed"] = p.get("condition_assumed", "")
        line["converted"] = p.get("converted", False)
    else:
        line["converted"] = p.get("converted", False)
    if p.get("fx"):
        line["fx"] = p["fx"]
    return line


def build(run_dir: Path, *, sweep: dict, books: list[dict], identified: dict, measurements: dict, room: dict, items: list[dict], prices: dict, ov: dict, metrics: dict, locale_comparison: dict | None) -> dict:
    review: list[dict] = []
    book_lines = []
    for b in books:
        if b["id"] in ov["excluded_books"]:
            continue
        idn = identified.get(b["id"], {})
        m = measurements["books"].get(b["id"], {})
        match = idn.get("match") or {}
        pr = prices["books"].get(b["id"], {})
        frame = b["best"]["file"]
        if idn.get("status") == "identified":
            status = "needs_appraisal" if pr.get("appraisal_reasons") else "identified"
        else:
            status = "unidentified"
        line = {
            "id": b["id"],
            "shelf": b["shelf"],
            "position": b["position"],
            "frame_ref": frame,
            "box_px": [round(v) for v in b["best"]["box_px"]],
            "status": status,
            "title": match.get("title", "") if status != "unidentified" else "",
            "author": ", ".join(match.get("authors", [])) if status != "unidentified" else "",
            "edition": (f"{match.get('publisher', '')} {match.get('published', '')}".strip() if "matches" in idn.get("edition_basis", "") else ""),
            "isbn": match.get("isbn", "") if status != "unidentified" else "",
            "spine_text_read": (idn.get("read") or {}).get("verbatim", ""),
            "catalogue_url": match.get("url", ""),
            "spine_height_cm": m.get("spine_height_cm"),
            "spine_thickness_cm": m.get("spine_thickness_cm"),
            "scale_method": m.get("scale_method", ""),
            "id_confidence": idn.get("id_confidence", 0.0) if status != "unidentified" else 0.0,
            "replacement_cost": _price_line(pr.get("replacement_cost") if status == "identified" else None),
            "used_value": _price_line(pr.get("used_value") if status == "identified" else None, used=True),
        }
        book_lines.append(line)
        if status == "unidentified":
            review.append({"ref_id": b["id"], "reason": f"unidentified book ({idn.get('reason', 'not read')}); dimensions recorded, not priced"})
        elif status == "needs_appraisal":
            review.append({"ref_id": b["id"], "reason": "needs human appraisal: " + "; ".join(pr["appraisal_reasons"])})
        else:
            if line["id_confidence"] < LOW_ID_CONF:
                review.append({"ref_id": b["id"], "reason": f"identification confidence {line['id_confidence']} below {LOW_ID_CONF}"})
            for note in pr.get("notes", []):
                review.append({"ref_id": b["id"], "reason": note + "; excluded from that total"})
            if line["replacement_cost"].get("converted") or line["used_value"].get("converted"):
                review.append({"ref_id": b["id"], "reason": "price converted from another market (no local listing)"})
        if m.get("scale_method") == "standard_format_prior":
            review.append({"ref_id": b["id"], "reason": "spine size from standard-format prior (too few identified books in frame to use the book ruler)"})
        elif m.get("spine_height_cm") is None:
            review.append({"ref_id": b["id"], "reason": "spine dimensions not measurable"})

    item_lines = []
    for it in items:
        if it["id"] in ov["excluded_items"]:
            continue
        pr = prices["items"].get(it["id"], {})
        rc = pr.get("replacement_cost") or {}
        dims = it.get("dimensions_cm") or {}
        line = {
            "id": it["id"],
            "category": it["category"],
            "description": f"{it['description']} ({it['material']})".strip(),
            "brand_model": it.get("brand_model", ""),
            "frame_ref": it["frame_ref"],
            "dimensions_cm": {"w": dims.get("w"), "h": dims.get("h"), "d": dims.get("d")},
            "dimensions_basis": dims.get("basis", "not measurable from this view"),
            "status": pr.get("status", "needs_appraisal"),
            "replacement_cost": {
                "low": rc.get("low"), "high": rc.get("high"), "currency": rc.get("currency", prices["currency"]),
                "source": rc.get("source", ""), "url": rc.get("url", ""), "retrieved_at": rc.get("retrieved_at", ""),
                "evidence_urls": rc.get("evidence_urls", []),
            },
            "confidence": it["confidence"],
        }
        item_lines.append(line)
        if line["status"] == "needs_appraisal":
            review.append({"ref_id": it["id"], "reason": pr.get("reason", "no sourced price")})
        if it["confidence"] < 0.5:
            review.append({"ref_id": it["id"], "reason": f"detection confidence {it['confidence']}"})

    for o in ov["unresolved"]:
        review.append({"ref_id": "override", "reason": f"voice override could not be matched to a line: {o.get('action')} '{o.get('target', '')}' at {o.get('t_ms', 0) / 1000:.0f}s"})
    for b_id, o in ov["excluded_books"].items():
        review.append({"ref_id": b_id, "reason": f"excluded by policyholder voice override ('{o.get('target', '')}') - verify"})
    for msg in room.get("review", []):
        review.append({"ref_id": "room", "reason": msg})
    if room.get("confidence", 1) < 0.6:
        review.append({"ref_id": "room", "reason": f"room geometry confidence {room.get('confidence')}"})

    totals = compute_totals(book_lines, item_lines, measurements)
    packet = {
        "sweep": sweep,
        "room": {
            "length_m": room.get("length_m"),
            "width_m": room.get("width_m"),
            "height_m": room.get("height_m"),
            "floor_area_m2": room.get("floor_area_m2"),
            "floor_area_ft2": room.get("floor_area_ft2"),
            "wall_area_m2": room.get("wall_area_m2"),
            "wall_area_ft2": room.get("wall_area_ft2"),
            "shelved_wall_area_m2": room.get("shelved_wall_area_m2"),
            "shelved_wall_area_ft2": room.get("shelved_wall_area_ft2"),
            "shape": room.get("shape", ""),
            "polygon_m": room.get("polygon_m"),
            "scale_method": (
                "Book ruler: catalogue heights of identified books -> distance to bookcase -> camera height; "
                f"focal length {room.get('focal_method', 'n/a')}; gravity and yaw from phone motion sensors"
            ),
            "wall_area_basis": room.get("wall_area_basis", ""),
            "confidence": room.get("confidence", 0),
        },
        "books": book_lines,
        "items": item_lines,
        "totals": totals,
        "review_queue": review,
        "locale_comparison": locale_comparison,
        "pipeline_metrics": metrics,
        "shelf_run_by_shelf_m": {k: v for k, v in measurements.get("shelves", {}).items()},
    }
    write_json(run_dir / "claim_packet.json", packet)
    return packet


def compute_totals(book_lines: list[dict], item_lines: list[dict], measurements: dict) -> dict:
    def amt(p):
        return p.get("amount") if p else None

    priced_new = [amt(b["replacement_cost"]) for b in book_lines if b["status"] == "identified" and amt(b["replacement_cost"]) is not None]
    priced_used = [amt(b["used_value"]) for b in book_lines if b["status"] == "identified" and amt(b["used_value"]) is not None]
    items_priced = [i for i in item_lines if i["status"] in ("priced", "range") and i["replacement_cost"]["low"] is not None]
    excluded = sum(
        1 for b in book_lines if b["status"] != "identified" or amt(b["replacement_cost"]) is None or amt(b["used_value"]) is None
    ) + sum(1 for i in item_lines if i not in items_priced)
    shelves_present = {b["shelf"] for b in book_lines}
    return {
        "book_count": len(book_lines),
        "books_identified": sum(1 for b in book_lines if b["status"] == "identified"),
        "books_needs_appraisal": sum(1 for b in book_lines if b["status"] == "needs_appraisal"),
        "books_unidentified": sum(1 for b in book_lines if b["status"] == "unidentified"),
        "shelf_run_m": round(sum(v for k, v in measurements.get("shelves", {}).items() if k in shelves_present), 2),
        "books_replacement_cost": round(sum(priced_new), 2),
        "books_used_value": round(sum(priced_used), 2),
        "items_replacement_cost_low": round(sum(i["replacement_cost"]["low"] for i in items_priced), 2),
        "items_replacement_cost_high": round(sum(i["replacement_cost"]["high"] for i in items_priced), 2),
        "excluded_from_totals": excluded,
    }


def load(run_dir: Path) -> dict:
    return read_json(run_dir / "claim_packet.json")
