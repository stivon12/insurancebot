"""Claim packet: lines, review queue and totals (computed in code), and the HTML report."""

from __future__ import annotations

import html
from pathlib import Path
from rapidfuzz import fuzz

from .common import LOCALES, write_json


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
def build(run_dir: Path, *, sweep: dict, books: list[dict], identified: dict, measurements: dict, room: dict, items: list[dict], prices: dict, ov: dict, metrics: dict, locale_comparison: dict | None, possible_books: list[dict] | None = None) -> dict:
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
            "thickness_basis": m.get("thickness_basis", ""),
            "box_basis": m.get("box_basis", ""),
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
        for note in b.get("review_notes", []):
            review.append({"ref_id": b["id"], "reason": note})
        if m.get("review_note"):
            review.append({"ref_id": b["id"], "reason": m["review_note"]})
        # A standard-format size is information on the line itself (scale_method), not a task for the reviewer.
        if m.get("scale_method") == "standard_format_prior":
            pass
        elif m.get("box_basis") == "loose (Gemini)" and m.get("spine_thickness_cm") is not None:
            review.append({"ref_id": b["id"], "reason": "spine thickness from a loose box (no tight detector box matched): may span neighbours"})
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
            "scale_method": room.get("scale_method", ""),
            "wall_area_basis": room.get("wall_area_basis", ""),
            "confidence": room.get("confidence", 0),
        },
        "books": book_lines,
        "items": item_lines,
        "totals": totals,
        "review_queue": review,
        "locale_comparison": locale_comparison,
        "possible_books_not_counted": possible_books or [],
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


e = html.escape
def _money(v, cur: str) -> str:
    return "–" if v is None else f"{e(cur)} {v:,.2f}"
def _link(url: str, label: str) -> str:
    return f'<a href="{e(url)}" target="_blank" rel="noopener">{e(label)}</a>' if url else e(label)
def _frame(ref: str) -> str:
    return f'<a class="thumb" href="{e(ref)}" target="_blank"><img src="{e(ref)}" loading="lazy" alt="evidence frame"></a>' if ref else ""
def _badge(status: str) -> str:
    label = {"identified": "Identified", "unidentified": "Unidentified", "needs_appraisal": "Appraisal", "priced": "Priced",
             "range": "Range", "excluded": "Excluded"}.get(status, status.replace("_", " ").title())
    return f'<span class="badge b-{e(status)}">{e(label)}</span>'
def _price_cell(p: dict, cur: str, extra: str = "") -> str:
    if p.get("amount") is None:
        return '<span class="muted">not priced</span>'
    conv = ' <span class="tag">converted</span>' if p.get("converted") else ""
    return (f'<div class="amt">{_money(p["amount"], cur)}{conv}</div>'
            f'<div class="src">{_link(p.get("url", ""), (p.get("source") or "")[:90])}</div>'
            f'<div class="when">{e(p.get("retrieved_at") or "")} {e(extra)}</div>')
def _num(v, unit: str = "") -> str:
    return "–" if v is None else f"{v}{unit}"
def render(packet: dict) -> str:
    s, r, t = packet["sweep"], packet["room"], packet["totals"]
    cur = s.get("currency", "")
    review = packet["review_queue"]
    review_by = {}
    for q in review:
        review_by.setdefault(q["ref_id"], []).append(q["reason"])

    book_rows = []
    for b in packet["books"]:
        rc, uv = b["replacement_cost"], b["used_value"]
        title = e(b["title"]) if b["title"] else '<span class="muted">not guessed</span>'
        notes = "".join(f'<div class="note">{e(n)}</div>' for n in review_by.get(b["id"], []))
        dims = f'{_num(b["spine_height_cm"])} × {_num(b["spine_thickness_cm"])}'
        book_rows.append((b["status"], 
            f'<tr id="{e(b["id"])}" data-status="{e(b["status"])}" data-search="{e((b["title"] + " " + b["author"] + " " + b["isbn"] + " " + b["spine_text_read"]).lower())}">'
            f'<td class="mono">{e(b["id"])}<div class="muted">{e(b["shelf"])} #{b["position"]}</div></td>'
            f'<td>{_badge(b["status"])}</td>'
            f'<td><div class="title">{title}</div><div class="muted">{e(b["author"])}{" · ISBN " + e(b["isbn"]) if b["isbn"] else ""}</div>'
            f'<div class="read">spine: “{e(b["spine_text_read"][:80])}”</div>{notes}</td>'
            f'<td data-sort="{b["spine_height_cm"] or 0}" class="num">{dims}<div class="muted">{e(b.get("box_basis") or "")}</div></td>'
            f'<td data-sort="{b["id_confidence"]}" class="num">{b["id_confidence"]:.2f}</td>'
            f'<td data-sort="{rc["amount"] or 0}">{_price_cell(rc, cur)}</td>'
            f'<td data-sort="{uv["amount"] or 0}">{_price_cell(uv, cur, uv.get("condition_assumed", ""))}</td>'
            f'<td>{_frame(b["frame_ref"])}</td></tr>'
        ))

    item_rows = []
    for i in packet["items"]:
        rc, d = i["replacement_cost"], i["dimensions_cm"]
        if rc["low"] is None:
            price = '<span class="muted">not priced</span>'
        else:
            amt = _money(rc["low"], cur) if rc["low"] == rc["high"] else f'{_money(rc["low"], cur)} – {_money(rc["high"], cur)}'
            price = (f'<div class="amt">{amt}</div><div class="src">{_link(rc.get("url", ""), (rc.get("source") or "")[:90])}</div>'
                     f'<div class="when">{e(rc.get("retrieved_at") or "")}</div>')
        notes = "".join(f'<div class="note">{e(n)}</div>' for n in review_by.get(i["id"], []))
        item_rows.append((i["status"],
            f'<tr id="{e(i["id"])}" data-status="{e(i["status"])}">'
            f'<td class="mono">{e(i["id"])}</td><td>{_badge(i["status"])}</td>'
            f'<td><div class="title">{e(i["category"].replace("_", " "))}</div><div>{e(i["description"])}</div>'
            f'<div class="muted">{e(i["brand_model"])}</div>{notes}</td>'
            f'<td class="num">{_num(d.get("w"))} × {_num(d.get("h"))}</td>'
            f'<td data-sort="{rc["low"] or 0}">{price}</td><td>{_frame(i["frame_ref"])}</td></tr>'
        ))

    def ref(rid: str) -> str:
        known = {b["id"] for b in packet["books"]} | {i["id"] for i in packet["items"]}
        target = rid if rid in known else ("room" if rid == "room" else "")
        return f'<a href="#{e(target)}" class="mono">{e(rid)}</a>' if target else f'<span class="mono">{e(rid)}</span>'

    # Review queue grouped by kind of problem, with counts, instead of one line per issue.
    import re as _re

    groups: dict[str, list[str]] = {}
    for q in review:
        key = _re.sub(r"\b[BI]\d{3}\b|\d+(?:\.\d+)?", "#", q["reason"]).split(";")[0].strip()
        groups.setdefault(key, []).append(q["ref_id"])
    review_html = "".join(
        f"<li><span class='cnt'>{len(ids)}</span><span>{e(k)}<div class='ids'>{' '.join(ref(i) for i in ids[:40])}</div></span></li>"
        for k, ids in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ) or "<li>Nothing to review.</li>"
    confirmed_books = [h for st, h in book_rows if st == "identified"]
    review_books = [h for st, h in book_rows if st != "identified"]
    priced_items = [h for st, h in item_rows if st in ("priced", "range")]
    review_items = [h for st, h in item_rows if st not in ("priced", "range")]
    possible = packet.get("possible_books_not_counted") or []
    possible_html = "".join(
        f'<div class="pb">{_frame(x["frame_ref"])}<div class="muted">{e(x["id"])}: {e(x["reason"])}</div></div>' for x in possible)

    comp = packet.get("locale_comparison") or {}
    pc, sc = comp.get("primary_currency", ""), comp.get("second_currency", "")
    comp_rows = "".join(
        f"<tr><td>{e(c['title'])}</td><td class='num'>{_money(c['primary']['new'], pc)}</td><td class='num'>{_money(c['primary']['used'], pc)}</td>"
        f"<td class='num'>{_money(c['second']['new'], sc)}</td><td class='num'>{_money(c['second']['used'], sc)}</td></tr>"
        for c in comp.get("books", [])
    )
    stages = packet.get("pipeline_metrics", {}).get("stages", {})
    metric_rows = "".join(
        f"<tr><td>{e(k)}</td><td class='num'>{v['latency_s']}</td><td class='num'>{v['llm_calls']}</td><td class='num'>${v['cost_usd']:.4f}</td></tr>"
        for k, v in stages.items()
    )
    pm = packet.get("pipeline_metrics", {})

    def area(m2, ft2) -> str:
        if m2 is None:
            return "–"
        return f'<span class="m2">{m2} m²</span><span class="ft2">{_num(ft2)} ft²</span>'

    room_ok = r.get("floor_area_m2") is not None
    cards = [
        ("Books", str(t["book_count"]), f'{t["books_identified"]} identified · {t["books_unidentified"]} unreadable · {t["books_needs_appraisal"]} appraisal'),
        ("Books — replacement", _money(t["books_replacement_cost"], cur), f'used value {_money(t["books_used_value"], cur)}'),
        ("Other contents", f'{_money(t["items_replacement_cost_low"], cur)} – {_money(t["items_replacement_cost_high"], cur)}', f'{len(packet["items"])} items'),
        ("Room floor", area(r.get("floor_area_m2"), r.get("floor_area_ft2")) if room_ok else "not measured", f'shelf run {t["shelf_run_m"]} m'),
        ("Excluded from totals", str(t["excluded_from_totals"]), f'{len(review)} lines to review'),
    ]
    cards_html = "".join(f'<div class="card"><div class="k">{e(k)}</div><div class="v">{v}</div><div class="s">{e(sub)}</div></div>' for k, v, sub in cards)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claim Packet {e(s['id'])}</title>
<style>
:root{{--bg:#f6f7f9;--panel:#fff;--fg:#14171c;--muted:#66707d;--line:#e3e6ea;--accent:#2563eb;--head:#fbfbfc;
--ok-bg:#e7f6ec;--ok:#15803d;--warn-bg:#fff4d6;--warn:#92400e;--bad-bg:#fde8e8;--bad:#b42318;--info-bg:#e8efff;--info:#1d4ed8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0b0d10;--panel:#15181d;--fg:#eef0f3;--muted:#9aa3ae;--line:#262b32;--accent:#7aa2ff;--head:#1a1e24;
--ok-bg:#12301d;--ok:#4ade80;--warn-bg:#352a10;--warn:#fbbf24;--bad-bg:#3a1a1a;--bad:#fb7185;--info-bg:#17223d;--info:#93b4ff}}}}
*{{box-sizing:border-box}}
body{{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;background:var(--bg);color:var(--fg);-webkit-font-smoothing:antialiased}}
a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}
header{{background:var(--panel);border-bottom:1px solid var(--line);padding:20px 16px 14px}}
.wrap{{max-width:1240px;margin:0 auto}}
h1{{margin:0;font-size:22px;letter-spacing:-.01em}}.meta{{color:var(--muted);font-size:13px;margin-top:4px}}
nav{{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line)}}
nav .wrap{{display:flex;gap:4px;overflow-x:auto;padding:8px 16px}}
nav a{{padding:6px 10px;border-radius:8px;color:var(--muted);font-weight:600;font-size:13px;white-space:nowrap}}nav a:hover{{background:var(--bg);text-decoration:none;color:var(--fg)}}
main{{padding:16px}}section{{margin:0 0 28px;scroll-margin-top:56px}}tr{{scroll-margin-top:96px}}
h2{{font-size:17px;margin:0 0 10px;display:flex;align-items:center;gap:8px}}h2 .count{{font-size:12px;color:var(--muted);font-weight:600;background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:1px 8px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px 14px}}
.card .k{{font-size:12px;color:var(--muted)}}.card .v{{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums;margin:2px 0}}.card .s{{font-size:12px;color:var(--muted)}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}}
.roomgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line)}}
.roomgrid div{{background:var(--panel);padding:10px 14px}}.roomgrid .k{{font-size:12px;color:var(--muted)}}.roomgrid .v{{font-weight:700;font-size:16px;font-variant-numeric:tabular-nums}}
.roomnote{{padding:10px 14px;font-size:12.5px;color:var(--muted);border-top:1px solid var(--line)}}
.toggle{{margin-left:auto;font-size:12px;border:1px solid var(--line);border-radius:8px;overflow:hidden;display:inline-flex}}
.toggle button{{font:inherit;border:0;background:transparent;color:var(--muted);padding:3px 10px;cursor:pointer}}.toggle button.on{{background:var(--accent);color:#fff}}
body.ft .m2{{display:none}}body:not(.ft) .ft2{{display:none}}
.review{{list-style:none;margin:0;padding:0}}.review li{{display:grid;grid-template-columns:70px 1fr;gap:10px;padding:8px 14px;border-top:1px solid var(--line);font-size:13.5px}}.review li:first-child{{border-top:0}}
.tools{{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 10px}}
.tools input,.tools select{{font:inherit;padding:8px 10px;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--fg)}}.tools input{{flex:1;min-width:180px}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;min-width:860px}}
th{{position:sticky;top:0;background:var(--head);text-align:left;font-size:12px;color:var(--muted);font-weight:600;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
th[data-col]{{cursor:pointer}}th[data-col]:after{{content:" ↕";opacity:.4}}
td{{padding:10px;border-bottom:1px solid var(--line);vertical-align:top}}tr:last-child td{{border-bottom:0}}
tr:target td{{background:var(--info-bg)}}
.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}}
.title{{font-weight:600}}.muted{{color:var(--muted);font-size:12.5px}}.read{{color:var(--muted);font-size:12px;font-style:italic;margin-top:2px}}
.note{{font-size:12px;color:var(--warn);margin-top:3px}}
.amt{{font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}}.src{{font-size:12px;max-width:260px}}.when{{font-size:11.5px;color:var(--muted)}}
.tag{{font-size:10.5px;font-weight:700;text-transform:uppercase;background:var(--warn-bg);color:var(--warn);border-radius:4px;padding:1px 5px}}
.badge{{display:inline-block;font-size:11.5px;font-weight:700;border-radius:999px;padding:2px 9px;white-space:nowrap}}
.b-identified,.b-priced{{background:var(--ok-bg);color:var(--ok)}}.b-unidentified,.b-range{{background:var(--warn-bg);color:var(--warn)}}
.b-needs_appraisal{{background:var(--bad-bg);color:var(--bad)}}.b-excluded{{background:var(--line);color:var(--muted)}}
.thumb img{{width:120px;height:80px;object-fit:cover;border-radius:8px;border:1px solid var(--line);display:block}}
details>summary{{list-style:none;cursor:pointer}}details>summary::-webkit-details-marker{{display:none}}
details>summary h2:before{{content:"▸";color:var(--muted);font-size:13px}}details[open]>summary h2:before{{content:"▾"}}
details>summary .muted{{display:block;margin:-6px 0 10px}}h3{{font-size:14px;margin:16px 0 8px}}
.lead{{color:var(--muted);font-size:13px;margin:-4px 0 10px}}
.review li{{grid-template-columns:44px 1fr}}.cnt{{font-weight:700;font-variant-numeric:tabular-nums}}.ids{{margin-top:3px;display:flex;flex-wrap:wrap;gap:6px}}
.pbs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}.pb .thumb img{{width:100%;height:100px}}
@media print{{details>summary h2:before{{content:""}}}}
footer{{color:var(--muted);font-size:12.5px;padding:0 16px 32px}}
@media print{{nav,.tools,.toggle{{display:none}}body{{background:#fff;color:#000}}.panel,.card{{border-color:#ccc}}th{{position:static}}a{{color:#000}}.thumb img{{width:90px;height:60px}}tr{{break-inside:avoid}}}}
</style></head><body>
<header><div class="wrap"><h1>Contents claim packet</h1>
<div class="meta">Sweep {e(s['id'])} · captured {e(s.get('captured_at', ''))} · {s.get('duration_s', 0)} s · {e(s.get('country', ''))} / {e(cur)} · {e((s.get('device') or '')[:80])}</div></div></header>
<nav><div class="wrap"><a href="#summary">Summary</a><a href="#books">Confirmed books ({len(confirmed_books)})</a><a href="#items">Priced items ({len(priced_items)})</a><a href="#room">Room</a><a href="#review">Needs review ({len(review)})</a><a href="#locale">Second market</a></div></nav>
<main class="wrap">
<section id="summary"><div class="cards">{cards_html}</div></section>

<section id="books"><h2>Confirmed books <span class="count">{len(confirmed_books)}</span></h2>
<p class="lead">Identified from the spine and matched to a catalogue record; priced where a source was found.</p>
<div class="tools"><input id="q" type="search" placeholder="Search title, author, ISBN or spine text"></div>
<div class="panel scroll"><table id="bt"><thead><tr><th data-col="0">ID / shelf</th><th data-col="1">Status</th><th data-col="2">Title</th><th data-col="3">H × T cm</th><th data-col="4">ID conf.</th><th data-col="5">Replacement (new)</th><th data-col="6">Used value</th><th>Frame</th></tr></thead>
<tbody>{''.join(confirmed_books) or '<tr><td colspan="8" class="muted">No book identified with confidence.</td></tr>'}</tbody></table></div></section>

<section id="items"><h2>Priced items <span class="count">{len(priced_items)}</span></h2>
<div class="panel scroll"><table><thead><tr><th>ID</th><th>Status</th><th>Item</th><th>W × H cm</th><th>Replacement</th><th>Frame</th></tr></thead><tbody>{''.join(priced_items) or '<tr><td colspan="6" class="muted">No item priced.</td></tr>'}</tbody></table></div></section>

<section id="room"><h2>Room <span class="toggle"><button class="on" data-u="m2">m²</button><button data-u="ft2">ft²</button></span></h2>
<div class="panel"><div class="roomgrid">
<div><div class="k">Length × width</div><div class="v">{_num(r['length_m'])} × {_num(r['width_m'])} m</div></div>
<div><div class="k">Ceiling height</div><div class="v">{_num(r['height_m'], ' m')}</div></div>
<div><div class="k">Floor area</div><div class="v">{area(r['floor_area_m2'], r.get('floor_area_ft2'))}</div></div>
<div><div class="k">Wall area</div><div class="v">{area(r['wall_area_m2'], r.get('wall_area_ft2'))}</div></div>
<div><div class="k">Shelved wall</div><div class="v">{area(r['shelved_wall_area_m2'], r.get('shelved_wall_area_ft2'))}</div></div>
<div><div class="k">Confidence</div><div class="v">{_num(r['confidence'])}</div></div>
</div><div class="roomnote">Shape {e(r.get('shape') or '–')}. {e(r.get('wall_area_basis') or '')} Scale: {e(r['scale_method'])}</div></div></section>

<section id="review"><details><summary><h2>Needs review <span class="count">{len(review)}</span></h2><span class="muted">Unreadable or unpriced lines, appraisal items and objects not counted as books. Excluded from totals unless priced.</span></summary>
<div class="panel"><ul class="review">{review_html}</ul></div>
<h3>Books not confirmed ({len(review_books)})</h3>
<div class="panel scroll"><table><thead><tr><th>ID / shelf</th><th>Status</th><th>Title</th><th>H × T cm</th><th>ID conf.</th><th>Replacement (new)</th><th>Used value</th><th>Frame</th></tr></thead><tbody>{''.join(review_books)}</tbody></table></div>
<h3>Items not priced ({len(review_items)})</h3>
<div class="panel scroll"><table><thead><tr><th>ID</th><th>Status</th><th>Item</th><th>W × H cm</th><th>Replacement</th><th>Frame</th></tr></thead><tbody>{''.join(review_items)}</tbody></table></div>
<h3>Possible books, not counted ({len(possible)})</h3>
<p class="muted">Shapes the spine reader listed but that are unreadable, seen once and not confirmed by the spine detector (often boxes or packaging).</p>
<div class="pbs">{possible_html}</div>
</details></section>

<section id="locale"><details><summary><h2>Second-market price check: {e(LOCALES.get(comp.get('second_country', ''), {}).get('name', '–'))} ({e(sc)})</h2><span class="muted">The brief asks for the same books priced in a second country to show prices are local. Not part of the totals.</span></summary>
<div class="panel scroll"><table><thead><tr><th>Title</th><th>New ({e(pc)})</th><th>Used ({e(pc)})</th><th>New ({e(sc)})</th><th>Used ({e(sc)})</th></tr></thead><tbody>{comp_rows}</tbody></table></div></details></section>

<section id="pipeline"><details><summary><h2>Pipeline cost and latency</h2></summary>
<div class="panel scroll"><table><thead><tr><th>Stage</th><th>Seconds</th><th>Model calls</th><th>Cost</th></tr></thead><tbody>{metric_rows}
<tr><td><b>Total</b></td><td class="num">{_num(pm.get('wall_clock_s'))} wall clock</td><td></td><td class="num"><b>${pm.get('total_cost_usd', 0):.4f}</b></td></tr></tbody></table></div></details></section>
</main>
<footer class="wrap">Totals are computed in code from the line items above; lines without a price source are excluded and listed in the review queue.
Prices are eBay active-listing asking prices or Google Shopping offers (via SerpApi), each with its source URL and retrieval time; nothing is priced from model memory.</footer>
<script>
(function(){{
  var rows=[].slice.call(document.querySelectorAll('#bt tbody tr[data-search]')), q=document.getElementById('q');
  q.addEventListener('input',function(){{var v=(q.value||'').toLowerCase();rows.forEach(function(r){{r.hidden=v&&r.dataset.search.indexOf(v)<0;}});}});
  window.addEventListener('beforeprint',function(){{document.querySelectorAll('details').forEach(function(d){{d.open=true;}});}});
  if(location.hash){{var t=document.querySelector(location.hash);if(t&&t.closest('details'))t.closest('details').open=true;}}
  document.querySelectorAll('a[href^="#"]').forEach(function(a){{a.addEventListener('click',function(){{var t=document.querySelector(a.getAttribute('href'));if(t&&t.closest('details'))t.closest('details').open=true;}});}});
  document.querySelectorAll('#bt th[data-col]').forEach(function(th){{var asc=true;th.addEventListener('click',function(){{
    var c=+th.dataset.col, tb=document.querySelector('#bt tbody');
    rows.sort(function(a,b){{var x=a.cells[c],y=b.cells[c],xs=x.dataset.sort,ys=y.dataset.sort;
      var r=(xs!==undefined&&ys!==undefined)?(+xs-+ys):x.textContent.localeCompare(y.textContent);return asc?r:-r;}});
    asc=!asc;rows.forEach(function(r){{tb.appendChild(r);}});}});}});
  document.querySelectorAll('.toggle button').forEach(function(b){{b.addEventListener('click',function(){{
    document.body.classList.toggle('ft',b.dataset.u==='ft2');
    document.querySelectorAll('.toggle button').forEach(function(x){{x.classList.toggle('on',x===b);}});}});}});
}})();
</script>
</body></html>"""
def write_report(run_dir: Path, packet: dict) -> Path:
    out = run_dir / "report.html"
    out.write_text(render(packet))
    return out
