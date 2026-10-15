"""Stage 9: human-readable HTML report built only from claim_packet.json."""

from __future__ import annotations

import html
from pathlib import Path


def _money(v, cur: str) -> str:
    return "-" if v is None else f"{cur} {v:,.2f}"


def _link(url: str, label: str) -> str:
    return f'<a href="{html.escape(url)}" target="_blank" rel="noopener">{html.escape(label)}</a>' if url else "-"


def _frame(ref: str) -> str:
    return f'<a href="{html.escape(ref)}" target="_blank"><img src="{html.escape(ref)}" loading="lazy" alt="frame"></a>' if ref else ""


def render(packet: dict) -> str:
    s, r, t = packet["sweep"], packet["room"], packet["totals"]
    cur = s.get("currency", "")
    e = html.escape
    rows_books = []
    for b in packet["books"]:
        rc, uv = b["replacement_cost"], b["used_value"]
        rows_books.append(
            f"<tr class='{b['status']}'><td>{b['id']}</td><td>{e(b['shelf'])} #{b['position']}</td><td>{e(b['status'])}</td>"
            f"<td>{e(b['title']) or '<i>(not guessed)</i>'}<br><small>{e(b['author'])} {e(b['isbn'])}</small></td>"
            f"<td>{b['spine_height_cm'] or '-'} x {b['spine_thickness_cm'] or '-'}</td><td>{b['id_confidence']:.2f}</td>"
            f"<td>{_money(rc['amount'], cur)}<br><small>{_link(rc['url'], rc['source'][:60])} {e(rc['retrieved_at'])}{' CONVERTED' if rc.get('converted') else ''}</small></td>"
            f"<td>{_money(uv['amount'], cur)}<br><small>{_link(uv['url'], uv['source'][:60])} {e(uv.get('condition_assumed', ''))}</small></td>"
            f"<td>{_frame(b['frame_ref'])}</td></tr>"
        )
    rows_items = []
    for i in packet["items"]:
        rc = i["replacement_cost"]
        d = i["dimensions_cm"]
        price = "-" if rc["low"] is None else (_money(rc["low"], cur) if rc["low"] == rc["high"] else f"{_money(rc['low'], cur)} - {_money(rc['high'], cur)}")
        rows_items.append(
            f"<tr class='{i['status']}'><td>{i['id']}</td><td>{e(i['category'])}</td><td>{e(i['description'])}<br><small>{e(i['brand_model'])}</small></td>"
            f"<td>{d['w'] or '-'} x {d['h'] or '-'}</td><td>{e(i['status'])}</td><td>{price}<br><small>{_link(rc['url'], rc['source'][:70])} {e(rc['retrieved_at'])}</small></td>"
            f"<td>{_frame(i['frame_ref'])}</td></tr>"
        )
    review = "".join(f"<li><b>{e(q['ref_id'])}</b>: {e(q['reason'])}</li>" for q in packet["review_queue"])
    comp = packet.get("locale_comparison") or {}
    comp_rows = "".join(
        f"<tr><td>{e(c['title'])}</td><td>{_money(c['primary']['new'], comp.get('primary_currency', ''))}</td><td>{_money(c['primary']['used'], comp.get('primary_currency', ''))}</td>"
        f"<td>{_money(c['second']['new'], comp.get('second_currency', ''))}</td><td>{_money(c['second']['used'], comp.get('second_currency', ''))}</td></tr>"
        for c in comp.get("books", [])
    )
    stages = packet.get("pipeline_metrics", {}).get("stages", {})
    metric_rows = "".join(f"<tr><td>{e(k)}</td><td>{v['latency_s']}</td><td>{v['llm_calls']}</td><td>${v['cost_usd']:.4f}</td></tr>" for k, v in stages.items())

    def num(v, unit=""):
        return "-" if v is None else f"{v}{unit}"

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claim Packet {e(s['id'])}</title>
<style>
:root{{--bg:#fff;--fg:#1b1b1b;--muted:#666;--line:#ddd;--warn:#fff4d6;--bad:#fde8e8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--muted:#999;--line:#333;--warn:#3a3218;--bad:#3a1d1d}}}}
body{{font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px;max-width:1300px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 24px}}td,th{{border-bottom:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}}
img{{width:96px;border-radius:4px}}small{{color:var(--muted)}}.unidentified{{background:var(--warn)}}.needs_appraisal{{background:var(--bad)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px}}.card{{border:1px solid var(--line);border-radius:6px;padding:10px}}
.card b{{display:block;font-size:18px}}.wrap{{overflow-x:auto}}
</style></head><body>
<h1>Contents claim packet</h1>
<p>Sweep {e(s['id'])} &middot; captured {e(s.get('captured_at', ''))} &middot; {e(s.get('device', ''))} &middot; {s.get('duration_s', 0)} s &middot; {e(s.get('country', ''))} / {e(cur)}</p>
<div class="grid">
<div class="card">Books<b>{t['book_count']}</b><small>{t['books_identified']} identified, {t['books_unidentified']} unidentified, {t['books_needs_appraisal']} appraisal</small></div>
<div class="card">Books replacement<b>{_money(t['books_replacement_cost'], cur)}</b><small>used value {_money(t['books_used_value'], cur)}</small></div>
<div class="card">Items replacement<b>{_money(t['items_replacement_cost_low'], cur)} - {_money(t['items_replacement_cost_high'], cur)}</b></div>
<div class="card">Shelf run<b>{t['shelf_run_m']} m</b></div>
<div class="card">Excluded from totals<b>{t['excluded_from_totals']}</b><small>see review queue</small></div>
</div>
<h2>Room</h2>
<p>{num(r['length_m'], ' m')} x {num(r['width_m'], ' m')}, ceiling {num(r['height_m'], ' m')} ({e(r.get('shape', ''))}).
Floor {num(r['floor_area_m2'], ' m²')} ({num(r.get('floor_area_ft2'), ' ft²')}); walls {num(r['wall_area_m2'], ' m²')} ({num(r.get('wall_area_ft2'), ' ft²')}), {e(r.get('wall_area_basis', ''))};
shelved wall {num(r['shelved_wall_area_m2'], ' m²')}. Confidence {r['confidence']}.<br><small>Scale: {e(r['scale_method'])}</small></p>
<h2>Review queue ({len(packet['review_queue'])})</h2><ul>{review}</ul>
<h2>Books</h2><div class="wrap"><table><tr><th>ID</th><th>Shelf</th><th>Status</th><th>Title</th><th>H x T cm</th><th>ID conf</th><th>Replacement</th><th>Used</th><th>Frame</th></tr>{''.join(rows_books)}</table></div>
<h2>Other contents</h2><div class="wrap"><table><tr><th>ID</th><th>Category</th><th>Description</th><th>W x H cm</th><th>Status</th><th>Replacement</th><th>Frame</th></tr>{''.join(rows_items)}</table></div>
<h2>Locale check: same books priced for {e(comp.get('second_country', '-'))}</h2><div class="wrap"><table><tr><th>Title</th><th>New ({e(comp.get('primary_currency', ''))})</th><th>Used</th><th>New ({e(comp.get('second_currency', ''))})</th><th>Used</th></tr>{comp_rows}</table></div>
<h2>Pipeline cost and latency</h2><table><tr><th>Stage</th><th>Seconds</th><th>LLM calls</th><th>Cost</th></tr>{metric_rows}</table>
<p><small>Totals are computed in code from the line items above. Prices are eBay active-listing asking prices or retailer page prices, each with its URL and retrieval time; nothing is priced from model memory.</small></p>
</body></html>"""


def run(run_dir: Path, packet: dict) -> Path:
    out = run_dir / "report.html"
    out.write_text(render(packet))
    return out
