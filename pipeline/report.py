"""Stage 9: human-readable HTML report built only from claim_packet.json.

One self-contained file (inline CSS and a small script for sorting and filtering). Every figure keeps its
evidence: frame thumbnails link to the saved frame, prices link to their source with the retrieval time.
"""

from __future__ import annotations

import html
from pathlib import Path

from .common import LOCALES

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
    statuses = sorted({b["status"] for b in packet["books"]})
    status_opts = "".join(f'<option value="{e(x)}">{e(x.replace("_", " "))}</option>' for x in statuses)

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


def run(run_dir: Path, packet: dict) -> Path:
    out = run_dir / "report.html"
    out.write_text(render(packet))
    return out
