import pytest

from pipeline import assemble, identify, pricing, scale
from pipeline.common import locale


def book(bid, shelf, pos, boxes, orientation="vertical"):
    obs = [{"frame_i": fi, "file": f"frames/f_{fi:06d}.jpg", "box_px": bx, "legible": True, "orientation": orientation} for fi, bx in boxes]
    return {"id": bid, "shelf": shelf, "position": pos, "observations": obs, "best": obs[0]}


def ident(height, thickness=None):
    dims = {"height_cm": height}
    if thickness:
        dims["thickness_cm"] = thickness
    return {"status": "identified", "dimensions": dims}


# ---------- scale: the book ruler ----------

def test_leave_one_out_scale_measures_unknown_book():
    # 0.1 cm per px in frame 1. Three ruler books + one unknown 230 px tall, 30 px thick.
    books = [
        book("B1", "U1-S1", 1, [(1, [0, 0, 20, 198])]),   # 19.8 cm B-format
        book("B2", "U1-S1", 2, [(1, [20, 0, 45, 240])]),  # 24.0 cm hardback
        book("B3", "U1-S1", 3, [(1, [45, 0, 63, 178])]),  # 17.8 cm A-format
        book("B4", "U1-S1", 4, [(1, [63, 0, 93, 230])]),
    ]
    idn = {"B1": ident(19.8), "B2": ident(24.0), "B3": ident(17.8)}
    out = scale.measure(books, idn)
    b4 = out["books"]["B4"]
    assert b4["scale_method"] == "book_ruler"
    assert b4["spine_height_cm"] == pytest.approx(23.0, abs=0.2)
    assert b4["spine_thickness_cm"] == pytest.approx(3.0, abs=0.05)
    # A ruler book is measured from the OTHER rulers, not its own catalogue height
    assert out["books"]["B1"]["spine_height_cm"] == pytest.approx(19.8, abs=0.2)


def test_outlier_ruler_is_rejected():
    books = [book(f"B{i}", "U1-S1", i, [(1, [i * 20, 0, i * 20 + 20, 200])]) for i in range(1, 6)]
    idn = {f"B{i}": ident(20.0) for i in range(1, 5)}
    idn["B4"] = ident(28.0)  # wrong edition matched: 40% off, must not move the scale
    out = scale.measure(books, idn)
    assert out["books"]["B5"]["spine_height_cm"] == pytest.approx(20.0, abs=0.1)


def test_aspect_check_drops_wrong_edition_ruler():
    b = book("B1", "U1-S1", 1, [(1, [0, 0, 20, 200])])  # observed thickness/height 0.1
    rulers = scale.ruler_books([b], {"B1": ident(24.0, thickness=4.0)})  # catalogue 0.167: 67% off
    assert rulers == {}


def test_format_prior_used_when_no_rulers():
    books = [book(f"B{i}", "U1-S1", i, [(1, [i * 20, 0, i * 20 + 20, h])]) for i, h in enumerate([198, 198, 240, 178], 1)]
    out = scale.measure(books, {})
    assert all(v["scale_method"] == "standard_format_prior" for v in out["books"].values())
    assert out["books"]["B1"]["spine_height_cm"] == pytest.approx(19.8, rel=0.05)


def test_shelf_run_counts_flat_stack_once():
    books = [
        book("B1", "U1-S1", 1, [(1, [0, 0, 30, 200])]),
        book("B2", "U1-S1", 2, [(1, [30, 150, 250, 180])], orientation="horizontal"),
        book("B3", "U1-S1", 3, [(1, [30, 180, 230, 200])], orientation="horizontal"),
    ]
    dims = {"B1": {"spine_thickness_cm": 3.0, "spine_height_cm": 20.0}, "B2": {"spine_height_cm": 22.0}, "B3": {"spine_height_cm": 20.0}}
    assert scale.shelf_runs(books, dims)["U1-S1"] == pytest.approx(0.25)  # 3 cm + max(22, 20) cm


# ---------- identification guard rails ----------

def test_unreadable_spine_is_not_guessed():
    v = identify.decide({"legible": False, "confidence": 0.0, "title": "", "verbatim": ""}, [{"title": "Dune", "authors": ["Frank Herbert"], "publisher": "", "isbn": "1"}])
    assert v["status"] == "unidentified"


def test_weak_match_is_unidentified():
    read = {"legible": True, "confidence": 0.9, "title": "The Dunes of Mars", "author": "Smith", "publisher": "", "verbatim": "The Dunes of Mars Smith"}
    v = identify.decide(read, [{"title": "Dune", "authors": ["Frank Herbert"], "publisher": "", "isbn": "1"}])
    assert v["status"] == "unidentified"


def test_edition_only_claimed_with_publisher_evidence():
    read = {"legible": True, "confidence": 0.9, "title": "Dune", "author": "Frank Herbert", "publisher": "Ace", "verbatim": "DUNE FRANK HERBERT ACE"}
    cands = [
        {"title": "Dune", "authors": ["Frank Herbert"], "publisher": "Hodder", "isbn": "111"},
        {"title": "Dune", "authors": ["Frank Herbert"], "publisher": "Ace Books", "isbn": "222"},
    ]
    v = identify.decide(read, cands)
    assert v["status"] == "identified" and v["match"]["isbn"] == "222" and "matches" in v["edition_basis"]


def test_parse_dimensions_units():
    assert identify.parse_dimensions("7.3 x 4.2 x 1.5 inches")["height_cm"] == pytest.approx(18.54, abs=0.01)
    d = identify.parse_dimensions("24.2 x 16.2 x 3.9 centimeters")
    assert d == {"height_cm": 24.2, "width_cm": 16.2, "thickness_cm": 3.9}


# ---------- pricing ----------

def listing(total, title="Dune by Frank Herbert", cur="USD"):
    return {"title": title, "price": total, "currency": cur, "shipping": 0.0, "total": total, "url": f"https://ebay.example/{total}"}


def test_book_figure_drops_lots_and_outliers():
    ls = [listing(9.0), listing(10.0), listing(11.0), listing(12.0), listing(300.0), listing(5.0, title="Lot of 10 books Dune")]
    fig = pricing.book_figure(ls, "USD")
    assert fig["amount"] == pytest.approx(10.5)
    assert all("300" not in u for u in fig["evidence_urls"])


def test_book_figure_ignores_other_currency():
    assert pricing.book_figure([listing(10.0, cur="GBP")], "USD") is None


def test_range_needs_five_listings():
    assert pricing.range_figure([listing(v) for v in (10, 20, 30, 40)], "USD") is None
    r = pricing.range_figure([listing(v) for v in (10, 20, 30, 40, 50)], "USD")
    assert r["low"] == 20 and r["high"] == 40


def test_extract_price_json_ld_and_isbn_check():
    page = '<script type="application/ld+json">{"@type":"Book","isbn":"9780441013593","offers":{"@type":"Offer","price":"18.00","priceCurrency":"USD"}}</script>'
    assert pricing.extract_price(page, "9780441013593") == {"amount": 18.0, "currency": "USD", "method": "json-ld"}
    assert pricing.extract_price(page, "9999999999999") is None
    assert pricing.extract_price("<p>Only $12.99 today!</p>") is None  # never parse free text


def test_appraisal_rules():
    assert pricing.appraisal_reasons({"published": "1887", "publisher": "", "isbn": ""}, {})
    assert pricing.appraisal_reasons({"published": "2001", "publisher": "The Folio Society", "isbn": "1"}, {})
    assert pricing.appraisal_reasons({"published": "2001", "publisher": "", "isbn": "1"}, {"special_marks": ["signed"], "verbatim": ""})
    assert pricing.appraisal_reasons({"published": "2001", "publisher": "Penguin", "isbn": "1"}, {"special_marks": [], "verbatim": "x"}) == []


async def test_art_defaults_to_appraisal_without_print_confirmation(tmp_path):
    item = {"id": "I1", "category": "art_portrait", "brand_model": "", "frame_ref": "x.jpg", "box_2d": [0, 0, 1000, 1000]}
    out = await pricing.price_item(tmp_path, pricing.Ebay(None, None), item, "US", {})
    assert out["status"] == "needs_appraisal" and out["replacement_cost"] is None


def test_locale_is_a_setting():
    assert locale("US")["currency"] == "USD" and locale("GB")["ebay_marketplace"] == "EBAY_GB"
    with pytest.raises(ValueError):
        locale("XX")


# ---------- assembly: totals come from lines ----------

def _price(amount):
    return {"amount": amount, "currency": "USD", "source": "eBay", "url": "https://e", "retrieved_at": "2026-10-15T00:00:00+00:00"}


def test_totals_add_up_and_unpriced_lines_excluded(tmp_path):
    books = [book(f"B{i}", "U1-S1", i, [(1, [0, 0, 20, 200])]) for i in range(1, 5)]
    identified = {
        "B1": {"status": "identified", "match": {"title": "A", "authors": ["X"], "isbn": "1"}, "id_confidence": 0.9, "edition_basis": "", "read": {}},
        "B2": {"status": "identified", "match": {"title": "B", "authors": ["Y"], "isbn": "2"}, "id_confidence": 0.9, "edition_basis": "", "read": {}},
        "B3": {"status": "identified", "match": {"title": "C", "authors": ["Z"], "isbn": "3"}, "id_confidence": 0.9, "edition_basis": "", "read": {}},
        "B4": {"status": "unidentified", "reason": "spine not legible", "read": {}},
    }
    prices = {
        "currency": "USD",
        "books": {
            "B1": {"replacement_cost": _price(10.0), "used_value": _price(4.0) | {"condition_assumed": "good"}, "notes": [], "appraisal_reasons": []},
            "B2": {"replacement_cost": None, "used_value": _price(3.0), "notes": ["no new-price source found"], "appraisal_reasons": []},
            "B3": {"replacement_cost": _price(500.0), "used_value": _price(450.0), "notes": [], "appraisal_reasons": ["over threshold"]},
        },
        "items": {
            "I1": {"status": "range", "replacement_cost": {"low": 50.0, "high": 90.0, "currency": "USD", "source": "eBay", "url": "u", "retrieved_at": "t"}},
            "I2": {"status": "needs_appraisal", "replacement_cost": None, "reason": "art"},
        },
    }
    items = [
        {"id": "I1", "category": "lamp", "description": "brass", "material": "metal", "brand_model": "", "frame_ref": "f.jpg", "confidence": 0.8, "sightings": ["f.jpg"]},
        {"id": "I2", "category": "art_portrait", "description": "oil", "material": "canvas", "brand_model": "", "frame_ref": "g.jpg", "confidence": 0.8, "sightings": ["g.jpg"]},
    ]
    meas = {"books": {}, "shelves": {"U1-S1": 0.8}}
    ov = {"excluded_books": {}, "excluded_items": {}, "books": {}, "items": {}, "unresolved": []}
    p = assemble.build(tmp_path, sweep={"id": "s", "currency": "USD"}, books=books, identified=identified, measurements=meas,
                       room={}, items=items, prices=prices, ov=ov, metrics={}, locale_comparison=None)
    t = p["totals"]
    assert t["book_count"] == 4 and t["books_identified"] == 2 and t["books_unidentified"] == 1 and t["books_needs_appraisal"] == 1
    assert t["books_replacement_cost"] == 10.0  # B2 has no new price, B3 is appraisal-only
    assert t["books_used_value"] == 7.0
    assert t["items_replacement_cost_low"] == 50.0 and t["items_replacement_cost_high"] == 90.0
    assert t["excluded_from_totals"] == 4  # B2 (missing new), B3, B4, I2
    b3 = next(b for b in p["books"] if b["id"] == "B3")
    assert b3["status"] == "needs_appraisal" and b3["replacement_cost"]["amount"] is None
    b4 = next(b for b in p["books"] if b["id"] == "B4")
    assert b4["title"] == "" and b4["isbn"] == ""
    refs = {q["ref_id"] for q in p["review_queue"]}
    assert {"B2", "B3", "B4", "I2"} <= refs
    for b in p["books"]:
        for k in ("id", "shelf", "position", "frame_ref", "status", "title", "author", "edition", "isbn", "spine_height_cm", "spine_thickness_cm", "id_confidence"):
            assert k in b
        if b["replacement_cost"]["amount"] is not None:
            assert b["replacement_cost"]["url"] and b["replacement_cost"]["retrieved_at"]


def test_override_excludes_books_in_view(tmp_path):
    books = [book("B1", "U1-S1", 1, [(1, [0, 0, 20, 200])]), book("B2", "U2-S1", 1, [(9, [0, 0, 20, 200])])]
    frames_t = {"frames/f_000001.jpg": 10_000, "frames/f_000009.jpg": 60_000}
    ov = assemble.resolve_overrides([{"target_type": "shelf", "action": "exclude", "target": "this shelf", "t_ms": 10_500}], books, {}, [], frames_t)
    assert set(ov["excluded_books"]) == {"B1"}
