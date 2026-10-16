import pytest

from pipeline import assemble, identify, pricing, scale
from pipeline.common import Metrics, locale


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


# ---------- spines: one book listed from overlapping bands ----------

def test_overlapping_bands_merge_and_in_frame_dedupe():
    from pipeline import spines
    bands = spines.merge_bands([{"y_top": 150, "y_bottom": 650}, {"y_top": 400, "y_bottom": 900}, {"y_top": 640, "y_bottom": 700}])
    assert len(bands) == 1 and bands[0]["y_top"] == 150 and bands[0]["y_bottom"] == 900
    two_rows = spines.merge_bands([{"y_top": 50, "y_bottom": 480}, {"y_top": 520, "y_bottom": 950}])
    assert len(two_rows) == 2  # real separate shelves stay separate
    same = [{"box_px": [100, 200, 140, 900], "row_in_frame": 0}, {"box_px": [102, 600, 141, 1000], "row_in_frame": 1},
            {"box_px": [150, 200, 190, 900], "row_in_frame": 0}]
    assert len(spines.dedupe_in_frame(same)) == 2  # cross-row duplicate removed
    loose = [{"box_px": [100, 200, 260, 900], "row_in_frame": 0}, {"box_px": [150, 200, 190, 900], "row_in_frame": 0}]
    assert len(spines.dedupe_in_frame(loose)) == 2  # a loose box never swallows a neighbour in the same row


def test_spine_box_match_one_to_one():
    import numpy as np
    from pipeline import spine_boxes
    gemini = [[0, 0, 120, 400], [120, 0, 160, 400]]           # first box is loose: spans two spines
    rf = np.array([[5, 10, 55, 395], [60, 10, 115, 395], [122, 10, 158, 398], [400, 0, 450, 100]])
    m = spine_boxes.match(gemini, rf)
    assert m[1] == 2                      # the tight box inside spine 1
    assert m[0] in (0, 1)                 # loose box gets one tight box, not both
    assert 3 not in m.values()            # a box elsewhere (background) is never used


def test_surname_fragment_counts_as_author_evidence():
    read = {"legible": True, "confidence": 0.9, "title": "Mrs Dalloway", "author": "", "publisher": "", "verbatim": "Mrs Dalloway WOO NE"}
    v = identify.decide(read, [{"title": "Mrs Dalloway", "authors": ["Virginia Woolf"], "publisher": "", "isbn": "1"}])
    assert v["status"] == "identified"
    wrong = identify.decide(read, [{"title": "Mrs Dalloway", "authors": ["Someone Else"], "publisher": "", "isbn": "2"}])
    assert wrong["status"] == "unidentified"  # 2-word title and no author evidence: not accepted


def test_one_shared_word_is_not_a_title_match():
    read = {"legible": True, "confidence": 0.9, "title": "Indonesia", "author": "", "publisher": "", "verbatim": "lonely planet Indonesia"}
    v = identify.decide(read, [{"title": "Soil Quality Standards for Indonesia", "authors": ["Indonesia. Ministry of Agriculture"], "publisher": "", "isbn": "1"}])
    assert v["status"] == "unidentified"
    read2 = {"legible": True, "confidence": 0.7, "title": "The MINISTRY Of UTMOST HAPPINESS", "author": "ARUND Roy NE", "publisher": "", "verbatim": "The MINISTRY Of UTMOST HAPPINESS ARUND Roy NE"}
    v2 = identify.decide(read2, [{"title": "Ministry of Utmost Happiness", "authors": ["Arundhati Roy"], "publisher": "", "isbn": "2"}])
    assert v2["status"] == "identified"


def _bk(id_, shelf, x0, x1, seg=0):
    return {"id": id_, "shelf": shelf, "segment": seg, "seg_box": [x0, 0, x1, 300], "observations": [{"frame_i": int(id_[1:])}]}


def _idn(isbn, conf=0.9, read="T A"):
    return {"status": "identified", "id_confidence": conf, "read": {"verbatim": read}, "match": {"isbn": isbn, "title": "T", "authors": ["A"]}}


def test_same_book_adjacent_on_same_shelf_is_merged():
    from pipeline.identify import merge_same_book

    books = [_bk("B001", "U1-S1", 0, 30), _bk("B002", "U1-S1", 35, 65), _bk("B003", "U1-S1", 70, 100)]
    idn = {"B001": _idn("111", 0.8), "B002": _idn("111", 0.9), "B003": _idn("222")}
    out, idn2 = merge_same_book(books, idn)
    assert [b["id"] for b in out] == ["B002", "B003"] and "B001" not in idn2
    assert len(out[0]["observations"]) == 2 and "merged duplicate" in out[0]["review_notes"][0]


def test_same_book_elsewhere_is_flagged_not_merged():
    from pipeline.identify import merge_same_book

    books = [_bk("B001", "U1-S1", 0, 30), _bk("B002", "U1-S3", 0, 30), _bk("B003", "U2-S1", 0, 30, seg=1), _bk("B004", "U1-S1", 400, 430)]
    idn = {k: _idn("111") for k in ("B001", "B002", "B003", "B004")}
    out, _ = merge_same_book(books, idn)
    assert len(out) == 4  # other shelf, other unit, far along the same shelf: possibly real second copies
    assert all("possible duplicate copy" in b["review_notes"][0] for b in out)


def test_same_view_test_on_homographies():
    import numpy as np

    from pipeline.keyframes import same_view

    assert same_view(np.eye(3), 960, 540)
    shifted = np.array([[1, 0, 200], [0, 1, 0], [0, 0, 1.0]])  # 21 % of the width
    assert not same_view(shifted, 960, 540)
    zoomed = np.diag([1.3, 1.3, 1.0])
    assert not same_view(zoomed, 960, 540)


def test_adjacent_same_match_but_different_spine_text_is_not_merged():
    from pipeline.identify import merge_same_book

    books = [_bk("B001", "U1-S1", 0, 30), _bk("B002", "U1-S1", 35, 65)]
    idn = {"B001": _idn("111", read="EL MONTSENY 51 passejades per descobrir"), "B002": _idn("111", read="EL MONTSENY 50 itineraris a peu")}
    out, _ = merge_same_book(books, idn)
    assert len(out) == 2


def test_shopping_matches_keeps_this_book_new_offers_only():
    res = [
        {"title": "The Invisible Man (Penguin Classics) Paperback", "extracted_price": 9.99, "source": "Waterstones"},
        {"title": "The Invisible Man - used, good", "extracted_price": 3.0, "source": "World of Books"},
        {"title": "The Invisible Man", "extracted_price": 12.5, "second_hand_condition": "used", "source": "x"},
        {"title": "Invisible Cities", "extracted_price": 8.0, "source": "y"},
        {"title": "The Invisible Man", "source": "no price"},
    ]
    out = pricing.shopping_matches(res, "The Invisible Man: A Grotesque Romance")
    assert [r["source"] for r in out] == ["Waterstones"]


async def test_serpapi_backup_reports_shop_page_and_its_price(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)  # SerpApi is used only when Serper is not configured
    calls = []

    async def fake(client, url, params, metrics, stage, cache=True, **kw):
        calls.append((params["engine"], cache))
        if params["engine"] == "google_shopping":
            return {"shopping_results": [
                {"title": t, "extracted_price": p, "source": s, "product_link": "https://www.google.com/x",
                 "immersive_product_page_token": "tok" + s} for t, p, s in [
                    ("Wilt by Tom Sharpe", 8.0, "A"), ("Wilt (Paperback)", 9.5, "Blackwells"), ("Wilt - Tom Sharpe", 14.0, "C"),
                    ("Wilt in Nowhere", 7.0, "D"), ("Wilt No Se Aclara", 6.0, "E"), ("Audiobook | Wilt", 3.0, "Audible")]]}
        return {"product_results": {"stores": [{"name": "Blackwells", "link": "https://blackwells.co.uk/wilt", "extracted_price": 9.49}]}}

    from pipeline import http
    monkeypatch.setattr(http, "get_json", fake)
    pricing._serp_calls.clear()
    r, _ = await pricing.retailer_price(None, {"title": "Wilt", "author": "Tom Sharpe", "isbn": ""}, "GB", Metrics())
    assert r["url"] == "https://blackwells.co.uk/wilt" and r["amount"] == 9.49 and "Blackwells" in r["source"]
    assert calls == [("google_shopping", False), ("google_immersive_product", False)]  # prices are never cached


def test_shopping_used_offers_kept_apart():
    res = [{"title": "Wilt", "extracted_price": 3.0, "source": "World of Books", "second_hand_condition": "pre-owned"},
           {"title": "Wilt", "extracted_price": 9.0, "source": "Waterstones"}]
    assert [r["source"] for r in pricing.shopping_matches(res, "Wilt", "Tom Sharpe", used=True)] == ["World of Books"]
    assert [r["source"] for r in pricing.shopping_matches(res, "Wilt", "Tom Sharpe")] == ["Waterstones"]


def test_unreadable_single_sighting_without_detector_is_not_counted():
    from pipeline.spines import countable

    junk = {"observations": [{"frame_i": 166, "legible": False, "is_book": "yes", "box_source": "gemini (loose)"}]}
    assert countable(junk, {"status": "unidentified"})[0] is False
    twice = {"observations": [{"frame_i": 1, "legible": False}, {"frame_i": 2, "legible": False}]}
    assert countable(twice, {})[0] is True
    detector = {"observations": [{"frame_i": 1, "legible": False, "box_source": "rfdetr-v1"}]}
    assert countable(detector, {})[0] is True
    legible = {"observations": [{"frame_i": 1, "legible": True, "is_book": "yes"}]}
    assert countable(legible, {})[0] is True
    unsure = {"observations": [{"frame_i": 1, "legible": True, "is_book": "unsure"}]}
    assert countable(unsure, {})[0] is False
    assert countable(junk, {"status": "identified"})[0] is True


def test_shopping_rejects_merch_and_outliers():
    res = [{"title": "Infinite Jest", "extracted_price": p, "source": s} for p, s in [(12.0, "A"), (13.5, "B"), (14.0, "C"), (95.0, "Rare")]]
    res += [{"title": "Infinite Jest Tote Bag", "extracted_price": 13.85, "source": "Redbubble"},
            {"title": "Infinite Jest - Study Guide", "extracted_price": 9.99, "source": "Lulu"}]
    out = pricing.shopping_matches(res, "Infinite Jest", "David Foster Wallace")
    assert sorted(r["source"] for r in out) == ["A", "B", "C"]


async def test_timeouts_are_not_retried(monkeypatch):
    import httpx
    from pipeline import http

    calls = []

    class Slow:
        async def get(self, url, params=None, timeout=None):
            calls.append(timeout)
            raise httpx.ReadTimeout("slow")

    r = await http.get_json(Slow(), "https://example.org/x", {}, Metrics(), "t", cache=False, timeout=25, attempts=2, retry_timeouts=False)
    assert r is None and calls == [25]


def test_report_puts_confirmed_results_before_review(tmp_path):
    import json
    from pathlib import Path

    from pipeline import report

    runs = sorted(p for p in Path("runs").glob("*/claim_packet.json"))
    if not runs:
        pytest.skip("no packet on disk")
    html = report.render(json.loads(runs[-1].read_text()))
    i_books, i_room, i_review = html.index('id="books"'), html.index('id="room"'), html.index('id="review"')
    assert i_books < i_room < i_review and "<details>" in html[i_review:i_review + 40]


def test_coach_notes_for_weak_captures():
    from pipeline import coach

    unreadable = [{"spines": 6, "legible": 1, "rows": 2}, {"spines": 5, "legible": 1, "rows": 2}]
    assert coach.shelf_notes(unreadable)[0][0] == "unreadable"
    empty = [{"spines": 0, "legible": 0, "rows": 0}] * 3
    assert coach.shelf_notes(empty)[0][0] == "no_books"
    good = [{"spines": 8, "legible": 7, "rows": 2}] * 3
    assert coach.shelf_notes(good) == []
    unmatched = [{"legible": True, "identified": False}] * 7
    assert coach.identify_notes(unmatched)[0][0] == "unmatched"
    # a quarter turn (gyro about the up axis) is not a full room turn
    imu = [{"t_ms": i * 20.0, "ax": 0, "ay": 9.8, "az": 0, "ra": 0, "rb": 0, "rg": 22.5} for i in range(250)]
    turn = [{"t_ms": t} for t in (0, 1000, 2000, 3000, 4000)]
    assert coach.room_turn_notes(turn, imu, "unit_overview")[0][0] == "turn"


async def test_coach_rate_limits():
    from pipeline import coach

    sent = []

    async def send(t):
        sent.append(t)

    c = coach.Coach(send)
    await c.offer([("a", "one"), ("b", "two")])
    await c.offer([("b", "two")])
    assert sent == ["one"]  # one note at a time, then a gap


def test_serper_price_parsing_and_used_detection():
    assert pricing.parse_price("$1,234.50", "USD") == 1234.5
    assert pricing.parse_price("£9.99", "GBP") == 9.99
    assert pricing.parse_price("CA$12.00", "USD") is None and pricing.parse_price("$12", "GBP") is None
    data = {"shopping": [
        {"title": "Infinite Jest", "source": "AbeBooks.com", "price": "$7.68", "link": "https://www.google.com/a"},
        {"title": "Infinite Jest (20th Anniversary Edition)", "source": "Barnes & Noble", "price": "$21.99", "link": "https://www.google.com/b"},
        {"title": "Infinite Jest", "source": "x", "price": "$5.00", "link": ""},
    ]}
    offers = pricing.serper_offers(data, "USD")
    assert [(o["source"], bool(o["second_hand_condition"])) for o in offers] == [("AbeBooks.com", True), ("Barnes & Noble", False)]


async def test_serper_book_price_one_search(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test")
    calls = []

    async def fake(client, q, country, metrics):
        calls.append(q)
        return {"shopping": [{"title": "Wilt", "source": s, "price": p, "link": "https://www.google.co.uk/x" + s}
                             for s, p in [("Waterstones", "£9.99"), ("Amazon.co.uk", "£8.99"), ("WHSmith", "£10.49"), ("World of Books", "£3.10")]]}

    monkeypatch.setattr(pricing, "_serper", fake)
    data = await pricing.shopping_search(None, {"title": "Wilt", "author": "Tom Sharpe"}, "GB", Metrics())
    new, _ = await pricing.retailer_price(None, {"title": "Wilt", "author": "Tom Sharpe"}, "GB", Metrics(), data=data)
    used, _ = await pricing.retailer_price(None, {"title": "Wilt", "author": "Tom Sharpe"}, "GB", Metrics(), used=True, data=data)
    assert new["amount"] == 9.99 and "Serper" in new["source"] and used["amount"] == 3.10
    assert len(calls) == 1
