"""Core checks: totals, identification guards, price figures, item merging, room geometry, spine rule A, eval."""

import asyncio
import json
import math

import numpy as np
import pytest

from eval.eval import evaluate
from pipeline import books, live, packet, pricing, room
from pipeline.common import Metrics, memo, memo_path, note_failure


def book(bid, shelf, pos, boxes, orientation="vertical"):
    obs = [{"frame_i": fi, "file": f"frames/f_{fi:06d}.jpg", "box_px": bx, "legible": True, "orientation": orientation} for fi, bx in boxes]
    return {"id": bid, "shelf": shelf, "position": pos, "observations": obs, "best": obs[0]}


def ident(height, thickness=None):
    dims = {"height_cm": height}
    if thickness:
        dims["thickness_cm"] = thickness
    return {"status": "identified", "dimensions": dims}


def test_leave_one_out_scale_measures_unknown_book():
    # 0.1 cm per px in frame 1. Three ruler books + one unknown 230 px tall, 30 px thick.
    bks = [
        book("B1", "U1-S1", 1, [(1, [0, 0, 20, 198])]),   # 19.8 cm B-format
        book("B2", "U1-S1", 2, [(1, [20, 0, 45, 240])]),  # 24.0 cm hardback
        book("B3", "U1-S1", 3, [(1, [45, 0, 63, 178])]),  # 17.8 cm A-format
        book("B4", "U1-S1", 4, [(1, [63, 0, 93, 230])]),
    ]
    idn = {"B1": ident(19.8), "B2": ident(24.0), "B3": ident(17.8)}
    out = books.measure(bks, idn)
    b4 = out["books"]["B4"]
    assert b4["scale_method"] == "book_ruler"
    assert b4["spine_height_cm"] == pytest.approx(23.0, abs=0.2)
    assert b4["spine_thickness_cm"] == pytest.approx(3.0, abs=0.05)
    # A ruler book is measured from the OTHER rulers, not its own catalogue height
    assert out["books"]["B1"]["spine_height_cm"] == pytest.approx(19.8, abs=0.2)


def test_unreadable_spine_is_not_guessed():
    v = books.decide({"legible": False, "confidence": 0.0, "title": "", "verbatim": ""}, [{"title": "Dune", "authors": ["Frank Herbert"], "publisher": "", "isbn": "1"}])
    assert v["status"] == "unidentified"


def test_weak_match_is_unidentified():
    read = {"legible": True, "confidence": 0.9, "title": "The Dunes of Mars", "author": "Smith", "publisher": "", "verbatim": "The Dunes of Mars Smith"}
    v = books.decide(read, [{"title": "Dune", "authors": ["Frank Herbert"], "publisher": "", "isbn": "1"}])
    assert v["status"] == "unidentified"


def test_author_name_alone_is_not_a_title():
    read = {"legible": True, "confidence": 0.9, "title": "", "author": "Nora Roberts", "publisher": "", "verbatim": "NORA ROBERTS"}
    assert books.decide(read, [{"title": "Nora Roberts", "authors": ["Nora Roberts"], "publisher": "", "isbn": "1"}])["status"] == "unidentified"
    read = {**read, "title": "Night Moves", "verbatim": "Nora Roberts Night Moves"}
    cands = [{"title": "Nora Roberts", "authors": ["Nora Roberts"], "publisher": "", "isbn": "1"},
             {"title": "Night Moves", "authors": ["Nora Roberts"], "publisher": "", "isbn": "2"}]
    assert books.decide(read, cands)["match"]["isbn"] == "2"


def test_series_name_on_many_spines_is_not_one_work():
    v = {"status": "identified", "read": {"title": "CHILDCRAFT", "author": ""}, "match": {"title": "Childcraft", "authors": ["World Book"]}}
    bks = [{"id": f"B{i}"} for i in range(3)]
    out = books.series_only(bks, {b["id"]: v for b in bks})
    assert all(x["status"] == "unidentified" for x in out.values())
    assert books.series_only(bks[:2], {b["id"]: v for b in bks[:2]})["B0"]["status"] == "identified"  # two copies stay


def listing(total, title="Dune by Frank Herbert", cur="USD"):
    return {"title": title, "price": total, "currency": cur, "shipping": 0.0, "total": total, "url": f"https://ebay.example/{total}"}


def test_book_figure_drops_lots_and_outliers():
    ls = [listing(9.0), listing(10.0), listing(11.0), listing(12.0), listing(300.0), listing(5.0, title="Lot of 10 books Dune")]
    fig = pricing.book_figure(ls, "USD")
    assert fig["amount"] == pytest.approx(10.5)
    assert all("300" not in u for u in fig["evidence_urls"])


def test_range_needs_five_listings():
    assert pricing.range_figure([listing(v) for v in (10, 20, 30, 40)], "USD") is None
    r = pricing.range_figure([listing(v) for v in (10, 20, 30, 40, 50)], "USD")
    assert r["low"] == 20 and r["high"] == 40


def test_appraisal_rules():
    assert pricing.appraisal_reasons({"published": "1887", "publisher": "", "isbn": ""}, {})
    assert pricing.appraisal_reasons({"published": "2001", "publisher": "The Folio Society", "isbn": "1"}, {})
    assert pricing.appraisal_reasons({"published": "2001", "publisher": "", "isbn": "1"}, {"special_marks": ["signed"], "verbatim": ""})
    assert pricing.appraisal_reasons({"published": "2001", "publisher": "Penguin", "isbn": "1"}, {"special_marks": [], "verbatim": "x"}) == []


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
    p = packet.build(tmp_path, sweep={"id": "s", "currency": "USD"}, books=books, identified=identified, measurements=meas,
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
    ov = packet.resolve_overrides([{"target_type": "shelf", "action": "exclude", "target": "this shelf", "t_ms": 10_500}], books, {}, [], frames_t)
    assert set(ov["excluded_books"]) == {"B1"}


def test_shopping_rejects_collectible_and_disagreeing_offers():
    res = [{"title": "Infinite Jest", "extracted_price": 103.69, "source": "Etsy"},
           {"title": "Infinite Jest", "extracted_price": 19.99, "source": "Target"},
           {"title": "Infinite Jest signed first edition", "extracted_price": 900.0, "source": "Rare Books"}]
    assert pricing.shopping_matches(res, "Infinite Jest", "David Foster Wallace") == []  # one honest offer is left
    res.append({"title": "Infinite Jest", "extracted_price": 21.0, "source": "Walmart"})
    assert [r["source"] for r in pricing.shopping_matches(res, "Infinite Jest", "David Foster Wallace")] == ["Target", "Walmart"]


def test_unreadable_single_sighting_without_detector_is_not_counted():
    from pipeline.books import countable

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


def test_two_views_must_agree(monkeypatch):
    import asyncio

    hit = lambda t, isbn: {"status": "identified", "match": {"title": t, "authors": ["A"], "isbn": isbn}, "id_confidence": 0.8}
    miss = {"status": "unidentified", "reason": "spine not legible"}
    book = {"id": "B1", "best": {"file": "a", "box_px": [0, 0, 1, 1], "frame_i": 0}, "observations": []}

    def run(v1, v2):
        reads = iter([v1, v2])

        async def fake_read(run_dir, client, obs, metrics):
            return next(reads)

        monkeypatch.setattr(books, "_read_view", fake_read)
        monkeypatch.setattr(books, "second_view", lambda run_dir, b: {"file": "b", "box_px": [0, 0, 1, 1]})
        return asyncio.run(books._one(None, None, book, None))[1]

    assert run(hit("Wilt", "1"), hit("Wilt", "1"))["views_agree"] == 2
    assert run(hit("Wilt", "1"), hit("Wilt on High", "2"))["status"] == "unidentified"
    assert run(hit("Wilt", "1"), hit("Wilt: A Novel", "3"))["status"] == "identified"  # same work, other edition
    assert run(miss, hit("Wilt", "1"))["match"]["title"] == "Wilt"  # a blurred view does not veto a clear one
    assert run(hit("Wilt", "1"), miss)["views_agree"] == 1


def test_item_seen_again_in_the_same_direction_is_merged():
    from pipeline import items as I

    res = lambda cat, desc: {"present": True, "insurable": True, "category": cat, "description": desc, "material": "",
                             "brand_model": "", "box_2d": [0, 0, 500, 500], "touches_floor": True}
    out: list = []
    f = {"file": "f1.jpg", "t_ms": 0, "w": 10, "h": 10}
    assert I.accept(out, res("furniture", "grey sofa"), f, 10.0)[1] == "new"
    assert I.accept(out, res("furniture", "grey fabric sofa"), {**f, "file": "f2.jpg"}, 25.0)[1] == "same"
    assert I.accept(out, res("furniture", "grey sofa"), {**f, "file": "f3.jpg", "t_ms": 20000}, 150.0)[1] == "new"  # another sofa, other side
    assert I.accept(out, res("furniture", "grey fabric sofa"), {**f, "file": "f4.jpg", "t_ms": 22000}, 200.0)[1] == "same"  # seen 2 s ago: same
    assert I.accept(out, res("lamp", "grey sofa lamp"), f, 10.0)[1] == "new"  # another category needs a near-identical description
    assert I.accept(out, {**res("decor", "bottle"), "insurable": False}, f, 0.0) == (None, "not an insurable contents item")
    assert len(out) == 3 and out[0]["sightings"] == ["f1.jpg", "f2.jpg"]


def test_item_query_names_the_object_and_searches_go_to_valuable_items():
    assert pricing.item_query({"description": "tissue box", "brand_model": "Scott", "material": "cardboard"}) == "Scott tissue box"
    assert pricing.item_query({"description": "HP printer", "brand_model": "HP"}) == "HP printer"
    assert pricing.item_query({"description": "dark grey sofa", "brand_model": "", "material": "fabric"}) == "dark grey sofa"
    items = [{"id": "a", "category": "decor", "confidence": 0.99}, {"id": "b", "category": "electronics", "confidence": 0.6},
             {"id": "c1", "category": "furniture", "confidence": 0.9}, {"id": "c2", "category": "furniture", "confidence": 0.8},
             {"id": "d", "category": "lamp", "confidence": 0.5}]
    assert [it["id"] for it in pricing.search_order(items)] == ["b", "c1", "d", "a", "c2"]


async def test_serpapi_used_when_serper_fails(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test")
    monkeypatch.setenv("SERPAPI_API_KEY", "test")

    async def serper_down(client, q, country, metrics):
        return None  # quota used up

    async def serpapi(client, params, metrics):
        return {"shopping_results": [{"title": "HP DeskJet printer", "extracted_price": 59.0}]}

    monkeypatch.setattr(pricing, "_serper", serper_down)
    monkeypatch.setattr(pricing, "_serpapi", serpapi)
    offers, provider = await pricing.shopping_offers(None, "HP printer", "US", Metrics())
    assert provider == "SerpApi" and offers[0]["extracted_price"] == 59.0


def test_eval_scores_against_truth(tmp_path):
    (tmp_path / "books.csv").write_text(
        "title,author,legible,height_cm,thickness_cm,price_new,price_used,price_source\n"
        "Dune,Frank Herbert,yes,17.8,3.4,10,5,x\n"
        "Emma,Jane Austen,yes,19.8,2.0,8,4,x\n"
        "(unreadable),,no,21,1.2,,,\n"
    )
    (tmp_path / "items.csv").write_text("name,category\nfloor lamp,lamp\nstriped wool rug,rug\n")
    (tmp_path / "room.json").write_text(json.dumps({"floor_area_m2": 13.0, "wall_area_m2": 36.0}))
    rc = lambda a: {"amount": a}
    packet = {
        "totals": {"book_count": 3},
        "books": [
            {"id": "B1", "status": "identified", "title": "Dune", "id_confidence": 0.9, "spine_height_cm": 18.0, "spine_thickness_cm": 3.3, "replacement_cost": rc(11)},
            {"id": "B2", "status": "identified", "title": "Persuasion", "id_confidence": 0.8, "spine_height_cm": 19.0, "spine_thickness_cm": 2.0, "replacement_cost": rc(8)},
            {"id": "B3", "status": "unidentified", "title": "", "id_confidence": 0, "spine_height_cm": 21, "spine_thickness_cm": 1.2, "replacement_cost": rc(None)},
        ],
        "items": [{"category": "lamp", "description": "brass floor lamp"}, {"category": "rug", "description": "striped rug"}],
        "room": {"floor_area_m2": 13.5, "wall_area_m2": 30.0},
        "pipeline_metrics": {"wall_clock_s": 120},
    }
    rows = {r["measure"]: r for r in evaluate(packet, tmp_path)}
    assert rows["Book count"]["pass"]
    assert not rows["Title identification"]["pass"]  # Persuasion is a confident wrong answer (1 of 2)
    assert "50%" in rows["Title identification"]["result"]
    assert rows["Non-book items"]["pass"] and rows["Floor area"]["pass"] and not rows["Wall area"]["pass"]
    assert rows["Time to packet"]["pass"]


def test_items_match_one_to_one_by_name():
    from eval.eval import match_items

    truth = [{"name": "flat screen TV", "category": "electronics"}, {"name": "grey sofa", "category": "seating"}]
    system = [{"description": "grey fabric sofa", "category": "seating"}, {"description": "grey sofa", "category": "seating"},
              {"description": "tissue box", "category": "electronics"}]
    assert match_items(truth, system) == (1, 2)  # TV missed; second sofa and the tissue box are extras


def test_result_built_on_failed_lookup_is_not_saved(tmp_path):
    async def failing():
        note_failure("example.org: HTTP 429")
        return {"candidates": []}

    async def ok():
        return {"candidates": ["x"]}

    async def go():
        r1 = await memo(tmp_path, "k", {"a": 1}, failing)
        assert r1 == {"candidates": []}
        assert not memo_path(tmp_path, "k", {"a": 1}).exists()  # retried next time, not cached as "no match"
        r2 = await memo(tmp_path, "k", {"a": 1}, ok)
        assert r2 == {"candidates": ["x"]}
        assert json.loads(memo_path(tmp_path, "k", {"a": 1}).read_text()) == r2

    asyncio.run(go())


def test_failure_in_nested_task_reaches_outer_memo(tmp_path):
    async def inner():
        note_failure("timeout")
        return 1

    async def outer():
        await asyncio.gather(memo(tmp_path, "in", 1, inner))
        return 2

    async def go():
        await memo(tmp_path, "out", 1, outer)
        assert not memo_path(tmp_path, "in", 1).exists()
        assert not memo_path(tmp_path, "out", 1).exists()

    asyncio.run(go())


def test_heading_integrates_a_clockwise_turn():
    h = live.Heading()
    # upright phone (gravity on +y), turning right at 30 deg/s about up for 3 s
    h.add([{"t_ms": t, "ax": 0, "ay": 9.8, "az": 0, "ra": 0, "rb": 0, "rg": -30.0} for t in range(0, 3001, 20)])
    assert abs(h.at(3000) - 90) < 3


def test_room_from_a_walk_through_an_l_shaped_room():
    # 4 x 3 m room missing a 2 x 1.5 m corner (9 m2, perimeter 14 m), 2.6 m ceiling, phone 1.4 m above the floor
    corners = [(0, 0), (4, 0), (4, 1.5), (2, 1.5), (2, 3), (0, 3)]
    rng = np.random.default_rng(0)
    walls = []
    for (x0, z0), (x1, z1) in zip(corners, corners[1:] + corners[:1]):
        t = rng.random(3000)
        walls.append(np.stack([x0 + t * (x1 - x0), rng.random(3000) * 2.6, z0 + t * (z1 - z0)], 1))
    floor = np.stack([rng.random(4000) * 2, np.zeros(4000), rng.random(4000) * 3], 1)
    ceil = floor + [0, 2.6, 0]
    world = np.concatenate(walls + [floor, ceil])
    tr = room.Tracker()
    for cx, cz in [(1, 0.5), (1, 1.5), (1, 2.5), (2.5, 0.7), (3.5, 0.7), (1.5, 1.0)]:
        C = np.array([cx, 1.4, cz])
        tr.placed.append({"R": np.eye(3), "s": 1.0, "C": C, "Q": (world - C).astype(np.float32), "f_px": 1500.0, "books": np.zeros((0, 3))})
    tr.seen = 6
    out = room.measure(__import__("pathlib").Path("/tmp"), tr)
    assert abs(out["floor_area_m2"] - 9.0) / 9.0 < 0.1 and abs(out["perimeter_m"] - 14.0) / 14.0 < 0.1
    assert abs(out["height_m"] - 2.6) < 0.1 and out["corners"] == 6
    assert "could not be tracked" in room.measure(__import__("pathlib").Path("/tmp"), room.Tracker())["review"][0]


def test_rule_a_keeps_detector_boxes_and_only_legible_extras():
    gem = [{"box_px": [0, 0, 20, 100], "text": "Dune", "legible": True}, {"box_px": [40, 0, 60, 100], "text": "", "legible": False},
           {"box_px": [80, 0, 100, 100], "text": "Emma", "legible": True}]
    rf = np.array([[2, 2, 18, 98], [120, 0, 140, 100]])
    out = books.combine(gem, rf)
    assert [o.get("text") for o in out] == ["Dune", "Emma", ""]  # matched, legible extra, detector-only
    assert out[0]["box_px"] == [2.0, 2.0, 18.0, 98.0] and out[0]["box_source"].startswith("rfdetr")


async def test_refusing_host_is_skipped_for_longer_each_time(tmp_path, monkeypatch):
    import httpx
    from pipeline import common
    monkeypatch.setattr(common, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(common.asyncio, "sleep", lambda *_: _noop())  # no back-off waits in the test
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: calls.append(1) or httpx.Response(429))) as c:
        await asyncio.gather(*(common.get_json(c, "https://refuse.test/v", {"q": str(i)}, Metrics(), "x") for i in range(20)))
        h = common._host("refuse.test")
        assert len(calls) == common.STRIKES_TO_COOL and common.host_down("refuse.test")
        first = h.cool_until
        h.cool_until = 0
        await asyncio.gather(*(common.get_json(c, "https://refuse.test/v", {"q": f"b{i}"}, Metrics(), "x") for i in range(5)))
        assert h.cools == 2 and h.cool_until - first > common.COOL_S * 0.9


async def _noop():
    return None


async def test_spine_reads_are_batched_and_matched_back_by_number(monkeypatch):
    calls = []

    async def fake_gemini(*, parts, **_):
        names = [p for p in parts if isinstance(p, str) and p.startswith("Spine ")]
        calls.append(len(names))
        n = len(names)  # answer in reverse order: results must still reach the right reader
        return {"spines": [{"n": i, "verbatim": f"book {i}"} for i in range(n, 0, -1)]}

    monkeypatch.setattr(books, "gemini_json", fake_gemini)
    monkeypatch.setattr(books, "image_part", lambda b: b)
    b = books._ReadBatcher()
    out = await asyncio.gather(*(b.read(f"img{i}".encode(), Metrics()) for i in range(13)))
    assert calls == [12, 1]
    assert [r["verbatim"] for r in out] == [f"book {i}" for i in range(1, 13)] + ["book 1"]


async def test_library_of_congress_record_becomes_a_candidate(tmp_path, monkeypatch):
    import httpx
    from pipeline import common
    monkeypatch.setattr(common, "CACHE_DIR", tmp_path)
    marc = ('<record xmlns="http://www.loc.gov/MARC21/slim"><datafield tag="020" ind1=" " ind2=" "><subfield code="a">9780385424738 (hc)</subfield></datafield>'
            '<datafield tag="100" ind1="1" ind2=" "><subfield code="a">Grisham, John.</subfield></datafield>'
            '<datafield tag="245" ind1="1" ind2="4"><subfield code="a">The rainmaker /</subfield><subfield code="c">John Grisham.</subfield></datafield>'
            '<datafield tag="300" ind1=" " ind2=" "><subfield code="a">434 p. ;</subfield><subfield code="c">25 cm.</subfield></datafield></record>')
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=marc))) as c:
        (cand,) = await books.loc_search(c, "The Rainmaker", "John Grisham", Metrics())
    assert (cand["title"], cand["authors"], cand["isbn"], cand["pages"]) == ("The rainmaker", ["John Grisham"], "9780385424738", 434)


def test_rulers_cover_each_frame_with_few_books():
    bks = [book(f"B{i}", "S1", i, [(1, [i, 0, i + 1, 10]), (2, [i, 0, i + 1, 10])]) for i in range(8)]
    bks += [book("B9", "S1", 9, [(3, [0, 0, 1, 10])])]
    idn = {b["id"]: {"status": "identified", "id_confidence": 0.9} for b in bks}
    chosen = books.pick_rulers(bks, idn)
    assert len(chosen) == 4  # 3 cover frames 1 and 2 together, 1 more for frame 3: not all 9 books
    assert "B9" in chosen
