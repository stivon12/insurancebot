"""End-to-end wiring test: full pipeline on a synthetic sweep with every external call stubbed.

Checks that stages connect, files are written, every frame_ref resolves and totals equal the line sums.
It does NOT measure accuracy - that needs a real sweep and ground truth (eval/eval.py).
"""

import json

import cv2
import numpy as np
import pytest

from pipeline import identify, items, pricing, room, spines
from pipeline import run as run_mod


def _texture(h, w, seed):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 230, np.uint8)
    for _ in range(400):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(5, 60)), int(y + rng.integers(5, 60))), rng.integers(0, 255, 3).tolist(), -1)
    return img


@pytest.fixture
def sweep_dir(tmp_path):
    d = tmp_path / "sweep"
    (d / "frames").mkdir(parents=True)
    shelf = _texture(1200, 3000, 1)
    rows, t, i = [], 0.0, 0
    def add(img, phase):
        nonlocal t, i
        i += 1
        name = f"frames/f_{i:06d}.jpg"
        cv2.imwrite(str(d / name), img)
        rows.append({"i": i, "file": name, "t_ms": t, "w": img.shape[1], "h": img.shape[0], "orient": 0, "phase": phase})
        t += 1000
    for k in range(6):
        add(_texture(1920, 1080, 100 + k), "room_turn")
    add(cv2.resize(shelf, (1080, 432)), "unit_overview")
    for k in range(6):  # pan along the shelf with overlap so ORB can chain the frames
        add(shelf[0:1200, k * 300 : k * 300 + 1200].copy(), "shelves")
    for k in range(3):
        add(_texture(1920, 1080, 200 + k), "pan")
    (d / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    imu = [{"t_ms": float(ms), "ax": 0.0, "ay": 9.8, "az": 0.0, "ra": 0.0, "rb": 0.0, "rg": -15.0} for ms in range(0, int(t), 20)]
    (d / "imu.jsonl").write_text("\n".join(json.dumps(s) for s in imu))
    (d / "sweep.json").write_text(json.dumps({"id": "test", "captured_at": "2026-10-15T00:00:00+00:00", "device": "test", "duration_s": 60, "country": "US"}))
    (d / "overrides.json").write_text(json.dumps([{"target_type": "item", "action": "is_print", "target": "portrait", "t_ms": 14000}]))
    return d


async def fake_gemini(*, model, parts, schema, metrics, stage, system=None, retries=2):
    metrics.add_llm(stage, model, None)
    if schema is spines.ROWS_SCHEMA:
        return {"rows": [{"y_top": 100, "y_bottom": 900, "has_books": True}]}
    if schema is spines.SPINES_SCHEMA:
        return {"spines": [{"box_2d": [50, x, 950, x + 90], "orientation": "vertical", "text": f"T{x}", "legible": True} for x in range(0, 1000, 100)]}
    if schema is identify.READ_SCHEMA:
        return {"verbatim": "DUNE HERBERT", "title": "Dune", "author": "Frank Herbert", "publisher": "", "legible": True, "confidence": 0.9, "special_marks": []}
    if schema is items.ITEMS_SCHEMA:
        return {"items": [
            {"image": 0, "same_as": "", "category": "lamp", "description": "brass", "material": "metal", "brand_model": "", "box_2d": [200, 400, 900, 600], "touches_floor": True, "count": 1, "confidence": 0.8},
            {"image": 0, "same_as": "", "category": "art_portrait", "description": "portrait", "material": "canvas", "brand_model": "", "box_2d": [100, 100, 300, 300], "touches_floor": False, "count": 1, "confidence": 0.8},
        ]}
    if schema is room.BOUNDARY_SCHEMA:
        return {"floor": [{"x": x, "y": 750, "at": "wall"} for x in range(0, 1001, 50)], "ceiling": [{"x": x, "y": 120} for x in range(0, 1001, 50)]}
    if schema is room.CASE_SCHEMA:
        return {"bookcases": [{"box_2d": [50, 50, 950, 950], "base_visible": True}]}
    raise AssertionError("unexpected schema")


async def fake_gb(client, title, author, metrics):
    return [{"source": "Google Books", "volume_id": "v1", "title": "Dune", "authors": ["Frank Herbert"], "publisher": "Ace", "published": "1990", "isbn": "9780441013593", "pages": 600, "url": "https://books.google.com/x"}]


async def fake_ol(client, title, author, metrics):
    return []


async def fake_dims(client, match, metrics):
    return {"height_cm": 17.8, "source": "Open Library", "url": "https://openlibrary.org/isbn/9780441013593"}


def _listings(base, cur="USD"):
    return [{"title": "Dune Frank Herbert paperback", "price": base + k, "currency": cur, "shipping": 0.0, "total": base + k, "url": f"https://www.ebay.com/itm/{k}", "condition": ""} for k in range(6)]


async def fake_search(self, country, *, condition, gtin="", q="", category=""):
    cur = "USD" if country == "US" else "GBP"
    return _listings(10.0 if condition == "NEW" else 4.0, cur)


async def fake_img(self, country, jpeg, condition="NEW"):
    return _listings(40.0, "USD" if country == "US" else "GBP")


async def test_full_pipeline_mocked(sweep_dir, monkeypatch):
    for mod in (spines, identify, items, room):
        monkeypatch.setattr(mod, "gemini_json", fake_gemini)
    monkeypatch.setattr(identify, "google_books", fake_gb)
    monkeypatch.setattr(identify, "open_library", fake_ol)
    monkeypatch.setattr(identify, "dimensions", fake_dims)
    monkeypatch.setattr(pricing.Ebay, "configured", property(lambda self: True))
    monkeypatch.setattr(pricing.Ebay, "search", fake_search)
    monkeypatch.setattr(pricing.Ebay, "search_by_image", fake_img)

    async def no_retailer(*a, **k):
        return None

    monkeypatch.setattr(pricing, "retailer_price", no_retailer)
    monkeypatch.setenv("SECOND_COUNTRY", "GB")

    packet = await run_mod.process(sweep_dir, None, "US")

    for f in ("keyframes.json", "spines.json", "identified.json", "measurements.json", "room.json", "items.json",
              "prices_US.json", "prices_GB.json", "claim_packet.json", "report.html", "metrics.json"):
        assert (sweep_dir / f).exists(), f
    assert packet["sweep"]["currency"] == "USD"
    assert packet["totals"]["book_count"] > 0
    for line in packet["books"] + packet["items"]:
        assert (sweep_dir / line["frame_ref"]).exists()
    priced = [b for b in packet["books"] if b["status"] == "identified" and b["replacement_cost"]["amount"] is not None]
    assert packet["totals"]["books_replacement_cost"] == pytest.approx(sum(b["replacement_cost"]["amount"] for b in priced))
    for b in priced:
        assert b["replacement_cost"]["url"].startswith("https://") and b["replacement_cost"]["retrieved_at"]
    art = next(i for i in packet["items"] if i["category"] == "art_portrait")
    assert art["status"] in ("range", "needs_appraisal")
    assert packet["locale_comparison"]["second_currency"] == "GBP"
    assert packet["pipeline_metrics"]["wall_clock_s"] < 300
    # Synthetic geometry gives a ~0.1 m camera height: the room must be withheld and flagged, not reported.
    assert packet["room"]["floor_area_m2"] is None
    assert any(q["ref_id"] == "room" and "implausible" in q["reason"] for q in packet["review_queue"])
