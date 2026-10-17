"""Non-book contents: Flash-Lite lists items on sharp new views (and checks the live model's logs); code merges re-sightings."""

from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
from rapidfuzz import fuzz

from . import room as G
from .common import LITE_MODEL, Metrics, gemini_json, image_part, memo, read_jsonl, write_json

STAGE = "items"
CATEGORIES = ["furniture", "shelving", "storage", "electronics", "appliance", "coffee_machine", "lamp", "art", "rug", "decor",
              "bag_luggage", "sports_music", "jewellery_watch", "other"]
SAME_HEADING_DEG = 25
RECENT_MS = 4000
MIN_VALUE_USD = 25  # the scanner's own rough value: drops bags, boxes and baskets it lists anyway
SURE = 0.9  # a scanned item counts at this confidence, or once it is seen a second time (a blurred wardrobe came back as a 0.8 washing machine)

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean", "description": "the claimed object is clearly visible in this photo"},
        "insurable": {"type": "boolean", "description": "a movable household item worth roughly 25 or more in money"},
        "category": {"type": "string", "enum": CATEGORIES},
        "description": {"type": "string", "description": "short: what it is, colour, style"},
        "material": {"type": "string"},
        "brand_model": {"type": "string", "description": "only if a brand or model name is legible; else empty"},
        "box_2d": {"type": "array", "items": {"type": "integer"}, "description": "[ymin,xmin,ymax,xmax] 0-1000"},
        "touches_floor": {"type": "boolean"},
    },
    "required": ["present", "insurable", "category", "description", "material", "brand_model", "box_2d", "touches_floor"],
}
CHECK_PROMPT = (
    "A home-contents insurance agent says this photo shows: {claim}. Check it. present=false if that object is not "
    "clearly in the photo. insurable=false for anything built into the building (toilets, sinks, taps, built-in "
    "cupboards, radiators, railings, light switches) and for low-value consumables (bottles, toiletries, papers, "
    "cables, food, packaging, tissues). Otherwise describe the object plainly as it is seen."
)


async def check(run_dir: Path, frame: dict, claim: str, metrics: Metrics) -> dict | None:
    """Flash-Lite's structured look at one logged item on the frame the agent saw it in."""
    async def ask():
        img = cv2.imread(str(Path(run_dir) / frame["file"]))
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        return await gemini_json(model=LITE_MODEL, parts=[image_part(jpg), CHECK_PROMPT.format(claim=claim)],
                                 schema=CHECK_SCHEMA, metrics=metrics, stage=STAGE)
    return await memo(run_dir, "item_checks", {"file": frame["file"], "claim": claim}, ask)


SCAN_SCHEMA = {"type": "object", "properties": {
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "description": {"type": "string", "description": "short: what it is, colour, style"},
        "material": {"type": "string"},
        "brand_model": {"type": "string", "description": "only if legible or unmistakable; else empty"},
        "est_value_usd": {"type": "integer", "description": "rough cost to replace it new"},
        "confidence": {"type": "number"},
        "box_2d": {"type": "array", "items": {"type": "integer"}, "description": "[ymin,xmin,ymax,xmax] 0-1000"},
        "touches_floor": {"type": "boolean"}},
        "required": ["category", "description", "material", "brand_model", "est_value_usd", "confidence", "box_2d", "touches_floor"]}}},
    "required": ["items"]}
SCAN_PROMPT = (
    "You are a home-contents insurance assessor looking at one phone-camera frame from a room tour. List every MOVABLE "
    "household possession clearly visible in this frame that would cost roughly 25 USD or more to replace: furniture "
    "(beds, wardrobes, freestanding cupboards, racks, tables, chairs), electronics (laptops, consoles, TVs, monitors, "
    "phones, speakers), appliances, lamps, art, rugs, bags and luggage, sports or music gear, decor. "
    "Do NOT list: books; things built into the building (doors, windows, switches, sockets, fuse/meter boxes, built-in "
    "cupboards, radiators, fixed lights); clothes, shoes, bedding, curtains and towels individually; consumables, "
    "bottles, toiletries, buckets, cables, papers, packaging. List each physical object once. If the frame is too "
    "blurred to recognise objects, list only what you can still identify with certainty. Do not guess brands."
)


async def scan(run_dir: Path, frame: dict, metrics: Metrics) -> list[dict]:
    """Flash-Lite lists the insurable items in one frame (768 px copy). On 14 frames of a phone run it found the laptop,
    console and backpack the live model never logged, at about $0.0007 a frame."""
    async def ask():
        img = cv2.imread(str(Path(run_dir) / frame["file"]))
        img = cv2.resize(img, None, fx=768 / img.shape[1], fy=768 / img.shape[1], interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        return await gemini_json(model=LITE_MODEL, parts=[image_part(jpg), SCAN_PROMPT], schema=SCAN_SCHEMA, metrics=metrics, stage=STAGE)
    r = await memo(run_dir, "item_scans", {"file": frame["file"]}, ask)
    return [{**x, "present": True, "insurable": True} for x in (r or {}).get("items", []) if x.get("est_value_usd", 25) >= MIN_VALUE_USD]


# eBay leaf categories for common valuables, keyed by words that name the product (also used to check a brand fits).
PRODUCT_CATEGORY = [("laptop", "177"), ("macbook", "177"), ("notebook", "177"), ("console", "139971"), ("playstation", "139971"),
                    ("xbox", "139971"), ("nintendo switch", "139971"), ("monitor", "80053"), ("television", "11071"), ("tv", "11071"),
                    ("headphone", "112529"), ("tablet", "171485"), ("ipad", "171485"), ("camera", "31388"), ("phone", "9355"),
                    ("iphone", "9355")]


def product(text: str) -> tuple[str, str]:
    """(product word, eBay category id) named in the text, or ('', '')."""
    text = text.lower()
    for noun, cat in PRODUCT_CATEGORY:
        if re.search(rf"\b{re.escape(noun)}", text):
            return noun, cat
    return "", ""


def brand_fits(it: dict, brand: str, description: str) -> bool:
    """A brand seen on a later look carries over only if it names the same kind of thing ('MacBook Pro' fits 'silver
    laptop', not 'flat screen television')."""
    have, got = product(it["description"])[1], product(f"{brand} {description}")[1]
    return have == got if have and got else fuzz.token_sort_ratio(it["description"], description) >= 60


def _turn(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def find_same(items: list[dict], new: dict, file: str = "", t_ms: float | None = None) -> dict | None:
    """The same object seen again. On a walking tour the heading to an object changes as the person moves, so:
    seen within the last few seconds with the same category and a similar description; or nearly the same direction
    with a similar description; or a near-identical description within a quarter turn. Two objects in one frame, or
    two different legible brands, are never merged."""
    for it in items:
        if file and file in it.get("sightings", []):
            continue
        if it.get("brand_model") and new.get("brand_model") and fuzz.token_set_ratio(it["brand_model"], new["brand_model"]) < 80:
            continue
        similar = it["category"] == new["category"] and fuzz.token_sort_ratio(it["description"], new["description"]) >= 55
        close = fuzz.token_sort_ratio(it["description"], new["description"]) >= 85
        if similar and t_ms is not None and it.get("last_t") is not None and t_ms - it["last_t"] <= RECENT_MS:
            return it
        if it.get("heading") is None or new.get("heading") is None:  # no motion data: only a near-identical description
            if close:
                return it
            continue
        turn = _turn(it["heading"], new["heading"])
        if (turn <= SAME_HEADING_DEG and (close or similar)) or (close and turn <= 90):
            return it
    return None


BEDDING = re.compile(r"\b(bed ?spreads?|bedding|bed ?sheets?|sheets?|duvets?|quilts?|comforters?|blankets?|pillows?|pillow ?cases?|throws?)\b", re.I)
# A bed (or sofa) described with its bedding on it is still the bed: 'bed frame with floral patterned bedding'.
FURNITURE = re.compile(r"\b(bed ?frame|bedstead|headboard|mattress|sofa|couch|chair|bed)\b(?! ?(spreads?|sheets?|linen|covers?))", re.I)


def is_bedding(description: str) -> bool:
    return bool(BEDDING.search(description)) and not FURNITURE.search(description)


def accept(items: list[dict], res: dict, frame: dict, heading: float | None, pending: list[dict] | None = None) -> tuple[dict | None, str]:
    """Add a checked item (or a re-sighting) to the list. Returns (item, 'new' | 'same' | 'pending' | why rejected).
    With `pending`, an unsure scan waits there until a second sighting confirms it."""
    if not res:
        return None, "check failed"
    if not res.get("present"):
        return None, "not visible in the frame"
    if not res.get("insurable"):
        return None, "not an insurable contents item"
    if is_bedding(res.get("description", "")):  # the scan prompt says so, but the model still listed a bedspread
        return None, "bedding is not listed as an item"
    if len(res.get("box_2d", [])) != 4:
        res["box_2d"] = [0, 0, 1000, 1000]
    new = {k: res.get(k, "") for k in ("category", "description", "material", "brand_model", "box_2d", "touches_floor")}
    if res.get("est_value_usd"):
        new["est_value_usd"] = res["est_value_usd"]  # the scanner's rough value: a sanity floor for prices
    new["heading"] = None if heading is None else round(heading % 360, 1)
    t = frame.get("t_ms")
    same = find_same(items, new, frame["file"], t)
    if same:
        if frame["file"] not in same["sightings"]:
            same["sightings"].append(frame["file"])
        same["last_t"] = t
        if new["brand_model"] and not same["brand_model"] and brand_fits(same, new["brand_model"], new["description"]):
            same["brand_model"] = new["brand_model"]
        return same, "same"
    conf = float(res.get("confidence", 0.9))
    if pending is not None and conf < SURE:
        seen = find_same(pending, new, frame["file"], t)
        if not seen:
            pending.append({**new, "sightings": [frame["file"]], "last_t": t})
            return None, "pending"
        pending.remove(seen)
        new["sightings"] = seen["sightings"]
        conf = max(conf, SURE)
    new.update({"id": f"I{len(items) + 1:03d}", "frame_ref": frame["file"], "confidence": round(conf, 2), "last_t": t,
                "sightings": new.get("sightings", []) + [frame["file"]],
                "frame": {k: frame[k] for k in ("t_ms", "w", "h") if k in frame} | {"orient": frame.get("orient", 0)}})
    items.append(new)
    return new, "new"


CONSOLIDATE_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"},
    "same_as": {"type": "integer", "description": "number of an EARLIER crop showing the same physical object, else -1"},
    "junk": {"type": "boolean", "description": "not an insurable possession worth about 25 USD (packaging, bag, box, basket, built-in)"}},
    "required": ["n", "same_as", "junk"]}}}, "required": ["items"]}
CONSOLIDATE_PROMPT = (
    "These numbered crops come from one walk through one room; the same object is often seen several times from "
    "different sides and described differently. For each crop n, give same_as = the number of an earlier crop that "
    "shows the SAME physical object (same size, colour, place and surroundings), or -1 if it is the first sighting. "
    "Similar but separate objects (two chairs side by side) are different. Mark junk=true for anything that is not an "
    "insurable possession worth about 25 USD: packaging, cardboard boxes, shopping bags, baskets, pouches, built-ins.\n"
)


def _crop(run_dir: Path, it: dict, side: int = 192) -> bytes:
    img = cv2.imread(str(Path(run_dir) / it["frame_ref"]))
    h, w = img.shape[:2]
    y0, x0, y1, x1 = (it.get("box_2d") or [0, 0, 1000, 1000])
    py, px = (y1 - y0) * 0.1, (x1 - x0) * 0.1
    c = img[int(max(0, y0 - py) * h / 1000):int(min(1000, y1 + py) * h / 1000), int(max(0, x0 - px) * w / 1000):int(min(1000, x1 + px) * w / 1000)]
    if c.size == 0:
        c = img
    s = side / max(c.shape[:2])
    return cv2.imencode(".jpg", cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


async def consolidate(run_dir: Path, items: list[dict], metrics: Metrics) -> list[dict]:
    """One look at a crop of every item at Finish: merge sightings of the same physical object and drop junk. Heading
    and wording alone can't tell, because a walking tour sees one table from several sides."""
    if len(items) < 2:
        return items
    items = items[:80]

    async def ask():
        parts = []
        for n, it in enumerate(items):
            parts += [f"crop {n}: {it['category']}, {it['description']}" + (f", {it['brand_model']}" if it.get("brand_model") else ""),
                      image_part(_crop(run_dir, it))]
        return await gemini_json(model=LITE_MODEL, parts=[CONSOLIDATE_PROMPT] + parts, schema=CONSOLIDATE_SCHEMA, metrics=metrics, stage=STAGE)
    r = await memo(run_dir, "item_consolidation", {"items": [[it["id"], it["frame_ref"]] for it in items]}, ask)
    verdict = {v["n"]: v for v in (r or {}).get("items", []) if 0 <= v.get("n", -1) < len(items)}
    keep, root = [], {}
    for n, it in enumerate(items):
        v = verdict.get(n, {"same_as": -1, "junk": False})
        j = v["same_as"]
        if v.get("junk"):
            continue
        first = items[root[j]] if 0 <= j < n and j in root else None  # root: the first kept sighting each crop belongs to
        if first and first.get("brand_model") and it.get("brand_model") and fuzz.token_set_ratio(first["brand_model"], it["brand_model"]) < 80:
            first = None  # two different legible brands are two objects
        if first:
            first["sightings"] = first.get("sightings", []) + [f for f in it.get("sightings", []) if f not in first.get("sightings", [])]
            if not first.get("brand_model") and it.get("brand_model") and brand_fits(first, it["brand_model"], it["description"]):
                first["brand_model"] = it["brand_model"]
            first["confidence"] = max(first.get("confidence", 0), it.get("confidence", 0))
            root[n] = root[j]
            continue
        root[n] = n
        keep.append(it)
    return keep


def approx_dims(frame: dict, box: list[int], touches_floor: bool, room: dict, imu: list[dict]) -> dict | None:
    """Width/height (cm) from the floor-contact ray and the measured camera height. None if not computable."""
    h_cam, f = room.get("camera_height_m"), room.get("focal_px")
    if not h_cam or not f or not touches_floor:
        return None
    up = G.up_at(imu, frame["t_ms"])
    if up is None:
        return None
    cam = G.Camera(f=f, cx=frame["w"] / 2, cy=frame["h"] / 2, up=up, azimuth=0.0, orient=frame.get("orient", 0))
    ymin, xmin, ymax, xmax = (v / 1000 for v in box)
    if min(ymin, xmin) < 0.01 or max(ymax, xmax) > 0.99:  # cut off by the frame edge: the box is not the whole item
        return None
    xb, yb = (xmin + xmax) / 2 * frame["w"], ymax * frame["h"]
    elev, _ = cam.elevation_azimuth(xb, yb)
    if elev > -math.radians(3):
        return None
    D = h_cam / math.tan(-elev)  # horizontal distance to the item's base
    fwd, right = cam._horizontal_basis()

    def g(x, y):
        r = cam.ray(x, y)
        n = float(r.dot(fwd))
        if n < 0.1:  # ray close to straight down or behind: the projection blows up
            raise ValueError
        return float(r.dot(cam.up)) / n, float(r.dot(right)) / n

    try:
        v_top, _ = g(xb, ymin * frame["h"])
        v_bot, _ = g(xb, yb)
        _, l0 = g(xmin * frame["w"], yb)
        _, l1 = g(xmax * frame["w"], yb)
    except ValueError:
        return None
    return {"w": round(D * (l1 - l0) * 100), "h": round(D * (v_top - v_bot) * 100), "d": None, "basis": "floor-contact ray x camera height"}


def add_dimensions(run_dir: Path, items: list[dict], room: dict) -> list[dict]:
    """Attach approximate dimensions once the room stage has measured camera height and focal length."""
    imu = read_jsonl(run_dir / "imu.jsonl")
    for it in items:
        it["dimensions_cm"] = approx_dims(it["frame"], it["box_2d"], it["touches_floor"], room, imu)
    write_json(run_dir / "items.json", items)
    return items
