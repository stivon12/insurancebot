"""Non-book contents: the live model logs an item, Flash-Lite checks it on that frame, code merges re-sightings."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
from rapidfuzz import fuzz

from . import geometry as G
from .common import LITE_MODEL, Metrics, gemini_json, image_part, memo, read_jsonl, write_json

STAGE = "items"
CATEGORIES = ["shelving", "furniture", "coffee_machine", "lamp", "art", "rug", "electronics", "decor"]
SAME_HEADING_DEG = 25

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


def _turn(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def find_same(items: list[dict], new: dict) -> dict | None:
    """Same category, seen in nearly the same direction, and a similar description: the same object again."""
    for it in items:
        if it["category"] != new["category"]:
            continue
        if it.get("heading") is None or new.get("heading") is None:  # no motion data: only a near-identical description
            if fuzz.token_sort_ratio(it["description"], new["description"]) >= 85:
                return it
        elif _turn(it["heading"], new["heading"]) <= SAME_HEADING_DEG and fuzz.token_set_ratio(it["description"], new["description"]) >= 45:
            return it
    return None


def accept(items: list[dict], res: dict, frame: dict, heading: float | None) -> tuple[dict | None, str]:
    """Add a checked item (or a re-sighting) to the list. Returns (item, 'new' | 'same' | why rejected)."""
    if not res:
        return None, "check failed"
    if not res.get("present"):
        return None, "not visible in the frame"
    if not res.get("insurable"):
        return None, "not an insurable contents item"
    if len(res.get("box_2d", [])) != 4:
        res["box_2d"] = [0, 0, 1000, 1000]
    new = {k: res[k] for k in ("category", "description", "material", "brand_model", "box_2d", "touches_floor")}
    new["heading"] = None if heading is None else round(heading % 360, 1)
    same = find_same(items, new)
    if same:
        same["sightings"].append(frame["file"])
        if new["brand_model"] and not same["brand_model"]:
            same["brand_model"] = new["brand_model"]
        return same, "same"
    new.update({"id": f"I{len(items) + 1:03d}", "frame_ref": frame["file"], "confidence": 0.9, "sightings": [frame["file"]],
                "frame": {k: frame[k] for k in ("t_ms", "w", "h") if k in frame} | {"orient": frame.get("orient", 0)}})
    items.append(new)
    return new, "new"


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
    xb, yb = (xmin + xmax) / 2 * frame["w"], ymax * frame["h"]
    elev, _ = cam.elevation_azimuth(xb, yb)
    if elev > -math.radians(3):
        return None
    D = h_cam / math.tan(-elev)  # horizontal distance to the item's base
    fwd, right = cam._horizontal_basis()

    def g(x, y):
        r = cam.ray(x, y)
        n = float(r.dot(fwd))
        return float(r.dot(cam.up)) / n, float(r.dot(right)) / n

    v_top, _ = g(xb, ymin * frame["h"])
    v_bot, _ = g(xb, yb)
    _, l0 = g(xmin * frame["w"], yb)
    _, l1 = g(xmax * frame["w"], yb)
    return {"w": round(D * (l1 - l0) * 100), "h": round(D * (v_top - v_bot) * 100), "d": None, "basis": "floor-contact ray x camera height"}


def add_dimensions(run_dir: Path, items: list[dict], room: dict) -> list[dict]:
    """Attach approximate dimensions once the room stage has measured camera height and focal length."""
    imu = read_jsonl(run_dir / "imu.jsonl")
    for it in items:
        it["dimensions_cm"] = approx_dims(it["frame"], it["box_2d"], it["touches_floor"], room, imu)
    write_json(run_dir / "items.json", items)
    return items
