"""Stage 6: non-book contents (furniture, lamps, art, rugs, electronics, decor).

Gemini Flash-Lite looks at batches of keyframes in sweep order and is given the
running item list, so it marks re-sightings as the same item instead of a new one.
Approximate dimensions come from geometry (camera height + the floor-contact ray),
not from the model.

Output: runs/<id>/items.json
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from .common import LITE_MODEL, Metrics, gemini_json, image_part, read_jsonl, write_json

STAGE = "items"
BATCH = 8

CATEGORIES = [
    "shelving", "seating", "table", "desk", "storage", "lamp", "art_portrait", "art_framed", "mirror", "rug",
    "electronics", "appliance", "coffee_machine", "decor", "plant", "clock", "textile", "other",
]

ITEMS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image": {"type": "integer", "description": "index of the image (0-based) where it is clearest"},
                    "same_as": {"type": "string", "description": "id of an already-listed item if this is a re-sighting, else empty"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "description": {"type": "string", "description": "short: colour, style"},
                    "material": {"type": "string"},
                    "brand_model": {"type": "string", "description": "only if a brand/model name is legible; else empty"},
                    "box_2d": {"type": "array", "items": {"type": "integer"}, "description": "[ymin,xmin,ymax,xmax] 0-1000"},
                    "touches_floor": {"type": "boolean"},
                    "count": {"type": "integer", "description": "number of identical physical items in this box, usually 1"},
                    "confidence": {"type": "number"},
                },
                "required": ["image", "same_as", "category", "description", "material", "brand_model", "box_2d", "touches_floor", "count", "confidence"],
            },
        }
    },
    "required": ["items"],
}

PROMPT = (
    "These photos are consecutive frames from a walk through one room. List every physical object that is NOT a "
    "book: furniture, shelving units, lamps, framed art and portraits, mirrors, rugs, electronics, appliances (e.g. "
    "coffee machine), decor, plants. Ignore walls, doors, windows, light switches and loose papers. If an object is "
    "already in the KNOWN ITEMS list, set same_as to its id instead of listing it as new. Report brand_model only if "
    "the name is actually legible in the image. Two identical objects are two items (or count=2)."
)


def _thumb(path: Path, width: int = 768) -> bytes:
    img = cv2.imread(str(path))
    s = width / img.shape[1]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def _known(items: list[dict]) -> str:
    if not items:
        return "KNOWN ITEMS: none yet."
    return "KNOWN ITEMS:\n" + "\n".join(f"{i['id']}: {i['category']} - {i['description']} ({i['material']})" for i in items)


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


async def run(run_dir: Path, keyframes: dict, metrics: Metrics) -> list[dict]:
    frames = sorted(
        [f for phase in ("room_turn", "unit_overview", "pan") for f in keyframes.get(phase, [])], key=lambda f: f["t_ms"]
    )
    items: list[dict] = []
    with metrics.time(STAGE):
        # Sequential batches: each batch sees the list built so far (that is how duplicates are avoided).
        for start in range(0, len(frames), BATCH):
            batch = frames[start : start + BATCH]
            parts = [image_part(_thumb(run_dir / f["file"])) for f in batch] + [_known(items), PROMPT]
            res = await gemini_json(model=LITE_MODEL, parts=parts, schema=ITEMS_SCHEMA, metrics=metrics, stage=STAGE)
            for it in (res or {}).get("items", []):
                if not 0 <= it.get("image", -1) < len(batch) or len(it.get("box_2d", [])) != 4:
                    continue
                frame = batch[it["image"]]
                existing = next((x for x in items if x["id"] == it.get("same_as")), None)
                if existing:
                    existing["sightings"].append(frame["file"])
                    if it.get("brand_model") and not existing["brand_model"]:
                        existing["brand_model"] = it["brand_model"]
                    continue
                for k in range(max(1, int(it.get("count") or 1))):
                    items.append(
                        {
                            "id": f"I{len(items) + 1:03d}",
                            "category": it["category"],
                            "description": it["description"],
                            "material": it["material"],
                            "brand_model": it.get("brand_model", ""),
                            "frame_ref": frame["file"],
                            "box_2d": it["box_2d"],
                            "touches_floor": it["touches_floor"],
                            "confidence": round(float(it.get("confidence", 0.5)), 2),
                            "frame": {k: frame[k] for k in ("t_ms", "w", "h") if k in frame} | {"orient": frame.get("orient", 0)},
                            "sightings": [frame["file"]],
                        }
                    )
    write_json(run_dir / "items.json", items)
    return items


def add_dimensions(run_dir: Path, items: list[dict], room: dict) -> list[dict]:
    """Attach approximate dimensions once the room stage has measured camera height and focal length."""
    imu = read_jsonl(run_dir / "imu.jsonl")
    for it in items:
        it["dimensions_cm"] = approx_dims(it["frame"], it["box_2d"], it["touches_floor"], room, imu)
    write_json(run_dir / "items.json", items)
    return items
