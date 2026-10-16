"""Stage 6: non-book contents (furniture, lamps, art, rugs, electronics, decor).

Gemini Flash-Lite looks at batches of keyframes in sweep order and is given the
running item list, so it marks re-sightings as the same item instead of a new one.
Approximate dimensions come from geometry (camera height + the floor-contact ray),
not from the model.

Output: runs/<id>/items.json
"""

from __future__ import annotations

import asyncio

import math
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from .common import LITE_MODEL, Metrics, gemini_json, image_part, memo, read_json, read_jsonl, write_json

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
    "already in the KNOWN ITEMS list, set same_as to its id instead of listing it as new; the LAST image (if present) "
    "is not a room frame but a sheet of the known items' crops, each labelled with its id, to compare against. "
    "Report brand_model only if the name is actually legible in the image. Two identical objects are two items (or "
    "count=2)."
)

GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "description": "sets of ids that show the SAME physical object; omit items that are unique",
            "items": {"type": "array", "items": {"type": "string"}},
        }
    },
    "required": ["groups"],
}
GROUP_PROMPT = (
    "Each tile is a crop of an object from a walk through ONE room, labelled with its id. Some objects were seen "
    "several times from different angles and listed twice. Group the ids that show the same physical object. Two "
    "matching chairs or lamps that are separate objects are NOT the same object: only group ids if they are clearly "
    "one object (same position cues, same surroundings, same details). If unsure, do not group."
)
TILE = 200


def _crop(run_dir: Path, it: dict) -> np.ndarray | None:
    img = cv2.imread(str(Path(run_dir) / it["frame_ref"]))
    if img is None:
        return None
    ymin, xmin, ymax, xmax = it["box_2d"]
    h, w = img.shape[:2]
    c = img[int(ymin / 1000 * h): int(ymax / 1000 * h), int(xmin / 1000 * w): int(xmax / 1000 * w)]
    return c if c.size else None


def contact_sheet(run_dir: Path, items: list[dict], cols: int = 6) -> bytes | None:
    """One image of the items' crops in a grid, each tile labelled with the item id."""
    tiles = []
    for it in items:
        c = _crop(run_dir, it)
        if c is None:
            continue
        s = (TILE - 24) / max(c.shape[:2])
        c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        tile = np.full((TILE, TILE, 3), 255, np.uint8)
        y0, x0 = 24 + (TILE - 24 - c.shape[0]) // 2, (TILE - c.shape[1]) // 2
        tile[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
        cv2.putText(tile, it["id"], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        tiles.append(tile)
    if not tiles:
        return None
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    rows[-1] += [np.full((TILE, TILE, 3), 255, np.uint8)] * (cols - len(rows[-1]))
    sheet = np.vstack([np.hstack(r) for r in rows])
    return cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


async def merge_duplicates(run_dir: Path, items: list[dict], metrics: Metrics) -> list[dict]:
    """One last look at all item crops together: ids the model groups as the same physical object (and of the same
    category) become one item with all sightings. Memoised on the list, so a rebuilt packet does not pay again."""
    if len(items) < 2:
        return items
    key = [(it["id"], it["frame_ref"], it["box_2d"]) for it in items]

    async def ask():
        sheet = contact_sheet(run_dir, items[:48])
        if sheet is None:
            return {"groups": []}
        return await gemini_json(model=LITE_MODEL, parts=[image_part(sheet), GROUP_PROMPT], schema=GROUP_SCHEMA,
                                 metrics=metrics, stage=STAGE)

    res = await memo(run_dir, "item_groups", key, ask) or {"groups": []}
    by_id = {it["id"]: it for it in items}
    drop: set[str] = set()
    for g in res.get("groups", []):
        ids = [i for i in g if i in by_id and i not in drop]
        if len(ids) < 2:
            continue
        keep = by_id[ids[0]]
        for i in ids[1:]:
            dup = by_id[i]
            if dup["category"] != keep["category"]:
                continue  # different kinds of object: never merged
            keep["sightings"] += dup["sightings"]
            keep.setdefault("merged_ids", []).append(i)
            if dup.get("brand_model") and not keep.get("brand_model"):
                keep["brand_model"] = dup["brand_model"]
            if dup.get("confidence", 0) > keep.get("confidence", 0):  # keep the clearer view as evidence
                for k in ("frame_ref", "box_2d", "touches_floor", "frame", "confidence"):
                    keep[k] = dup[k]
            drop.add(i)
    return [it for it in items if it["id"] not in drop]


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


def frames_for(keyframes: dict) -> list[dict]:
    return sorted([f for phase in ("room_turn", "unit_overview", "pan") for f in keyframes.get(phase, [])], key=lambda f: f["t_ms"])


_locks: dict[str, asyncio.Lock] = {}


async def run(run_dir: Path, keyframes: dict, metrics: Metrics, tick=None, complete_batches_only: bool = False) -> list[dict]:
    """Running item list, extended in time order with keyframes not yet looked at.

    The state (frames already looked at + items so far) is kept in runs/<id>/cache/items_state.json, so the live
    worker can build the list during the sweep and the post-sweep run only adds the frames it had not reached.
    Each batch sees the list built so far: that is how re-sightings are recognised instead of re-counted."""
    lock = _locks.setdefault(str(run_dir), asyncio.Lock())
    async with lock:
        state_p = Path(run_dir) / "cache" / "items_state.json"
        state = read_json(state_p, None) or {"done": [], "items": []}
        done = set(state["done"])
        items: list[dict] = state["items"]
        all_frames = frames_for(keyframes)
        frames = [f for f in all_frames if f["file"] not in done]
        if tick and len(frames) < len(all_frames):
            tick(len(all_frames) - len(frames))  # already looked at during the sweep
        if complete_batches_only:  # during the sweep: leave the last, still-filling batch for later
            frames = frames[: len(frames) // BATCH * BATCH]
        with metrics.time(STAGE):
            for start in range(0, len(frames), BATCH):
                batch = frames[start : start + BATCH]
                parts = [image_part(_thumb(run_dir / f["file"])) for f in batch]
                sheet = contact_sheet(run_dir, items[-30:]) if items else None  # what the known items look like
                parts += ([image_part(sheet)] if sheet else []) + [_known(items), PROMPT]
                res = await gemini_json(model=LITE_MODEL, parts=parts, schema=ITEMS_SCHEMA, metrics=metrics, stage=STAGE)
                if res is None:
                    break  # model call failed: leave these frames for the next pass
                if tick:
                    tick(len(batch))
                _apply(res, batch, items)
                state["done"] += [f["file"] for f in batch]
                write_json(state_p, state)
        if not complete_batches_only:  # the packet build: one last pass for re-sightings listed twice
            with metrics.time(STAGE):
                items = await merge_duplicates(run_dir, [dict(it) for it in items], metrics)
    write_json(run_dir / "items.json", items)
    return items


def _apply(res: dict, batch: list[dict], items: list[dict]) -> None:
    for it in res.get("items", []):
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


def add_dimensions(run_dir: Path, items: list[dict], room: dict) -> list[dict]:
    """Attach approximate dimensions once the room stage has measured camera height and focal length."""
    imu = read_jsonl(run_dir / "imu.jsonl")
    for it in items:
        it["dimensions_cm"] = approx_dims(it["frame"], it["box_2d"], it["touches_floor"], room, imu)
    write_json(run_dir / "items.json", items)
    return items
