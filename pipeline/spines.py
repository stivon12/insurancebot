"""Stage 2: detect and count book spines, merged across overlapping frames.

Per shelf keyframe:
  1. Gemini finds the shelf rows (horizontal bands of books).
  2. Each row crop is listed left-to-right twice with different padding; if the
     two counts disagree by more than 1 the row is split in halves and re-listed.
Across frames:
  3. Consecutive keyframes are chained with ORB + RANSAC homographies into one
     coordinate frame per "segment"; spine boxes are projected there and merged
     by overlap, so a book seen in five frames is counted once.

Output: runs/<id>/spines.json
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np

from .common import VISION_MODEL, Metrics, gemini_json, image_part, read_json, write_json

STAGE = "spines"

ROWS_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "y_top": {"type": "integer", "description": "0-1000, top of the band of books"},
                    "y_bottom": {"type": "integer", "description": "0-1000, bottom of the band (shelf board)"},
                    "has_books": {"type": "boolean"},
                },
                "required": ["y_top", "y_bottom", "has_books"],
            },
        }
    },
    "required": ["rows"],
}

SPINES_SCHEMA = {
    "type": "object",
    "properties": {
        "spines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[ymin, xmin, ymax, xmax] normalised 0-1000 within this image",
                    },
                    "orientation": {"type": "string", "enum": ["vertical", "horizontal"]},
                    "text": {"type": "string", "description": "Verbatim visible spine text; empty if none legible"},
                    "legible": {"type": "boolean"},
                },
                "required": ["box_2d", "orientation", "text", "legible"],
            },
        }
    },
    "required": ["spines"],
}

ROWS_PROMPT = (
    "This photo shows part of a bookcase. Return every horizontal band (shelf row) that contains books, "
    "top to bottom, as y_top/y_bottom on a 0-1000 scale of image height. Include books lying flat or stacked. "
    "Do not invent rows that are cut off by the image edge with no visible books."
)

SPINES_PROMPT = (
    "This image is one shelf row of a bookcase. List EVERY book spine from left to right, one entry per physical "
    "book, including thin ones, books lying flat or stacked (orientation=horizontal, list a stack top to bottom), "
    "and books whose text you cannot read. Give a tight box_2d around each spine. For text, transcribe exactly the "
    "characters you can see on that spine (title, author, publisher), not what you think the book is. If you cannot "
    "read it, set text to \"\" and legible=false. Never guess a title. Skip spines cut off at the left or right edge "
    "if less than half is visible."
)


def _crop(img: np.ndarray, y0: float, y1: float, x0: float = 0.0, x1: float = 1.0) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape[:2]
    ya, yb = max(0, int(y0 * h)), min(h, int(y1 * h))
    xa, xb = max(0, int(x0 * w)), min(w, int(x1 * w))
    return img[ya:yb, xa:xb], (xa, ya)


def _jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def _to_px(box: list[int], crop_shape: tuple[int, int], offset: tuple[int, int]) -> list[float]:
    """box_2d [ymin,xmin,ymax,xmax] (0-1000 in crop) -> [x0,y0,x1,y1] pixels in the full frame."""
    ch, cw = crop_shape
    ymin, xmin, ymax, xmax = (max(0, min(1000, v)) for v in box[:4])
    ox, oy = offset
    return [ox + xmin / 1000 * cw, oy + ymin / 1000 * ch, ox + xmax / 1000 * cw, oy + ymax / 1000 * ch]


async def _list_spines(img: np.ndarray, offset: tuple[int, int], metrics: Metrics) -> list[dict]:
    if img.size == 0:
        return []
    res = await gemini_json(
        model=VISION_MODEL, parts=[image_part(_jpeg(img)), SPINES_PROMPT], schema=SPINES_SCHEMA, metrics=metrics, stage=STAGE
    )
    out = []
    for s in (res or {}).get("spines", []):
        if len(s.get("box_2d", [])) != 4:
            continue
        out.append({**s, "box_px": _to_px(s["box_2d"], img.shape[:2], offset)})
    out.sort(key=lambda s: s["box_px"][0])
    return out


def _merge_halves(left: list[dict], right: list[dict], split_x: float) -> list[dict]:
    """Halves overlap; keep left spines whose centre is left of the split, right ones to the right."""
    keep = [s for s in left if (s["box_px"][0] + s["box_px"][2]) / 2 < split_x]
    keep += [s for s in right if (s["box_px"][0] + s["box_px"][2]) / 2 >= split_x]
    return sorted(keep, key=lambda s: s["box_px"][0])


async def _row(img: np.ndarray, row: dict, metrics: Metrics) -> dict:
    y0, y1 = row["y_top"] / 1000, row["y_bottom"] / 1000
    pad = (y1 - y0) * 0.06
    crop_a, off_a = _crop(img, y0 - pad, y1 + pad)
    crop_b, off_b = _crop(img, y0 - 3 * pad, y1 + 3 * pad)
    a, b = await asyncio.gather(_list_spines(crop_a, off_a, metrics), _list_spines(crop_b, off_b, metrics))
    check = "agree"
    spines = a
    if abs(len(a) - len(b)) > 1:
        # Disagreement: re-list each half with 10% overlap and stitch at the midpoint.
        w = img.shape[1]
        crop_l, off_l = _crop(img, y0 - pad, y1 + pad, 0.0, 0.55)
        crop_r, off_r = _crop(img, y0 - pad, y1 + pad, 0.45, 1.0)
        left, right = await asyncio.gather(_list_spines(crop_l, off_l, metrics), _list_spines(crop_r, off_r, metrics))
        spines = _merge_halves(left, right, 0.5 * w)
        check = "split"
    return {"spines": spines, "count_a": len(a), "count_b": len(b), "check": check}


async def _frame(run_dir: Path, kf: dict, metrics: Metrics) -> dict:
    img = cv2.imread(str(run_dir / kf["file"]))
    if img is None:
        return {"frame": kf, "observations": [], "rows": []}
    res = await gemini_json(
        model=VISION_MODEL, parts=[image_part(_jpeg(img)), ROWS_PROMPT], schema=ROWS_SCHEMA, metrics=metrics, stage=STAGE
    )
    rows = [r for r in (res or {}).get("rows", []) if r.get("has_books") and r["y_bottom"] > r["y_top"]]
    row_results = await asyncio.gather(*(_row(img, r, metrics) for r in rows))
    obs = []
    for r_idx, rr in enumerate(row_results):
        for s in rr["spines"]:
            obs.append({"frame_i": kf["i"], "file": kf["file"], "row_in_frame": r_idx, **s})
    checks = [{k: rr[k] for k in ("count_a", "count_b", "check")} for rr in row_results]
    return {"frame": kf, "observations": obs, "rows": checks}


# ---------- cross-frame merge ----------

def _gray(run_dir: Path, file: str, width: int = 960) -> tuple[np.ndarray, float]:
    img = cv2.imread(str(run_dir / file), cv2.IMREAD_GRAYSCALE)
    s = width / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def homography(a: np.ndarray, b: np.ndarray, min_inliers: int = 25) -> np.ndarray | None:
    """Homography mapping image b into image a, or None if the views do not overlap enough."""
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(db, da, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or int(mask.sum()) < min_inliers:
        return None
    return H


def chain_segments(run_dir: Path, keyframes: list[dict]) -> dict[int, tuple[int, np.ndarray]]:
    """frame_i -> (segment id, 3x3 matrix mapping full-res frame px into segment px)."""
    out: dict[int, tuple[int, np.ndarray]] = {}
    seg, prev, prev_to_seg, prev_scale = -1, None, None, 1.0
    for kf in keyframes:
        g, s = _gray(run_dir, kf["file"])
        S = np.diag([s, s, 1.0])
        H = homography(prev, g) if prev is not None else None
        if H is None:
            seg += 1
            to_seg = np.eye(3)  # a new segment starts in this frame's own pixel coordinates
        else:
            # full-res b -> small b -> small a -> full-res a -> segment
            to_seg = prev_to_seg @ np.linalg.inv(np.diag([prev_scale, prev_scale, 1.0])) @ H @ S
        out[kf["i"]] = (seg, to_seg)
        prev, prev_to_seg, prev_scale = g, to_seg, s
    return out


def _project(box: list[float], M: np.ndarray) -> list[float]:
    x0, y0, x1, y1 = box
    pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    p = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge(observations: list[dict], segs: dict[int, tuple[int, np.ndarray]], iou_thr: float = 0.3) -> list[dict]:
    clusters: list[dict] = []
    for o in observations:
        seg, M = segs.get(o["frame_i"], (-1, np.eye(3)))
        box = _project(o["box_px"], M)
        best, best_iou = None, iou_thr
        for c in clusters:
            if c["segment"] != seg or any(x["frame_i"] == o["frame_i"] for x in c["observations"]):
                continue  # one book cannot appear twice in the same frame
            v = _iou(box, c["seg_box"])
            if v > best_iou:
                best, best_iou = c, v
        if best is None:
            clusters.append({"segment": seg, "seg_box": box, "observations": [o]})
        else:
            n = len(best["observations"])
            best["seg_box"] = [(bv * n + v) / (n + 1) for bv, v in zip(best["seg_box"], box)]
            best["observations"].append(o)
    return clusters


def assign_shelves(clusters: list[dict]) -> list[dict]:
    """Group merged spines into rows per segment by vertical position, then order left to right."""
    books = []
    for seg in sorted({c["segment"] for c in clusters}):
        cs = sorted([c for c in clusters if c["segment"] == seg], key=lambda c: (c["seg_box"][1] + c["seg_box"][3]) / 2)
        heights = [c["seg_box"][3] - c["seg_box"][1] for c in cs]
        gap = 0.5 * float(np.median(heights)) if heights else 0.0
        rows: list[list[dict]] = []
        last_cy = None
        for c in cs:
            cy = (c["seg_box"][1] + c["seg_box"][3]) / 2
            if last_cy is None or cy - last_cy > gap:
                rows.append([])
            rows[-1].append(c)
            last_cy = cy
        for r_idx, row in enumerate(rows):
            for pos, c in enumerate(sorted(row, key=lambda c: c["seg_box"][0])):
                books.append({**c, "shelf": f"U{seg + 1}-S{r_idx + 1}", "position": pos + 1})
    for n, b in enumerate(books):
        b["id"] = f"B{n + 1:03d}"
        # Best evidence frame: largest legible observation, else largest.
        b["best"] = max(
            b["observations"],
            key=lambda o: (o["legible"], (o["box_px"][2] - o["box_px"][0]) * (o["box_px"][3] - o["box_px"][1])),
        )
    return books


async def run(run_dir: Path, keyframes: dict, metrics: Metrics) -> list[dict]:
    kfs = keyframes.get("shelves", [])
    with metrics.time(STAGE):
        per_frame = await asyncio.gather(*(_frame(run_dir, kf, metrics) for kf in kfs))
        observations = [o for pf in per_frame for o in pf["observations"]]
        segs = chain_segments(run_dir, kfs)
        books = assign_shelves(merge(observations, segs))
    row_checks = [{"frame_i": pf["frame"]["i"], "rows": pf["rows"]} for pf in per_frame]
    out = {"books": books, "row_checks": row_checks, "keyframes_used": len(kfs), "observations": len(observations)}
    write_json(run_dir / "spines.json", out)
    return books
