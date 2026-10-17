"""Books: spine detection (RF-DETR + Gemini), reading, catalogue identification, spine sizes."""

from __future__ import annotations

import asyncio
import cv2
import httpx
import importlib.util
import numpy as np
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from rapidfuzz import fuzz

from .common import LITE_MODEL, Metrics, VISION_MODEL, gather_until, gemini_json, get_json, image_part, memo, write_json


SPINES_STAGE = "spines"
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
                    "is_book": {"type": "string", "enum": ["yes", "unsure"],
                                "description": "yes = clearly a printed book; unsure = could be a box, folder, case or other object"},
                },
                "required": ["box_2d", "orientation", "text", "legible", "is_book"],
            },
        }
    },
    "required": ["spines"],
}
ROWS_PROMPT = (
    "This photo may show part of a bookcase. Return every horizontal band (shelf row) that contains printed books, "
    "top to bottom, as y_top/y_bottom on a 0-1000 scale of image height. Include books lying flat or stacked. "
    "Set has_books=false for shelves holding only boxes, packaging, folders, electronics, cables, ornaments or other "
    "objects. Do not invent rows that are cut off by the image edge with no visible books. If there are no books "
    "in the photo, return no rows."
)
SPINES_PROMPT = (
    "This image is one shelf row. List every BOOK spine from left to right, one entry per physical book, including "
    "thin ones, books lying flat or stacked (orientation=horizontal, list a stack top to bottom, one entry per book: "
    "count the separate page blocks and covers on the stack's side, not the stack as one), and books whose "
    "text you cannot read. For a book lying cover up, its text may be the title and author printed on that visible "
    "cover. Only printed books: do NOT list boxes, product packaging, cartons, folders, binders, "
    "magazine files, DVD/game cases, electronics, cables, cloth or ornaments. If you are not sure an object is a "
    "book, list it with is_book=\"unsure\". Give a tight box_2d around each spine. For text, transcribe exactly the "
    "characters you can see on that spine (title, author, publisher), not what you think the book is. If you cannot "
    "read it, set text to \"\" and legible=false. Never guess a title. Skip spines cut off at the left or right edge "
    "if less than half is visible."
)
PROMPT_VERSION = 3  # bump when the prompts or schema change, so frames read with old prompts are not reused
def countable(book: dict, idn: dict) -> tuple[bool, str]:
    """Whether a merged spine counts as a book in the packet."""
    obs = book["observations"]
    if idn.get("status") == "identified":
        return True, "identified"
    sightings = len({o["frame_i"] for o in obs})
    detector = any(str(o.get("box_source", "")).startswith("rfdetr") for o in obs)
    unsure = all(o.get("is_book") == "unsure" for o in obs)
    legible = any(o.get("legible") for o in obs)
    if legible and not unsure:
        return True, "legible spine text"
    if sightings >= 2:
        return True, f"seen in {sightings} frames"
    if detector:
        return True, "spine detector agrees"
    why = "model unsure it is a book" if unsure else "unreadable"
    return False, f"{why}, seen once, spine detector found no spine there"
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
        model=VISION_MODEL, parts=[image_part(_jpeg(img)), SPINES_PROMPT], schema=SPINES_SCHEMA, metrics=metrics, stage=SPINES_STAGE
    )
    out = []
    for s in (res or {}).get("spines", []):
        if len(s.get("box_2d", [])) != 4:
            continue
        out.append({**s, "box_px": _to_px(s["box_2d"], img.shape[:2], offset)})
    out.sort(key=lambda s: s["box_px"][0])
    return out
def _overlaps(p: list[float], q: list[float], min_iou: float = 0.3) -> bool:
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
    return union > 0 and inter / union >= min_iou
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
    if abs(len(a) - len(b)) == 1:
        # Off by one: keep a, and add b's spines that a did not list as 'unsure', so they count only with support
        # (a second sighting or the spine detector, see countable). Taking a alone dropped real books from short rows
        extra = [{**x, "is_book": "unsure"} for x in b if not any(_overlaps(x["box_px"], y["box_px"]) for y in a)]
        if extra:
            spines, check = sorted(a + extra, key=lambda s: s["box_px"][0]), "union"
    elif abs(len(a) - len(b)) > 1:
        # Disagreement: re-list each half with 10% overlap and stitch at the midpoint.
        w = img.shape[1]
        crop_l, off_l = _crop(img, y0 - pad, y1 + pad, 0.0, 0.55)
        crop_r, off_r = _crop(img, y0 - pad, y1 + pad, 0.45, 1.0)
        left, right = await asyncio.gather(_list_spines(crop_l, off_l, metrics), _list_spines(crop_r, off_r, metrics))
        spines = _merge_halves(left, right, 0.5 * w)
        check = "split"
    return {"spines": spines, "count_a": len(a), "count_b": len(b), "check": check}
def merge_bands(rows: list[dict], min_overlap: float = 0.3) -> list[dict]:
    """Union row bands that overlap vertically: the model sometimes returns several bands over one shelf row
    (top, middle, bottom of the same spines), which would list every book once per band."""
    out: list[dict] = []
    for r in sorted(rows, key=lambda r: r["y_top"]):
        if out:
            last = out[-1]
            inter = min(last["y_bottom"], r["y_bottom"]) - max(last["y_top"], r["y_top"])
            smaller = min(last["y_bottom"] - last["y_top"], r["y_bottom"] - r["y_top"])
            if smaller > 0 and inter / smaller > min_overlap:
                last["y_top"], last["y_bottom"] = min(last["y_top"], r["y_top"]), max(last["y_bottom"], r["y_bottom"])
                continue
        out.append(dict(r))
    return out
def dedupe_in_frame(obs: list[dict], x_overlap: float = 0.6) -> list[dict]:
    """One physical spine can be listed from two neighbouring row crops of the same frame."""
    keep: list[dict] = []
    for o in sorted(obs, key=lambda o: -(o["box_px"][3] - o["box_px"][1])):
        x0, y0, x1, y1 = o["box_px"]
        dup = False
        for k in keep:
            if k.get("row_in_frame") == o.get("row_in_frame"):
                continue
            a0, b0, a1, b1 = k["box_px"]
            ix = min(x1, a1) - max(x0, a0)
            iy = min(y1, b1) - max(y0, b0)
            if iy > 0 and ix > x_overlap * min(x1 - x0, a1 - a0):
                dup = True
                break
        if not dup:
            keep.append(o)
    return keep
async def _frame(run_dir: Path, kf: dict, metrics: Metrics) -> dict:
    """One keyframe's spine list. Memoised per frame file, so frames read during the sweep are not read again."""
    r = await memo(run_dir, "spine_frames", {"file": kf["file"], "prompt": PROMPT_VERSION}, lambda: _frame_uncached(run_dir, kf, metrics))
    return {**(r or {"observations": [], "rows": []}), "frame": kf}
async def _frame_uncached(run_dir: Path, kf: dict, metrics: Metrics) -> dict | None:
    img = cv2.imread(str(run_dir / kf["file"]))
    if img is None:
        return {"frame": kf, "observations": [], "rows": []}
    res = await gemini_json(
        model=VISION_MODEL, parts=[image_part(_jpeg(img)), ROWS_PROMPT], schema=ROWS_SCHEMA, metrics=metrics, stage=SPINES_STAGE
    )
    if res is None:
        return None  # model call failed: not saved, retried later
    rows = merge_bands([r for r in (res or {}).get("rows", []) if r.get("has_books") and r["y_bottom"] > r["y_top"]])
    row_results = await asyncio.gather(*(_row(img, r, metrics) for r in rows))
    obs = []
    for r_idx, rr in enumerate(row_results):
        for s in rr["spines"]:
            obs.append({"frame_i": kf["i"], "file": kf["file"], "row_in_frame": r_idx, **s})
    n_before = len(obs)
    obs = dedupe_in_frame(obs)
    checks = [{k: rr[k] for k in ("count_a", "count_b", "check")} for rr in row_results] + [{"in_frame_duplicates_removed": n_before - len(obs)}]
    return {"frame": kf, "observations": obs, "rows": checks}
SAME_VIEW_SHIFT, SAME_VIEW_ZOOM = 0.15, 0.15
def same_view(H, width: float, height: float) -> bool:
    """H maps the candidate frame into the last kept one (same pixel scale). Same view = the candidate's centre
    moved less than 15 % of the width and the apparent size changed less than 15 %."""
    import numpy as np

    c = np.array([width / 2, height / 2, 1.0])
    p = H @ c
    p = p[:2] / p[2]
    shift = float(np.hypot(p[0] - c[0], p[1] - c[1])) / width
    zoom = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
    return shift < SAME_VIEW_SHIFT and abs(zoom - 1) < SAME_VIEW_ZOOM
def _gray(run_dir: Path, file: str, width: int = 960) -> tuple[np.ndarray, float]:
    img = cv2.imread(str(run_dir / file), cv2.IMREAD_GRAYSCALE)
    s = width / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s
def features(g: np.ndarray):
    return cv2.ORB_create(3000).detectAndCompute(g, None)
@lru_cache(maxsize=1024)
def _frame_features(path: str) -> tuple:
    g, s = _gray(Path(path).parent, Path(path).name)
    return features(g), s, g.shape
def frame_features(run_dir: Path, file: str) -> tuple:
    """(ORB features, scale to the 960 px copy, grey shape) per frame file, cached: during the sweep the worker
    re-runs keyframe selection and frame tracking every few seconds."""
    return _frame_features(str(Path(run_dir) / file))
def homography(a, b, min_inliers: int = 25) -> np.ndarray | None:
    """Homography mapping image b into image a, or None if the views do not overlap enough.
    a and b are grey images or precomputed (keypoints, descriptors) from features()."""
    ka, da = a if isinstance(a, tuple) else features(a)
    kb, db = b if isinstance(b, tuple) else features(b)
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
REATTACH_MIN_INLIERS = 40  # stricter than frame-to-frame: a wrong re-attachment would merge different books
ANCHORS_PER_SEGMENT = 6
def chain_segments(run_dir: Path, keyframes: list[dict]) -> dict[int, tuple[int, np.ndarray]]:
    """frame_i -> (segment id, 3x3 matrix mapping full-res frame px into segment px)."""
    out: dict[int, tuple[int, np.ndarray]] = {}
    anchors: dict[int, list[tuple]] = {}  # segment -> [(features, to_seg, scale)]
    seg_count, seg, prev = 0, -1, None
    for kf in keyframes:
        feat, s, _ = frame_features(run_dir, kf["file"])
        S = np.diag([s, s, 1.0])
        to_seg = None
        if prev is not None:
            H = homography(prev[0], feat)
            if H is not None:
                # full-res b -> small b -> small a -> full-res a -> segment
                to_seg = prev[1] @ np.linalg.inv(np.diag([prev[2], prev[2], 1.0])) @ H @ S
        if to_seg is None:
            for cand in sorted(anchors, reverse=True):  # most recent segment first
                if cand == seg:
                    continue
                for a_feat, a_to_seg, a_scale in anchors[cand]:
                    H = homography(a_feat, feat, REATTACH_MIN_INLIERS)
                    if H is not None:
                        seg, to_seg = cand, a_to_seg @ np.linalg.inv(np.diag([a_scale, a_scale, 1.0])) @ H @ S
                        break
                if to_seg is not None:
                    break
        if to_seg is None:
            seg, seg_count = seg_count, seg_count + 1
            to_seg = np.eye(3)  # a new segment starts in this frame's own pixel coordinates
        out[kf["i"]] = (seg, to_seg)
        prev = (feat, to_seg, s)
        a = anchors.setdefault(seg, [])
        a.append(prev)
        if len(a) > ANCHORS_PER_SEGMENT:  # keep anchors spread along the segment: drop every other one
            del a[1::2]
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
def _centre_in(a: list[float], b: list[float]) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
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
            if v <= iou_thr and _centre_in(box, c["seg_box"]):
                v = iou_thr + 1e-3  # thin boxes (flat stacks) barely overlap after projection: same place = same book
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


BOX_REPO = "Geraldine/rf-detr-nano-bookshelf"
BOX_REVISION = os.getenv("SPINE_BOX_MODEL_REV", "v1")
BOX_THRESHOLD = 0.6  # false boxes on a no-books room scored median 0.43; real spines 0.63-0.96
BOX_TILES = 3
_detector = None
def detector_available() -> bool:
    return importlib.util.find_spec("rfdetr") is not None
def _load_detector():
    global _detector
    if _detector is None:
        from huggingface_hub import hf_hub_download
        from rfdetr import RFDETRNano

        ckpt = hf_hub_download(BOX_REPO, "checkpoints/checkpoint_best_ema.pth", revision=BOX_REVISION)
        _detector = RFDETRNano(pretrain_weights=ckpt)
        try:
            _detector.optimize_for_inference()
        except Exception:
            pass
    return _detector
def nms(boxes: np.ndarray, scores: np.ndarray, overlap: float = 0.5) -> list[int]:
    """Greedy NMS on overlap of the smaller box (tiles re-detect the same spine at slightly different sizes)."""
    order = np.argsort(-scores)
    keep: list[int] = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        ih = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        o = iw * ih / np.maximum(np.minimum(area[i], area[rest]), 1e-6)
        order = rest[o < overlap]
    return keep
def detect(img_rgb: np.ndarray, threshold: float = BOX_THRESHOLD, tiles: int = BOX_TILES) -> np.ndarray:
    """Spine boxes [x0,y0,x1,y1] in pixels: whole image plus overlapping vertical strips (thin spines are tiny at
    the model's 384 px input), merged by NMS."""
    from PIL import Image

    model = _load_detector()
    H, W = img_rgb.shape[:2]
    windows = [(0, W)]
    if tiles > 1:
        step = W / (tiles - (tiles - 1) * 0.25)
        windows += [(int(k * step * 0.75), int(min(W, k * step * 0.75 + step))) for k in range(tiles)]
    boxes, scores = [], []
    for x0, x1 in windows:
        det = model.predict(Image.fromarray(img_rgb[:, x0:x1]), threshold=threshold)
        if len(det):
            b = det.xyxy.astype(float).copy()
            b[:, [0, 2]] += x0
            boxes.append(b)
            scores.append(det.confidence)
    if not boxes:
        return np.zeros((0, 4))
    B, S = np.concatenate(boxes), np.concatenate(scores)
    return B[nms(B, S)]
def match(gemini_boxes: list[list[float]], rf: np.ndarray, min_v_overlap: float = 0.5) -> dict[int, int]:
    """One-to-one: Gemini spine i -> RF-DETR box j whose centre is inside spine i's box and which overlaps it
    vertically by >= 50 % of the shorter height. Closest centres are paired first."""
    pairs = []
    for i, g in enumerate(gemini_boxes):
        gx0, gy0, gx1, gy1 = g
        gc = (gx0 + gx1) / 2
        for j, r in enumerate(rf):
            rc_x, rc_y = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
            vo = min(gy1, r[3]) - max(gy0, r[1])
            if gx0 <= rc_x <= gx1 and gy0 <= rc_y <= gy1 and vo >= min_v_overlap * min(gy1 - gy0, r[3] - r[1]):
                pairs.append((abs(rc_x - gc), i, j))
    pairs.sort()
    out: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in pairs:
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out
def combine(gemini_obs: list[dict], rf: np.ndarray) -> list[dict]:
    """Rule A: every RF-DETR box is a spine (with Gemini's text when matched); unmatched Gemini spines count only if
    legible. On 15 unseen 4TU shelf photos: +4.7 % count error, against +15.9 % for Gemini alone."""
    if not gemini_obs:  # Gemini saw no shelf of books here: detector boxes alone are chairs and cushions as often as spines
        return []
    bands = {}
    for o in gemini_obs:
        y0, y1 = o["box_px"][1], o["box_px"][3]
        lo, hi = bands.get(o.get("row_in_frame", 0), (y0, y1))
        bands[o.get("row_in_frame", 0)] = (min(lo, y0), max(hi, y1))
    on_row = lambda b: any(lo - 0.1 * (hi - lo) <= (b[1] + b[3]) / 2 <= hi + 0.1 * (hi - lo) for lo, hi in bands.values())  # noqa: E731
    rf = np.array([b for b in rf if on_row(b)]).reshape(-1, 4)  # boxes off Gemini's book rows are boxes, baskets, batteries
    m = match([o["box_px"] for o in gemini_obs], rf)
    by_rf = {j: i for i, j in m.items()}
    base = gemini_obs[0] if gemini_obs else {}
    out = []
    for j, box in enumerate(rf):
        g = gemini_obs[by_rf[j]] if j in by_rf else {"text": "", "legible": False, "is_book": "yes", "orientation": "vertical"}
        out.append({**{k: base[k] for k in ("frame_i", "file") if k in base}, **g, "box_px": [float(v) for v in box],
                    "box_source": f"rfdetr-{BOX_REVISION}"})
    out += [{**o, "box_source": "gemini"} for i, o in enumerate(gemini_obs) if i not in m and o.get("legible")]
    return sorted(out, key=lambda o: o["box_px"][0])


IDENTIFY_STAGE = "identify"
READ_SCHEMA = {
    "type": "object",
    "properties": {
        "verbatim": {"type": "string", "description": "Exactly the characters visible on the spine"},
        "title": {"type": "string", "description": "Title as printed on the spine, or empty"},
        "author": {"type": "string", "description": "Author as printed on the spine, or empty"},
        "publisher": {"type": "string", "description": "Publisher/imprint as printed, or empty"},
        "legible": {"type": "boolean"},
        "confidence": {"type": "number", "description": "0-1: how sure you are the transcription is exact"},
        "special_marks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Visible marks such as 'signed', 'first edition', 'limited edition', leather, gilt; empty if none",
        },
    },
    "required": ["verbatim", "title", "author", "publisher", "legible", "confidence", "special_marks"],
}
READ_PROMPT = (
    "This is a single book spine photographed on a shelf (it may be rotated). Transcribe exactly what is printed. "
    "Split it into title, author and publisher only using what is printed on this spine. Do not complete partial "
    "words from memory and do not name a book you cannot actually read: if the text is unreadable, return empty "
    "fields, legible=false and confidence 0."
)
# Matching thresholds (0-100). A spine is "identified" only above these.
TITLE_MIN = 88
AUTHOR_MIN = 75
READ_CONF_MIN = 0.5
UA = f"library-claim-agent/0.1 ({os.getenv('CONTACT_EMAIL') or 'contact unset'})"
def norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    s = re.sub(r"\b(the|a|an|novel|a novel)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()
def crop_spine(run_dir: Path, obs: dict, pad: float = 0.12) -> bytes:
    img = cv2.imread(str(run_dir / obs["file"]))
    x0, y0, x1, y1 = obs["box_px"]
    w, h = x1 - x0, y1 - y0
    xa, ya = max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h))
    xb, yb = min(img.shape[1], int(x1 + pad * w)), min(img.shape[0], int(y1 + pad * h))
    crop = img[ya:yb, xa:xb].copy()
    # Grey out the neighbours so the reader sees one book only.
    keep = np.zeros(crop.shape[:2], bool)
    keep[max(0, int(y0) - ya):int(y1) - ya, max(0, int(x0) - xa):int(x1) - xa] = True
    crop[~keep] = (crop[~keep] * 0.2 + 100).astype(np.uint8)
    # Upscale tiny crops so small spine type is readable.
    if min(crop.shape[:2]) < 120:
        f = 120 / max(1, min(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
async def _get(client: httpx.AsyncClient, url: str, params: dict, metrics: Metrics) -> dict | None:
    return await get_json(client, url, params, metrics, IDENTIFY_STAGE)
def _isbn13(ids: list[dict]) -> str:
    for t in ("ISBN_13", "ISBN_10"):
        for i in ids or []:
            if i.get("type") == t:
                return i["identifier"]
    return ""
async def google_books(client, title: str, author: str, metrics: Metrics) -> list[dict]:
    # Plain query: field operators (intitle:/inauthor:) returned 0 results from this API in testing (2026-10-15).
    # Candidates are verified by fuzzy title/author scoring in decide(), not trusted from search ranking.
    q = f"{title} {author}".strip()
    params = {"q": q, "maxResults": 8, "printType": "books"}
    if os.getenv("GOOGLE_BOOKS_API_KEY"):
        params["key"] = os.getenv("GOOGLE_BOOKS_API_KEY")
    data = await _get(client, "https://www.googleapis.com/books/v1/volumes", params, metrics)
    out = []
    for item in (data or {}).get("items", []):
        v = item.get("volumeInfo", {})
        out.append(
            {
                "source": "Google Books",
                "volume_id": item.get("id"),
                "title": v.get("title", "") + (f": {v['subtitle']}" if v.get("subtitle") else ""),
                "authors": v.get("authors", []),
                "publisher": v.get("publisher", ""),
                "published": v.get("publishedDate", ""),
                "isbn": _isbn13(v.get("industryIdentifiers")),
                "pages": v.get("pageCount"),
                "url": v.get("infoLink") or f"https://books.google.com/books?id={item.get('id')}",
            }
        )
    return out
async def open_library(client, title: str, author: str, metrics: Metrics) -> list[dict]:
    params = {
        "title": title,
        "limit": 8,
        "fields": "key,title,author_name,isbn,publisher,first_publish_year,number_of_pages_median",
    }
    if author:
        params["author"] = author
    data = await _get(client, "https://openlibrary.org/search.json", params, metrics)
    out = []
    for d in (data or {}).get("docs", []):
        isbns = d.get("isbn") or []
        out.append(
            {
                "source": "Open Library",
                "title": d.get("title", ""),
                "authors": d.get("author_name", []),
                "publisher": (d.get("publisher") or [""])[0],
                # Search results are work-level: first_publish_year is when the WORK first appeared (1813 for any
                # Pride and Prejudice), not this copy's edition, so it must not drive the antiquarian/pre-ISBN rules.
                "published": "",
                "first_published": str(d.get("first_publish_year") or ""),
                "isbn": next((i for i in isbns if len(i) == 13), isbns[0] if isbns else ""),
                # A work-level record lists the ISBNs of many editions: kept for size and price lookups.
                "isbns": [i for i in isbns if len(i) == 13][:12],
                "pages": d.get("number_of_pages_median"),
                "url": f"https://openlibrary.org{d.get('key', '')}",
            }
        )
    return out
def surname_on_spine(read: dict, authors: list[str]) -> bool:
    """Spines often print a truncated or partial surname ('WOO' for Woolf, 'HARPE' for Harpe(r), 'ARUND' for
    Arundhati). True if any 3+ letter token on the spine starts a candidate author's name part, or vice versa."""
    tokens = {t for t in norm(read.get("author", "") + " " + read.get("verbatim", "")).split() if len(t) >= 3}
    for a in authors:
        for part in (p for p in norm(a).split() if len(p) >= 3):
            if any(part.startswith(t) or t.startswith(part) for t in tokens):
                return True
    return False
def _coverage(words: list[str], pool: list[str]) -> float:
    """Share of `words` (3+ letters) that appear, fuzzily, among `pool`."""
    words = [w for w in words if len(w) >= 3]
    if not words:
        return 1.0
    return sum(any(fuzz.ratio(w, p) >= 85 for p in pool) for w in words) / len(words)
def title_coverage(read: dict, cand: dict) -> tuple[float, float]:
    """(main-title words of the candidate found on the spine, title words read from the spine found in the candidate).
    token_set_ratio alone scores 100 when one shared word ('Indonesia') is a subset; both directions must be covered."""
    main = re.split(r"[:/;]| - ", cand["title"])[0]
    spine = norm(read.get("title", "") + " " + read.get("verbatim", "")).split()
    return _coverage(norm(main).split(), spine), _coverage(norm(read.get("title", "")).split(), norm(cand["title"]).split())
def score(read: dict, cand: dict) -> tuple[float, float]:
    rt = norm(read.get("title") or read.get("verbatim"))
    ct = norm(cand["title"])
    title = max(fuzz.token_set_ratio(rt, ct), fuzz.partial_ratio(ct, norm(read.get("verbatim", ""))) if len(ct) > 6 else 0)
    cand_cov, read_cov = title_coverage(read, cand)
    if cand_cov < 0.7 or read_cov < 0.5:
        title = min(title, 60.0)  # shares a word or two, not the title
    ra = norm(read.get("author", ""))
    author = max((fuzz.token_set_ratio(ra, norm(a)) for a in cand["authors"]), default=0) if ra else -1
    if surname_on_spine(read, cand["authors"]):
        author = max(author, 85.0)
    return float(title), float(author)
AUDIO_EDITION = re.compile(r"\b(audio|audible|brilliance|blackstone audio|tantor|recorded books|mp3|audiobook|audio cd)\b", re.I)
def decide(read: dict, cands: list[dict]) -> dict:
    """Pick the best candidate or return an unidentified verdict with the reason."""
    if not read or not read.get("legible") or read.get("confidence", 0) < READ_CONF_MIN:
        return {"status": "unidentified", "reason": "spine not legible"}
    scored = []
    for c in cands:
        t, a = score(read, c)
        ok = t >= TITLE_MIN and (a >= AUTHOR_MIN or (a < 0 and len(norm(read.get("title", "")).split()) >= 3 and t >= 95))
        # Prefer print: an audiobook record of the same title is the wrong product to price and has no spine size.
        print_pref = 0 if AUDIO_EDITION.search(f"{c.get('publisher', '')} {c.get('title', '')}") else 1
        scored.append((ok, print_pref, t + max(a, 0), t, a, c))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    scored = [(ok, tot, t, a, c) for ok, _, tot, t, a, c in scored]
    if not scored or not scored[0][0]:
        return {"status": "unidentified", "reason": "no catalogue match above threshold"}
    _, _, t, a, best = scored[0]
    if a < 0:  # no author on the spine: the title must point to one work only ('Playing to Win' is several books)
        works = {tuple(sorted(norm(x) for x in c["authors"])) for ok, _, tt, _, c in scored if ok and tt >= TITLE_MIN}
        if len(works) > 1:
            return {"status": "unidentified", "reason": "title fits several books and no author was read"}
    # Edition: only claim one when the printed publisher matches the candidate's publisher.
    rp = norm(read.get("publisher", ""))
    edition_match = next(
        (c for ok, _, _, _, c in scored if ok and rp and c["isbn"] and fuzz.token_set_ratio(rp, norm(c["publisher"])) >= 80),
        None,
    )
    chosen = edition_match or best
    conf = 0.5 * t / 100 + 0.3 * (a / 100 if a >= 0 else 0.6) + 0.2 * read.get("confidence", 0)
    # Other catalogue records that passed the same title/author check: other editions of the same work.
    # Used only to find dimensions and price listings when the chosen record has none.
    alternates = [c for ok, _, _, _, c in scored if ok and c is not chosen and c.get("isbn")][:6]
    return {
        "status": "identified",
        "match": chosen,
        "alternates": alternates,
        "title_score": t,
        "author_score": a,
        "edition_basis": "publisher on spine matches catalogue record" if edition_match else "representative edition; publisher not confirmed",
        "id_confidence": round(min(conf, 0.99), 3),
    }
def parse_dimensions(text: str) -> dict | None:
    """'24.2 x 16.2 x 3.9 centimeters' / '7.3 x 4.2 x 1.5 inches' -> {'height_cm','width_cm','thickness_cm'}."""
    if not text:
        return None
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return None
    unit = 2.54 if re.search(r"inch|\bin\b|\"", text.lower()) else (0.1 if "mm" in text.lower() else 1.0)
    vals = sorted((n * unit for n in nums if n > 0), reverse=True)
    if not vals:
        return None
    out = {"height_cm": round(vals[0], 2)}  # height is the largest dimension of a book
    if len(vals) >= 3:
        out["width_cm"], out["thickness_cm"] = round(vals[1], 2), round(vals[-1], 2)
    return out
async def _dims_for(client, rec: dict, metrics: Metrics) -> dict | None:
    # Google Books first: far higher rate limit than Open Library (~3 req/s).
    if rec.get("volume_id"):
        params = {"key": os.getenv("GOOGLE_BOOKS_API_KEY")} if os.getenv("GOOGLE_BOOKS_API_KEY") else {}
        data = await _get(client, f"https://www.googleapis.com/books/v1/volumes/{rec['volume_id']}", params, metrics)
        d = (data or {}).get("volumeInfo", {}).get("dimensions") or {}
        if d.get("height"):
            dims = {k + "_cm": parse_dimensions(d[k])["height_cm"] for k in ("height", "width", "thickness") if d.get(k)}
            if 10 <= dims["height_cm"] <= 50:
                return {**dims, "source": "Google Books", "url": rec["url"]}
    if rec.get("isbn"):
        data = await _get(client, f"https://openlibrary.org/isbn/{rec['isbn']}.json", {}, metrics)
        dims = parse_dimensions((data or {}).get("physical_dimensions", ""))
        if dims and 10 <= dims["height_cm"] <= 50:
            return {**dims, "source": "Open Library", "url": f"https://openlibrary.org/isbn/{rec['isbn']}"}
    return None
LOC_SRU = "https://lx2.loc.gov/sru/lcdb"
def parse_marc_300(xml: str) -> dict | None:
    """MARC 300 (physical description) of the first record: '$a xlii, 435 p. ; $c 20 cm.' -> height and pages.
    Cataloguers round height UP to the next whole centimetre, so 0.5 cm is taken off."""
    f = re.search(r'<datafield[^>]*tag="300"[^>]*>(.*?)</datafield>', xml, re.S)
    if not f:
        return None
    sub = dict(re.findall(r'<subfield code="(\w)">(.*?)</subfield>', f.group(1), re.S))
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", sub.get("c", ""))
    if not m:
        return None
    out = {"height_cm": round(float(m.group(1)) - 0.5, 1)}
    a = sub.get("a", "")
    roman = re.match(r"\s*([ivxlc]+)\s*,", a, re.I)
    pages = [int(n) for n in re.findall(r"(\d+)\s*(?:p\b|pages)", a)]
    if pages:
        out["pages"] = max(pages) + (_roman(roman.group(1)) if roman else 0)
    return out
def _roman(s: str) -> int:
    v = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
    s, total = s.lower(), 0
    for i, ch in enumerate(s):
        total += -v[ch] if i + 1 < len(s) and v[ch] < v[s[i + 1]] else v[ch]
    return total
async def loc_dimensions(client, isbn: str, metrics: Metrics) -> dict | None:
    """Library of Congress catalogue (free, no key): height from MARC 300 $c, which most library records carry
    even when Google Books and Open Library have no dimensions."""
    params = {"version": "1.1", "operation": "searchRetrieve", "query": f"bath.isbn={isbn}", "recordSchema": "marcxml",
              "maximumRecords": "1"}
    data = await get_json(client, LOC_SRU, params, metrics, IDENTIFY_STAGE, as_text=True, timeout=15)
    d = parse_marc_300((data or {}).get("text", ""))
    if d and 10 <= d["height_cm"] <= 50:
        return {**d, "source": "Library of Congress (MARC 300, rounded-up cm less 0.5)",
                "url": f"https://www.loc.gov/books/?q={isbn}"}
    return None
async def work_editions(client, match: dict, alternates: list[dict] | None, metrics: Metrics) -> list[str]:
    """ISBNs of this work's editions: the chosen record and alternates, plus one Open Library work search (its
    records list many editions' ISBNs) when fewer than 3 are known. Used for size and price lookups only."""
    isbns = edition_isbns(match, alternates)
    if len(isbns) < 3:
        title = re.split(r"[:/;]| - ", match.get("title", ""))[0].strip()
        surname = (norm((match.get("authors") or [""])[0]).split() or [""])[-1]  # 'A.G. Lafley' vs 'A. G. Lafley'
        for rec in await open_library(client, title, surname, metrics):
            if main_title(rec["title"]) == main_title(match["title"]):
                for i in [rec.get("isbn")] + rec.get("isbns", []):
                    if i and i not in isbns:
                        isbns.append(i)
    # Same-language editions first: ISBN registration group 978-0/978-1 is English, 978-3 German, 978-88 Italian...
    group = _isbn_group(match.get("isbn", ""))
    return sorted(isbns, key=lambda i: (i != match.get("isbn"), _isbn_group(i) != group))[:15]
def _isbn_group(isbn: str) -> str:
    if not isbn.startswith(("978", "979")) or len(isbn) != 13:
        return ""
    return "en" if isbn[3] in "01" else isbn[3:5]
async def dimensions(client, match: dict, metrics: Metrics, alternates: list[dict] | None = None,
                     isbns: list[str] | None = None) -> dict | None:
    """Catalogue physical dimensions: the chosen edition first, then other editions of the same work."""
    for k, rec in enumerate(([match] + (alternates or []))[:3]):
        dims = await _dims_for(client, rec, metrics)
        if dims:
            return {**dims, "isbn": rec.get("isbn", ""), "edition_basis": "chosen edition" if k == 0 else "another edition of the same work"}
    for isbn in (isbns or edition_isbns(match, alternates))[:8]:
        dims = await loc_dimensions(client, isbn, metrics)
        if dims:
            return {**dims, "isbn": isbn, "edition_basis": "chosen edition" if isbn == match.get("isbn") else "another edition of the same work"}
    return None
def edition_isbns(match: dict, alternates: list[dict] | None = None) -> list[str]:
    """The chosen edition's ISBN first, then other editions of the same work (alternate records and the edition
    lists of work-level records), without repeats."""
    out: list[str] = []
    for rec in [match] + (alternates or []):
        for i in [rec.get("isbn")] + list(rec.get("isbns") or []):
            if i and i not in out:
                out.append(i)
    return out
def main_title(t: str) -> str:
    return norm(re.split(r"[:/;]| - ", t or "")[0])
QUERY_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "author": {"type": "string"}},
    "required": ["title", "author"],
}
QUERY_PROMPT = (
    "Text read from a book spine (may contain OCR errors, truncated words and library call numbers such as "
    "'796.51 Gre' or 'NE Smi'). Return the book's title and author as they would be written in a catalogue, "
    "fixing only errors and truncations whose intended words are evident from the text itself. Leave a field "
    "empty if it is not on the spine. This is used only as a search query."
)
PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer", "description": "index of the matching record, or -1 if none matches"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["choice", "confidence", "reason"],
}
PICK_PROMPT = (
    "Which catalogue record is the book whose spine text is given? A record matches only if the spine's own words "
    "(title words, author name or fragment) identify that work; another edition, translation or volume of the same "
    "work counts. Do not choose a record because it is popular or plausible on a shelf. If none clearly matches, "
    "answer -1."
)
PICK_MIN_COVERAGE = 0.5  # even after the model picks, half the record's main-title words must be on the spine
async def _search(client, title: str, author: str, metrics: Metrics, open_library_too: bool = True) -> list[dict]:
    jobs = [google_books(client, title, author, metrics)]
    if open_library_too:
        jobs.append(open_library(client, title, author, metrics))
    return [c for part in await asyncio.gather(*jobs) for c in part]
def _dedupe(cands: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in cands:
        k = (norm(c["title"]), tuple(norm(a) for a in c["authors"][:1]), c.get("isbn"))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out
async def resolve(client, read: dict, metrics: Metrics) -> dict:
    """Spine read -> catalogue verdict."""
    if not read.get("legible") or read.get("confidence", 0) < READ_CONF_MIN or not (read.get("title") or read.get("verbatim")):
        return decide(read, [])
    title, author = read.get("title") or read.get("verbatim"), read.get("author", "")
    # Google Books first and alone: it allows far more requests per second than Open Library (~3/s).
    cands = await _search(client, title, author, metrics, open_library_too=False)
    v = decide(read, cands)
    if v["status"] == "identified" and v["id_confidence"] >= 0.85:
        return v | {"match_method": "fuzzy"}
    # Open Library with author, plus title-only searches in parallel (a misread author filters out the right record).
    extra = await asyncio.gather(
        open_library(client, title, author, metrics),
        *([google_books(client, read["title"], "", metrics), open_library(client, read["title"], "", metrics)] if author and read.get("title") else []),
    )
    cands = _dedupe(cands + [c for part in extra for c in part])
    v = decide(read, cands)
    if v["status"] == "identified":
        return v | {"match_method": "fuzzy"}
    # 3. small-model help for the hard cases
    q = await gemini_json(model=LITE_MODEL, parts=[QUERY_PROMPT, f"Spine text: {read.get('verbatim', '')}"], schema=QUERY_SCHEMA, metrics=metrics, stage=IDENTIFY_STAGE)
    if q and (q.get("title") or "").strip():
        cands = _dedupe(cands + await _search(client, q["title"], q.get("author", ""), metrics))
    v = decide(read, cands)
    if v["status"] == "identified":
        return v | {"match_method": "fuzzy after query rewrite", "query": q}
    ranked = sorted(cands, key=lambda c: -score(read, c)[0])[:8]
    if not ranked:
        return v | {"query": q}
    listing = "\n".join(f"{i}. {c['title']} | {', '.join(c['authors'][:2])} | {c.get('publisher', '')} {c.get('published', '')}" for i, c in enumerate(ranked))
    pick = await gemini_json(
        model=LITE_MODEL,
        parts=[PICK_PROMPT, f"Spine text: {read.get('verbatim', '')}\nRead as: title={read.get('title', '')!r}, author={read.get('author', '')!r}", "Records:\n" + listing],
        schema=PICK_SCHEMA, metrics=metrics, stage=IDENTIFY_STAGE,
    )
    i = (pick or {}).get("choice", -1)
    if not isinstance(i, int) or not 0 <= i < len(ranked):
        return v | {"query": q, "pick": pick}
    chosen = ranked[i]
    cand_cov, _ = title_coverage(read, chosen)
    if cand_cov < PICK_MIN_COVERAGE:
        return {"status": "unidentified", "reason": "model pick rejected: too few of its title words on the spine", "query": q, "pick": pick}
    conf = round(min(0.85, 0.85 * float(pick.get("confidence", 0))), 3)
    # Other editions: same main title ('Playing to Win' == 'Playing to Win: How Strategy Really Works') and author.
    first = norm((chosen.get("authors") or [""])[0]).split()[-1:] or [""]
    alternates = [c for c in ranked if c is not chosen and c.get("isbn") and main_title(c["title"]) == main_title(chosen["title"])
                  and (not first[0] or any(first[0] in norm(a) for a in c.get("authors", [])))][:6]
    return {
        "status": "identified", "match": chosen, "alternates": alternates, "title_score": score(read, chosen)[0],
        "author_score": score(read, chosen)[1], "edition_basis": "representative edition; publisher not confirmed",
        "id_confidence": conf, "match_method": "small-model pick among catalogue records", "query": q, "pick": pick,
    }
def sharpness(run_dir: Path, obs: dict) -> float:
    """Variance of the Laplacian over the spine crop (higher = sharper)."""
    img = cv2.imread(str(run_dir / obs["file"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    x0, y0, x1, y1 = (int(v) for v in obs["box_px"])
    crop = img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
def second_view(run_dir: Path, book: dict) -> dict | None:
    """The sharpest view of this spine from another frame, at least 60 % of the evidence view's size."""
    best = book["best"]
    area = lambda o: (o["box_px"][2] - o["box_px"][0]) * (o["box_px"][3] - o["box_px"][1])
    others = [o for o in book["observations"] if o.get("frame_i") != best.get("frame_i") and area(o) >= 0.6 * area(best)]
    others = sorted(others, key=area, reverse=True)[:6]  # bound the image reads
    return max(others, key=lambda o: sharpness(run_dir, o)) if others else None
async def _read_view(run_dir: Path, client, obs: dict, metrics: Metrics) -> dict:
    key = {"file": obs["file"], "box": [round(v) for v in obs["box_px"]]}
    return await memo(run_dir, "spine_reads", key, lambda: _read_and_resolve(run_dir, client, obs, metrics))
async def _one(run_dir: Path, client, book: dict, metrics: Metrics) -> tuple[str, dict]:
    """Read the spine in its evidence view and, when it was seen in another frame, in the sharpest other view, and combine (reads memoised per crop, so spines read..."""
    v1 = await _read_view(run_dir, client, book["best"], metrics)
    other = second_view(run_dir, book)
    if other is None:
        return book["id"], v1
    v2 = await _read_view(run_dir, client, other, metrics)
    k1, k2 = work_key(v1), work_key(v2)
    if k1 and k2 and k1 != k2 and main_title(v1["match"]["title"]) != main_title(v2["match"]["title"]):
        return book["id"], {
            "read": v1.get("read", {}), "status": "unidentified",
            "reason": f"two views of this spine resolve to different books ({v1['match']['title']!r} vs {v2['match']['title']!r})",
            "conflicting_reads": [v1, v2],
        }
    if k1 and k2:
        return book["id"], {**v1, "views_agree": 2, "id_confidence": round(min(0.99, v1.get("id_confidence", 0) + 0.1), 3)}
    if k2 and not k1:
        return book["id"], {**v2, "views_agree": 1, "evidence_view": {"file": other["file"], "box_px": other["box_px"]},
                            "edition_basis": v2.get("edition_basis", "") + "; identified from a second view"}
    return book["id"], {**v1, "views_agree": 1 if k1 else 0}
async def _read_and_resolve(run_dir: Path, client, obs: dict, metrics: Metrics) -> dict:
    read = await gemini_json(
        model=VISION_MODEL,
        parts=[image_part(crop_spine(run_dir, obs)), READ_PROMPT],
        schema=READ_SCHEMA,
        metrics=metrics,
        stage=IDENTIFY_STAGE,
    )
    read = read or {}
    verdict = await resolve(client, read, metrics)
    if verdict["status"] == "identified":
        verdict["edition_isbns"] = await work_editions(client, verdict["match"], verdict["alternates"], metrics)
        verdict["dimensions"] = await dimensions(client, verdict["match"], metrics, verdict["alternates"], verdict["edition_isbns"])
    return {"read": read, **verdict}
def work_key(idn: dict) -> str | None:
    """Same book = same main title + first author (the ISBN is often a representative edition, so views disagree on it)."""
    if idn.get("status") != "identified":
        return None
    m = idn["match"]
    return "work:" + norm(re.split(r"[:/;]", m["title"])[0]) + "|" + norm((m.get("authors") or [""])[0])
SAME_SPINE_TEXT = 80
def _touching(a: dict, b: dict, frame_i: int) -> bool:
    pa = next(o["box_px"] for o in a["observations"] if o["frame_i"] == frame_i)
    pb = next(o["box_px"] for o in b["observations"] if o["frame_i"] == frame_i)
    gap = max(pa[0], pb[0]) - min(pa[2], pb[2])
    return gap <= 0.3 * min(pa[2] - pa[0], pb[2] - pb[0])


def merge_same_book(books: list[dict], identified: dict) -> tuple[list[dict], dict]:
    """Second guard against counting one physical book twice, after frame tracking."""
    by_key: dict[str, list[dict]] = {}
    for b in books:
        k = work_key(identified.get(b["id"], {}))
        if k:
            by_key.setdefault(k, []).append(b)
    gone: set[str] = set()
    for k, group in by_key.items():
        group.sort(key=lambda b: b["seg_box"][0])
        for i, a in enumerate(group):
            if a["id"] in gone:
                continue
            for b in group[i + 1 :]:
                if b["id"] in gone:
                    continue
                shared = {o["frame_i"] for o in a["observations"]} & {o["frame_i"] for o in b["observations"]}
                if shared and not _touching(a, b, min(shared)):
                    continue  # apart in one frame: two copies (touching = the spine and the angled cover of one book)
                if a.get("segment") == b.get("segment"):
                    wa, wb = a["seg_box"][2] - a["seg_box"][0], b["seg_box"][2] - b["seg_box"][0]
                    ca, cb = (a["seg_box"][0] + a["seg_box"][2]) / 2, (b["seg_box"][0] + b["seg_box"][2]) / 2
                    if a["shelf"] != b["shelf"] or abs(ca - cb) > 2 * max(wa, wb, 1.0):
                        continue
                # else: a free tour films one shelf from several unlinked angles, so the same work seen apart is one copy
                ra, rb = (norm((identified[x["id"]].get("read") or {}).get("verbatim", "")) for x in (a, b))
                if ra and rb and fuzz.token_set_ratio(ra, rb) < SAME_SPINE_TEXT:
                    continue  # different spine wording ('50 itineraris' vs '51 passejades'): one of them is misidentified
                keep, drop = (a, b) if identified[a["id"]].get("id_confidence", 0) >= identified[b["id"]].get("id_confidence", 0) else (b, a)
                keep["observations"] = keep["observations"] + drop["observations"]
                keep.setdefault("review_notes", []).append(f"merged sighting {drop['id']} of the same book: say so if there are two copies")
                gone.add(drop["id"])
                if drop is a:
                    break
        left = [b for b in group if b["id"] not in gone]
        if len(left) > 1:
            for b in left:
                others = ", ".join(o["id"] for o in left if o is not b)
                b.setdefault("review_notes", []).append(f"possible duplicate copy of {others} (same book seen elsewhere): confirm it is a second copy")
    return [b for b in books if b["id"] not in gone], {k: v for k, v in identified.items() if k not in gone}
async def identify_all(run_dir: Path, books: list[dict], metrics: Metrics, tick=None, deadline: float | None = None) -> dict[str, dict]:
    """deadline (time.perf_counter): books still being looked up then are left unidentified, so the packet is on time."""
    late = lambda i: (books[i]["id"], {"read": {}, "status": "unidentified", "reason": "not looked up before the packet deadline"})  # noqa: E731
    with metrics.time(IDENTIFY_STAGE):
        async with httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True) as client:
            async def one(b):
                r = await _one(run_dir, client, b, metrics)
                if tick:
                    tick()
                return r

            results = await gather_until([one(b) for b in books], deadline, late)
    out = dict(results)
    write_json(run_dir / "identified.json", out)
    return out


SCALE_STAGE = "scale"
OUTLIER_TOL = 0.08
ASPECT_TOL = 0.4
# Common book heights in cm: A-format, B-format, US trade, Demy, Royal, US hardcover, large hardcover.
STANDARD_HEIGHTS_CM = [17.8, 19.8, 21.0, 21.6, 23.4, 24.0, 25.4, 28.0]
# Thickness from page count: paper caliper ~0.0057 cm/page (white) to ~0.0064 (cream), Amazon KDP spine formula;
# real stock varies (300-530 pages per inch), so this estimate is +-20-25 % on its own. Plus covers.
CM_PER_PAGE = 0.0060
COVER_CM = 0.1
PAGE_REL_SD = 0.22
PIXEL_REL_SD = 0.12  # ~+-3 px on a ~50 px spine; thin spines are worse (see _pixel_sd)
THICK_DISAGREE = 0.20
OTHER_EDITION_TOL = 0.12
def spine_px(box: list[float]) -> tuple[float, float]:
    """(long side, short side) in pixels: a spine's height and thickness whatever its orientation."""
    w, h = box[2] - box[0], box[3] - box[1]
    return max(w, h), min(w, h)
def robust_median(values: list[float], tol: float = OUTLIER_TOL) -> tuple[float | None, int]:
    if not values:
        return None, 0
    m = float(np.median(values))
    kept = [v for v in values if abs(v / m - 1) <= tol]
    return (float(np.median(kept)), len(kept)) if kept else (m, len(values))
def format_prior_scale(px_heights: list[float]) -> float | None:
    """Scale (cm/px) that best snaps observed spine heights to standard formats."""
    if len(px_heights) < 3:
        return None
    med = float(np.median(px_heights))
    best, best_cost = None, float("inf")
    for s in np.linspace(14.0 / med, 30.0 / med, 600):
        cm = np.array(px_heights) * s
        cost = float(np.median([min(abs(c - f) / f for f in STANDARD_HEIGHTS_CM) for c in cm]))
        if cost < best_cost:
            best, best_cost = float(s), cost
    return best
def ruler_books(books: list[dict], identified: dict[str, dict], other_editions: bool = True) -> dict[str, float]:
    """book_id -> catalogue height (cm) for identified books whose spine shape matches the catalogue edition."""
    out = {}
    for b in books:
        idn = identified.get(b["id"], {})
        dims = idn.get("dimensions") if idn.get("status") == "identified" else None
        if not dims or not dims.get("height_cm"):
            continue
        if not other_editions and "another edition" in dims.get("edition_basis", ""):
            continue
        if dims.get("thickness_cm"):
            long_px, short_px = spine_px(b["best"]["box_px"])
            expected = dims["thickness_cm"] / dims["height_cm"]
            observed = short_px / long_px if long_px else 0
            if not observed or max(observed / expected, expected / observed) - 1 > ASPECT_TOL:
                continue  # probably a different edition (e.g. hardback vs paperback): not a reliable ruler
        out[b["id"]] = float(dims["height_cm"])
    return out
def frame_scales(books: list[dict], rulers: dict[str, float]) -> dict[int, list[tuple[str, float]]]:
    """frame_i -> [(ruler book id, cm/px from that book)]."""
    per_frame: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for b in books:
        if b["id"] not in rulers:
            continue
        for o in b["observations"]:
            long_px, _ = spine_px(o["box_px"])
            if long_px > 0:
                per_frame[o["frame_i"]].append((b["id"], rulers[b["id"]] / long_px))
    return per_frame
def _pixel_sd(thick_px: float) -> float:
    """Relative error of a pixel thickness: about 3 px of box error, never better than PIXEL_REL_SD."""
    return max(PIXEL_REL_SD, 3.0 / thick_px) if thick_px > 0 else 1.0
def thickness(height_cm: float, ratio: float | None, thick_px: float, pages: int | None, catalogue: float | None) -> tuple[float | None, str, str | None]:
    """(thickness cm, basis, review note). Catalogue thickness first; otherwise the scale-free pixel ratio
    (short/long side x height) fused with the page-count estimate by inverse variance."""
    if catalogue and 0.2 <= catalogue <= 15:
        return round(catalogue, 2), "catalogue", None
    est = []
    if ratio:
        est.append((height_cm * ratio, _pixel_sd(thick_px), "pixel ratio x height"))
    if pages and 20 <= pages <= 3000:
        est.append((pages * CM_PER_PAGE + COVER_CM, PAGE_REL_SD, f"{pages} pages x paper caliper"))
    if not est:
        return None, "none", None
    if len(est) == 1:
        return round(est[0][0], 2), est[0][2], None
    (a, sa, la), (b, sb, lb) = est
    wa, wb = 1 / sa**2, 1 / sb**2
    fused = (a * wa + b * wb) / (wa + wb)
    note = None
    if abs(a / b - 1) > THICK_DISAGREE:
        note = f"spine thickness sources disagree: {la} {a:.1f} cm vs {lb} {b:.1f} cm (reported {fused:.1f} cm, weighted)"
    return round(fused, 2), f"{la} + {lb} (weighted)", note
def measure(books: list[dict], identified: dict[str, dict]) -> dict:
    rulers = ruler_books(books, identified)
    # Only a book's own edition sets the scale for its neighbours; another edition's height may be a different format.
    per_frame = frame_scales(books, ruler_books(books, identified, other_editions=False))
    px_heights_by_frame: dict[int, list[float]] = defaultdict(list)
    for b in books:
        for o in b["observations"]:
            px_heights_by_frame[o["frame_i"]].append(spine_px(o["box_px"])[0])

    results: dict[str, dict] = {}
    for b in books:
        obs = []  # per observation: long px, short px, cm/px (or None), method, rulers, tight box
        for o in b["observations"]:
            long_px, short_px = spine_px(o["box_px"])
            if long_px <= 0:
                continue
            others = [s for bid, s in per_frame.get(o["frame_i"], []) if bid != b["id"]]
            scale, n = robust_median(others) if len(others) >= 2 else (None, 0)
            method = "book_ruler"
            if scale is None:
                scale, method = format_prior_scale(px_heights_by_frame[o["frame_i"]]), "standard_format_prior"
            obs.append({"long": long_px, "short": short_px, "scale": scale, "method": method if scale else None, "n": n,
                        "tight": str(o.get("box_source", "")).startswith("rfdetr")})
        idn = identified.get(b["id"], {})
        ok = idn.get("status") == "identified"
        dims = (idn.get("dimensions") or {}) if ok else {}
        pages = (dims.get("pages") or (idn.get("match") or {}).get("pages")) if ok else None
        if not obs or (b["id"] not in rulers and not any(x["scale"] for x in obs)):
            results[b["id"]] = {"spine_height_cm": None, "spine_thickness_cm": None, "scale_method": "none", "frames_used": 0}
            continue
        scaled = [x for x in obs if x["scale"]]
        use = [x for x in scaled if x["method"] == "book_ruler"] or scaled or obs  # prefer frames with the book ruler
        if any(x["tight"] for x in use):
            use = [x for x in use if x["tight"]]  # prefer tight detector boxes over loose model boxes
        px_h = float(np.median([x["long"] * x["scale"] for x in use])) if all(x["scale"] for x in use) else None
        px_t = float(np.median([x["short"] * x["scale"] for x in use])) if all(x["scale"] for x in use) else None
        # The book's own catalogue height when it fits what is seen (ruler_books' aspect check); the pixel
        # measurement is kept alongside for checking.
        other_edition = "another edition" in dims.get("edition_basis", "")
        if b["id"] in rulers and other_edition and (not px_h or abs(rulers[b["id"]] / px_h - 1) > OTHER_EDITION_TOL):
            # Another edition's size (hardback 24 cm vs paperback 19.8 cm) is used only when the pixels agree.
            h, method = px_h, ("book_ruler" if use[0]["method"] == "book_ruler" else "standard_format_prior")
        elif b["id"] in rulers:
            h, method = rulers[b["id"]], f"catalogue height ({dims.get('source', 'catalogue')}; {dims.get('edition_basis', '')})"
        else:
            h, method = px_h, ("book_ruler" if use[0]["method"] == "book_ruler" else "standard_format_prior")
        ratio = float(np.median([x["short"] / x["long"] for x in use]))  # scale-free
        thick_px = float(np.median([x["short"] for x in use]))
        t, t_basis, t_note = thickness(h, ratio, thick_px, pages, dims.get("thickness_cm") if b["id"] in rulers else None)
        results[b["id"]] = {
            "spine_height_cm": round(float(h), 1),
            "spine_thickness_cm": t,
            "scale_method": method,
            "thickness_basis": t_basis,
            "pixel_height_cm": round(px_h, 1) if px_h else None,
            "pixel_thickness_cm": round(px_t, 2) if px_t else None,
            "rulers_in_frame": int(max(x["n"] for x in use)),
            "frames_used": len(use),
            "box_basis": "tight (RF-DETR)" if any(x["tight"] for x in use) else "loose (Gemini)",
            **({"review_note": t_note} if t_note else {}),
        }
    return {"books": results, "ruler_count": len(rulers), "shelves": shelf_runs(books, results)}
def shelf_runs(books: list[dict], dims: dict[str, dict]) -> dict[str, float]:
    """Occupied shelf run per shelf (m): sum of upright thicknesses; a flat stack occupies its longest book."""
    runs: dict[str, float] = defaultdict(float)
    by_shelf: dict[str, list[dict]] = defaultdict(list)
    for b in books:
        by_shelf[b["shelf"]].append(b)
    for shelf, bs in by_shelf.items():
        stack: list[float] = []
        for b in sorted(bs, key=lambda b: b["position"]):
            d = dims.get(b["id"], {})
            if b["best"].get("orientation") == "horizontal":
                if d.get("spine_height_cm"):
                    stack.append(d["spine_height_cm"])
                continue
            if stack:
                runs[shelf] += max(stack)
                stack = []
            runs[shelf] += d.get("spine_thickness_cm") or 0.0
        if stack:
            runs[shelf] += max(stack)
    return {k: round(v / 100, 3) for k, v in runs.items()}
def measure_all(run_dir: Path, books: list[dict], identified: dict[str, dict], metrics) -> dict:
    with metrics.time(SCALE_STAGE):
        out = measure(books, identified)
    write_json(run_dir / "measurements.json", out)
    return out
