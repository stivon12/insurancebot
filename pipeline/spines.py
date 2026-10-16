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
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .common import VISION_MODEL, Metrics, gemini_json, image_part, read_json, write_json, memo

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
    """Whether a merged spine counts as a book in the packet.

    Identified or legible spines count. A spine with no legible text counts only with independent support: seen in
    at least 2 keyframes, or matched by the open spine detector (RF-DETR) box. A single blurry sighting of an
    unreadable 'spine' was, in a real sweep, a box, a router or cables 8 times out of 8. Spines the model itself
    marked 'unsure' need the same support even when legible."""
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
        model=VISION_MODEL, parts=[image_part(_jpeg(img)), SPINES_PROMPT], schema=SPINES_SCHEMA, metrics=metrics, stage=STAGE
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
        # (a 3-book stack read as 1 and 2: counted 1).
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
    """One physical spine can be listed from two neighbouring row crops of the same frame. Boxes FROM DIFFERENT
    ROWS that overlap vertically and share most of their width are the same book; keep the taller box.
    Never compare within one row's list: model boxes on dense spines are loose (one box can span several
    books), so a box-overlap test inside a row deletes real neighbours (measured: 23 of 43 on one shelf)."""
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
        model=VISION_MODEL, parts=[image_part(_jpeg(img)), ROWS_PROMPT], schema=ROWS_SCHEMA, metrics=metrics, stage=STAGE
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


# ---------- cross-frame merge ----------

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
    """frame_i -> (segment id, 3x3 matrix mapping full-res frame px into segment px).

    Each keyframe is matched to the previous one. When that fails (fast pan, blur, or the camera came back to a
    shelf it already covered), it is matched against a few anchor frames of every earlier segment before a new
    segment is started; otherwise returning to a shelf would count all of its books again."""
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


def frames_for(keyframes: dict) -> list[dict]:
    kfs = keyframes.get("shelves", [])
    if not kfs:
        # The live agent did not tag a shelves phase: fall back to every non-room keyframe rather than count nothing.
        kfs = sorted([f for p, fs in keyframes.items() if p not in ("room_turn", "room_tilt") for f in fs], key=lambda f: f["t_ms"])
    return kfs


async def run(run_dir: Path, keyframes: dict, metrics: Metrics, tick=None) -> list[dict]:
    kfs = frames_for(keyframes)
    with metrics.time(STAGE):
        async def one(kf):
            r = await _frame(run_dir, kf, metrics)
            if tick:
                tick()
            return r

        per_frame = await asyncio.gather(*(one(kf) for kf in kfs))
        observations = [o for pf in per_frame for o in pf["observations"]]
        segs = chain_segments(run_dir, kfs)
        books = assign_shelves(merge(observations, segs))
    row_checks = [{"frame_i": pf["frame"]["i"], "rows": pf["rows"]} for pf in per_frame]
    out = {"books": books, "row_checks": row_checks, "keyframes_used": len(kfs), "observations": len(observations)}
    write_json(run_dir / "spines.json", out)
    return books
