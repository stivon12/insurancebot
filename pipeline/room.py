"""Stage 5: room dimensions, floor/wall area and shelved wall area from the same sweep.

Metric scale chain (no markers, no LiDAR):
  1. Focal length f: measured during the 360-degree turn as image shift / tan(gyro yaw change).
  2. Bookcase overview frames: identified books (catalogue heights) mapped in by ORB homography
     from the close-up shelf frames give the horizontal distance D to the bookcase face; the
     ray to the bookcase base then gives the camera height h. The same D gives each bookcase's
     width and height (shelved wall area).
  3. Room turn: Gemini marks where floor meets wall (and wall meets ceiling) across each frame;
     the marks are refined to the strongest image edge nearby. Each floor-boundary pixel is a
     ray; with h it lands on the floor at a metric point. The points around the turn give the
     room outline (Manhattan polygon fit, any number of corners).
  4. Ceiling height = h + wall distance * tan(elevation of the wall-ceiling line).

Output: runs/<id>/room.json
"""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from .common import LITE_MODEL, VISION_MODEL, Metrics, gemini_json, image_part, memo_path, memo, read_json, read_jsonl, write_json
from .scale import ruler_books
from .spines import _gray, homography

STAGE = "room"
SQFT_PER_M2 = 10.7639
CAM_HEIGHT_RANGE = (0.7, 2.1)
DEFAULT_HFOV_DEG = 69.0  # typical phone main camera, used only if the focal length cannot be measured

BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "floor": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer", "description": "0-1000; -1 if not visible"},
                    "at": {"type": "string", "enum": ["wall", "furniture", "door", "none"]},
                },
                "required": ["x", "y", "at"],
            },
        },
        "ceiling": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer", "description": "0-1000; -1 if not visible"}},
                "required": ["x", "y"],
            },
        },
    },
    "required": ["floor", "ceiling"],
}

BOUNDARY_PROMPT = (
    "Photo taken from inside a room. For each x in 0, 50, 100, ..., 1000 (fraction of image width x1000) give: "
    "(a) floor: the y (0-1000 from the top) where the floor meets the wall; if furniture stands in front of the wall "
    "there, give where the floor meets the bottom of that furniture and set at='furniture'; at='door' for an open "
    "doorway; y=-1 and at='none' if the floor is not visible at that x. "
    "(b) ceiling: the y where the wall meets the ceiling, or -1 if not visible. Be precise; do not guess hidden lines."
)

CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "bookcases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {"type": "array", "items": {"type": "integer"}, "description": "[ymin,xmin,ymax,xmax] 0-1000"},
                    "base_visible": {"type": "boolean", "description": "Is the bottom edge where it meets the floor visible?"},
                },
                "required": ["box_2d", "base_visible"],
            },
        }
    },
    "required": ["bookcases"],
}

CASE_PROMPT = (
    "Give a tight box_2d for every bookcase/shelving unit in this photo, from its top to where it meets the floor. "
    "Set base_visible=false if the bottom is cut off by the image edge."
)


def _jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


class Poses:
    """Per-frame camera model from the IMU log and a focal length."""

    def __init__(self, imu: list[dict]):
        self.imu = imu
        self.t, self.yaw = G.integrate_yaw(imu)

    def azimuth(self, t_ms: float) -> float:
        return float(np.interp(t_ms, self.t, self.yaw))

    def camera(self, frame: dict, f: float) -> G.Camera | None:
        up = G.up_at(self.imu, frame["t_ms"])
        if up is None:
            return None
        return G.Camera(
            f=f, cx=frame["w"] / 2, cy=frame["h"] / 2, up=up, azimuth=self.azimuth(frame["t_ms"]), orient=frame.get("orient", 0)
        )


def estimate_focal(run_dir: Path, frames: list[dict], poses: Poses) -> tuple[float, str]:
    """f = horizontal image shift of the frame centre / tan(gyro yaw change), median over frame pairs."""
    est = []
    for a, b in zip(frames, frames[1:]):
        d_yaw = poses.azimuth(b["t_ms"]) - poses.azimuth(a["t_ms"])
        if not math.radians(3) < abs(d_yaw) < math.radians(30):
            continue
        ga, sa = _gray(run_dir, a["file"])
        gb, sb = _gray(run_dir, b["file"])
        H = homography(ga, gb, min_inliers=30)
        if H is None:
            continue
        c = np.array([[[b["w"] * sb / 2, b["h"] * sb / 2]]], dtype=np.float32)
        x_in_a = cv2.perspectiveTransform(c, H)[0, 0, 0] / sa
        dx = a["w"] / 2 - x_in_a  # turning right moves the old centre left
        f = dx / math.tan(d_yaw)
        if f > 0:
            est.append(f)
    if len(est) >= 3:
        return float(np.median(est)), f"measured: image shift / gyro yaw over {len(est)} frame pairs"
    w = frames[0]["w"] if frames else 1080
    h = frames[0]["h"] if frames else 1920
    f = max(w, h) / 2 / math.tan(math.radians(DEFAULT_HFOV_DEG) / 2)
    return f, f"ASSUMED: {DEFAULT_HFOV_DEG} deg field of view on the long image side (focal length not measurable)"


def refine_y(gray: np.ndarray, x: float, y: float, search_frac: float = 0.03) -> float:
    """Snap a model-marked boundary to the strongest horizontal edge within +-3% of image height."""
    h, w = gray.shape
    xi = int(min(max(x, 4), w - 5))
    r = int(search_frac * h)
    y0, y1 = int(max(1, y - r)), int(min(h - 2, y + r))
    if y1 <= y0:
        return y
    strip = cv2.GaussianBlur(gray[y0 - 1 : y1 + 2, xi - 4 : xi + 5].astype(np.float32), (3, 3), 0)
    grad = np.abs(cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)[1:-1]
    if grad.size == 0 or grad.max() < 8:
        return y  # no clear edge: keep the model's mark
    return float(y0 + int(np.argmax(grad)))


BOUNDARY_VIEWS = 12  # floor/ceiling lines feed only the gyro cross-check: 12 evenly spaced views, not every keyframe
# Used only when no identified book gives a catalogue height (no book ruler): a phone held at chest height by a
# standing adult. Rough (about +-15 %), so the room is then reported with low confidence and a review note.
# Phone held upright at chest height when no book ruler gives the real height. Area scales with height squared, so
# the prior depends on how the policyholder holds the phone (welcome screen), and its error band is stated wide.
CAMERA_HEIGHT_PRIOR_M = {
    "standing": float(os.getenv("CAMERA_HEIGHT_PRIOR_M", "1.40")),
    "seated": float(os.getenv("CAMERA_HEIGHT_PRIOR_SEATED_M", "1.05")),
}
PRIOR_ERROR = "about +-20 % on lengths, +-40 % on areas"


def height_prior(run_dir: Path) -> tuple[float, str]:
    posture = (read_json(Path(run_dir) / "sweep.json", None) or {}).get("posture", "unknown")
    if posture == "seated":
        return CAMERA_HEIGHT_PRIOR_M["seated"], "seated"
    return CAMERA_HEIGHT_PRIOR_M["standing"], "standing" if posture == "standing" else "posture not given; standing assumed"


def boundary_views(turn: list[dict], tilt: list[dict]) -> list[dict]:
    def spread(fs, n):
        if len(fs) <= n:
            return fs
        return [fs[int(round(i * (len(fs) - 1) / (n - 1)))] for i in range(n)]

    return spread(turn, BOUNDARY_VIEWS * 2 // 3) + spread(tilt, BOUNDARY_VIEWS - BOUNDARY_VIEWS * 2 // 3)


async def _boundaries(run_dir: Path, frame: dict, metrics: Metrics) -> dict:
    """Floor and ceiling lines in one room-turn frame. Memoised per frame (marked during the sweep when possible)."""

    async def ask():
        img = cv2.imread(str(run_dir / frame["file"]))
        return await gemini_json(
            model=LITE_MODEL, parts=[image_part(_jpeg(img)), BOUNDARY_PROMPT], schema=BOUNDARY_SCHEMA, metrics=metrics, stage=STAGE
        )

    res = await memo(run_dir, "room_boundaries", {"file": frame["file"], "model": LITE_MODEL}, ask)
    img = cv2.imread(str(run_dir / frame["file"]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    out = {"floor": [], "ceiling": []}
    for key in ("floor", "ceiling"):
        for s in (res or {}).get(key, []):
            if s.get("y", -1) < 0 or s.get("at") == "none":
                continue
            x, y = s["x"] / 1000 * w, s["y"] / 1000 * h
            out[key].append({"x": x, "y": refine_y(gray, x, y), "at": s.get("at", "wall")})
    return out


def _g(cam: G.Camera, x: float, y: float) -> tuple[float, float]:
    """(vertical, lateral) offset per metre of forward distance for the ray through (x, y)."""
    r = cam.ray(x, y)
    fwd, right = cam._horizontal_basis()
    n = float(r.dot(fwd))
    return float(r.dot(cam.up)) / n, float(r.dot(right)) / n


async def _overview(run_dir: Path, frame: dict, cam: G.Camera, shelf_frames: list[dict], books: list[dict], rulers: dict, metrics: Metrics) -> dict | None:
    """Distance to the bookcase face, camera height, and bookcase width/height from one overview frame."""
    go, so = _gray(run_dir, frame["file"], width=1280)
    obs_by_frame: dict[int, list[tuple[str, list[float]]]] = {}
    for b in books:
        if b["id"] in rulers:
            for o in b["observations"]:
                obs_by_frame.setdefault(o["frame_i"], []).append((b["id"], o["box_px"]))
    distances = []
    for sf in shelf_frames:
        if sf["i"] not in obs_by_frame:
            continue
        gs, ss = _gray(run_dir, sf["file"], width=1280)
        H = homography(go, gs, min_inliers=20)
        if H is None:
            continue
        M = np.diag([1 / so, 1 / so, 1.0]) @ H @ np.diag([ss, ss, 1.0])
        for bid, box in obs_by_frame[sf["i"]]:
            cx = (box[0] + box[2]) / 2
            pts = np.float32([[cx, box[1]], [cx, box[3]]]).reshape(-1, 1, 2)
            (xt, yt), (xb, yb) = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            span = _g(cam, xt, yt)[0] - _g(cam, xb, yb)[0]
            if span > 1e-4:
                distances.append(rulers[bid] / 100 / span)
    if len(distances) < 2:
        return None
    D = float(np.median(distances))
    img = cv2.imread(str(run_dir / frame["file"]))
    res = await gemini_json(
        model=VISION_MODEL, parts=[image_part(_jpeg(img)), CASE_PROMPT], schema=CASE_SCHEMA, metrics=metrics, stage=STAGE
    )
    h_img, w_img = img.shape[:2]
    cases = [c for c in (res or {}).get("bookcases", []) if len(c.get("box_2d", [])) == 4]
    if not cases:
        return {"distance_m": D, "n_rulers": len(distances)}
    # The bookcase nearest the image centre is the one the books were mapped onto.
    case = min(cases, key=lambda c: abs((c["box_2d"][1] + c["box_2d"][3]) / 2 - 500))
    ymin, xmin, ymax, xmax = case["box_2d"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xc = (xmin + xmax) / 2000 * w_img
    y_base = refine_y(gray, xc, ymax / 1000 * h_img)
    y_top = ymin / 1000 * h_img
    v_base, _ = _g(cam, xc, y_base)
    v_top, _ = _g(cam, xc, y_top)
    _, l0 = _g(cam, xmin / 1000 * w_img, y_base)
    _, l1 = _g(cam, xmax / 1000 * w_img, y_base)
    return {
        "distance_m": round(D, 3),
        "n_rulers": len(distances),
        "camera_height_m": round(-D * v_base, 3) if case["base_visible"] else None,
        "case_height_m": round(D * (v_top - v_base), 3) if case["base_visible"] else None,
        "case_width_m": round(D * (l1 - l0), 3),
        "azimuth_deg": round(math.degrees(cam.azimuth) % 360, 1),
        "frame_ref": frame["file"],
    }


def _unique_cases(overviews: list[dict]) -> list[dict]:
    """One bookcase per ~25 degrees of heading (several overview frames can show the same unit)."""
    out: list[dict] = []
    for o in sorted(overviews, key=lambda o: o.get("n_rulers", 0), reverse=True):
        if o.get("case_height_m") is None:
            continue
        if all(abs((o["azimuth_deg"] - u["azimuth_deg"] + 180) % 360 - 180) > 25 for u in out):
            out.append(o)
    return out


def _cast(poly: np.ndarray, az: float) -> float | None:
    """Horizontal distance from the camera (origin) to the polygon along azimuth az."""
    d = np.array([math.sin(az), math.cos(az)])
    best = None
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        m = np.array([[d[0], -e[0]], [d[1], -e[1]]])
        if abs(np.linalg.det(m)) < 1e-12:
            continue
        t, u = np.linalg.solve(m, a)
        if t > 0 and 0 <= u <= 1 and (best is None or t < best):
            best = float(t)
    return best


def room_from_points(floor_pts: list[tuple[float, np.ndarray, str]]) -> np.ndarray | None:
    """Fit the outline from floor points (azimuth, xy, kind). Prefer wall contact over furniture per 5-degree bin."""
    if len(floor_pts) < 20:
        return None
    bins: dict[int, list[tuple[float, np.ndarray, str]]] = {}
    for az, p, kind in floor_pts:
        bins.setdefault(int((math.degrees(az) % 360) // 5), []).append((az, p, kind))
    kept = []
    for b, pts in bins.items():
        walls = [x for x in pts if x[2] == "wall"]
        use = walls or pts
        # furthest robust point in the bin: furniture and clutter only ever bring the boundary closer
        dists = np.array([np.linalg.norm(x[1]) for x in use])
        k = int(np.argsort(dists)[int(0.7 * (len(dists) - 1))])
        kept.append(use[k])
    kept.sort(key=lambda x: x[0] % (2 * math.pi))
    return G.fit_room_polygon(np.array([x[1] for x in kept]))


async def run(run_dir: Path, keyframes: dict, books: list[dict], identified: dict, metrics: Metrics) -> dict:
    with metrics.time(STAGE):
        imu = read_jsonl(run_dir / "imu.jsonl")
        out: dict = {"review": []}
        if len(imu) < 50:
            out["review"].append("no motion-sensor data: room geometry cannot be computed")
            write_json(run_dir / "room.json", out)
            return out
        poses = Poses(imu)
        turn = keyframes.get("room_turn", [])
        tilt = keyframes.get("room_tilt", [])
        f, f_method = estimate_focal(run_dir, turn or tilt, poses)
        out["focal_px"], out["focal_method"] = round(f, 1), f_method

        rulers = ruler_books(books, identified)
        shelf_frames = keyframes.get("shelves", [])
        overview_tasks = []
        for fr in keyframes.get("unit_overview", []):
            cam = poses.camera(fr, f)
            if cam:
                overview_tasks.append(_overview(run_dir, fr, cam, shelf_frames, books, rulers, metrics))
        boundary_frames = boundary_views(turn, tilt)
        overviews, boundaries = await asyncio.gather(
            asyncio.gather(*overview_tasks), asyncio.gather(*(_boundaries(run_dir, fr, metrics) for fr in boundary_frames))
        )
        overviews = [o for o in overviews if o]
        all_heights = [o["camera_height_m"] for o in overviews if o.get("camera_height_m")]
        # A phone held by a standing adult is 0.7-2.1 m above the floor; anything else is a measurement failure.
        heights = [v for v in all_heights if CAM_HEIGHT_RANGE[0] <= v <= CAM_HEIGHT_RANGE[1]]
        out["overviews"] = overviews
        if all_heights and not heights:
            out["review"].append(f"implausible camera heights {all_heights} m from bookcase overviews: room not measured")
        if all_heights and not heights:  # rulers gave nonsense: the geometry is broken, don't paper over it
            write_json(run_dir / "room.json", out)
            return out
        prior_scale = not heights
        if prior_scale:
            # No book ruler: still measure the room, scaled by a typical phone height, and say so.
            prior, posture = height_prior(run_dir)
            heights = [prior]
            out["camera_height_prior"] = {"height_m": prior, "posture": posture, "error": PRIOR_ERROR}
            out["review"].append(
                f"room scale from a typical phone height ({prior} m, {posture}; {PRIOR_ERROR}): no identified book "
                "had a catalogue height to use as a ruler. Check one wall with a tape measure")
        h = float(np.median(heights))
        out["camera_height_m"] = round(h, 3)
        out["camera_height_basis"] = "typical phone height prior" if prior_scale else "book ruler on bookcase overviews"

        floor_pts, ceil_obs = [], []
        for fr, bd in zip(boundary_frames, boundaries):
            cam = poses.camera(fr, f)
            if not cam:
                continue
            for s in bd["floor"]:
                p = G.floor_point(cam, s["x"], s["y"], h)
                if p is not None and np.linalg.norm(p) < 15:
                    floor_pts.append((cam.elevation_azimuth(s["x"], s["y"])[1], p, s["at"]))
            for s in bd["ceiling"]:
                elev, az = cam.elevation_azimuth(s["x"], s["y"])
                if elev > math.radians(2):
                    ceil_obs.append((elev, az))
        gyro = _gyro_room(floor_pts, ceil_obs, h)
        da3 = await asyncio.to_thread(_da3_room, run_dir, boundary_frames, poses, f, f_method, h)
        out["methods"] = {"gyro_floor_rays": gyro, "da3_book_scaled": da3}
        # Gyro rays first: the DA3 outline was tuned on one ARKitScenes room and measured +84 % floor area on the
        # held-out room (docs/ROOM_DA3_CHECK.md), so it is a cross-check, and the figure only when gyro fails.
        primary = gyro if "error" not in gyro else da3
        if "error" in primary:
            out["review"].append(f"room outline not measurable: DA3 - {da3.get('error')}; gyro - {gyro.get('error')}")
            write_json(run_dir / "room.json", out)
            return out
        out["primary_method"] = primary["method"]
        disagree = None
        if "error" not in da3 and "error" not in gyro:
            disagree = abs(da3["floor_area_m2"] / gyro["floor_area_m2"] - 1)
            out["methods_floor_area_disagreement"] = round(disagree, 3)
            if disagree > 0.10:
                out["review"].append(
                    f"room methods disagree by {disagree:.0%} on floor area (DA3 {da3['floor_area_m2']} m2, "
                    f"gyro {gyro['floor_area_m2']} m2): check against a tape measure"
                )
        else:
            out["review"].append(f"only one room method succeeded ({primary['method']}); no independent cross-check")
            if primary is da3:
                out["review"].append("room figure from DA3 depth only, a method not validated (+84 % floor area on its "
                                     "held-out test room): treat as indicative and check with a tape measure")
        # Ceiling: from the primary method, else whichever measured one.
        height = primary.get("height_m") or gyro.get("height_m") or da3.get("height_m")
        if height is None:
            out["review"].append("ceiling not seen well enough by either method: ceiling height and wall area unknown")
        area, perim = primary["floor_area_m2"], primary["perimeter_m"]
        cases = _unique_cases(overviews)
        shelved = sum(c["case_width_m"] * c["case_height_m"] for c in cases)
        corners = primary["corners"]
        out.update(
            {
                "polygon_m": primary["polygon_m"],
                "corners": corners,
                "shape": "rectangle" if corners == 4 else f"rectilinear polygon with {corners} corners",
                "length_m": primary["length_m"],
                "width_m": primary["width_m"],
                "height_m": height,
                "floor_area_m2": area,
                "floor_area_ft2": round(area * SQFT_PER_M2, 1),
                "perimeter_m": perim,
                "wall_area_m2": round(perim * height, 2) if height else None,
                "wall_area_ft2": round(perim * height * SQFT_PER_M2, 1) if height else None,
                "wall_area_basis": "gross: perimeter x ceiling height, doors and windows not subtracted",
                "shelved_wall_area_m2": round(shelved, 2),
                "shelved_wall_area_ft2": round(shelved * SQFT_PER_M2, 1),
                "bookcases": cases,
            }
        )
        spread = float(np.std(heights) / h) if len(heights) > 1 else 0.15
        conf = 0.9 - 2 * spread - (0.2 if "ASSUMED" in f_method else 0)
        conf -= 0.2 if disagree is None else min(0.4, 2 * disagree)
        if prior_scale or primary is da3:
            conf = min(conf, 0.3)
        out["confidence"] = round(max(0.1, min(0.9, conf)), 2)
    write_json(run_dir / "room.json", out)
    return out


def _gyro_room(floor_pts: list, ceil_obs: list, h: float) -> dict:
    """Outline from floor-line rays (gravity + gyro + camera height); ceiling from wall-ceiling rays."""
    poly = room_from_points(floor_pts)
    if poly is None:
        return {"error": "too few floor-boundary points from the room turn"}
    ceilings = [h + d * math.tan(e) for e, az in ceil_obs if (d := _cast(poly, az))]
    height = float(np.median(ceilings)) if len(ceilings) >= 5 else None
    length, width = G.oriented_extent(poly)
    return {
        "method": "gyro floor-line rays + camera height",
        "polygon_m": [[round(x, 3), round(y, 3)] for x, y in poly],
        "corners": len(poly),
        "length_m": round(length, 2),
        "width_m": round(width, 2),
        "height_m": round(height, 2) if height and 1.9 <= height <= 5.0 else None,
        "floor_area_m2": round(G.polygon_area(poly), 2),
        "perimeter_m": round(G.polygon_perimeter(poly), 2),
        "n_floor_points": len(floor_pts),
        "n_ceiling_points": len(ceilings),
    }


def _da3_inputs(frames: list[dict], f: float, f_method: str) -> tuple[list[dict], list | None]:
    from . import room_da3

    views = room_da3.pick_views(sorted(frames, key=lambda fr: fr["t_ms"]))
    Ks = None
    if "ASSUMED" not in f_method:  # pass the measured focal length; otherwise DA3 estimates it
        Ks = [[[round(f, 1), 0, fr["w"] / 2], [0, round(f, 1), fr["h"] / 2], [0, 0, 1]] for fr in views]
    return views, Ks


def _da3_depth(run_dir: Path, views: list[dict], Ks: list | None) -> dict:
    from . import room_da3

    key = {"views": [fr["file"] for fr in views], "K": Ks}
    return room_da3.run_worker([str(run_dir / fr["file"]) for fr in views], Ks, cache=memo_path(run_dir, "da3", key, ".npz"))


async def prewarm(run_dir: Path, keyframes: dict, metrics: Metrics) -> None:
    """During the sweep, once the room turn is recorded: mark floor/ceiling lines and run DA3 now, so run() only
    has the book-dependent part (camera height from the book ruler) left after the sweep."""
    from . import room_da3

    imu = read_jsonl(run_dir / "imu.jsonl")
    frames = keyframes.get("room_turn", []) + keyframes.get("room_tilt", [])
    if len(imu) < 50 or not frames:
        return
    poses = Poses(imu)
    f, f_method = estimate_focal(run_dir, keyframes.get("room_turn", []) or keyframes.get("room_tilt", []), poses)
    jobs = [_boundaries(run_dir, fr, metrics) for fr in boundary_views(keyframes.get("room_turn", []), keyframes.get("room_tilt", []))]
    views, Ks = _da3_inputs(frames, f, f_method)
    if room_da3.available() and len(views) >= 6:
        jobs.append(asyncio.to_thread(_da3_depth, run_dir, views, Ks))
    await asyncio.gather(*jobs, return_exceptions=True)


def _da3_room(run_dir: Path, frames: list[dict], poses: "Poses", f: float, f_method: str, h: float) -> dict:
    from . import room_da3

    if not room_da3.available():
        return {"error": "Depth Anything 3 not installed (uv sync --extra depth)"}
    views, Ks = _da3_inputs(frames, f, f_method)
    if len(views) < 6:
        return {"error": f"only {len(views)} room-turn frames"}
    ups = []
    for fr in views:
        u = G.up_at(poses.imu, fr["t_ms"])
        ups.append(room_da3.device_up_to_camera(u, fr.get("orient", 0)) if u is not None else None)
    try:
        z = _da3_depth(run_dir, views, Ks)
    except Exception as e:
        return {"error": f"DA3 failed: {type(e).__name__}: {str(e)[-300:]}"}
    return room_da3.measure(z, ups, h)
