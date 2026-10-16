"""Room size from a walking tour: camera path from phone orientation + MoGe-2 depth at matched keypoints, then an occupancy grid."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import write_json

SQFT_PER_M2 = 10.7639
MOGE_ID = "Ruicheng/moge-2-vits-normal"
# MoGe's own metric scale was 6-13 % short of LiDAR and its floor height 18 % low on three iPhone captures, while a
# 1.40 m phone height was within 5 %: depth gives the shape, the phone height gives the scale.
CAMERA_HEIGHT_PRIOR_M = 1.40
PRIOR_ERROR = 0.10
_model = None


# ---------- camera and gravity (W3C DeviceMotion: x right, y up, z out of the screen) ----------

@dataclass
class Camera:
    f: float
    cx: float
    cy: float
    up: np.ndarray
    azimuth: float
    orient: int = 0

    def _image_axes(self) -> tuple[np.ndarray, np.ndarray]:
        t = math.radians(self.orient)
        return np.array([math.cos(t), math.sin(t), 0.0]), np.array([math.sin(t), -math.cos(t), 0.0])

    def ray(self, x: float, y: float) -> np.ndarray:
        ix, idn = self._image_axes()
        r = (x - self.cx) * ix + (y - self.cy) * idn + self.f * np.array([0.0, 0.0, -1.0])
        return r / np.linalg.norm(r)

    def _horizontal_basis(self) -> tuple[np.ndarray, np.ndarray]:
        fwd = np.array([0.0, 0.0, -1.0])
        fwd_h = fwd - fwd.dot(self.up) * self.up
        fwd_h /= max(np.linalg.norm(fwd_h), 1e-6)
        return fwd_h, np.cross(fwd_h, self.up)

    def elevation_azimuth(self, x: float, y: float) -> tuple[float, float]:
        r = self.ray(x, y)
        fwd, right = self._horizontal_basis()
        return math.asin(max(-1.0, min(1.0, float(r.dot(self.up))))), self.azimuth + math.atan2(float(r.dot(right)), float(r.dot(fwd)))


def up_at(imu: list[dict], t_ms: float, window_ms: float = 150) -> np.ndarray | None:
    near = [(s["ax"], s["ay"], s["az"]) for s in imu if abs(s["t_ms"] - t_ms) <= window_ms]
    if not near:
        return None
    a = np.median(np.array(near, float), axis=0)
    return a / max(np.linalg.norm(a), 1e-6)


# ---------- rectilinear polygon ----------

def _rdp(points: np.ndarray, eps: float) -> list[int]:
    if len(points) < 3:
        return list(range(len(points)))
    a, b = points[0], points[-1]
    ab, rel = b - a, points - a
    n = np.linalg.norm(ab)
    d = np.linalg.norm(rel, axis=1) if n < 1e-9 else np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / n
    i = int(np.argmax(d))
    if d[i] > eps:
        return _rdp(points[: i + 1], eps)[:-1] + [j + i for j in _rdp(points[i:], eps)]
    return [0, len(points) - 1]


def dominant_angle(points: np.ndarray) -> float:
    d = np.diff(np.vstack([points, points[:1]]), axis=0)
    w, ang = np.linalg.norm(d, axis=1), np.mod(np.arctan2(d[:, 1], d[:, 0]), math.pi / 2)
    return float(np.mod(math.atan2(np.sum(w * np.sin(4 * ang)), np.sum(w * np.cos(4 * ang))) / 4, math.pi / 2))


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(-theta), math.sin(-theta)
    return np.array([[c, -s], [s, c]])


def fit_rectangle(points: np.ndarray) -> np.ndarray:
    R = _rot(dominant_angle(points))
    p = points @ R.T
    (x0, y0), (x1, y1) = np.percentile(p, 3, axis=0), np.percentile(p, 97, axis=0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ np.linalg.inv(R).T


def fit_room_polygon(points: np.ndarray, eps: float = 0.12) -> np.ndarray:
    """Manhattan polygon from wall points ordered by bearing: straight runs snapped to two axes, corners at meets."""
    R = _rot(dominant_angle(points))
    closed = np.vstack([points @ R.T, (points @ R.T)[:1]])
    walls = []
    for a, b in zip(_rdp(closed, eps)[:-1], _rdp(closed, eps)[1:]):
        seg = closed[a: b + 1]
        d = seg[-1] - seg[0]
        if len(seg) >= 2 and np.linalg.norm(d) >= 0.25:
            axis, col = ("y", 1) if abs(d[0]) >= abs(d[1]) else ("x", 0)
            walls.append([axis, float(np.median(seg[:, col])), float(np.linalg.norm(d)), seg[-1].copy()])
    merged: list[list] = []
    for w in walls:
        if merged and merged[-1][0] == w[0] and abs(merged[-1][1] - w[1]) < 2 * eps:
            m = merged[-1]
            m[1], m[2], m[3] = (m[1] * m[2] + w[1] * w[2]) / (m[2] + w[2]), m[2] + w[2], w[3]
        else:
            merged.append(w)
    if len(merged) > 1 and merged[0][0] == merged[-1][0] and abs(merged[0][1] - merged[-1][1]) < 2 * eps:
        last = merged.pop()
        merged[0][1] = (merged[0][1] * merged[0][2] + last[1] * last[2]) / (merged[0][2] + last[2])
    if len(merged) < 4:
        return fit_rectangle(points)
    corners = []
    for i in range(len(merged)):
        a, b = merged[i], merged[(i + 1) % len(merged)]
        if a[0] != b[0]:
            corners.append([a[1] if a[0] == "x" else b[1], a[1] if a[0] == "y" else b[1]])
        else:  # a short step wall was not seen: join the parallel runs where the first one ended
            e = a[3]
            corners += [[a[1], e[1]], [b[1], e[1]]] if a[0] == "x" else [[e[0], a[1]], [e[0], b[1]]]
    return np.array(corners) @ np.linalg.inv(R).T


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(poly: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)))


def oriented_extent(poly: np.ndarray) -> tuple[float, float]:
    p = poly @ _rot(dominant_angle(poly)).T
    ext = p.max(axis=0) - p.min(axis=0)
    return float(max(ext)), float(min(ext))


# ---------- per-frame depth ----------

def available() -> bool:
    try:
        import moge  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    global _model
    if _model is None:
        import torch
        from moge.model.v2 import MoGeModel

        dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        _model = (MoGeModel.from_pretrained(MOGE_ID).to(dev).eval(), dev)
    return _model


def _infer(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    model, dev = _load()
    x = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(x, resolution_level=5, use_fp16=False)
    return out["points"].cpu().numpy(), out["mask"].cpu().numpy() > 0.5, float(out["intrinsics"].cpu().numpy()[1, 1] * img_rgb.shape[0])


def points(img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """MoGe-2 metric points of one frame (camera axes: x right, y down, z forward) and its focal length in pixels."""
    P, mask, f_px = _infer(img_rgb)
    return P[mask], f_px


def orientation(up_dev: np.ndarray, heading_deg: float) -> np.ndarray:
    """World-from-camera rotation (world y up, bearing clockwise from x=0,z=1) from phone gravity and gyro heading."""
    up = np.array([up_dev[0], -up_dev[1], -up_dev[2]])  # device -> camera (x right, y down, z forward)
    up /= np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up
    fwd /= max(np.linalg.norm(fwd), 1e-6)
    right = -np.cross(up, fwd)
    h = math.radians(heading_deg)
    world = np.array([[math.cos(h), 0, -math.sin(h)], [0, 1, 0], [math.sin(h), 0, math.cos(h)]]).T  # columns: right, up, fwd
    return world @ np.array([right, up, fwd])


_sift = None


def frame_points(img_rgb: np.ndarray, spine_boxes: list | None = None) -> dict:
    """One frame: MoGe points (subsampled), SIFT keypoints lifted to 3D for tracking, and the points on book spines."""
    global _sift
    P, mask, f_px = _infer(img_rgb)
    _sift = _sift or cv2.SIFT_create(4000)
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    s = 960 / g.shape[0]
    kp, des = _sift.detectAndCompute(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), None)
    uv = np.array([k.pt for k in kp]).reshape(-1, 2) / s
    u, v = np.clip(uv[:, 0].astype(int), 0, P.shape[1] - 1), np.clip(uv[:, 1].astype(int), 0, P.shape[0] - 1)
    X = P[v, u]
    ok = mask[v, u] & (X[:, 2] > 0.2) & (X[:, 2] < 8)
    books = []
    for x0, y0, x1, y1 in (np.array(b, int) for b in spine_boxes or []):
        sub = P[max(0, y0):y1, max(0, x0):x1][mask[max(0, y0):y1, max(0, x0):x1]]
        books.append(sub[:: max(1, len(sub) // 200)])
    return {"Q": P[mask][::97].astype(np.float32), "X": X[ok], "des": des[ok] if des is not None else None, "f_px": f_px,
            "books": np.concatenate(books) if books else np.zeros((0, 3), np.float32)}


class Tracker:
    """Camera path through a walking tour. Orientation comes from the phone; each step comes from keypoints matched to
    a placed frame and lifted to 3D by MoGe depth, solving the move and the depth-scale ratio together (RANSAC)."""

    MIN_INLIERS = 6

    def __init__(self) -> None:
        self.placed: list[dict] = []
        self.lost = 0
        self.seen = 0
        self._bf = cv2.BFMatcher()
        self._rng = np.random.default_rng(0)

    def add(self, fp: dict, R: np.ndarray) -> bool:
        self.seen += 1
        f = {**fp, "R": R}
        if not self.placed:
            f["C"], f["s"] = np.zeros(3), 1.0
            self.placed.append(f)
            return True
        cands = self.placed[-3:][::-1]
        if self.lost >= 2:  # lost for a while: try every placed frame, best descriptor overlap first
            cands += sorted(self.placed[:-3], key=lambda a: -self._overlap(a, f))[:5]
        for a in cands:
            res = self._step(a, f)
            if res:
                dC, sb = res
                f["s"], f["C"] = a["s"] * sb, a["C"] + a["s"] * dC
                self.placed.append(f)
                self.lost = 0
                return True
        self.lost += 1
        return False

    def _matches(self, a: dict, b: dict) -> list:
        if a["des"] is None or b["des"] is None or len(a["des"]) < 10 or len(b["des"]) < 10:
            return []
        return [p for p, q in (x for x in self._bf.knnMatch(b["des"], a["des"], k=2) if len(x) == 2) if p.distance < 0.75 * q.distance]

    def _overlap(self, a: dict, b: dict) -> int:
        return len(self._matches(a, b))

    def _step(self, a: dict, b: dict) -> tuple[np.ndarray, float] | None:
        m = self._matches(a, b)
        if len(m) < 8:
            return None
        A = (a["R"] @ a["X"][[x.trainIdx for x in m]].T).T
        B = (b["R"] @ b["X"][[x.queryIdx for x in m]].T).T
        best = None
        for _ in range(200):
            i, j = self._rng.choice(len(m), 2, replace=False)
            sb = float(np.linalg.norm(A[i] - A[j]) / max(np.linalg.norm(B[i] - B[j]), 1e-9))
            if not 0.5 < sb < 2.0:
                continue
            d = A - sb * B
            inl = np.linalg.norm(d - np.median(d[[i, j]], 0), axis=1) < 0.08 + 0.03 * np.linalg.norm(A, axis=1)
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < self.MIN_INLIERS:
            return None
        a_, b_ = A[best], B[best]
        # symmetric scale (spread ratio): a regression of A on B is biased low when both depths are noisy
        sb = float(np.median(np.linalg.norm(a_ - np.median(a_, 0), axis=1)) / max(np.median(np.linalg.norm(b_ - np.median(b_, 0), axis=1)), 1e-9))
        return (np.median(a_ - sb * b_, 0), sb) if 0.5 < sb < 2.0 else None


# ---------- the room ----------

GRID = 0.05
WALL_BAND = (1.5, 2.3)  # m above the floor: above most furniture, below most ceilings
DOOR_CLOSE_M = 0.45
ROUGH_CLOSE_M = 1.2
MIN_PLACED = 0.7


def _flood(wall: np.ndarray, seed: tuple[int, int], close_m: float = DOOR_CLOSE_M) -> np.ndarray:
    r = int(close_m / GRID)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fill = (cv2.dilate(wall, ker) == 0).astype(np.uint8) * 255
    if fill[seed[1], seed[0]] == 0:
        fy, fx = np.nonzero(fill)
        if not len(fx):
            return np.zeros_like(wall)
        k = int(np.argmin((fx - seed[0]) ** 2 + (fy - seed[1]) ** 2))
        seed = (int(fx[k]), int(fy[k]))
    cv2.floodFill(fill, np.zeros((fill.shape[0] + 2, fill.shape[1] + 2), np.uint8), seed, 128)
    room = (fill == 128).astype(np.uint8)
    return cv2.dilate(room, ker) & (wall == 0)  # give back the doorway closing, up to the wall cells


def measure(run_dir: Path, tracker: Tracker | None) -> dict:
    """Room from the tracked tour: each placed frame's points in the world, wall band on a 5 cm grid, flood fill from
    the walked path (doorways closed by 45 cm), area and perimeter from the filled cells; scale from the phone height."""
    out: dict = {"review": []}
    placed = tracker.placed if tracker else []
    if len(placed) < 5:
        out["review"].append("the camera path could not be tracked through the tour: room size unknown, measure with a tape")
        return _save(run_dir, out)
    share = len(placed) / max(tracker.seen, 1)
    out.update(frames_tracked=len(placed), frames_seen=tracker.seen, tracked_share=round(share, 2))
    world = [((f["R"] @ (f["s"] * f["Q"]).T).T + f["C"], f) for f in placed]
    hgt = np.concatenate([w[:, 1] for w, _ in world])
    cam_y = float(np.median([f["C"][1] for f in placed]))
    hist, e = np.histogram(hgt, bins=np.arange(hgt.min(), hgt.max() + 0.02, 0.02))
    c = (e[:-1] + e[1:]) / 2
    low = np.where(c < cam_y - 0.6, hist, 0)
    if low.max() < 50:
        out["review"].append("the floor was never in view: room scale unknown, measure with a tape")
        return _save(run_dir, out)
    floor = float(c[np.argmax(low)])
    k = CAMERA_HEIGHT_PRIOR_M / (cam_y - floor)  # tracker units -> metres
    high = np.where(c > cam_y + 0.3, hist, 0)
    height = (float(c[np.argmax(high)]) - floor) * k if high.max() > 0.2 * low.max() else None
    if height is not None and not 1.9 <= height <= 5.0:
        height = None
    pts = np.concatenate([w for w, _ in world])
    rel = (pts[:, 1] - floor) * k
    band = pts[(rel > WALL_BAND[0]) & (rel < WALL_BAND[1])][:, [0, 2]] * k
    track = np.array([f["C"][[0, 2]] for f in placed]) * k
    lo = np.minimum(band.min(0), track.min(0)) - 1.0
    shape = (np.maximum(band.max(0), track.max(0)) - lo + 1.0) / GRID
    W, H = int(shape[0]) + 1, int(shape[1]) + 1
    cnt = np.zeros((H, W))
    q = ((band - lo) / GRID).astype(int)
    np.add.at(cnt, (q[:, 1], q[:, 0]), 1)
    wall = (cnt >= 2).astype(np.uint8) * 255
    seed = tuple(int(v) for v in ((track.mean(0) - lo) / GRID))
    leaks = lambda m: bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())  # noqa: E731
    room = _flood(wall, seed)
    closed, rough = not leaks(room), None
    if not closed:  # gaps wider than a doorway: close harder, else take the hull of the walls near the path
        room = _flood(wall, seed, ROUGH_CLOSE_M)
        rough = f"gaps in the walls seen were closed with straight lines up to {ROUGH_CLOSE_M} m"
        if leaks(room) or not room.any():
            near = np.zeros_like(wall)
            cv2.circle(near, seed, int(5 / GRID), 1, -1)
            ys, xs = np.nonzero((wall > 0) & (near > 0))
            pts_ = np.concatenate([np.stack([xs, ys], 1), ((track - lo) / GRID).astype(int)]).astype(np.int32)
            room = np.zeros_like(wall)
            cv2.fillPoly(room, [cv2.convexHull(pts_)], 1)
            rough = "the walls seen never closed: area is the hull of the walls seen within 5 m of the path"
    _save_map(run_dir, wall, room, ((track - lo) / GRID).astype(int))
    out.update(map="room_map.png", outline_closed=closed, wall_seen_m=round(float((wall > 0).sum() * GRID / 2), 1))
    cont, _ = cv2.findContours(room.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        out["review"].append("no outline could be formed from the walls seen: room size unknown, measure with a tape")
        return _save(run_dir, out)
    big = max(cont, key=cv2.contourArea)
    area = float((room.sum() + 0.5 * len(big)) * GRID * GRID)  # cells inside, plus half the boundary: to the wall line
    perim = float(cv2.arcLength(big, True) * GRID)
    outline = big[:, 0, :] * GRID + lo
    poly = fit_room_polygon(outline.astype(float))
    length, width = oriented_extent(poly)
    shelved = 0.0
    bp = [((f["R"] @ (f["s"] * f["books"]).T).T + f["C"]) for f in placed if len(f["books"])]
    if bp:  # spine points on the grid: each cell's height span, walls about two cells thick
        bp = np.concatenate(bp)
        bq = ((bp[:, [0, 2]] * k - lo) / GRID).astype(int)
        cells: dict[tuple, list] = {}
        for (x, y_), hh in zip(map(tuple, bq), (bp[:, 1] - floor) * k):
            cells.setdefault((x, y_), []).append(hh)
        shelved = sum(GRID * (np.percentile(v, 95) - np.percentile(v, 5)) for v in cells.values() if len(v) >= 5) / 2
    if rough:
        out["review"].append(f"rough room size: {rough} (see room_map.png); measure with a tape")
    if share < MIN_PLACED:
        out["review"].append(f"the camera path was tracked through only {share:.0%} of the tour: parts of the room may be missing")
    if height is None:
        out["review"].append("ceiling not seen: ceiling height and wall area unknown")
    out["review"].append(f"room scale assumes the phone was held at about {CAMERA_HEIGHT_PRIOR_M} m (+-{PRIOR_ERROR:.0%} on lengths): check one wall with a tape")
    conf = 0.2 if rough else 0.6 if share >= MIN_PLACED else 0.4
    out.update(
        camera_height_m=CAMERA_HEIGHT_PRIOR_M, camera_height_rel_error=PRIOR_ERROR,
        focal_px=round(float(np.median([f["f_px"] for f in placed])), 1), frames_tracked=len(placed), frames_seen=tracker.seen,
        polygon_m=[[round(x, 3), round(y, 3)] for x, y in poly], corners=len(poly),
        shape="rectangle" if len(poly) == 4 else f"rectilinear polygon with {len(poly)} corners",
        length_m=round(length, 2), width_m=round(width, 2), height_m=round(height, 2) if height else None,
        floor_area_m2=round(area, 2), floor_area_ft2=round(area * SQFT_PER_M2, 1), perimeter_m=round(perim, 2),
        wall_area_m2=round(perim * height, 2) if height else None, wall_area_ft2=round(perim * height * SQFT_PER_M2, 1) if height else None,
        wall_area_basis="gross: perimeter x ceiling height, doors and windows not subtracted",
        shelved_wall_area_m2=round(shelved, 2), shelved_wall_area_ft2=round(shelved * SQFT_PER_M2, 1),
        scale_method=f"camera path tracked frame to frame (phone orientation + MoGe-2 depth at matched keypoints); walls on a "
                     f"{GRID * 100:.0f} cm grid, flood-filled from the walked path; scaled so the floor is {CAMERA_HEIGHT_PRIOR_M} m below the phone",
        confidence=conf)
    return _save(run_dir, out)


def _save_map(run_dir: Path, wall: np.ndarray, room: np.ndarray, track: np.ndarray) -> None:
    """Top view for the report: walls seen white, the measured floor green, the walked path red, 1 m grid."""
    img = np.full((*wall.shape, 3), 245, np.uint8)
    img[::int(1 / GRID)], img[:, ::int(1 / GRID)] = 225, 225
    img[room > 0] = (170, 220, 170)
    img[wall > 0] = (40, 40, 40)
    for x, y in track:
        cv2.circle(img, (int(x), int(y)), 2, (40, 40, 220), -1)
    cv2.imwrite(str(Path(run_dir) / "room_map.png"), cv2.resize(cv2.flip(img, 0), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))


def _save(run_dir: Path, out: dict) -> dict:
    write_json(Path(run_dir) / "room.json", out)
    return out
