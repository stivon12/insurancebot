"""Room size from the turn: MoGe-2 metric depth per frame + phone gravity and gyro heading; agent corners as a check."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import memo, write_json

SQFT_PER_M2 = 10.7639
MOGE_ID = "Ruicheng/moge-2-vits-normal"
BIN_DEG = 2
MIN_SEEN_DEG, FULL_SEEN_DEG = 200, 300
CAM_HEIGHT_RANGE = (0.8, 2.0)
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


def frame_geometry(img_rgb: np.ndarray, up_dev: np.ndarray, heading_deg: float, spine_boxes: list | None = None) -> dict:
    """One turn frame: camera height above the floor, focal length, wall distance per bearing, ceiling, book spans."""
    import torch

    model, dev = _load()
    x = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(x, resolution_level=5, use_fp16=False)
    P, mask = out["points"].cpu().numpy(), out["mask"].cpu().numpy() > 0.5
    f_px = float(out["intrinsics"].cpu().numpy()[1, 1] * img_rgb.shape[0])
    up = np.array([up_dev[0], -up_dev[1], -up_dev[2]])  # device -> camera (x right, y down, z forward)
    up /= np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up
    fwd /= max(np.linalg.norm(fwd), 1e-6)
    right = np.cross(up, fwd) * -1  # clockwise side seen from above
    Q = P[mask]
    h = Q @ up
    hist, e = np.histogram(h, bins=np.arange(-3, 3, 0.02))
    c = (e[:-1] + e[1:]) / 2
    low = np.where(c < -0.5, hist, 0)
    floor = float(c[np.argmax(low)]) if low.max() > 0.03 * len(h) / 50 else None
    high = np.where(c > 0.4, hist, 0)
    ceiling = float(c[np.argmax(high)]) if high.max() > 0.02 * len(h) / 50 else None
    cam_h = -floor if floor is not None and CAM_HEIGHT_RANGE[0] <= -floor <= CAM_HEIGHT_RANGE[1] else None
    base = -(cam_h or CAMERA_HEIGHT_PRIOR_M)
    walls: dict[int, float] = {}
    for lo, hi in ((2.05, 2.45), (1.0, 2.05)):  # above doors where visible, else above most furniture
        band = (h > base + lo) & (h < base + hi)
        xs, ys = Q[band] @ right, Q[band] @ fwd
        k = (((heading_deg + np.degrees(np.arctan2(xs, ys))) % 360) // BIN_DEG).astype(int)
        r = np.hypot(xs, ys)
        for b in np.unique(k):
            if int(b) * BIN_DEG not in walls and (k == b).sum() >= 30:
                walls[int(b) * BIN_DEG] = float(np.median(r[k == b]))
    books = {}
    for box in spine_boxes or []:
        x0, y0, x1, y1 = (int(v) for v in box)
        sub, sm = P[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        if sm.sum() < 20:
            continue
        S = sub[sm]
        bh = S @ up
        bb = (heading_deg + np.degrees(np.arctan2(S @ right, S @ fwd))) % 360
        for k in np.unique((bb // BIN_DEG).astype(int)):
            lo, hi = float(np.percentile(bh[(bb // BIN_DEG).astype(int) == k], 5)), float(np.percentile(bh[(bb // BIN_DEG).astype(int) == k], 95))
            cur = books.get(int(k) * BIN_DEG, [lo, hi])
            books[int(k) * BIN_DEG] = [min(cur[0], lo), max(cur[1], hi)]
    return {"cam_h": cam_h, "f_px": f_px, "walls": walls, "ceiling_above": ceiling, "books": books}


async def measure_frame(run_dir: Path, frame: dict, up_dev: np.ndarray, heading_deg: float, spine_boxes: list | None) -> dict:
    import asyncio

    def run():
        img = cv2.cvtColor(cv2.imread(str(Path(run_dir) / frame["file"])), cv2.COLOR_BGR2RGB)
        return frame_geometry(img, up_dev, heading_deg, spine_boxes)
    key = {"file": frame["file"], "model": MOGE_ID, "heading": round(heading_deg, 1)}
    return await memo(run_dir, "room_frames", key, lambda: asyncio.to_thread(run))


# ---------- the room ----------

def measure(run_dir: Path, frames: list[dict], corners: list[float] | None = None, layout: dict | None = None) -> dict:
    """Combine turn frames: one camera height (median), each frame's depth rescaled to it, walls per bearing."""
    out: dict = {"review": [], "corners_marked": len(corners or []), "layout": layout}
    frames = [f for f in frames if f and f.get("walls") is not None]
    hs = [f["cam_h"] for f in frames if f.get("cam_h")]
    if not frames:
        out["review"].append("no room-turn frames were measured: room size unknown")
        return _save(run_dir, out)
    H = CAMERA_HEIGHT_PRIOR_M
    mh = float(np.median(hs)) if hs else H
    out.update(camera_height_m=H, camera_height_rel_error=PRIOR_ERROR, focal_px=round(float(np.median([f["f_px"] for f in frames])), 1),
               camera_height_basis=f"typical phone height held upright ({H} m, +-{PRIOR_ERROR:.0%}); depth model's own estimate {mh:.2f} m")
    out["review"].append(f"room scale assumes the phone was held at about {H} m (+-{PRIOR_ERROR:.0%} on lengths): check one wall with a tape")
    walls: dict[int, list[float]] = {}
    books: dict[int, list[float]] = {}
    ceil = []
    for f in frames:
        s = H / (f.get("cam_h") or mh)
        for b, d in f["walls"].items():
            walls.setdefault(int(b), []).append(d * s)
        for b, (lo, hi) in f["books"].items():
            cur = books.get(int(b), [lo * s, hi * s])
            books[int(b)] = [min(cur[0], lo * s), max(cur[1], hi * s)]
        if f.get("ceiling_above"):
            ceil.append(f["ceiling_above"] * s)
    seen = len(walls) * BIN_DEG
    out["seen_deg"] = seen
    if seen < MIN_SEEN_DEG:
        out["review"].append(f"walls seen over only {seen} degrees of the turn: room outline not measured")
        return _save(run_dir, out)
    med = {b: float(np.median(v)) for b, v in walls.items()}
    keys = sorted(med)
    for b in keys:  # a bearing much farther than its neighbours looks through a doorway or window: drop it
        near = [med[k] for k in keys if k != b and min(abs(k - b), 360 - abs(k - b)) <= 20]
        if len(near) >= 4 and med[b] > 1.4 * float(np.median(near)):
            walls.pop(b)
    bearings = sorted(walls)
    pts = np.array([[np.median(walls[b]) * math.sin(math.radians(b + BIN_DEG / 2)),
                     np.median(walls[b]) * math.cos(math.radians(b + BIN_DEG / 2))] for b in bearings])
    poly = fit_room_polygon(pts)
    area, perim = polygon_area(poly), polygon_perimeter(poly)
    length, width = oriented_extent(poly)
    height = H + float(np.median(ceil)) if len(ceil) >= 3 and 1.9 <= H + float(np.median(ceil)) <= 5.0 else None
    if height is None:
        out["review"].append("ceiling not seen in enough turn frames: ceiling height and wall area unknown")
    wall_d = {b: float(np.median(v)) for b, v in walls.items()}
    shelved = sum(math.radians(BIN_DEG) * wall_d.get(b, 0) * (hi - lo) for b, (lo, hi) in books.items() if hi > lo)
    if seen < FULL_SEEN_DEG:
        out["review"].append(f"walls seen over {seen} degrees of 360: unseen walls are closed with straight lines, area may be low")
    if corners and abs(len(corners) - len(poly)) > 1:
        out["review"].append(f"the agent marked {len(corners)} corners but the outline has {len(poly)}: check the shape")
    conf = 0.6 if seen >= FULL_SEEN_DEG else 0.4
    conf -= 0.2 if not hs else 0
    conf -= 0.1 if any("corners" in r for r in out["review"]) else 0
    out.update(
        polygon_m=[[round(x, 3), round(y, 3)] for x, y in poly], corners=len(poly),
        shape="rectangle" if len(poly) == 4 else f"rectilinear polygon with {len(poly)} corners",
        length_m=round(length, 2), width_m=round(width, 2), height_m=round(height, 2) if height else None,
        floor_area_m2=round(area, 2), floor_area_ft2=round(area * SQFT_PER_M2, 1), perimeter_m=round(perim, 2),
        wall_area_m2=round(perim * height, 2) if height else None, wall_area_ft2=round(perim * height * SQFT_PER_M2, 1) if height else None,
        wall_area_basis="gross: perimeter x ceiling height, doors and windows not subtracted",
        shelved_wall_area_m2=round(shelved, 2), shelved_wall_area_ft2=round(shelved * SQFT_PER_M2, 1),
        scale_method=f"MoGe-2 depth per turn frame (shape), scaled so the floor is {H} m below the phone; bearing from the gyro, "
                     "gravity from the accelerometer; rectilinear fit",
        confidence=round(max(0.1, conf), 2))
    return _save(run_dir, out)


def _save(run_dir: Path, out: dict) -> dict:
    write_json(Path(run_dir) / "room.json", out)
    return out
