"""Room geometry from Depth Anything 3 on the room-turn frames, scaled to metres by the book ruler.

Why this split: in the author's roomscope benchmark (Faro laser, M1 16 GB), DA3 multi-view gave a good
room SHAPE in ~15 s per room, but its own metric scale was 3-12 % off per capture, which alone breaks a
10 % floor-area bar. Here DA3 supplies shape only; scale comes from the camera height measured with
catalogue book heights (room.py), and "up" comes from the phone's gravity sensor.

Steps
  1. DA3-BASE multi-view on up to 16 turn/tilt keyframes -> per-view depth + poses (arbitrary units).
  2. Rotate the cloud so gravity is -z, using each frame's accelerometer "up" mapped through its DA3 pose.
  3. Floor and ceiling = the lowest and highest dense height bands; camera height in DA3 units = median
     camera height above that floor.  scale = camera height in metres (book ruler) / camera height in DA3 units.
  4. Walls from points in the upper wall band (75-95 % of room height), where furniture and bookcases
     rarely reach, so a bookcase front is not mistaken for the wall. Per 2-degree bearing around the
     camera centre the far robust point is a wall sample; the samples go to the same Manhattan
     polygon fit as the gyro method.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from . import geometry as G

MAX_VIEWS = 16
CONF_PERCENTILE = 30.0
PIXEL_STRIDE = 2
WALL_BAND = (0.30, 0.90)  # fraction of room height above the floor: above most clutter, below the ceiling
BEARING_BINS = 180          # 2-degree bearings around the camera centre
SMOOTH_BINS = 9             # circular median over 18 degrees removes doorway/window spikes
RDP_EPS = 0.25              # m: wall breaks smaller than this are not new walls
# Tuned on one real capture (ARKitScenes 42444946, 8 eye-level photos, scale fixed from the true ceiling):
# 75-95 % band saw walls over only 98 deg; 30-90 % + 18-deg smoothing gave perimeter +1.6 %, floor area +9 %
# against the LiDAR/laser reference. See docs/ROOM_DA3_CHECK.md.


def available() -> bool:
    return importlib.util.find_spec("depth_anything_3") is not None and importlib.util.find_spec("torch") is not None


def pick_views(frames: list[dict], n: int = MAX_VIEWS) -> list[dict]:
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def run_worker(images: list[str], Ks: list | None, timeout: int = 240) -> dict:
    with tempfile.TemporaryDirectory() as td:
        jp, op = Path(td) / "job.json", Path(td) / "out.npz"
        jp.write_text(json.dumps({"images": images, "K": Ks}))
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, "-m", "pipeline.da3_worker", str(jp), str(op)], check=True, cwd=root, timeout=timeout,
                       capture_output=True, text=True)
        return dict(np.load(op))


def cloud(z: dict) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Per-view world points (confident pixels, strided), camera centres, camera-to-world rotations."""
    pts, cams, rots = [], [], []
    for i in range(int(z["n"])):
        d, K, c, T = z[f"depth_{i}"].astype(np.float64), z[f"K_{i}"], z[f"conf_{i}"], z[f"pose_{i}"]
        H, W = d.shape
        vv, uu = np.mgrid[0:H:PIXEL_STRIDE, 0:W:PIXEL_STRIDE] + 0.5
        dd = d[::PIXEL_STRIDE, ::PIXEL_STRIDE]
        cc = c[::PIXEL_STRIDE, ::PIXEL_STRIDE]
        m = np.isfinite(dd) & (dd > 0)
        if m.any():
            m &= cc >= np.percentile(cc[m], CONF_PERCENTILE)
        P = np.stack([(uu - K[0, 2]) / K[0, 0] * dd, (vv - K[1, 2]) / K[1, 1] * dd, dd], -1)[m]
        pts.append(P @ T[:3, :3].T + T[:3, 3])
        cams.append(T[:3, 3])
        rots.append(T[:3, :3])
    return pts, np.array(cams), rots


def up_in_world(rots: list[np.ndarray], ups_cam: list[np.ndarray | None]) -> np.ndarray | None:
    """Gravity 'up' in the DA3 world frame: each frame's accelerometer up (camera axes) rotated by its pose."""
    v = [R @ u for R, u in zip(rots, ups_cam) if u is not None]
    if len(v) < 3:
        return None
    m = np.median(np.array(v), axis=0)
    return m / np.linalg.norm(m)


def up_from_cloud(P: np.ndarray, rots: list[np.ndarray]) -> np.ndarray:
    """Fallback when there is no gravity data: start from the cameras' mean image-up (phones are held roughly
    level) and search directions within 25 degrees for the one whose height histogram is sharpest - floor and
    ceiling are large planes, so they collapse into tall peaks only along true vertical."""
    u0 = np.mean([R @ np.array([0.0, -1.0, 0.0]) for R in rots], axis=0)
    u0 /= np.linalg.norm(u0)
    sub = P[:: max(1, len(P) // 60000)]
    a = np.cross(u0, [1.0, 0.0, 0.0])
    a = a if np.linalg.norm(a) > 0.1 else np.cross(u0, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(u0, a)
    best, best_score = u0, -1.0
    for tilt in np.radians(np.arange(0, 26, 1.0)):
        for phi in np.radians(np.arange(0, 360, 10.0 if tilt > 0 else 360)):
            u = np.cos(tilt) * u0 + np.sin(tilt) * (np.cos(phi) * a + np.sin(phi) * b)
            h = sub @ u
            hist, _ = np.histogram(h, bins=200)
            score = float(np.sort(hist)[-2:].sum())  # floor + ceiling peaks
            if score > best_score:
                best, best_score = u, score
    return best / np.linalg.norm(best)


def device_up_to_camera(up_dev: np.ndarray, orient: int) -> np.ndarray:
    """Accelerometer up (device axes) -> OpenCV camera axes (x right, y down, z forward) for the screen orientation."""
    cam = G.Camera(f=1.0, cx=0.0, cy=0.0, up=up_dev, azimuth=0.0, orient=orient)
    ix, idn = cam._image_axes()
    return np.array([up_dev.dot(ix), up_dev.dot(idn), up_dev.dot(np.array([0.0, 0.0, -1.0]))])


def _band_peak(z: np.ndarray, lo: float, hi: float, bin_w: float) -> float | None:
    """Centre of the densest height bin in [lo, hi], if it is a real plane (well above the band's typical bin)."""
    sel = z[(z >= lo) & (z <= hi)]
    if sel.size < 200:
        return None
    hist, edges = np.histogram(sel, bins=max(10, int((hi - lo) / bin_w)))
    k = int(np.argmax(hist))
    if hist[k] < 3 * max(1.0, float(np.median(hist))):
        return None
    return float((edges[k] + edges[k + 1]) / 2)


def measure(z: dict, ups_cam: list[np.ndarray | None], camera_height_m: float | None, ceiling_m: float | None = None) -> dict:
    """camera_height_m (book ruler) sets the scale. ceiling_m is only for offline checks against a known ceiling."""
    pts, cams, rots = cloud(z)
    P = np.concatenate(pts) if pts else np.empty((0, 3))
    if len(P) < 5000:
        return {"error": "too few confident DA3 points"}
    up = up_in_world(rots, ups_cam)
    up_source = "phone gravity sensor"
    if up is None:
        up, up_source = up_from_cloud(P, rots), "floor/ceiling planes in the cloud (no gravity data)"
    h = P @ up
    hc = cams @ up
    lo, hi = np.percentile(h, [0.2, 99.8])  # not tighter: the ceiling can be only a few % of the points
    span = hi - lo
    floor = _band_peak(h, lo, lo + 0.35 * span, span / 120)
    ceil = _band_peak(h, hi - 0.35 * span, hi, span / 120)
    if ceil is not None and float(np.median(hc)) >= ceil:
        ceil = None  # cameras above the "ceiling": that band was something else
    if floor is None:
        return {"error": "floor not found in the DA3 cloud"}
    cam_h_units = float(np.median(hc - floor))
    if cam_h_units <= 0:
        return {"error": "cameras below the detected floor: reconstruction inconsistent"}
    if camera_height_m:
        s = camera_height_m / cam_h_units  # metres per DA3 unit, from the book ruler
    elif ceiling_m and ceil is not None:
        s = ceiling_m / (ceil - floor)  # offline validation only
    else:
        return {"error": "no metric scale"}
    height = (ceil - floor) * s if ceil is not None else None
    if height is not None and not 1.9 <= height <= 5.0:
        height = None  # implausible: the 'ceiling' band was something else

    # horizontal frame
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    centre = cams.mean(axis=0)
    rel = P - centre
    xy = np.stack([rel @ e1, rel @ e2], 1) * s
    zr = (h - floor) * s
    H_ref = height or 2.4
    band = (zr >= WALL_BAND[0] * H_ref) & (zr <= WALL_BAND[1] * H_ref)
    q = xy[band]
    r = np.linalg.norm(q, axis=1)
    az = np.arctan2(q[:, 0], q[:, 1])
    width = 2 * math.pi / BEARING_BINS
    radius = np.full(BEARING_BINS, np.nan)
    for b in range(BEARING_BINS):
        m = (az >= -math.pi + b * width) & (az < -math.pi + (b + 1) * width) & (r > 0.3) & (r < 15)
        if m.sum() >= 8:
            radius[b] = np.percentile(r[m], 90)  # the far side of the bin's points: the wall, not what stands before it
    seen = int(np.isfinite(radius).sum())
    if seen * 360 / BEARING_BINS < 200:
        return {"error": f"walls seen over only {seen * 360 // BEARING_BINS} degrees"}
    half = SMOOTH_BINS // 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-empty windows stay NaN and are dropped below
        smooth = np.array([np.nanmedian(np.take(radius, range(b - half, b + half + 1), mode="wrap")) for b in range(BEARING_BINS)])
    ang = -math.pi + (np.arange(BEARING_BINS) + 0.5) * width
    ok = np.isfinite(smooth)
    samples = list(np.stack([smooth * np.sin(ang), smooth * np.cos(ang)], 1)[ok])
    poly = G.fit_room_polygon(np.array(samples), eps=RDP_EPS)
    area, perim = G.polygon_area(poly), G.polygon_perimeter(poly)
    length, width = G.oriented_extent(poly)
    return {
        "method": "DA3 multi-view shape + book-ruler camera height",
        "scale_m_per_unit": round(s, 5),
        "up_source": up_source,
        "camera_height_m": round(cam_h_units * s, 3),
        "camera_height_units": round(cam_h_units, 4),
        "views": int(z["n"]),
        "da3_seconds": round(float(z["seconds"]), 1),
        "wall_coverage_deg": seen * 360 // BEARING_BINS,
        "polygon_m": [[round(x, 3), round(y, 3)] for x, y in poly],
        "corners": len(poly),
        "length_m": round(length, 2),
        "width_m": round(width, 2),
        "height_m": round(height, 2) if height else None,
        "floor_area_m2": round(area, 2),
        "perimeter_m": round(perim, 2),
    }
