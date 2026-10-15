"""Camera rays from phone motion sensors, plus room-polygon fitting. Pure numpy, unit-tested.

Conventions
- Device frame (W3C DeviceMotion): x right, y up (screen top), z out of the screen.
- The rear camera looks along device -z.
- `up` is the unit vector opposite gravity in device coordinates, from accelerationIncludingGravity.
- Azimuth increases clockwise seen from above; yaw is integrated from the gyro rate about `up`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    f: float  # focal length, px
    cx: float
    cy: float
    up: np.ndarray  # unit vector, device coords
    azimuth: float  # radians, heading of the optical axis
    orient: int = 0  # screen orientation angle (0, 90, 180, 270)

    def _image_axes(self) -> tuple[np.ndarray, np.ndarray]:
        t = math.radians(self.orient)
        image_x = np.array([math.cos(t), math.sin(t), 0.0])
        image_down = np.array([math.sin(t), -math.cos(t), 0.0])
        return image_x, image_down

    def ray(self, x: float, y: float) -> np.ndarray:
        """Unit ray (device coords) through pixel (x, y)."""
        ix, idn = self._image_axes()
        r = (x - self.cx) * ix + (y - self.cy) * idn + self.f * np.array([0.0, 0.0, -1.0])
        return r / np.linalg.norm(r)

    def _horizontal_basis(self) -> tuple[np.ndarray, np.ndarray]:
        fwd = np.array([0.0, 0.0, -1.0])
        fwd_h = fwd - fwd.dot(self.up) * self.up
        if np.linalg.norm(fwd_h) < 1e-6:  # camera pointing straight up/down: use image-up as forward
            fwd_h = -self._image_axes()[1] - (-self._image_axes()[1]).dot(self.up) * self.up
        fwd_h /= np.linalg.norm(fwd_h)
        right = np.cross(fwd_h, self.up)
        return fwd_h, right

    def elevation_azimuth(self, x: float, y: float) -> tuple[float, float]:
        r = self.ray(x, y)
        fwd, right = self._horizontal_basis()
        elev = math.asin(max(-1.0, min(1.0, float(r.dot(self.up)))))
        az = self.azimuth + math.atan2(float(r.dot(right)), float(r.dot(fwd)))
        return elev, az

    def optical_depth_factor(self, x: float, y: float) -> float:
        """Component of the unit ray along the optical axis (for converting optical depth to ray length)."""
        return float(-self.ray(x, y)[2])


def floor_point(cam: Camera, x: float, y: float, cam_height: float) -> np.ndarray | None:
    """Where the ray through (x, y) hits the floor, as (east, north) metres from the camera."""
    elev, az = cam.elevation_azimuth(x, y)
    if elev > -math.radians(2):  # at or above the horizon: never reaches the floor
        return None
    d = cam_height / math.tan(-elev)
    return np.array([d * math.sin(az), d * math.cos(az)])


def gravity_up(samples: list[tuple[float, float, float]]) -> np.ndarray:
    a = np.median(np.array(samples, dtype=float), axis=0)
    n = np.linalg.norm(a)
    return a / n if n > 0 else np.array([0.0, 1.0, 0.0])


def integrate_yaw(imu: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(t_ms array, azimuth radians) from gyro rate projected on the up vector. Clockwise positive."""
    if len(imu) < 2:
        return np.array([0.0]), np.array([0.0])
    t = np.array([s["t_ms"] for s in imu], dtype=float)
    acc = np.array([[s["ax"], s["ay"], s["az"]] for s in imu], dtype=float)
    # rotationRate: alpha about z, beta about x, gamma about y (deg/s) -> vector in device xyz
    w = np.radians(np.array([[s["rb"], s["rg"], s["ra"]] for s in imu], dtype=float))
    # Low-pass the accelerometer so hand motion does not swing the up vector.
    k = 9
    kernel = np.ones(k) / k
    acc_s = np.stack([np.convolve(acc[:, i], kernel, mode="same") for i in range(3)], axis=1)
    up = acc_s / np.maximum(np.linalg.norm(acc_s, axis=1, keepdims=True), 1e-6)
    rate_left = np.einsum("ij,ij->i", w, up)  # counter-clockwise about up = turning left
    dt = np.diff(t, prepend=t[0]) / 1000.0
    return t, -np.cumsum(rate_left * dt)


def up_at(imu: list[dict], t_ms: float, window_ms: float = 150) -> np.ndarray | None:
    near = [(s["ax"], s["ay"], s["az"]) for s in imu if abs(s["t_ms"] - t_ms) <= window_ms]
    return gravity_up(near) if near else None


# ---------- room polygon ----------

def _rdp(points: np.ndarray, eps: float) -> list[int]:
    """Ramer-Douglas-Peucker on an open polyline; returns kept indices."""
    if len(points) < 3:
        return list(range(len(points)))
    a, b = points[0], points[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-9:
        d = np.linalg.norm(points - a, axis=1)
    else:
        rel = points - a
        d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / n
    i = int(np.argmax(d))
    if d[i] > eps:
        left = _rdp(points[: i + 1], eps)
        right = _rdp(points[i:], eps)
        return left[:-1] + [j + i for j in right]
    return [0, len(points) - 1]


def dominant_angle(points: np.ndarray) -> float:
    """Manhattan orientation (radians in [0, pi/2)) from length-weighted edge directions."""
    d = np.diff(np.vstack([points, points[:1]]), axis=0)
    lengths = np.linalg.norm(d, axis=1)
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), math.pi / 2)
    # circular mean on the 90-degree circle
    c = np.sum(lengths * np.cos(4 * ang))
    s = np.sum(lengths * np.sin(4 * ang))
    return float(np.mod(math.atan2(s, c) / 4, math.pi / 2))


def fit_room_polygon(points: np.ndarray, eps: float = 0.12) -> np.ndarray:
    """Rectilinear (Manhattan) polygon from floor-boundary points ordered by azimuth around the camera.

    Points are simplified into straight runs, each run is snapped to the nearest
    axis, collinear neighbours are merged, and corners are where consecutive
    perpendicular walls intersect. Works for rectangles and L/T-shaped rooms.
    """
    theta = dominant_angle(points)
    c, s = math.cos(-theta), math.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    p = points @ R.T
    closed = np.vstack([p, p[:1]])
    idx = _rdp(closed, eps)
    walls = []  # [axis, coordinate, weight, end_point]: axis 'x' means the wall line x = coordinate
    for a, b in zip(idx[:-1], idx[1:]):
        seg = closed[a : b + 1]
        if len(seg) < 2:
            continue
        d = seg[-1] - seg[0]
        length = float(np.linalg.norm(d))
        if length < 0.25:
            continue
        axis, col = ("y", 1) if abs(d[0]) >= abs(d[1]) else ("x", 0)
        walls.append([axis, float(np.median(seg[:, col])), length, seg[-1].copy()])
    # merge consecutive walls on the same axis and line (wrap around)
    merged: list[list] = []
    for w in walls:
        if merged and merged[-1][0] == w[0] and abs(merged[-1][1] - w[1]) < 2 * eps:
            m = merged[-1]
            m[1] = (m[1] * m[2] + w[1] * w[2]) / (m[2] + w[2])
            m[2] += w[2]
            m[3] = w[3]
        else:
            merged.append(w)
    if len(merged) > 1 and merged[0][0] == merged[-1][0] and abs(merged[0][1] - merged[-1][1]) < 2 * eps:
        last = merged.pop()
        merged[0][1] = (merged[0][1] * merged[0][2] + last[1] * last[2]) / (merged[0][2] + last[2])
        merged[0][2] += last[2]
    if len(merged) < 4:
        # Too few walls seen: fall back to the axis-aligned bounding rectangle of the points.
        x0, y0 = np.percentile(p, 2, axis=0)
        x1, y1 = np.percentile(p, 98, axis=0)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return corners @ np.linalg.inv(R).T
    corners = []
    n = len(merged)
    for i in range(n):
        a, b = merged[i], merged[(i + 1) % n]
        if a[0] != b[0]:
            x = a[1] if a[0] == "x" else b[1]
            y = a[1] if a[0] == "y" else b[1]
            corners.append([x, y])
        else:
            # Parallel neighbours (a short step wall was not observed): join them with a
            # perpendicular through the point where the first one ended.
            e = a[3]
            if a[0] == "x":
                corners += [[a[1], e[1]], [b[1], e[1]]]
            else:
                corners += [[e[0], a[1]], [e[0], b[1]]]
    corners = np.array(corners)
    return corners @ np.linalg.inv(R).T


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(poly: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)))


def oriented_extent(poly: np.ndarray) -> tuple[float, float]:
    """(length, width) of the polygon along its own Manhattan axes."""
    theta = dominant_angle(poly)
    c, s = math.cos(-theta), math.sin(-theta)
    p = poly @ np.array([[c, -s], [s, c]]).T
    ext = p.max(axis=0) - p.min(axis=0)
    return float(max(ext)), float(min(ext))
