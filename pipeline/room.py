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


def up_at(imu: list[dict], t_ms: float, window_ms: float = 300) -> np.ndarray | None:
    """Gravity at a frame from the motion samples just before it (the phone sends those ahead of the frame)."""
    near = [(s["ax"], s["ay"], s["az"]) for s in imu if t_ms - window_ms <= s["t_ms"] <= t_ms + 50]
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


def _infer(img_rgb: np.ndarray, level: int = 5) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    model, dev = _load()
    x = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(x, resolution_level=level, use_fp16=False)
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
    R = world @ np.array([right, up, fwd])
    R[0] *= -1  # (right, up, fwd) is a left-handed set in camera axes: mirror world x so R is a rotation (det +1)
    return R


E2W = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])  # earth (z up) -> world (y up), a rotation


def _euler_matrix(a: float, b: float, g: float) -> np.ndarray:
    """W3C deviceorientation: device -> earth is Rz(alpha) Rx(beta) Ry(gamma)."""
    a, b, g = math.radians(a), math.radians(b), math.radians(g)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    Ry = np.array([[math.cos(g), 0, math.sin(g)], [0, 1, 0], [-math.sin(g), 0, math.cos(g)]])
    return Rz @ Rx @ Ry


def _quat_matrix(q: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / max(np.linalg.norm(q), 1e-9)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def camera_in_device(screen_deg: float = 0) -> np.ndarray:
    """Rear camera axes (x right, y down, z forward) in W3C device axes, with the picture turned by the screen angle."""
    t = math.radians(-screen_deg)
    Rc = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    return np.diag([1, -1, -1.0]) @ Rc


def yaw_deg(R: np.ndarray) -> float:
    f = R @ np.array([0, 0, 1.0])
    return math.degrees(math.atan2(f[0], f[2]))


class Orientation:
    """The phone's fused orientation (RelativeOrientationSensor quaternions, or deviceorientation angles), absolute,
    so a dropped event costs nothing. The convention is chosen once per sweep, as the one whose 'up' agrees with
    gravity from the accelerometer; each frame gets it interpolated to its capture time."""

    MAX_GAP_MS = 150
    DECIDE_AFTER = 40

    def __init__(self) -> None:
        self.t: list[float] = []
        self.M: list[np.ndarray] = []  # device -> earth as sent, before the convention is chosen
        self.mode: str | None = None  # 'as_sent' | 'transposed' | 'off'
        self.agreement_deg: float | None = None
        self.src: str | None = None  # 'sensor' (RelativeOrientationSensor) or 'euler' (deviceorientation)

    def add(self, rows: list[dict], imu: list[dict]) -> None:
        for r in rows:
            src = "sensor" if "q" in r else "euler"
            if src != self.src:  # one source per sweep: the sensor wins once it starts reading
                if self.src == "sensor":
                    continue
                self.t, self.M, self.mode, self.src = [], [], None, src
            M = _quat_matrix(r["q"]) if "q" in r else _euler_matrix(r["a"], r["b"], r["g"]) if "a" in r else None
            if M is not None and (not self.t or r["t_ms"] > self.t[-1]):
                self.t.append(r["t_ms"])
                self.M.append(M)
        if self.mode is None and len(self.t) >= self.DECIDE_AFTER:
            self._decide(imu)

    def _decide(self, imu: list[dict]) -> None:
        errs: dict[str, list[float]] = {"as_sent": [], "transposed": []}
        for t, M in list(zip(self.t, self.M))[-self.DECIDE_AFTER:]:
            up = up_at(imu, t, 60)
            if up is None:
                continue
            for name, R in (("as_sent", M), ("transposed", M.T)):
                errs[name].append(math.degrees(math.acos(float(np.clip(np.dot(R.T @ np.array([0, 0, 1.0]), up), -1, 1)))))
        if min(len(v) for v in errs.values()) < 10:
            return  # no gravity yet: decide later
        name = min(errs, key=lambda k: float(np.median(errs[k])))
        self.agreement_deg = round(float(np.median(errs[name])), 1)
        self.mode = name if self.agreement_deg < 20 else "off"

    @property
    def usable(self) -> bool:
        return self.mode in ("as_sent", "transposed")

    def at(self, t_ms: float, screen_deg: float = 0) -> np.ndarray | None:
        """World-from-camera rotation at t_ms, or None if no sample lies within 150 ms."""
        if not self.usable or not self.t:
            return None
        i = int(np.searchsorted(self.t, t_ms))
        a, b = max(i - 1, 0), min(i, len(self.t) - 1)
        if min(abs(self.t[a] - t_ms), abs(self.t[b] - t_ms)) > self.MAX_GAP_MS:
            return None
        Ma, Mb = (self.M[k] if self.mode == "as_sent" else self.M[k].T for k in (a, b))
        if a != b and self.t[b] - self.t[a] < 2 * self.MAX_GAP_MS:  # interpolate on the rotation between the two
            w = (t_ms - self.t[a]) / (self.t[b] - self.t[a])
            rv = cv2.Rodrigues(Ma.T @ Mb)[0]
            M = Ma @ cv2.Rodrigues(rv * min(max(w, 0), 1))[0]
        else:
            M = Ma if abs(self.t[a] - t_ms) <= abs(self.t[b] - t_ms) else Mb
        return E2W @ M @ camera_in_device(screen_deg)


_sift = None
TRACK_W = 640  # width of the phone's tracking frames
RESEED, KEY_EVERY = 80, 3  # a new depth keyframe when fewer points remain, or every third tracking frame
RAY_DEG = 2.0  # a tracked point agrees with a position if its ray passes within this angle of its 3D point
REFINE_MAX_DEG = 6  # the image may correct the phone's rotation by up to this much per frame
TURN_PX = 3.0  # median pixel error under which the points' motion counts as a pure turn
RESTART_AFTER = 2  # failed relinks before the path restarts from the last known position
SCALE_PULL = 0.2


def track_gray(img_rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (TRACK_W, round(TRACK_W * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA) if g.shape[1] != TRACK_W else g


def frame_points(img_rgb: np.ndarray, spine_boxes: list | None = None) -> dict:
    """One frame: MoGe points (subsampled), corners and SIFT keypoints lifted to 3D for tracking, and the points on book
    spines. A 640 px tracking frame runs MoGe at its lowest resolution level (0.19 s on an M1, against 0.43 s)."""
    global _sift
    small = img_rgb.shape[1] <= TRACK_W
    P, mask, f_px = _infer(img_rgb, 0 if small else 5)
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
    gray = track_gray(img_rgb)
    corners = cv2.goodFeaturesToTrack(gray, 500, 0.01, 8)
    kp = corners.reshape(-1, 2) if corners is not None else np.zeros((0, 2), np.float32)
    sc = g.shape[1] / TRACK_W
    cu, cv_ = np.clip((kp[:, 0] * sc).astype(int), 0, P.shape[1] - 1), np.clip((kp[:, 1] * sc).astype(int), 0, P.shape[0] - 1)
    kX = P[cv_, cu]
    kok = mask[cv_, cu] & (kX[:, 2] > 0.2) & (kX[:, 2] < 8)
    return {"Q": P[mask][::25 if small else 97].astype(np.float32), "X": X[ok], "des": des[ok] if des is not None else None, "f_px": f_px,
            "books": np.concatenate(books) if books else np.zeros((0, 3), np.float32),
            "gray": gray, "kp": kp[kok].astype(np.float32), "kpX": kX[kok], "P": P[::4, ::4], "Pmask": mask[::4, ::4], "full_w": g.shape[1]}


class Tracker:
    """Camera path from the phone's 5 fps tracking frames. Corner points ride along by optical flow; each frame is
    placed by PnP against their 3D positions, starting from the phone's rotation and refining it from the image. A
    pure turn (the points move as a rotation alone predicts) keeps the position. When points thin out a frame becomes
    a keyframe: MoGe depth, its depth scale from the tracked points, fresh points. When the track is lost, a keyframe
    relinks to recent keyframes by SIFT matches lifted to 3D."""

    MIN_INLIERS = 6
    MIN_TRACKED = 15

    def __init__(self) -> None:
        self.placed: list[dict] = []
        self.lost = 0
        self.seen = 0
        self.tracked_frames = 0
        self._bf = cv2.BFMatcher()
        self._rng = np.random.default_rng(0)
        self.prev = None  # last grey tracking image
        self.pts = self.Xw = None  # tracked corners (640 px) and their world points
        self.K = None
        self.pose = None  # pose of the last tracked frame, or None
        self.since_key = 0
        self.C = None  # current camera position
        self.R_prev = None  # rotation of the last tracking frame
        self.track_seen = 0  # tracking frames offered: the honest denominator for 'tracked'
        self.turned = 0  # tracking frames placed as a pure turn
        self.breaks = 0  # times the path was lost and restarted from the last known position

    def _flow(self, gray: np.ndarray) -> bool:
        """Carry the points into this image; True if enough survive the forward-backward check."""
        if self.pts is None or self.prev is None or len(self.pts) < self.MIN_TRACKED or gray.shape != self.prev.shape:
            return False
        lk = {"winSize": (21, 21), "maxLevel": 3}
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts.reshape(-1, 1, 2), None, **lk)
        p0, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **lk)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((p0 - self.pts.reshape(-1, 1, 2)).reshape(-1, 2), axis=1) < 1.0)
        self.from_pts = self.pts[ok]  # where the surviving points were in the previous image (for the turn test)
        self.pts, self.Xw, self.prev = p1.reshape(-1, 2)[ok], self.Xw[ok], gray
        if len(self.pts) < self.MIN_TRACKED:
            self.pts = None
            return False
        return True

    def _position(self, R: np.ndarray) -> np.ndarray | None:
        """Camera position from the tracked points with the rotation known (from the phone): each point's ray must
        pass through its 3D position, which is linear in the position. RANSAC on pairs, then least squares."""
        if self.pts is None or len(self.pts) < 6:
            return None
        d = (np.linalg.inv(self.K) @ np.c_[self.pts, np.ones(len(self.pts))].T).T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        Rcw = R.T
        sk = np.zeros((len(d), 3, 3))
        sk[:, 0, 1], sk[:, 0, 2], sk[:, 1, 2] = -d[:, 2], d[:, 1], -d[:, 0]
        sk[:, 1, 0], sk[:, 2, 0], sk[:, 2, 1] = d[:, 2], -d[:, 1], d[:, 0]
        A = sk @ Rcw  # A_i C = A_i X_i
        b = np.einsum("nij,nj->ni", A, self.Xw)

        def fit(idx):
            return np.linalg.lstsq(A[idx].reshape(-1, 3), b[idx].reshape(-1), rcond=None)[0]

        def err(C):
            v = (Rcw @ (self.Xw - C).T).T
            return np.degrees(np.arccos(np.clip(np.sum(v * d, 1) / np.maximum(np.linalg.norm(v, axis=1), 1e-9), -1, 1)))
        best = None
        for _ in range(60):
            inl = err(fit(self._rng.choice(len(d), 2, replace=False))) < RAY_DEG
            if best is None or inl.sum() > best.sum():
                best = inl
        if best.sum() < 6:
            return None
        C = fit(np.nonzero(best)[0])
        keep = err(C) < RAY_DEG
        self.pts, self.Xw = self.pts[keep], self.Xw[keep]
        return C if len(self.pts) >= 6 else None

    def _pose(self, R: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """PnP from the phone's rotation and the last position, refining both; kept if the rotation moved less than
        REFINE_MAX_DEG from the phone's. Else the position alone with the phone's rotation."""
        if self.pts is not None and len(self.pts) >= 12:
            rvec, tvec = cv2.Rodrigues(R.T)[0], (-R.T @ self.C).reshape(3, 1)
            ok, rvec, tvec, inl = cv2.solvePnPRansac(self.Xw.astype(np.float64), self.pts.astype(np.float64), self.K, None,
                                                     rvec=rvec, tvec=tvec, useExtrinsicGuess=True, reprojectionError=3.0,
                                                     iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok and inl is not None and len(inl) >= 12:
                Rr = cv2.Rodrigues(rvec)[0].T
                if math.degrees(np.linalg.norm(cv2.Rodrigues(R.T @ Rr)[0])) < REFINE_MAX_DEG:
                    i = inl.ravel()
                    self.pts, self.Xw = self.pts[i], self.Xw[i]
                    return Rr, (-Rr @ tvec).ravel()
        C = self._position(R)
        return None if C is None else (R, C)

    def _pure_turn(self, R: np.ndarray) -> bool:
        """The points moved as a turn about the camera alone predicts (no walking between the two frames)."""
        if self.R_prev is None or getattr(self, "from_pts", None) is None or len(self.from_pts) < 8:
            return False
        H = self.K @ (R.T @ self.R_prev) @ np.linalg.inv(self.K)
        p = (H @ np.c_[self.from_pts, np.ones(len(self.from_pts))].T).T
        p = p[:, :2] / p[:, 2:3]
        return float(np.median(np.linalg.norm(p - self.pts, axis=1))) < TURN_PX

    def track(self, gray: np.ndarray, R: np.ndarray) -> bool:
        """A tracking frame with the phone's orientation R. True if it should become a depth keyframe (points thinning,
        every third frame, or lost)."""
        self.track_seen += 1
        self.pose = None
        if self.C is None:
            self.prev, self.R_prev = gray, R
            return True
        self.from_pts = None
        pose = self._pose(R) if self._flow(gray) else None
        if pose is None and self.pts is not None and self._pure_turn(R):
            pose = (R, self.C)  # turning on the spot: the position stays, and that is right
            self.turned += 1
        self.R_prev = R
        if pose is None:
            self.prev = gray
            return True
        self.pose = pose
        self.C = pose[1]
        self.tracked_frames += 1
        self.since_key += 1
        return len(self.pts) < RESEED or self.since_key >= KEY_EVERY

    def _seed(self, f: dict) -> None:
        self.pts = f["kp"].copy()
        self.Xw = (f["R"] @ (f["s"] * f["kpX"]).T).T + f["C"]
        self.prev = f["gray"]
        for k in ("P", "Pmask", "gray", "kp", "kpX"):  # only needed while placing: a tour keeps hundreds of keyframes
            f.pop(k, None)

    def add(self, fp: dict, R: np.ndarray) -> bool:
        """Make the frame just tracked a keyframe, using its depth (fp); its rotation is the refined one when the frame
        was tracked, else the phone's R."""
        self.seen += 1
        if self.pose is not None:
            R = self.pose[0]
        f = {**fp, "R": R}
        h, w = fp["gray"].shape
        fl = fp["f_px"] * TRACK_W / fp["full_w"]
        self.K = np.array([[fl, 0, w / 2], [0, fl, h / 2], [0, 0, 1.0]])
        self.since_key = 0
        if not self.placed:
            f["C"], f["s"] = np.zeros(3), 1.0
            return self._place(f)
        if self.pose is not None:
            C = self.pose[1]
            s_new = self.placed[-1]["s"]
            if self.pts is not None and len(self.pts) >= 10:
                z_pred = (R.T @ (self.Xw - C).T)[2]  # depth of the tracked points in this camera, in map units
                sc = fp["full_w"] / TRACK_W / 4  # 640 px -> the 1/4 depth map
                u = np.clip((self.pts[:, 0] * sc).astype(int), 0, fp["P"].shape[1] - 1)
                v = np.clip((self.pts[:, 1] * sc).astype(int), 0, fp["P"].shape[0] - 1)
                z_m = fp["P"][v, u, 2]
                ok = fp["Pmask"][v, u] & (z_m > 0.2) & (z_pred > 0.2)
                if ok.sum() >= 10:
                    s_new = float(np.median(z_pred[ok] / z_m[ok]))
            # MoGe's own metric scale wanders between frames but doesn't drift: pull the chained scale toward it
            f["C"], f["s"] = C, float(np.exp((1 - SCALE_PULL) * np.log(s_new)))
            return self._place(f)
        if self._relink(f):
            return self._place(f, already=True)
        if self.lost >= RESTART_AFTER and self.C is not None:
            # Lost for a while: carry on from the last known position (an error of however far the phone moved
            # while lost) rather than lose the rest of the tour. Counted as a break in the path.
            self.breaks += 1
            f["C"], f["s"] = self.C, self.placed[-1]["s"]
            return self._place(f)
        self.prev, self.pts = fp["gray"], None
        return False

    def _place(self, f: dict, already: bool = False) -> bool:
        if not already:
            self.placed.append(f)
        self.C, self.lost, self.R_prev = f["C"], 0, f["R"]
        self._seed(f)
        return True

    def _relink(self, f: dict) -> bool:
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
    seen = tracker.track_seen or tracker.seen
    share = (tracker.tracked_frames if tracker.track_seen else len(placed)) / max(seen, 1)
    out.update(frames_tracked=tracker.tracked_frames if tracker.track_seen else len(placed), frames_seen=seen,
               tracked_share=round(share, 2), keyframes=len(placed), turns_on_the_spot=tracker.turned, path_breaks=tracker.breaks)
    if tracker.breaks:
        out["review"].append(f"the camera path was lost {tracker.breaks} times and restarted from the last known position: "
                             "the outline may be shifted where that happened")
    world = [((f["R"] @ (f["s"] * f["Q"]).T).T + f["C"], f) for f in placed]
    hgt = np.concatenate([w[:, 1] for w, _ in world])
    cam_y = float(np.median([f["C"][1] for f in placed]))
    hist, e = np.histogram(hgt, bins=np.arange(hgt.min(), hgt.max() + 0.02, 0.02))
    c = (e[:-1] + e[1:]) / 2
    low = np.where(c < cam_y - 0.6, hist, 0)
    if low.max() < 50:
        out["review"].append("the floor was never in view: room scale unknown, measure with a tape")
        return _save(run_dir, out)
    # The floor is the LOWEST strong horizontal layer (a bed or desk top can be denser), the ceiling the highest.
    floor = float(c[np.nonzero(low >= 0.3 * low.max())[0][0]])
    k = CAMERA_HEIGHT_PRIOR_M / (cam_y - floor)  # tracker units -> metres
    high = np.where(c > cam_y + 0.3, hist, 0)
    height = (float(c[np.nonzero(high >= 0.3 * high.max())[0][-1]]) - floor) * k if high.max() > 0.2 * low.max() else None
    if height is not None and not 1.9 <= height <= 5.0:
        height = None
    # Walls between furniture height and just under the ceiling: a fixed band took in low ceilings as solid 'wall'.
    band_lo, band_hi = (min(WALL_BAND[0], height - 0.6), height - 0.3) if height else WALL_BAND
    pts = np.concatenate([w for w, _ in world])
    rel = (pts[:, 1] - floor) * k
    band = pts[(rel > band_lo) & (rel < band_hi)][:, [0, 2]] * k
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
    cont, _ = cv2.findContours(room.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # every boundary cell
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
