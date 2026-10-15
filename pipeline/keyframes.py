"""Stage 1: pick sharp, non-redundant keyframes per sweep phase.

Input:  runs/<id>/frames.jsonl  (one row per saved frame: i, file, t_ms, phase)
Output: runs/<id>/keyframes.json  {phase: [frame rows + sharpness]}
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .common import read_jsonl, write_json

# Window (ms) per phase: keep the sharpest frame in each window.
WINDOW_MS = {"room_turn": 700, "room_tilt": 700, "unit_overview": 1500, "shelves": 600, "pan": 1500}
DEFAULT_WINDOW_MS = 1000
# Phases where a still camera adds nothing: a frame that shows the same view as the last kept keyframe is dropped
# (the first one is kept). Room turns keep their time sampling for the room geometry.
DEDUPE_PHASES = {"shelves", "unit_overview", "pan"}
SAME_VIEW_INLIERS = 60
SAME_VIEW_SHIFT = 0.15  # fraction of the frame width
SAME_VIEW_ZOOM = 0.15


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


def drop_same_views(rows: list[dict], run_dir: Path) -> list[dict]:
    from .spines import frame_features, homography

    kept: list[dict] = []
    last = None  # (features, grey width, grey height)
    for r in rows:
        f, _, (gh, gw) = frame_features(run_dir, r["file"])
        if last is not None:
            H = homography(last[0], f, SAME_VIEW_INLIERS)
            if H is not None and same_view(H, last[1], last[2]):
                # Keep the FIRST frame of a view (already the sharpest of its window and above the blur floor), not
                # the sharpest overall: during the sweep the worker has usually read it already, and swapping in a
                # slightly sharper later frame would make it read the same shelf again.
                continue
        kept.append(r)
        last = (f, gw, gh)
    return kept


@lru_cache(maxsize=20000)
def _sharpness(path: str) -> float:
    return sharpness(Path(path), cached=False)


def sharpness(path: Path, cached: bool = True) -> float:
    """Variance of the Laplacian on a 640px-wide grey copy (higher = sharper). Cached per file: the worker
    re-selects keyframes every few seconds during the sweep."""
    if cached:
        return _sharpness(str(path))
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    scale = 640 / img.shape[1]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def select(frames: list[dict], run_dir: Path) -> dict[str, list[dict]]:
    by_phase: dict[str, list[dict]] = defaultdict(list)
    for f in frames:
        by_phase[f.get("phase") or "pan"].append(f)

    out: dict[str, list[dict]] = {}
    for phase, rows in by_phase.items():
        rows.sort(key=lambda r: r["t_ms"])
        for r in rows:
            r["sharpness"] = sharpness(run_dir / r["file"])
        floor = 0.35 * float(np.median([r["sharpness"] for r in rows])) if rows else 0.0
        window = WINDOW_MS.get(phase, DEFAULT_WINDOW_MS)
        picked: dict[int, dict] = {}
        for r in rows:
            if r["sharpness"] < floor:
                continue  # motion-blurred; a neighbour in the window will be sharper
            slot = int(r["t_ms"] // window)
            if slot not in picked or r["sharpness"] > picked[slot]["sharpness"]:
                picked[slot] = r
        out[phase] = [picked[k] for k in sorted(picked)]
        if phase in DEDUPE_PHASES:
            out[phase] = drop_same_views(out[phase], run_dir)
    return out


def run(run_dir: Path) -> dict[str, list[dict]]:
    frames = read_jsonl(run_dir / "frames.jsonl")
    keyframes = select(frames, run_dir)
    write_json(run_dir / "keyframes.json", keyframes)
    return keyframes
