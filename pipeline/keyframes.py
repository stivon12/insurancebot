"""Stage 1: pick sharp, non-redundant keyframes per sweep phase.

Input:  runs/<id>/frames.jsonl  (one row per saved frame: i, file, t_ms, phase)
Output: runs/<id>/keyframes.json  {phase: [frame rows + sharpness]}
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .common import read_jsonl, write_json

# Window (ms) per phase: keep the sharpest frame in each window.
WINDOW_MS = {"room_turn": 700, "room_tilt": 700, "unit_overview": 1500, "shelves": 600, "pan": 1500}
DEFAULT_WINDOW_MS = 1000


def sharpness(path: Path) -> float:
    """Variance of the Laplacian on a 640px-wide grey copy (higher = sharper)."""
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
    return out


def run(run_dir: Path) -> dict[str, list[dict]]:
    frames = read_jsonl(run_dir / "frames.jsonl")
    keyframes = select(frames, run_dir)
    write_json(run_dir / "keyframes.json", keyframes)
    return keyframes
