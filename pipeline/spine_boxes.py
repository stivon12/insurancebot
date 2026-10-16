"""Stage 2b (optional): tighten spine boxes with an open, pretrained spine detector before measuring.

Why: Gemini counts and reads spines well, but its boxes on dense shelves are loose (one box can span 2-4 thin
spines), and spine thickness is measured from box width. RF-DETR-nano bookshelf v1 (Geraldine/rf-detr-nano-bookshelf,
Apache-2.0, trained by its author on Roboflow bookshelf-recognition-2; no training here) gives tight per-spine boxes
but misses thin spines, so it cannot be the counter (docs/SPINE_MODEL_CHECKS.md). Here it only REFINES: each Gemini
spine is matched one-to-one to an RF-DETR box whose centre lies inside it and that overlaps it vertically; matched
spines are measured with the tight box, unmatched ones keep the Gemini box and are flagged as loose.

v1, not v2: v2 boxes distant background books and misses foreground spines (+122 % count on held-out photos).

Runs locally on CPU (~0.3-0.7 s per frame). Skipped if `rfdetr` is not installed (uv sync --extra depth).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np

from .common import write_json

STAGE = "spine_boxes"
REPO = "Geraldine/rf-detr-nano-bookshelf"
REVISION = os.getenv("SPINE_BOX_MODEL_REV", "v1")
THRESHOLD = 0.3
TILES = 3

_model = None


def available() -> bool:
    return importlib.util.find_spec("rfdetr") is not None


def _load():
    global _model
    if _model is None:
        from huggingface_hub import hf_hub_download
        from rfdetr import RFDETRNano

        ckpt = hf_hub_download(REPO, "checkpoints/checkpoint_best_ema.pth", revision=REVISION)
        _model = RFDETRNano(pretrain_weights=ckpt)
        try:
            _model.optimize_for_inference()
        except Exception:
            pass
    return _model


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


def detect(img_rgb: np.ndarray, threshold: float = THRESHOLD, tiles: int = TILES) -> np.ndarray:
    """Spine boxes [x0,y0,x1,y1] in pixels: whole image plus overlapping vertical strips (thin spines are tiny at
    the model's 384 px input), merged by NMS."""
    from PIL import Image

    model = _load()
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


def refine(run_dir: Path, books: list[dict], metrics) -> dict:
    """Replace observation boxes with matched tight boxes in place; keep the Gemini box as box_px_gemini."""
    if not available():
        out = {"skipped": "rfdetr not installed (uv sync --extra depth)"}
        write_json(run_dir / "spine_boxes.json", out)
        return out
    with metrics.time(STAGE):
        by_frame: dict[str, list[dict]] = {}
        for b in books:
            for o in b["observations"]:
                by_frame.setdefault(o["file"], []).append(o)
        refined = total = 0
        for file, obs in by_frame.items():
            img = cv2.imread(str(run_dir / file))
            rf = detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            m = match([o["box_px"] for o in obs], rf)
            for i, o in enumerate(obs):
                total += 1
                o["box_px_gemini"] = o["box_px"]
                if i in m:
                    o["box_px"] = [float(v) for v in rf[m[i]]]
                    o["box_source"] = f"rfdetr-{REVISION}"
                    refined += 1
                else:
                    o["box_source"] = "gemini (loose)"
    out = {"model": f"{REPO}@{REVISION}", "observations": total, "refined": refined, "frames": len(by_frame)}
    write_json(run_dir / "spine_boxes.json", out)
    return out


def combine(gemini_obs: list[dict], rf: np.ndarray) -> list[dict]:
    """Rule A: every RF-DETR box is a spine (with Gemini's text when matched); unmatched Gemini spines count only if
    legible. On 15 unseen 4TU shelf photos: +4.7 % count error, against +15.9 % for Gemini alone."""
    m = match([o["box_px"] for o in gemini_obs], rf)
    by_rf = {j: i for i, j in m.items()}
    base = gemini_obs[0] if gemini_obs else {}
    out = []
    for j, box in enumerate(rf):
        g = gemini_obs[by_rf[j]] if j in by_rf else {"text": "", "legible": False, "is_book": "yes", "orientation": "vertical"}
        out.append({**{k: base[k] for k in ("frame_i", "file") if k in base}, **g, "box_px": [float(v) for v in box],
                    "box_source": f"rfdetr-{REVISION}"})
    out += [{**o, "box_source": "gemini"} for i, o in enumerate(gemini_obs) if i not in m and o.get("legible")]
    return sorted(out, key=lambda o: o["box_px"][0])
