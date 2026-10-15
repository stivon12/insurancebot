"""Stage 4: metric spine dimensions, using identified books as the ruler.

Metric scale source: catalogue heights of identified editions (Open Library /
Google Books physical dimensions). In each frame, the centimetres-per-pixel at
the shelf plane is the median of catalogue_height / pixel_height over the
identified books visible in that frame, after rejecting outliers more than 8%
from the median. Each book is measured with a scale computed WITHOUT its own
catalogue height (leave-one-out), so a book never measures itself.

Fallback when a frame has fewer than 2 other ruler books: fit the scale that best
snaps the frame's spine heights onto standard trade formats (low confidence,
routed to review).

Output: runs/<id>/measurements.json
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import write_json

STAGE = "scale"
OUTLIER_TOL = 0.08
ASPECT_TOL = 0.4
# Common book heights in cm: A-format, B-format, US trade, Demy, Royal, US hardcover, large hardcover.
STANDARD_HEIGHTS_CM = [17.8, 19.8, 21.0, 21.6, 23.4, 24.0, 25.4, 28.0]


def spine_px(box: list[float]) -> tuple[float, float]:
    """(long side, short side) in pixels: a spine's height and thickness whatever its orientation."""
    w, h = box[2] - box[0], box[3] - box[1]
    return max(w, h), min(w, h)


def robust_median(values: list[float], tol: float = OUTLIER_TOL) -> tuple[float | None, int]:
    if not values:
        return None, 0
    m = float(np.median(values))
    kept = [v for v in values if abs(v / m - 1) <= tol]
    return (float(np.median(kept)), len(kept)) if kept else (m, len(values))


def format_prior_scale(px_heights: list[float]) -> float | None:
    """Scale (cm/px) that best snaps observed spine heights to standard formats."""
    if len(px_heights) < 3:
        return None
    med = float(np.median(px_heights))
    best, best_cost = None, float("inf")
    for s in np.linspace(14.0 / med, 30.0 / med, 600):
        cm = np.array(px_heights) * s
        cost = float(np.median([min(abs(c - f) / f for f in STANDARD_HEIGHTS_CM) for c in cm]))
        if cost < best_cost:
            best, best_cost = float(s), cost
    return best


def ruler_books(books: list[dict], identified: dict[str, dict]) -> dict[str, float]:
    """book_id -> catalogue height (cm) for identified books whose spine shape matches the catalogue edition."""
    out = {}
    for b in books:
        idn = identified.get(b["id"], {})
        dims = idn.get("dimensions") if idn.get("status") == "identified" else None
        if not dims or not dims.get("height_cm"):
            continue
        if dims.get("thickness_cm"):
            long_px, short_px = spine_px(b["best"]["box_px"])
            expected = dims["thickness_cm"] / dims["height_cm"]
            observed = short_px / long_px if long_px else 0
            if not observed or max(observed / expected, expected / observed) - 1 > ASPECT_TOL:
                continue  # probably a different edition (e.g. hardback vs paperback): not a reliable ruler
        out[b["id"]] = float(dims["height_cm"])
    return out


def frame_scales(books: list[dict], rulers: dict[str, float]) -> dict[int, list[tuple[str, float]]]:
    """frame_i -> [(ruler book id, cm/px from that book)]."""
    per_frame: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for b in books:
        if b["id"] not in rulers:
            continue
        for o in b["observations"]:
            long_px, _ = spine_px(o["box_px"])
            if long_px > 0:
                per_frame[o["frame_i"]].append((b["id"], rulers[b["id"]] / long_px))
    return per_frame


def measure(books: list[dict], identified: dict[str, dict]) -> dict:
    rulers = ruler_books(books, identified)
    per_frame = frame_scales(books, rulers)
    px_heights_by_frame: dict[int, list[float]] = defaultdict(list)
    for b in books:
        for o in b["observations"]:
            px_heights_by_frame[o["frame_i"]].append(spine_px(o["box_px"])[0])

    results: dict[str, dict] = {}
    for b in books:
        heights, thicks, methods, n_rulers, tight = [], [], [], [], []
        for o in b["observations"]:
            others = [s for bid, s in per_frame.get(o["frame_i"], []) if bid != b["id"]]
            scale, n = robust_median(others) if len(others) >= 2 else (None, 0)
            method = "book_ruler"
            if scale is None:
                scale = format_prior_scale(px_heights_by_frame[o["frame_i"]])
                method = "standard_format_prior"
            if scale is None:
                continue
            long_px, short_px = spine_px(o["box_px"])
            heights.append(long_px * scale)
            thicks.append(short_px * scale)
            methods.append(method)
            n_rulers.append(n)
            tight.append(str(o.get("box_source", "")).startswith("rfdetr"))
        if not heights:
            results[b["id"]] = {"spine_height_cm": None, "spine_thickness_cm": None, "scale_method": "none", "frames_used": 0}
            continue
        ruler_idx = [i for i, m in enumerate(methods) if m == "book_ruler"]
        use = ruler_idx or list(range(len(heights)))  # prefer frames measured with the book ruler
        if any(tight[i] for i in use):
            use = [i for i in use if tight[i]]  # prefer tight detector boxes over loose model boxes
        results[b["id"]] = {
            "spine_height_cm": round(float(np.median([heights[i] for i in use])), 1),
            "spine_thickness_cm": round(float(np.median([thicks[i] for i in use])), 2),
            "scale_method": "book_ruler" if ruler_idx else "standard_format_prior",
            "rulers_in_frame": int(max(n_rulers[i] for i in use)),
            "frames_used": len(use),
            "box_basis": "tight (RF-DETR)" if any(tight[i] for i in use) else "loose (Gemini)",
        }
    return {"books": results, "ruler_count": len(rulers), "shelves": shelf_runs(books, results)}


def shelf_runs(books: list[dict], dims: dict[str, dict]) -> dict[str, float]:
    """Occupied shelf run per shelf (m): sum of upright thicknesses; a flat stack occupies its longest book."""
    runs: dict[str, float] = defaultdict(float)
    by_shelf: dict[str, list[dict]] = defaultdict(list)
    for b in books:
        by_shelf[b["shelf"]].append(b)
    for shelf, bs in by_shelf.items():
        stack: list[float] = []
        for b in sorted(bs, key=lambda b: b["position"]):
            d = dims.get(b["id"], {})
            if b["best"].get("orientation") == "horizontal":
                if d.get("spine_height_cm"):
                    stack.append(d["spine_height_cm"])
                continue
            if stack:
                runs[shelf] += max(stack)
                stack = []
            runs[shelf] += d.get("spine_thickness_cm") or 0.0
        if stack:
            runs[shelf] += max(stack)
    return {k: round(v / 100, 3) for k, v in runs.items()}


def run(run_dir: Path, books: list[dict], identified: dict[str, dict], metrics) -> dict:
    with metrics.time(STAGE):
        out = measure(books, identified)
    write_json(run_dir / "measurements.json", out)
    return out
