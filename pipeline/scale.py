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

Height: the book's own catalogue height (Google Books, Open Library, Library of Congress) when its spine shape fits
that edition; otherwise pixels x the ruler/format scale. Thickness: catalogue thickness, else the scale-free pixel
ratio (short/long side) x height fused with a page-count estimate; disagreement over 20 % goes to review.

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
# Thickness from page count: paper caliper ~0.0057 cm/page (white) to ~0.0064 (cream), Amazon KDP spine formula;
# real stock varies (300-530 pages per inch), so this estimate is +-20-25 % on its own. Plus covers.
CM_PER_PAGE = 0.0060
COVER_CM = 0.1
PAGE_REL_SD = 0.22
PIXEL_REL_SD = 0.12  # ~+-3 px on a ~50 px spine; thin spines are worse (see _pixel_sd)
THICK_DISAGREE = 0.20
OTHER_EDITION_TOL = 0.12


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


def ruler_books(books: list[dict], identified: dict[str, dict], other_editions: bool = True) -> dict[str, float]:
    """book_id -> catalogue height (cm) for identified books whose spine shape matches the catalogue edition."""
    out = {}
    for b in books:
        idn = identified.get(b["id"], {})
        dims = idn.get("dimensions") if idn.get("status") == "identified" else None
        if not dims or not dims.get("height_cm"):
            continue
        if not other_editions and "another edition" in dims.get("edition_basis", ""):
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


def _pixel_sd(thick_px: float) -> float:
    """Relative error of a pixel thickness: about 3 px of box error, never better than PIXEL_REL_SD."""
    return max(PIXEL_REL_SD, 3.0 / thick_px) if thick_px > 0 else 1.0


def thickness(height_cm: float, ratio: float | None, thick_px: float, pages: int | None, catalogue: float | None) -> tuple[float | None, str, str | None]:
    """(thickness cm, basis, review note). Catalogue thickness first; otherwise the scale-free pixel ratio
    (short/long side x height) fused with the page-count estimate by inverse variance."""
    if catalogue and 0.2 <= catalogue <= 15:
        return round(catalogue, 2), "catalogue", None
    est = []
    if ratio:
        est.append((height_cm * ratio, _pixel_sd(thick_px), "pixel ratio x height"))
    if pages and 20 <= pages <= 3000:
        est.append((pages * CM_PER_PAGE + COVER_CM, PAGE_REL_SD, f"{pages} pages x paper caliper"))
    if not est:
        return None, "none", None
    if len(est) == 1:
        return round(est[0][0], 2), est[0][2], None
    (a, sa, la), (b, sb, lb) = est
    wa, wb = 1 / sa**2, 1 / sb**2
    fused = (a * wa + b * wb) / (wa + wb)
    note = None
    if abs(a / b - 1) > THICK_DISAGREE:
        note = f"spine thickness sources disagree: {la} {a:.1f} cm vs {lb} {b:.1f} cm (reported {fused:.1f} cm, weighted)"
    return round(fused, 2), f"{la} + {lb} (weighted)", note


def measure(books: list[dict], identified: dict[str, dict]) -> dict:
    rulers = ruler_books(books, identified)
    # Only a book's own edition sets the scale for its neighbours; another edition's height may be a different format.
    per_frame = frame_scales(books, ruler_books(books, identified, other_editions=False))
    px_heights_by_frame: dict[int, list[float]] = defaultdict(list)
    for b in books:
        for o in b["observations"]:
            px_heights_by_frame[o["frame_i"]].append(spine_px(o["box_px"])[0])

    results: dict[str, dict] = {}
    for b in books:
        obs = []  # per observation: long px, short px, cm/px (or None), method, rulers, tight box
        for o in b["observations"]:
            long_px, short_px = spine_px(o["box_px"])
            if long_px <= 0:
                continue
            others = [s for bid, s in per_frame.get(o["frame_i"], []) if bid != b["id"]]
            scale, n = robust_median(others) if len(others) >= 2 else (None, 0)
            method = "book_ruler"
            if scale is None:
                scale, method = format_prior_scale(px_heights_by_frame[o["frame_i"]]), "standard_format_prior"
            obs.append({"long": long_px, "short": short_px, "scale": scale, "method": method if scale else None, "n": n,
                        "tight": str(o.get("box_source", "")).startswith("rfdetr")})
        idn = identified.get(b["id"], {})
        ok = idn.get("status") == "identified"
        dims = (idn.get("dimensions") or {}) if ok else {}
        pages = (dims.get("pages") or (idn.get("match") or {}).get("pages")) if ok else None
        if not obs or (b["id"] not in rulers and not any(x["scale"] for x in obs)):
            results[b["id"]] = {"spine_height_cm": None, "spine_thickness_cm": None, "scale_method": "none", "frames_used": 0}
            continue
        scaled = [x for x in obs if x["scale"]]
        use = [x for x in scaled if x["method"] == "book_ruler"] or scaled or obs  # prefer frames with the book ruler
        if any(x["tight"] for x in use):
            use = [x for x in use if x["tight"]]  # prefer tight detector boxes over loose model boxes
        px_h = float(np.median([x["long"] * x["scale"] for x in use])) if all(x["scale"] for x in use) else None
        px_t = float(np.median([x["short"] * x["scale"] for x in use])) if all(x["scale"] for x in use) else None
        # The book's own catalogue height when it fits what is seen (ruler_books' aspect check); the pixel
        # measurement is kept alongside for checking.
        other_edition = "another edition" in dims.get("edition_basis", "")
        if b["id"] in rulers and other_edition and (not px_h or abs(rulers[b["id"]] / px_h - 1) > OTHER_EDITION_TOL):
            # Another edition's size (hardback 24 cm vs paperback 19.8 cm) is used only when the pixels agree.
            h, method = px_h, ("book_ruler" if use[0]["method"] == "book_ruler" else "standard_format_prior")
        elif b["id"] in rulers:
            h, method = rulers[b["id"]], f"catalogue height ({dims.get('source', 'catalogue')}; {dims.get('edition_basis', '')})"
        else:
            h, method = px_h, ("book_ruler" if use[0]["method"] == "book_ruler" else "standard_format_prior")
        ratio = float(np.median([x["short"] / x["long"] for x in use]))  # scale-free
        thick_px = float(np.median([x["short"] for x in use]))
        t, t_basis, t_note = thickness(h, ratio, thick_px, pages, dims.get("thickness_cm") if b["id"] in rulers else None)
        results[b["id"]] = {
            "spine_height_cm": round(float(h), 1),
            "spine_thickness_cm": t,
            "scale_method": method,
            "thickness_basis": t_basis,
            "pixel_height_cm": round(px_h, 1) if px_h else None,
            "pixel_thickness_cm": round(px_t, 2) if px_t else None,
            "rulers_in_frame": int(max(x["n"] for x in use)),
            "frames_used": len(use),
            "box_basis": "tight (RF-DETR)" if any(x["tight"] for x in use) else "loose (Gemini)",
            **({"review_note": t_note} if t_note else {}),
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
