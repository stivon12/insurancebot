"""Spine sizes without a tape: each ruler book's height measured from pixels using only the OTHER rulers in its frame
(leave-one-out, as packed in measurements.json), against its catalogue height. A proxy for the brief's 20 hand-measured
spines: catalogue heights are themselves rounded, and a ruler is a book whose box already looked right."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TOL = 0.15  # the brief's pass bar for spine dimensions


def score(measurements: dict) -> dict:
    rows = []
    for bid, m in measurements["books"].items():
        if "catalogue" in m.get("scale_method", "") and m.get("pixel_height_cm") and m.get("rulers_in_frame", 0) >= 1:
            rows.append((bid, m["spine_height_cm"], m["pixel_height_cm"], m.get("box_basis", "")))
    err = np.array([abs(p / c - 1) for _, c, p, _ in rows])
    tight = np.array(["tight" in b for *_, b in rows])
    out = {"books": len(rows), "within_15pct": int((err <= TOL).sum()), "median_error_pct": round(float(np.median(err)) * 100, 1)}
    for name, mask in (("tight_boxes", tight), ("loose_boxes", ~tight)):
        if mask.any():
            out[name] = {"books": int(mask.sum()), "within_15pct": int((err[mask] <= TOL).sum()),
                         "median_error_pct": round(float(np.median(err[mask])) * 100, 1)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="a sweep folder with measurements.json")
    print(json.dumps(score(json.loads((ap.parse_args().run / "measurements.json").read_text())), indent=1))


if __name__ == "__main__":
    main()
