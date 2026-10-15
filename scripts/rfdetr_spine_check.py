"""Count check of a pretrained open spine detector (Geraldine/rf-detr-nano-bookshelf, Apache-2.0) on the same
Library Dataset photos used for the Gemini count check.

Use revision v1 only for a fair score: v2 was distilled on 285 photos of llabres/library-dataset itself.

  uv pip install rfdetr supervision   # not a project dependency (experiment)
  uv run python scripts/rfdetr_spine_check.py --runs runs/libds_count --rev v1 --out runs/rfdetr_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.5) -> list[int]:
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        o = inter / np.minimum(area(boxes[i]), area(boxes[order[1:]]))  # overlap of the smaller box
        order = order[1:][o < iou]
    return keep


def detect(model, img_rgb: np.ndarray, threshold: float, tiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole image plus `tiles` overlapping vertical strips (thin spines are tiny at 384 px), merged by NMS."""
    from PIL import Image

    H, W = img_rgb.shape[:2]
    boxes, scores = [], []
    windows = [(0, W)]
    if tiles > 1:
        step = W / (tiles - (tiles - 1) * 0.25)
        windows += [(int(k * step * 0.75), int(min(W, k * step * 0.75 + step))) for k in range(tiles)]
    for x0, x1 in windows:
        det = model.predict(Image.fromarray(img_rgb[:, x0:x1]), threshold=threshold)
        if len(det):
            b = det.xyxy.copy()
            b[:, [0, 2]] += x0
            boxes.append(b)
            scores.append(det.confidence)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0)
    B, S = np.concatenate(boxes), np.concatenate(scores)
    k = nms(B, S)
    return B[k], S[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs/libds_count"), help="Gemini count run: same photos + truth counts")
    ap.add_argument("--rev", default="v1")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--tiles", type=int, default=3)
    ap.add_argument("--min-rel-height", type=float, default=0.0,
                    help="drop boxes shorter than this fraction of the tallest-quartile box height (background books)")
    ap.add_argument("--out", type=Path, default=Path("runs/rfdetr_v1"))
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from rfdetr import RFDETRNano

    ckpt = hf_hub_download("Geraldine/rf-detr-nano-bookshelf", "checkpoints/checkpoint_best_ema.pth", revision=a.rev)
    model = RFDETRNano(pretrain_weights=ckpt)
    try:
        model.optimize_for_inference()
    except Exception:
        pass
    ref = json.loads((a.runs / "summary.json").read_text())["per_image"]
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in ref:
        img = cv2.imread(str(a.runs / Path(r["image"]).stem / "frames" / "f_000001.jpg"))
        t = time.perf_counter()
        B, S = detect(model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), a.threshold, a.tiles)
        if a.min_rel_height and len(B):
            h = B[:, 3] - B[:, 1]
            B = B[h >= a.min_rel_height * np.percentile(h, 75)]
        dt = time.perf_counter() - t
        widths = (B[:, 2] - B[:, 0]) if len(B) else np.zeros(0)
        rows.append({"image": r["image"], "true": r["true_count"], "rfdetr": len(B), "gemini_flash": r["pred_count"], "seconds": round(dt, 2),
                     "median_box_width_px": float(np.median(widths)) if len(widths) else None})
        vis = img.copy()
        for x0, y0, x1, y1 in B.astype(int):
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)
        f = 1400 / vis.shape[1]
        cv2.imwrite(str(a.out / f"{Path(r['image']).stem}.jpg"), cv2.resize(vis, None, fx=f, fy=f))
        print(f"{r['image']}: rfdetr {len(B)} | gemini {r['pred_count']} | true {r['true_count']} | {dt:.1f}s", flush=True)
    T = sum(x["true"] for x in rows)
    for k in ("rfdetr", "gemini_flash"):
        P = sum(x[k] for x in rows)
        exact = sum(x[k] == x["true"] for x in rows)
        pm1 = sum(abs(x[k] - x["true"]) <= 1 for x in rows)
        print(f"{k}: {P}/{T} ({(P - T) / T:+.1%}) | exact {exact}/{len(rows)} | within +-1 {pm1}/{len(rows)}")
    print(f"rfdetr mean seconds/photo {np.mean([x['seconds'] for x in rows]):.2f} (CPU/MPS, tiles={a.tiles})")
    (a.out / "summary.json").write_text(json.dumps({"rev": a.rev, "threshold": a.threshold, "tiles": a.tiles, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
