"""Offline check of the DA3 room method on a folder of room photos with a known reference.

  uv run python scripts/da3_room_check.py <image_dir> --ceiling 3.06    (scale from a known ceiling height)
  uv run python scripts/da3_room_check.py <image_dir> --cam-height 1.4  (scale from a known camera height)

No gravity data is assumed (up is found from the cloud). Prints the room the DA3 method measures.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import room_da3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image_dir", type=Path)
ap.add_argument("--ceiling", type=float)
ap.add_argument("--cam-height", type=float)
ap.add_argument("--max-views", type=int, default=16)
a = ap.parse_args()
imgs = sorted(p for p in a.image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
imgs = imgs if len(imgs) <= a.max_views else [imgs[round(k * (len(imgs) - 1) / (a.max_views - 1))] for k in range(a.max_views)]
z = room_da3.run_worker([str(p) for p in imgs], None)
out = room_da3.measure(z, [None] * len(imgs), a.cam_height, a.ceiling)
out.pop("polygon_m", None)
print(json.dumps(out, indent=1))
