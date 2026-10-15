"""Depth Anything 3 multi-view worker, run as a subprocess (adapted from the author's roomscope project).

    python -m pipeline.da3_worker <job.json> <out.npz>

job.json: {"images": [path, ...], "K": [3x3, ...] | null}
  K (optional) is the pinhole matrix of each image at its stored resolution; without it DA3 estimates it.

DA3-BASE (0.12 B, Apache-2.0) runs on all images jointly and returns per view: z-depth, confidence,
camera-to-world pose (OpenCV axes) and intrinsics, all at its processing resolution (long side 504). The
reconstruction has an arbitrary scale: metric scale is applied by the caller from the book ruler.

Run in its own process because torch and other native libraries can clash over OpenMP on macOS, and to free
the model's memory as soon as it is done. DA3 imports export/UI modules this project never calls; they are
stubbed so they need not be installed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

_UNUSED = {"pycolmap", "gradio", "typer", "gsplat", "xformers", "open3d", "sklearn"}


class _Stub(types.ModuleType):
    __path__: list = []

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return type(k, (), {"__init__": lambda s, *a, **kw: None})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in _UNUSED:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubFinder())

MODEL_ID = "depth-anything/DA3-BASE"
PROCESS_RES = 504


def main(job_path: str, out_path: str) -> None:
    import torch
    from depth_anything_3.api import DepthAnything3
    from PIL import Image

    job = json.loads(Path(job_path).read_text())
    paths, Ks = job["images"], job.get("K")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    t0 = time.time()
    model = DepthAnything3.from_pretrained(MODEL_ID).to(dev).eval()
    kw = {"intrinsics": np.asarray(Ks, np.float32)} if Ks else {}
    with torch.no_grad():
        pred = model.inference(imgs, process_res=PROCESS_RES, **kw)
    out = {"n": len(paths)}
    for i, im in enumerate(imgs):
        d = np.asarray(pred.depth[i], np.float32)
        h, w = d.shape
        H0, W0 = im.shape[:2]
        E = np.eye(4)
        E[:3, :4] = np.asarray(pred.extrinsics[i])[:3, :4]
        if Ks:
            K = np.asarray(Ks[i], np.float64).copy()
            K[0] *= w / W0
            K[1] *= h / H0
        else:
            K = np.asarray(pred.intrinsics[i], np.float64)
        out[f"depth_{i}"] = d
        out[f"conf_{i}"] = np.asarray(pred.conf[i], np.float32)
        out[f"pose_{i}"] = np.linalg.inv(E)  # world-to-camera -> camera-to-world
        out[f"K_{i}"] = K
    out["seconds"] = time.time() - t0
    out["device"] = dev
    np.savez_compressed(out_path, **out)
    print(f"da3 multiview: {len(paths)} images in {out['seconds']:.1f}s on {dev}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
