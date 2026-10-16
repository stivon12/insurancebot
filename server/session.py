"""Sweep recorder: saves every frame, IMU sample, override and transcript line to runs/<sweep_id>/."""

from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline.common import RUNS, locale, now_iso, write_json

MAX_FRAME_BYTES = 2_500_000


@dataclass
class Sweep:
    id: str
    dir: Path
    country: str
    device: str
    posture: str = "unknown"  # standing / seated: sets the camera-height prior when no book ruler is available
    started_at: str = field(default_factory=now_iso)
    started_mono: float = field(default_factory=time.monotonic)
    phase: str = "pan"
    unit_label: str = ""
    frame_n: int = 0
    last_t_ms: float = 0.0
    overrides: list[dict] = field(default_factory=list)
    live_items: list[dict] = field(default_factory=list)
    recaptures: list[dict] = field(default_factory=list)
    finished: bool = False
    ended_mono: float | None = None
    # capture-quality monitor state (t_ms when a bad streak started; last hint time)
    blur_since: float | None = None
    glare_since: float | None = None
    last_hint_ms: float = -1e9

    @classmethod
    def create(cls, country: str, device: str, posture: str = "unknown") -> "Sweep":
        # The id is the only key to this sweep's photos and packet (served under /runs): 128 random bits, not
        # guessable from the timestamp.
        sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex
        d = RUNS / sid
        (d / "frames").mkdir(parents=True, exist_ok=True)
        s = cls(id=sid, dir=d, country=country, device=device, posture=posture)
        s.write_meta()
        return s

    def write_meta(self) -> None:
        loc = locale(self.country)
        write_json(
            self.dir / "sweep.json",
            {
                "id": self.id,
                "captured_at": self.started_at,
                "device": self.device,
                "duration_s": round((self.ended_mono or time.monotonic()) - self.started_mono, 1),
                "country": self.country,
                "currency": loc["currency"],
                "posture": self.posture,
            },
        )

    def _append(self, name: str, row: dict) -> None:
        with open(self.dir / name, "a") as f:
            f.write(json.dumps(row) + "\n")

    def save_frame(self, msg: dict) -> tuple[dict, bytes]:
        data = base64.b64decode(msg["data"], validate=True)
        if len(data) > MAX_FRAME_BYTES or not data.startswith(b"\xff\xd8\xff"):
            raise ValueError("frame must be a JPEG under 2.5 MB")
        self.frame_n += 1
        name = f"frames/f_{self.frame_n:06d}.jpg"
        (self.dir / name).write_bytes(data)
        row = {
            "i": self.frame_n,
            "file": name,
            "t_ms": float(msg["t"]),
            "w": int(msg["w"]),
            "h": int(msg["h"]),
            "orient": int(msg.get("orient", 0)),
            "phase": self.phase,
            "unit": self.unit_label,
            "blur": float(msg.get("blur", 0)),
            "glare": float(msg.get("glare", 0)),
        }
        self._append("frames.jsonl", row)
        self.last_t_ms = row["t_ms"]
        return row, data

    def save_imu(self, samples: list[dict]) -> None:
        with open(self.dir / "imu.jsonl", "a") as f:
            for s in samples[:500]:
                f.write(json.dumps({k: float(s.get(k) or 0) for k in ("t_ms", "ax", "ay", "az", "ra", "rb", "rg")}) + "\n")

    def transcript(self, who: str, text: str) -> None:
        self._append("transcript.jsonl", {"t_ms": self.last_t_ms, "who": who, "text": text})

    def add_override(self, args: dict) -> dict:
        ov = {**args, "t_ms": self.last_t_ms, "frame_ref": f"frames/f_{self.frame_n:06d}.jpg"}
        self.overrides.append(ov)
        write_json(self.dir / "overrides.json", self.overrides)
        return ov

    def quality_hint(self, row: dict, blur_bad: bool, glare_bad: bool) -> str | None:
        """Turn per-frame camera checks into at most one spoken-hint request every 8 seconds."""
        t = row["t_ms"]
        self.blur_since = (self.blur_since or t) if blur_bad else None
        self.glare_since = (self.glare_since or t) if glare_bad else None
        if t - self.last_hint_ms < 8000:
            return None
        hint = None
        if self.blur_since is not None and t - self.blur_since > 1200:
            hint = "[capture] the last second of video is motion-blurred; ask them to slow down or hold steadier."
        elif self.glare_since is not None and t - self.glare_since > 1200:
            hint = "[capture] strong glare or overexposure in view; ask them to tilt or shift to avoid it."
        if hint:
            self.last_hint_ms = t
        return hint


def downscale_jpeg(data: bytes, width: int = 768) -> bytes:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return data
    if img.shape[1] > width:
        s = width / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
