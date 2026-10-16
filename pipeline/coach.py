"""Capture coaching: turn weak intermediate results from the live worker into short [capture] notes for the agent.

The worker reads shelves, identifies books and checks the room turn while the sweep is still running. When those
results show a capture problem that the policyholder can fix on the spot (spines too small or blurred to read,
the camera on something other than books, an incomplete room turn), a note is sent to the voice agent, which
asks for a better view. Each note names what was seen and what to ask for; the agent's prompt lists the same
failure cases. Pure functions over worker state, so they are unit-tested without a model.
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import geometry as G

MIN_GAP_S = 12  # at most one note every 12 s, so the agent is not flooded
SAME_NOTE_GAP_S = 45  # and the same kind of note at most every 45 s


def shelf_notes(recent: list[dict]) -> list[tuple[str, str]]:
    """recent: the last few shelf keyframes read, newest last: [{"spines": n, "legible": n, "rows": n, "unit": str}]."""
    out = []
    if len(recent) < 2:
        return out
    last = recent[-3:]
    spines = sum(r["spines"] for r in last)
    legible = sum(r["legible"] for r in last)
    if all(r["rows"] == 0 for r in last):
        out.append(("no_books", "[capture] the last few shelf views show no books. If they have moved past the bookcase, "
                    "call set_phase with pan; if not, ask them to point the phone squarely at the shelf."))
    elif spines >= 8 and legible / spines < 0.35:
        out.append(("unreadable", f"[capture] only {legible} of {spines} spines in the last shelf views were readable. Ask "
                    "them to step closer (about an arm's length), hold the phone level facing the spines, and move more "
                    "slowly along this shelf."))
    return out


def identify_notes(recent_verdicts: list[dict]) -> list[tuple[str, str]]:
    """recent_verdicts: the last spine readings resolved (newest last), each {"legible": bool, "identified": bool}."""
    last = recent_verdicts[-8:]
    legible = [v for v in last if v["legible"]]
    if len(legible) >= 6 and sum(v["identified"] for v in legible) / len(legible) < 0.3:
        return [("unmatched", "[capture] spines are being read but most titles do not match a catalogue record, often "
                 "because only part of the title or no author is visible. Ask them to pause on each section for a "
                 "second so the full spine, top to bottom, is in view.")]
    return []


def room_turn_notes(turn_frames: list[dict], imu: list[dict], phase_now: str) -> list[tuple[str, str]]:
    """After the room turn: was it a full circle, with enough sharp frames?"""
    if phase_now in ("room_turn", "") or not turn_frames or len(imu) < 50:
        return []
    t0, t1 = min(f["t_ms"] for f in turn_frames), max(f["t_ms"] for f in turn_frames)
    t, yaw = G.integrate_yaw(imu)
    m = (t >= t0) & (t <= t1)
    covered = math.degrees(float(np.ptp(yaw[m]))) if m.sum() > 2 else 0.0
    if covered < 300:
        return [("turn", f"[capture] the room turn covered only about {covered:.0f} degrees, so the room size cannot be "
                 "measured well. When convenient, ask them to stand in the middle of the room and turn slowly through a "
                 "full circle once more, with the floor line along the walls in view (call set_phase room_turn first).")]
    if len(turn_frames) < 8:
        return [("turn_sharp", "[capture] few sharp frames from the room turn. Ask them to turn more slowly and keep "
                 "the phone steady, then call set_phase room_turn and repeat the turn.")]
    return []


class Coach:
    """Rate-limits notes and remembers which were sent."""

    def __init__(self, send):
        self.send = send  # async callable(text)
        self.last_any = 0.0
        self.last_kind: dict[str, float] = {}
        self.sent: list[dict] = []

    async def offer(self, notes: list[tuple[str, str]]) -> None:
        now = time.monotonic()
        for kind, text in notes:
            if now - self.last_any < MIN_GAP_S or now - self.last_kind.get(kind, -1e9) < SAME_NOTE_GAP_S:
                continue
            self.last_any = self.last_kind[kind] = now
            self.sent.append({"kind": kind, "text": text})
            await self.send(text)
            return  # one note at a time
