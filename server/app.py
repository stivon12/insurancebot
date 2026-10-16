"""FastAPI app: serves the phone client and bridges it to Gemini Live; the sweep engine processes everything live."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import websockets
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.genai import types

from pipeline.common import LOCALES, RUNS, country as default_country, gemini, locale, now_iso, write_json
from pipeline.live import SweepEngine

from .live_config import LIVE_MODEL, build_config

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WEB = Path(__file__).resolve().parents[1] / "web"
app = FastAPI(title="Library Contents Claim Agent")
app.mount("/runs", StaticFiles(directory=RUNS), name="runs")
app.mount("/static", StaticFiles(directory=WEB), name="static")

RECONNECT_GRACE_S = float(os.getenv("LCA_RECONNECT_GRACE_S", "120"))
CONFIRM_S = 20.0
SPOKEN_YES = re.compile(r"\b(yes|yeah|yep|yup|sure|go ahead|finish it|please finish|ok finish|okay finish)\b", re.I)
BLUR_NOTE_GAP_MS = 45000
QUIET_BEFORE_OBSERVE_S = 10  # the Live model only acts on speech or text: nudge it after a quiet spell
TURN_DONE_DEG = 330


@dataclass
class Sweep:
    id: str
    dir: Path
    country: str
    device: str
    started_at: str = field(default_factory=now_iso)
    started_mono: float = field(default_factory=time.monotonic)
    phase: str = "room_turn"
    unit_label: str = ""
    frame_n: int = 0
    last_t_ms: float = 0.0
    overrides: list[dict] = field(default_factory=list)
    finished: bool = False
    ended_mono: float | None = None
    blur_since: float | None = None
    last_blur_note: float = -1e12
    last_talk: float = field(default_factory=time.monotonic)  # last user speech, agent audio or nudge
    turn_yaw: list | None = None  # [min, max] gyro heading since the room turn started
    turn_noted: bool = False
    # live connection state (outlives one WebSocket: a phone that drops reconnects to the same sweep)
    engine: SweepEngine | None = None
    live: object | None = None
    gem: asyncio.Task | None = None
    handle: str | None = None
    greeted: bool = False
    ws: object | None = None
    grace: asyncio.Task | None = None
    packet_msg: dict | None = None
    confirm_until: float = 0.0
    building: asyncio.Task | None = None
    owes_reply: bool = False  # the user spoke and the agent has not said anything since

    def write_meta(self) -> None:
        write_json(self.dir / "sweep.json", {"id": self.id, "captured_at": self.started_at, "device": self.device,
                                             "duration_s": round((self.ended_mono or time.monotonic()) - self.started_mono, 1),
                                             "country": self.country, "currency": locale(self.country)["currency"]})

    def append(self, name: str, row: dict) -> None:
        with open(self.dir / name, "a") as f:
            f.write(json.dumps(row) + "\n")

    def save_frame(self, msg: dict) -> tuple[dict, bytes]:
        data = base64.b64decode(msg["data"], validate=True)
        if len(data) > 2_500_000 or not data.startswith(b"\xff\xd8\xff"):
            raise ValueError("frame must be a JPEG under 2.5 MB")
        self.frame_n += 1
        name = f"frames/f_{self.frame_n:06d}.jpg"
        (self.dir / name).write_bytes(data)
        row = {"i": self.frame_n, "file": name, "t_ms": float(msg["t"]), "w": int(msg["w"]), "h": int(msg["h"]),
               "orient": int(msg.get("orient", 0)), "phase": self.phase, "unit": self.unit_label,
               "blur": float(msg.get("blur", 0)), "glare": float(msg.get("glare", 0))}
        self.append("frames.jsonl", row)
        self.last_t_ms = row["t_ms"]
        return {**row, "blur_bad": bool(msg.get("blur_bad"))}, data

    def save_imu(self, samples: list[dict]) -> list[dict]:
        rows = [{k: float(s.get(k) or 0) for k in ("t_ms", "ax", "ay", "az", "ra", "rb", "rg")} for s in samples[:500]]
        with open(self.dir / "imu.jsonl", "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        return rows

    def blur_note(self, row: dict) -> str | None:
        """One 'slow down' note at most every 45 s, only while reading shelves."""
        t = row["t_ms"]
        self.blur_since = (self.blur_since or t) if row["blur_bad"] else None
        if self.phase != "shelves" or self.blur_since is None or t - self.blur_since < 1500 or t - self.last_blur_note < BLUR_NOTE_GAP_MS:
            return None
        self.last_blur_note = t
        return "[capture] the shelf view has been blurred for a second or two."

    def add_override(self, args: dict) -> dict:
        ov = {**args, "t_ms": self.last_t_ms, "frame_ref": f"frames/f_{self.frame_n:06d}.jpg"}
        self.overrides.append(ov)
        write_json(self.dir / "overrides.json", self.overrides)
        return ov


SWEEPS: dict[str, Sweep] = {}


def downscale_jpeg(data: bytes, width: int = 768) -> bytes:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.shape[1] <= width:
        return data
    img = cv2.resize(img, None, fx=width / img.shape[1], fy=width / img.shape[1], interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
async def health():
    return {"live_model": LIVE_MODEL, "gemini_key": bool(os.getenv("GEMINI_API_KEY")), "country": default_country(), "countries": list(LOCALES)}


@app.post("/api/sweeps")
async def create_sweep(request: Request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex  # unguessable: the id is the only key to the photos
    d = RUNS / sid
    (d / "frames").mkdir(parents=True, exist_ok=True)
    s = Sweep(id=sid, dir=d, country=(body.get("country") or default_country()).upper(), device=(body.get("device") or "")[:200])
    s.write_meta()
    SWEEPS[sid] = s
    return {"id": sid, "country": s.country, "currency": locale(s.country)["currency"]}


@app.get("/api/sweeps/{sid}")
async def sweep_status(sid: str):
    d = RUNS / sid
    if not d.exists():
        return JSONResponse({"error": "unknown sweep"}, status_code=404)
    return {"id": sid, "packet": (d / "claim_packet.json").exists(), "report": f"/runs/{sid}/report.html"}


def packet_summary(p: dict) -> str:
    t, r, cur = p["totals"], p["room"], p["sweep"]["currency"]
    room = f"floor about {r['floor_area_m2']} square metres" if r.get("floor_area_m2") else "room size not measured (in the review list)"
    return (f"{t['book_count']} books ({t['books_identified']} identified). Books: replacement {cur} {t['books_replacement_cost']:,.2f}, "
            f"used {cur} {t['books_used_value']:,.2f}. Other contents: {cur} {t['items_replacement_cost_low']:,.2f} to "
            f"{t['items_replacement_cost_high']:,.2f} across {len(p['items'])} items; {room}. {len(p['review_queue'])} lines to review.")


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    sweep = SWEEPS.get(ws.query_params.get("sweep_id", ""))
    if not sweep:
        await ws.close(code=4404)
        return
    reattach = sweep.engine is not None
    if sweep.grace:
        sweep.grace.cancel()
        sweep.grace = None
    sweep.ws = ws

    async def send(msg: dict) -> None:
        if sweep.ws is not None:
            try:
                await sweep.ws.send_json(msg)
            except Exception:
                pass

    async def note(text: str) -> None:
        if sweep.live and not sweep.finished:
            sweep.append("transcript.jsonl", {"t_ms": sweep.last_t_ms, "who": "app", "text": text})
            try:
                await sweep.live.send_realtime_input(text=text)
            except Exception as e:
                log.warning("note not sent: %s", e)

    async def finish() -> None:
        if sweep.building:
            return
        sweep.finished, sweep.ended_mono = True, time.monotonic()
        sweep.write_meta()
        await send({"type": "finished"})

        async def build():
            await asyncio.sleep(4)  # let the agent finish its sentence, then close the voice session
            if sweep.gem:
                sweep.gem.cancel()
            try:
                packet = await sweep.engine.finish()
                sweep.packet_msg = {"type": "packet", "report": f"/runs/{sweep.id}/report.html", "json": f"/runs/{sweep.id}/claim_packet.json",
                                    "summary": packet_summary(packet), "totals": packet["totals"], "currency": packet["sweep"]["currency"],
                                    "items": len(packet["items"]), "review": len(packet["review_queue"]),
                                    "floor_area_m2": packet["room"].get("floor_area_m2")}
                await send(sweep.packet_msg)
            except Exception as e:  # report honestly; never invent a summary
                log.exception("packet failed")
                await send({"type": "error", "text": f"processing failed: {e}"})
            finally:
                await sweep.engine.close()

        sweep.building = asyncio.create_task(build())

    def turn_note(row: dict) -> str | None:
        """Once per turn: the gyro says the full circle is done."""
        h = row.get("heading")
        if sweep.phase != "room_turn" or h is None or sweep.turn_noted:
            return None
        sweep.turn_yaw = [min(sweep.turn_yaw[0], h), max(sweep.turn_yaw[1], h)] if sweep.turn_yaw else [h, h]
        if sweep.turn_yaw[1] - sweep.turn_yaw[0] < TURN_DONE_DEG:
            return None
        sweep.turn_noted = True
        return "[capture] the room turn is complete. Call log_room now, then set_phase shelves and guide them to the bookcases."

    def observe_note(row: dict) -> str | None:
        if time.monotonic() - sweep.last_talk < QUIET_BEFORE_OBSERVE_S:
            return None
        sweep.last_talk = time.monotonic()
        return "[observe]"

    def with_inventory(result: dict) -> dict:
        inv = sweep.engine.inventory()
        return {**result, "inventory": {k: inv[k] for k in ("books_seen", "books_identified", "items", "corners_marked")}}

    async def run_tool(fc) -> tuple[dict, types.FunctionResponseScheduling]:
        a, idle = dict(fc.args or {}), types.FunctionResponseScheduling.WHEN_IDLE
        silent = idle if sweep.owes_reply else types.FunctionResponseScheduling.SILENT  # a SILENT call answering speech ends the turn mute
        eng = sweep.engine
        if fc.name == "set_phase":
            sweep.phase, sweep.unit_label = a.get("phase", "pan"), a.get("unit_label", "")
            if sweep.phase == "room_turn":
                sweep.turn_yaw, sweep.turn_noted = None, False
            await send({"type": "phase", "phase": sweep.phase, "unit": sweep.unit_label})
            return {"ok": True}, silent
        if fc.name == "log_item":
            return with_inventory(await eng.log_item(a)), silent
        if fc.name == "mark_corner":
            return eng.mark_corner(), silent
        if fc.name == "log_room":
            return eng.log_room(a), silent
        if fc.name == "sync_inventory":
            return eng.inventory(), idle
        if fc.name == "user_override":
            ov = sweep.add_override(a)
            await send({"type": "status", "text": f"noted: {ov['action']} - {ov['target']}"})
            return {"recorded": True}, silent
        if fc.name == "set_locale":
            c = str(a.get("country", "")).upper()
            if c not in LOCALES:
                return {"error": f"supported: {list(LOCALES)}"}, idle
            sweep.country = eng.country = c
            sweep.write_meta()
            await send({"type": "locale", "country": c, "currency": locale(c)["currency"]})
            return {"country": c, "currency": locale(c)["currency"]}, idle
        if fc.name == "finish_sweep":  # never on the model's word alone: a tap or a clear spoken yes ends it
            sweep.confirm_until = time.monotonic() + CONFIRM_S
            await send({"type": "confirm_finish"})
            return {"status": "a Finish card is on screen: ask them to tap Finish or say yes"}, idle
        return {"error": "unknown tool"}, silent

    async def gemini_loop() -> None:
        if not os.getenv("GEMINI_API_KEY"):
            await send({"type": "error", "text": "GEMINI_API_KEY is not set on the server; frames are still recorded."})
            return
        while not sweep.finished:
            loc = locale(sweep.country)
            try:
                async with gemini().aio.live.connect(model=LIVE_MODEL, config=build_config(loc["name"], loc["currency"], sweep.handle)) as s:
                    sweep.live = s
                    if not sweep.greeted:
                        sweep.greeted = True
                        await s.send_client_content(turns=types.Content(role="user", parts=[types.Part(text="(The policyholder opened the app. Greet them.)")]), turn_complete=True)
                    while not sweep.finished:  # receive() ends at every turn_complete: keep the session, read the next turn
                        async for msg in s.receive():
                            u = msg.session_resumption_update
                            if u and u.resumable and u.new_handle:
                                sweep.handle = u.new_handle
                            if msg.go_away:
                                raise ConnectionResetError("go_away")
                            sc = msg.server_content
                            if sc:
                                if sc.interrupted:
                                    await send({"type": "interrupted"})
                                if sc.input_transcription and sc.input_transcription.text:
                                    sweep.last_talk, sweep.owes_reply = time.monotonic(), True
                                    text = sc.input_transcription.text
                                    sweep.append("transcript.jsonl", {"t_ms": sweep.last_t_ms, "who": "user", "text": text})
                                    if time.monotonic() < sweep.confirm_until and SPOKEN_YES.search(text):
                                        sweep.confirm_until = 0.0
                                        await send({"type": "confirm_cleared"})
                                        await finish()
                                    await send({"type": "transcript", "who": "user", "text": text})
                                if sc.output_transcription and sc.output_transcription.text:
                                    sweep.append("transcript.jsonl", {"t_ms": sweep.last_t_ms, "who": "agent", "text": sc.output_transcription.text})
                                    await send({"type": "transcript", "who": "agent", "text": sc.output_transcription.text})
                                for part in (sc.model_turn.parts if sc.model_turn else []) or []:
                                    if part.inline_data and part.inline_data.mime_type.startswith("audio/pcm"):
                                        sweep.last_talk, sweep.owes_reply = time.monotonic(), False
                                        await send({"type": "audio", "data": base64.b64encode(part.inline_data.data).decode()})
                                if sc.turn_complete:
                                    await send({"type": "turn_complete"})
                            if msg.tool_call:
                                responses = []
                                for fc in msg.tool_call.function_calls or []:
                                    result, sched = await run_tool(fc)
                                    responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result, scheduling=sched))
                                if responses:
                                    await s.send_tool_response(function_responses=responses)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("live session error: %s; reconnecting", e)
                await asyncio.sleep(1.0)
            finally:
                sweep.live = None

    async def client_loop() -> None:
        while True:
            msg = await ws.receive_json()
            kind, live = msg.get("type"), sweep.live
            if sweep.finished and kind in ("audio", "frame", "imu"):
                continue
            try:
                if kind == "audio" and live:
                    pcm = np.frombuffer(base64.b64decode(msg["data"]), np.int16)
                    if pcm.size and np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) > 500:  # the user is talking: not quiet
                        sweep.last_talk = time.monotonic()
                    await live.send_realtime_input(audio=types.Blob(data=base64.b64decode(msg["data"]), mime_type="audio/pcm;rate=16000"))
                elif kind == "frame":
                    row, data = await asyncio.to_thread(sweep.save_frame, msg)
                    sweep.engine.on_frame(row)
                    if live:
                        await live.send_realtime_input(video=types.Blob(data=await asyncio.to_thread(downscale_jpeg, data), mime_type="image/jpeg"))
                    hint = sweep.blur_note(row) or turn_note(row) or observe_note(row)
                    if hint:
                        await note(hint)
                elif kind == "imu":
                    sweep.engine.on_imu(await asyncio.to_thread(sweep.save_imu, msg.get("samples", [])))
                elif kind == "end":
                    sweep.confirm_until = 0.0
                    await finish()
                elif kind == "cancel_finish":
                    sweep.confirm_until = 0.0
                    await note("[capture] they tapped Keep going: carry on guiding.")
            except (ValueError, KeyError) as e:
                await send({"type": "error", "text": str(e)})
            except websockets.exceptions.ConnectionClosed as e:  # the Live session closed under us: gemini_loop reconnects
                log.warning("live send dropped: %s", e)

    if not reattach:
        sweep.engine = SweepEngine(sweep.dir, sweep.country, emit=send)
    if sweep.finished:
        await send({"type": "finished"})
        if sweep.packet_msg:
            await send(sweep.packet_msg)
    elif not sweep.gem or sweep.gem.done():
        sweep.gem = asyncio.create_task(gemini_loop())
    try:
        await client_loop()
    except WebSocketDisconnect:
        pass
    finally:
        if sweep.ws is ws:
            sweep.ws = None
            sweep.write_meta()
            if not sweep.finished:
                async def grace() -> None:
                    await asyncio.sleep(RECONNECT_GRACE_S)
                    if sweep.ws is None and not sweep.finished:
                        log.info("sweep %s: phone did not come back; building the packet", sweep.id)
                        await finish()

                sweep.grace = asyncio.create_task(grace())
