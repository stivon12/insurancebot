"""FastAPI app: serves the phone web client and bridges it to Gemini Live.

Browser  --WS /ws/live-->  this server  --Live API-->  Gemini
  audio (16 kHz PCM16), frames (JPEG, ~3 fps saved, ~1 fps forwarded), IMU, text
The two-loop bridge (client->Gemini, Gemini->client) and non-blocking tools follow the
pattern of the Insurance Claim Live Agent Team reference; the tools, recorder and
post-sweep pipeline are this project's own.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.genai import types
from rapidfuzz import fuzz

from pipeline.common import LOCALES, RUNS, country as default_country, gemini, locale
from pipeline.live_worker import SweepWorker
from pipeline.run import process

from .live_config import LIVE_MODEL, build_config
from .session import Sweep, downscale_jpeg

log = logging.getLogger("live")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WEB = Path(__file__).resolve().parents[1] / "web"
app = FastAPI(title="Library Contents Claim Agent")
app.mount("/runs", StaticFiles(directory=RUNS), name="runs")
app.mount("/static", StaticFiles(directory=WEB), name="static")

SWEEPS: dict[str, Sweep] = {}
# Test-only: scripts/replay_sweep.py replays a recorded sweep; frames then carry their recorded phase, and the voice
# agent can be left off (?live=0). Never set this for a real sweep.
REPLAY = os.getenv("LCA_ALLOW_REPLAY") == "1"


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
async def health():
    return {
        "live_model": LIVE_MODEL,
        "gemini_key": bool(os.getenv("GEMINI_API_KEY")),
        "ebay_keys": bool(os.getenv("EBAY_CLIENT_ID") and os.getenv("EBAY_CLIENT_SECRET")),
        "country": default_country(),
        "countries": list(LOCALES),
    }


@app.post("/api/sweeps")
async def create_sweep(request: Request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    device = (body.get("device") or request.headers.get("user-agent", ""))[:200]
    posture = body.get("posture") if body.get("posture") in ("standing", "seated") else "unknown"
    s = Sweep.create(country=(body.get("country") or default_country()).upper(), device=device, posture=posture)
    SWEEPS[s.id] = s
    return {"id": s.id, "country": s.country, "currency": locale(s.country)["currency"]}


@app.get("/api/sweeps/{sid}")
async def sweep_status(sid: str):
    d = RUNS / sid
    if not d.exists():
        return JSONResponse({"error": "unknown sweep"}, status_code=404)
    return {"id": sid, "packet": (d / "claim_packet.json").exists(), "report": f"/runs/{sid}/report.html"}


def is_repeat(item: dict, logged: list[dict]) -> bool:
    """Same kind and nearly the same label as something already in the live preview list."""
    label = str(item.get("label", "")).lower().strip()
    return any(
        it.get("kind") == item.get("kind") and fuzz.token_sort_ratio(label, str(it.get("label", "")).lower()) >= 80
        for it in logged
    )


@dataclass
class LiveState:
    live: object | None = None
    closed: bool = False
    pipeline: asyncio.Task | None = None
    worker: SweepWorker | None = None
    gem: asyncio.Task | None = None
    turn_done: asyncio.Event | None = None
    gen: int = 0  # bumped on every finish/resume, so a delayed close from an earlier finish does nothing
    handle: str | None = None  # Live session-resumption handle (kept across a finish so Continue can reconnect)
    greeted: bool = False
    resume_note: str | None = None


def packet_summary(p: dict) -> str:
    t, r, cur = p["totals"], p["room"], p["sweep"]["currency"]
    room = (
        f"room about {r['length_m']} by {r['width_m']} metres, floor {r['floor_area_m2']} square metres"
        if r.get("floor_area_m2")
        else "room size could not be measured and is in the review list"
    )
    return (
        f"{t['book_count']} books: {t['books_identified']} identified, {t['books_unidentified']} unreadable, "
        f"{t['books_needs_appraisal']} sent for appraisal. Priced books: replacement {cur} {t['books_replacement_cost']:,.2f}, "
        f"used value {cur} {t['books_used_value']:,.2f}. Other contents: {cur} {t['items_replacement_cost_low']:,.2f} to "
        f"{t['items_replacement_cost_high']:,.2f} across {len(p['items'])} items. Shelf run {t['shelf_run_m']} metres; {room}. "
        f"{len(p['review_queue'])} lines need a human check."
    )


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    sweep = SWEEPS.get(ws.query_params.get("sweep_id", ""))
    if not sweep:
        await ws.close(code=4404)
        return
    state = LiveState()

    async def send(msg: dict) -> None:
        try:
            await ws.send_json(msg)
        except Exception:
            pass

    async def finish() -> None:
        if state.pipeline:
            return
        sweep.finished = True
        sweep.ended_mono = time.monotonic()
        sweep.write_meta()
        # The phone stops its camera, mic and sensors on this message; nothing more is recorded or sent to the model.
        await send({"type": "finished"})

        state.gen += 1
        gen = state.gen

        async def close_live():
            # Let the agent finish its current sentence (e.g. "processing now"), then close the Live session so no
            # model session stays open while the packet is built. The result is shown on screen, not spoken.
            state.turn_done = asyncio.Event()
            try:
                await asyncio.wait_for(state.turn_done.wait(), timeout=6)
            except asyncio.TimeoutError:
                pass
            if state.gen != gen:
                return  # the sweep was continued in the meantime
            state.closed = True
            if state.gem:
                state.gem.cancel()
            log.info("sweep %s finished: live session closed", sweep.id)

        asyncio.create_task(close_live())

        async def go():
            async def progress(snap: dict):
                await send({"type": "progress", **snap})

            try:
                metrics = None
                if state.worker:
                    await state.worker.stop()  # no new work; jobs already started finish and are reused below
                    metrics = state.worker.metrics
                packet = await process(sweep.dir, progress, sweep.country, metrics)
                summary = packet_summary(packet)
                await send({
                    "type": "packet", "report": f"/runs/{sweep.id}/report.html", "json": f"/runs/{sweep.id}/claim_packet.json",
                    "summary": summary, "totals": packet["totals"], "currency": packet["sweep"]["currency"],
                    "items": len(packet["items"]), "review": len(packet["review_queue"]),
                    "floor_area_m2": packet["room"].get("floor_area_m2"),
                })
            except Exception as e:  # report honestly; never invent a summary
                log.exception("pipeline failed")
                await send({"type": "error", "text": f"processing failed: {e}"})
            finally:
                if state.worker:
                    await state.worker.close()

        state.pipeline = asyncio.create_task(go())

    async def coach_agent(text: str) -> None:
        """A [capture] note from the live worker (weak reads, empty shelves, incomplete room turn) to the voice agent."""
        if state.live and not sweep.finished:
            log.info("coach -> agent: %s", text)
            sweep.transcript("coach", text)
            try:
                await state.live.send_realtime_input(text=text)
            except Exception as e:
                log.warning("coach note not sent: %s", e)

    async def resume() -> None:
        """Continue a sweep that was finished by mistake: stop building the packet, record again, reconnect the agent."""
        if not sweep.finished:
            return
        state.gen += 1
        old_metrics = state.worker.metrics if state.worker else None
        if state.pipeline and not state.pipeline.done():
            state.pipeline.cancel()
            await asyncio.gather(state.pipeline, return_exceptions=True)  # its finally closes the old worker
        elif state.worker:
            await state.worker.close()
        state.pipeline = None
        sweep.finished = False
        sweep.ended_mono = None
        sweep.write_meta()
        if os.getenv("LCA_WORKER", "1") == "1":
            state.worker = SweepWorker(sweep.dir, sweep.country, emit=send, coach_send=coach_agent)
            if old_metrics:
                state.worker.metrics = old_metrics  # keep counting cost from where it was
            state.worker.start()  # work already done is reused from runs/<id>/cache
        state.resume_note = (
            f"(The policyholder tapped Continue sweep after finishing by mistake. Camera and microphone are on again. "
            f"Last phase: {sweep.phase}. Ask briefly what is left to cover and keep guiding.)"
        )
        if state.gem and not state.gem.done():
            state.gem.cancel()
            await asyncio.gather(state.gem, return_exceptions=True)
        state.closed = False
        if use_live:
            state.gem = asyncio.create_task(gemini_loop())
        log.info("sweep %s continued", sweep.id)
        await send({"type": "resumed"})

    async def run_tool(fc) -> tuple[dict, types.FunctionResponseScheduling]:
        a = dict(fc.args or {})
        silent, idle = types.FunctionResponseScheduling.SILENT, types.FunctionResponseScheduling.WHEN_IDLE
        if fc.name == "set_phase":
            sweep.phase, sweep.unit_label = a.get("phase", "pan"), a.get("unit_label", "")
            await send({"type": "phase", "phase": sweep.phase, "unit": sweep.unit_label})
            return {"ok": True}, silent
        if fc.name == "log_items":
            for it in a.get("items", []):
                if not is_repeat(it, sweep.live_items):  # the camera resting on or returning to a shelf re-logs it
                    sweep.live_items.append({**it, "t_ms": sweep.last_t_ms, "frame_ref": f"frames/f_{sweep.frame_n:06d}.jpg"})
            await send({"type": "inventory", "items": sweep.live_items})
            return {"ok": True}, silent
        if fc.name == "request_recapture":
            sweep.recaptures.append({**a, "t_ms": sweep.last_t_ms})
            await send({"type": "status", "text": f"re-capture: {a.get('target')} ({a.get('reason')})"})
            return {"ok": True}, silent
        if fc.name == "user_override":
            ov = sweep.add_override(a)
            await send({"type": "override", "override": ov})
            return {"recorded": True}, idle
        if fc.name == "set_locale":
            c = str(a.get("country", "")).upper()
            if c not in LOCALES:
                return {"error": f"unsupported country; supported: {list(LOCALES)}"}, idle
            sweep.country = c
            sweep.write_meta()
            if state.worker:
                state.worker.country = c  # price in the confirmed market from now on
            await send({"type": "locale", "country": c, "currency": locale(c)["currency"]})
            return {"country": c, "currency": locale(c)["currency"]}, idle
        if fc.name == "finish_sweep":
            await finish()
            return {"status": "processing started. Tell the policyholder in one short sentence that the camera is off and the packet will appear on screen in a few minutes; the voice session then ends."}, types.FunctionResponseScheduling.INTERRUPT
        return {"error": "unknown tool"}, silent

    async def gemini_loop() -> None:
        if not os.getenv("GEMINI_API_KEY"):
            await send({"type": "error", "text": "GEMINI_API_KEY is not set on the server (see .env.example); frames are still being recorded."})
            return
        while not state.closed:
            loc = locale(sweep.country)
            try:
                async with gemini().aio.live.connect(model=LIVE_MODEL, config=build_config(loc["name"], loc["currency"], state.handle)) as s:
                    state.live = s
                    note = None
                    if not state.greeted:
                        state.greeted = True
                        note = "(The policyholder opened the app; camera and microphone are on. Greet them now.)"
                    elif state.resume_note:
                        note, state.resume_note = state.resume_note, None
                    if note:
                        await s.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=note)]), turn_complete=True)
                    reconnect = False
                    while not reconnect and not state.closed:
                        got = False
                        async for msg in s.receive():
                            got = True
                            if msg.session_resumption_update and msg.session_resumption_update.resumable and msg.session_resumption_update.new_handle:
                                state.handle = msg.session_resumption_update.new_handle
                            if msg.go_away:
                                log.info("GoAway (time left %s): reconnecting with resumption handle", msg.go_away.time_left)
                                reconnect = True
                                break
                            sc = msg.server_content
                            if sc:
                                if sc.interrupted:
                                    await send({"type": "interrupted"})
                                if sc.input_transcription and sc.input_transcription.text:
                                    sweep.transcript("user", sc.input_transcription.text)
                                    await send({"type": "transcript", "who": "user", "text": sc.input_transcription.text})
                                if sc.output_transcription and sc.output_transcription.text:
                                    sweep.transcript("agent", sc.output_transcription.text)
                                    await send({"type": "transcript", "who": "agent", "text": sc.output_transcription.text})
                                for part in (sc.model_turn.parts if sc.model_turn else []) or []:
                                    if part.inline_data and part.inline_data.mime_type.startswith("audio/pcm"):
                                        await send({"type": "audio", "data": base64.b64encode(part.inline_data.data).decode()})
                                if sc.turn_complete:
                                    await send({"type": "turn_complete"})
                                    if state.turn_done:
                                        state.turn_done.set()
                            if msg.tool_call:
                                responses = []
                                for fc in msg.tool_call.function_calls or []:
                                    result, sched = await run_tool(fc)
                                    responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result, scheduling=sched))
                                if responses:
                                    await s.send_tool_response(function_responses=responses)
                        if not got:
                            reconnect = True  # the server closed the stream
            except Exception as e:
                if state.closed:
                    break
                log.warning("live session error: %s; reconnecting", e)
                await send({"type": "status", "text": "reconnecting voice agent..."})
                await asyncio.sleep(1.0)
            finally:
                state.live = None

    async def client_loop() -> None:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            live = state.live
            try:
                if sweep.finished and kind in ("audio", "frame", "imu", "text"):
                    continue  # the sweep is over: nothing more is recorded or forwarded
                if kind == "audio" and live:
                    await live.send_realtime_input(audio=types.Blob(data=base64.b64decode(msg["data"]), mime_type="audio/pcm;rate=16000"))
                elif kind == "frame":
                    if REPLAY and msg.get("phase"):
                        sweep.phase, sweep.unit_label = msg["phase"], msg.get("unit", "")
                    row, data = await asyncio.to_thread(sweep.save_frame, msg)
                    if live and msg.get("live"):
                        small = await asyncio.to_thread(downscale_jpeg, data)
                        await live.send_realtime_input(video=types.Blob(data=small, mime_type="image/jpeg"))
                    hint = sweep.quality_hint(row, bool(msg.get("blur_bad")), bool(msg.get("glare_bad")))
                    if hint and live:
                        await live.send_realtime_input(text=hint)
                elif kind == "imu":
                    await asyncio.to_thread(sweep.save_imu, msg.get("samples", []))
                elif kind == "text" and live:
                    await live.send_realtime_input(text=str(msg.get("text", ""))[:2000])
                elif kind == "end":
                    await finish()
                elif kind == "resume":
                    await resume()
            except (ValueError, KeyError) as e:
                await send({"type": "error", "text": str(e)})

    if os.getenv("LCA_WORKER", "1") == "1":  # process the sweep while it is being recorded
        state.worker = SweepWorker(sweep.dir, sweep.country, emit=send, coach_send=coach_agent)
        state.worker.start()
    use_live = not (REPLAY and ws.query_params.get("live") == "0")
    state.gem = asyncio.create_task(gemini_loop() if use_live else asyncio.sleep(0))
    try:
        await client_loop()
    except WebSocketDisconnect:
        pass
    finally:
        state.closed = True
        state.gem.cancel()  # close the Live session first; the packet does not need it
        sweep.write_meta()
        if state.pipeline and not state.pipeline.done():
            await asyncio.shield(state.pipeline)  # finish the packet even if the phone disconnects
        elif state.worker and not state.pipeline:
            await state.worker.close()  # disconnected without finishing: stop background work
