"""Replay a recorded sweep into a running server, as the phone would send it, and time what comes back.

Test tool only. Start the server with replay allowed (frames then carry their recorded phase):
  LCA_ALLOW_REPLAY=1 uv run uvicorn server.app:app --port 8000
  uv run python scripts/replay_sweep.py runs/<recorded sweep> [--frames 60] [--speed 1] [--live]

Prints every server message with seconds since start, sends `end`, keeps sending frames for 2 s afterwards (they
must be ignored), and stops when the packet or an error arrives.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import httpx
import websockets


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--frames", type=int, default=0, help="replay only the first N frames (0 = all)")
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice real time")
    ap.add_argument("--live", action="store_true", help="also run the voice agent (costs a little)")
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    rows = [json.loads(line) for line in (a.run / "frames.jsonl").read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["i"])
    if a.frames:
        rows = rows[: a.frames]
    imu_path = a.run / "imu.jsonl"
    imu = [json.loads(line) for line in imu_path.read_text().splitlines()] if imu_path.exists() else []
    async with httpx.AsyncClient(base_url=a.url) as c:
        sw = (await c.post("/api/sweeps", json={"device": "replay of " + a.run.name})).json()
    ws_url = a.url.replace("http", "ws", 1) + f"/ws/live?sweep_id={sw['id']}" + ("" if a.live else "&live=0")
    t0 = time.perf_counter()
    log: list[tuple[float, dict]] = []
    marks: dict[str, float] = {}

    def stamp() -> float:
        return round(time.perf_counter() - t0, 1)

    async with websockets.connect(ws_url, max_size=None) as ws:
        done = asyncio.Event()

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "audio":
                    continue
                log.append((stamp(), m))
                brief = {k: v for k, v in m.items() if k != "type"}
                print(f"{stamp():7.1f}s  {m['type']:<10} {json.dumps(brief, ensure_ascii=False)[:160]}", flush=True)
                if m["type"] in ("packet", "error"):
                    marks[m["type"]] = stamp()
                    done.set()

        rd = asyncio.create_task(reader())
        start_t = rows[0]["t_ms"]
        ii = 0
        for r in rows:
            target = (r["t_ms"] - start_t) / 1000 / a.speed
            wait = target - (time.perf_counter() - t0)
            if wait > 0:
                await asyncio.sleep(wait)
            batch = []
            while ii < len(imu) and imu[ii]["t_ms"] <= r["t_ms"]:
                batch.append(imu[ii])
                ii += 1
            if batch:
                await ws.send(json.dumps({"type": "imu", "samples": batch}))
            data = base64.b64encode((a.run / r["file"]).read_bytes()).decode()
            await ws.send(json.dumps({"type": "frame", "data": data, "t": r["t_ms"], "w": r["w"], "h": r["h"], "orient": r.get("orient", 0),
                                      "live": r["i"] % 3 == 0, "blur": r.get("blur", 0), "glare": r.get("glare", 0),
                                      "phase": r.get("phase"), "unit": r.get("unit", "")}))
        marks["end"] = stamp()
        print(f"{stamp():7.1f}s  >> end ({len(rows)} frames sent)", flush=True)
        await ws.send(json.dumps({"type": "end"}))
        last = rows[-1]
        for k in range(6):  # frames after the end must be ignored by the server
            await asyncio.sleep(0.33)
            data = base64.b64encode((a.run / last["file"]).read_bytes()).decode()
            await ws.send(json.dumps({"type": "frame", "data": data, "t": last["t_ms"] + 333 * (k + 1), "w": last["w"], "h": last["h"]}))
        try:
            await asyncio.wait_for(done.wait(), a.timeout)
        except asyncio.TimeoutError:
            print("timed out waiting for the packet", flush=True)
        rd.cancel()
    saved = sum(1 for _ in (Path("runs") / sw["id"] / "frames").glob("*.jpg"))
    print(f"\nsweep {sw['id']}: frames sent {len(rows)}, frames saved {saved} (post-end frames ignored: {saved == len(rows)})")
    if "packet" in marks:
        print(f"end -> packet: {marks['packet'] - marks['end']:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
