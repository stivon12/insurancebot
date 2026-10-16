# Library Contents Claim Agent

A live voice-and-vision agent for a home-library contents claim. The policyholder walks the room once with a phone. The agent guides the sweep, logs what it sees, and returns a claim packet:
- **Books:** counted from spines, identified, measured, and priced new and used at local prices.
- **Other items:** insurable non-book items, priced.
- **Room:** floor, wall and shelved wall area.

Every number links to a saved frame. Every price links to a URL and carries a retrieval date.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and [ngrok](https://ngrok.com/). Phones only allow the camera and microphone over HTTPS.

```bash
uv sync --extra models                    # Python 3.12, plus torch / RF-DETR for the local models
uv pip install --no-deps "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" \
                         "git+https://github.com/EasternJournalist/utils3d.git@35a17ba705d921c7a559ec6067c8eaa5c03ddb37"
cp .env.example .env                      # add GEMINI_API_KEY and SERPER_API_KEY at least
uv run uvicorn server.app:app --port 8000
ngrok http 8000                           # open the https URL on the phone and tap Start sweep
uv run pytest
```

MoGe is installed without its declared dependencies because they pin Linux-only packages. Inference doesn't need them.

To re-build a packet from a recorded sweep: `uv run python -m pipeline.live runs/<sweep_id>`. To score it: `uv run python -m eval.eval --packet runs/<id>/claim_packet.json --truth ground_truth/`.

## How a sweep works

1. The agent confirms the country and currency: US/USD by default, UK/GBP as the second market.
2. **One slow turn from the middle of the room.** While it turns, the agent marks corners (`mark_corner`), then reports the layout (`log_room`). Each frame at 1 fps gets MoGe-2 depth, which builds the room outline during the turn.
3. **Each bookcase, one shelf at a time.** Shelf frames are read as they arrive:
   - RF-DETR spine boxes, plus readable Gemini spines;
   - masked single-spine OCR;
   - catalogue lookup and price lookup, as soon as each book appears.
4. **The rest of the room.**
   - The agent logs each insurable item with `log_item`.
   - The server checks that item on the frame and merges re-sightings by gyro heading.
   - The price search starts at once.
5. **Finish.** The server waits for running jobs, then assembles the packet. Nothing is re-read.

The agent never asks for a part to be repeated. Gaps go to the review queue.

## Files

```
server/app.py          FastAPI app, phone WebSocket, Gemini Live bridge, tools, sweep recorder
server/live_config.py  prompt, tools and Live session settings
pipeline/live.py       per-sweep engine: frame queue, books / items / room jobs, packet build, replay CLI
pipeline/books.py      spines (Gemini + RF-DETR, rule A), cross-frame merge, OCR, catalogue ID, spine sizes
pipeline/items.py      check of agent-logged items, merge by heading, dimensions
pipeline/room.py       MoGe-2 per frame, wall bearings, rectilinear outline, areas
pipeline/pricing.py    ISBN shop pages, Google Shopping (Serper/SerpApi), eBay, FX, appraisal rules
pipeline/packet.py     packet lines, review queue, totals in code, HTML report
pipeline/common.py     config, Gemini calls, memo, metrics, HTTP
eval/eval.py           packet vs. hand ground truth, one row per pass bar
web/                   phone client
```

## Reference app

This builds on the Insurance Claim Live Agent Team example (`awesome-llm-apps/voice_ai_agents/insurance_claim_live_agent_team`).
- **Kept:**
  - the browser-to-Live WebSocket bridge;
  - non-blocking tools;
  - 1 fps video;
  - the live model logging what it sees, with the server re-checking the frame;
  - a running state fed back to the model;
  - work done after every event.
- **Changed:** the tools now measure and price instead of describing, and there's a recorder and per-frame processing.
- **Removed:** the policy rules, the sketches and the avatar.

## Assumptions

- **Wall area** is gross: perimeter × ceiling height.
- **Shelf run** is occupied run.
- **Book prices:**
  - new price: the exact-ISBN shop page first, otherwise Google Shopping offers that agree;
  - used price: the cheaper half of AbeBooks used offers, capped at the new price.
- **Appraisal:** items at or above `APPRAISAL_THRESHOLD` (default 250 in the local currency), plus art, signed and rare books, go to appraisal.
- **Room scale:** the phone's height, assumed to be 1.40 m when held upright (±10%, stated in the packet). Depth gives the room's shape.
