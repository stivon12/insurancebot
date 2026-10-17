# Library Contents Claim Agent

A live voice-and-camera agent for a home-library contents claim. The policyholder walks the room once with a phone and talks to the agent. The agent asks for another look when it is unsure, and fills in an inventory live. When the policyholder says they're done, it builds a claim packet with:
- every book counted from its spine, identified, measured, and priced new and used;
- every other item of value, priced;
- the room's floor, wall and shelved wall area.

Every number links to a saved frame. Every price has a URL and a retrieval date.

- [ARCHITECTURE.md](ARCHITECTURE.md): how input becomes a packet; models, scale, prices, latency, cost and memory.
- [RESULTS.md](RESULTS.md): each pass bar against what was achieved.
- [FAILURE_LOG.md](FAILURE_LOG.md): the worst errors, their root causes and fixes, and what two more weeks would buy.

## The demo data

The sweeps were split across two places. They don't come from one room.

| Part | Where it was tested | Run |
|---|---|---|
| **Books** (count, titles, spine sizes, prices) | A small library corner in a coworking space. This is the screen-recorded demo sweep (1 min 46 s, 448 spines counted). Filming people and making any sound weren't allowed, so it covers one area, with no room tour and no other items. | `submission/bookstore/` |
| **Room measurement and other items** | My bedroom (no books). | `submission/bedroom/` |

**So the bookstore packet's `room` block and its 3 shelving "items" aren't real measurements.** The room has confidence 0.2, and the review queue says so.

**Frames:** the repo has only 640 px thumbnails of the frames, at the same `frames/...` paths the packets use.
- **Bookstore:** every frame the packet references, except 5 that showed my feet (`f_000001`, `f_000003`, `f_000006`, `f_000012`, `f_000013`), which are withheld.
- **Bedroom:** only the PS5, TV and MacBook frames (`f_000001`, `f_000040`, `f_000051`). The other bedroom frames are withheld for privacy, so those `frame_ref`s don't open in this repo.

The packets are exactly as the sweeps produced them. The fixes made later are measured in [FAILURE_LOG.md](FAILURE_LOG.md).

## Quick start (about 10 minutes)

You need macOS or Linux, [uv](https://docs.astral.sh/uv/), [ngrok](https://ngrok.com/) (phones only allow the camera and mic over HTTPS), and a Gemini API key.

```bash
uv sync --extra models          # Python 3.12, torch, RF-DETR (about 1 GB)
uv pip install --no-deps "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" \
                         "git+https://github.com/EasternJournalist/utils3d.git@35a17ba705d921c7a559ec6067c8eaa5c03ddb37"
cp .env.example .env            # set GEMINI_API_KEY and SERPER_API_KEY; eBay keys are optional but give the best prices
uv run uvicorn server.app:app --port 8000
ngrok http 8000                 # open the https URL on the phone, tap Start
uv run pytest -q                # 23 tests
```

- **MoGe-2** is installed without its declared dependencies, because they pin Linux-only packages that inference doesn't need. Run that line again after any `uv sync`, which removes it.
- **Model weights** download on first use: RF-DETR (461 MB) and MoGe-2 (268 MB), into the Hugging Face cache.
- **Install time:** from a clean clone with a warm uv cache, install plus tests took under 1 minute. On a fresh machine, the downloads (about 1 GB of packages and 0.7 GB of weights) set the time.

**Other commands:**
- **Rebuild a packet from a saved sweep:** `uv run python -m pipeline.live runs/<sweep_id>`. Work done during the sweep is reused from the run's cache.
- **Score a packet against your ground truth:** `uv run python -m eval.eval --packet runs/<id>/claim_packet.json --truth <dir>`. The file formats are in `ground_truth_template/`.

## How a sweep works

1. **Locale:** the agent confirms the country and currency (US/USD by default, GB/GBP as the second market).
2. **Tour:** it asks for a plain walk round the room: "show me everything you own in it… tell me when you're done".
3. **During the tour,** the server reads every frame:
   - spines are detected, read and identified, and their prices looked up;
   - other valuable items are found by a per-frame scan;
   - the camera path is tracked to build the room's outline.

   The inventory fills in on the phone as this happens.
4. **The agent speaks only when it needs something:**
   - a closer look at unreadable spines or a glimpsed item;
   - "original or print?" for pictures;
   - a part of the room it hasn't seen;
   - "slow down" when the view blurs.

   You can correct it by voice ("skip that shelf", "that one is signed").
5. **Saying "I'm done"** ends the tour. The packet is built within the time limit:
   - identification stops 150 s after the end;
   - pricing stops at 240 s;
   - anything not finished goes to the review queue.

## Files

```
server/app.py          FastAPI app: phone WebSocket, Gemini Live bridge, tools, sweep recorder
server/live_config.py  agent prompt, tools, Live session settings
pipeline/live.py       per-sweep engine: jobs per frame, doubts for the agent, packet build, replay CLI
pipeline/books.py      spine boxes (RF-DETR + Gemini), cross-frame merge, spine reading, catalogue ID, spine sizes
pipeline/items.py      per-frame item scan, merging sightings, duplicate merge at the end
pipeline/room.py       phone orientation, camera tracking, MoGe-2 depth, floor grid, areas
pipeline/pricing.py    eBay Browse API, shop pages, Google Shopping (Serper/SerpApi), FX, appraisal rules
pipeline/packet.py     packet lines, review queue, totals in code, HTML report
pipeline/common.py     config, Gemini calls, HTTP (rate limits, cache), metrics
eval/eval.py           packet vs. hand ground truth, one row per pass bar
web/                   phone client (capture worker, mic worklet)
submission/            the two demo packets, reports and thumbnails
```

## Reference app (Insurance Claim Live Agent Team)

- **Kept:**
  - the browser ↔ Gemini Live WebSocket bridge;
  - non-blocking tools;
  - video into the Live session at about 1 fps;
  - a claim file that builds while the user talks.
- **Changed:**
  - the agent directs the capture (re-looks, print or original, coverage);
  - background jobs measure, count and price instead of describing;
  - every sweep is recorded, so it can be replayed and scored.
- **Thrown away:**
  - the policy-rule agents;
  - the narrative notebook;
  - the avatar;
  - letting the model's own words become packet content. Here a model never writes a number, price or total.

## Models, APIs and tools

- **Models:**
  - Gemini Live `gemini-3.8-live` for voice;
  - `gemini-3.8-flash` to read spines;
  - `gemini-3.1-flash-lite` for checks, item scans and merges;
  - RF-DETR Nano bookshelf weights (`Geraldine/rf-detr-nano-bookshelf`, Apache-2.0) for spine boxes;
  - MoGe-2 ViT-S (MIT) for depth.
- **APIs:**
  - Google Books, Open Library and Library of Congress for catalogue data;
  - eBay Browse API, Serper (Google Shopping), SerpApi as fallback, and AbeBooks / Barnes & Noble pages for prices;
  - Frankfurter for exchange rates.
- **Code:** written with Claude Code as the AI coding tool. Everything in it can be explained line by line.

## Assumptions

- **Wall area** is gross: perimeter × ceiling height, with doors and windows not subtracted.
- **Shelf run** is the occupied run of each shelf.
- **Shelved wall area** for each shelf is its run × (its tallest spine + 5 cm).
- **Room scale** assumes the phone is held about 1.40 m above the floor (±10%, stated in the packet).
- **Book prices:**
  - new = the 25th–75th percentile of new fixed-price eBay listings for that ISBN or title, or a shop page;
  - used = the cheaper used listings, capped at the new price.
- **Items:** a sourced range from the 25th–75th percentile of new listings in the item's eBay category, with accessories removed. Below about $25 an item isn't logged.
- **Appraisal:** these are not auto-priced and go to "needs appraisal":
  - anything at or above `APPRAISAL_THRESHOLD` (default 250 in local currency);
  - art not confirmed as a print;
  - signed, first-edition or rare books.
