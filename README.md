# Library Contents Claim Agent

This is a live voice and vision agent for a contents claim. The policyholder walks a home library once with their phone, and the agent guides the sweep as it happens. The output is a claim packet:
- every book counted and read from its spine, measured in centimetres, and priced new and used at local prices;
- every non-book item listed and priced;
- the room's floor area, wall area and shelved wall area.

Every number links to a saved frame. Every price links to a URL and has a retrieval date.

The app runs in any modern phone browser: iOS Safari or Android Chrome.

## Quick start (about 10 minutes)

Prerequisites: [uv](https://docs.astral.sh/uv/) and [ngrok](https://ngrok.com/). Phone browsers only allow the camera, microphone and motion sensors over HTTPS, so the server needs an HTTPS tunnel.

```bash
git clone <repo> && cd library-claim-agent
uv sync                                  # Python 3.12 + dependencies
cp .env.example .env                     # then fill in the keys below
uv run uvicorn server.app:app --port 8000
ngrok http 8000                          # in a second terminal; open the https URL on the phone
```

Keys in `.env`:

| Key | Where to get it | Cost |
|---|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | pay-per-use; estimated about $1 per sweep (measured cost goes in `metrics.json` and the failure log) |
| `EBAY_CLIENT_ID`, `EBAY_CLIENT_SECRET` | https://developer.ebay.com → create a **Production** keyset (OAuth client credentials) | free, 5,000 calls a day |
| `GOOGLE_BOOKS_API_KEY` (optional) | Google Cloud console → enable the Books API | free |

On the phone:
1. Open the ngrok URL and tap **Start sweep**.
2. Allow the camera, microphone and motion sensors.
3. Follow the agent.

The packet appears on screen when processing finishes. Everything is saved under `runs/<sweep_id>/`:
- `frames/`, `frames.jsonl`, `imu.jsonl`
- one JSON file per pipeline stage
- `claim_packet.json`, `report.html`, `metrics.json`

To re-run the post-sweep pipeline on a saved sweep (for example after changing a stage):

```bash
uv run python -m pipeline.run runs/<sweep_id>              # primary country from .env
uv run python -m pipeline.run runs/<sweep_id> --country GB # same sweep priced for the UK
uv run python -m eval.eval --packet runs/<sweep_id>/claim_packet.json --truth ground_truth/
uv run pytest
```

## How a sweep works

1. **Greeting:** the agent greets you, confirms the country and currency (US/USD by default, UK/GBP as the second locale), and explains the sweep.
2. **Room turn:** you stand in the middle of the room and turn slowly through a full circle, then turn again with the phone tilted up toward the ceiling line. This is for room geometry.
3. **Each bookcase:**
   - step back so the whole bookcase is in view, for metric scale;
   - then walk along it one shelf at a time, for spines.
4. **The rest of the room:** a slow pan for the remaining contents.

While you walk, the agent:
- reacts to blur and glare reported by the in-browser camera checks;
- asks for a re-capture of a specific shelf;
- asks "original or print?" about framed art;
- takes voice corrections such as "skip that shelf, those aren't mine" or "that one's a first edition".

5. **After the sweep:** the background stages run, and the agent reads a summary whose numbers are computed in code.

## Design summary

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline diagram, which model does what, where metric scale comes from, and where prices come from.

- **The live model only talks and directs.** Gemini Live (`gemini-3.8-live`) handles conversation, capture guidance and voice overrides. It never produces a count, measurement or price.
- **Separate, testable stages,** each writing its own JSON file in `runs/<id>/`:
  `keyframes → spines → identify → scale → room → items → pricing → assemble → report`.
- **Metric scale has no markers and no LiDAR. The books are the ruler.**
  - Spine sizes: identified books' catalogue heights (Open Library / Google Books) give centimetres per pixel at the shelf. A book is never measured with its own catalogue height (leave-one-out median).
  - Room: the same book ruler gives the distance to each bookcase, and so the camera height. The phone's gravity sensor and gyro then turn floor-line pixels into metres.
- **Prices:**
  - eBay Browse API on the policyholder's marketplace, as active-listing asking prices including shipping, labelled as such.
  - Retailer product pages as a fallback. A grounded web search finds the URL, and code reads the price from the page's schema.org markup.
  - Frankfurter (ECB) rates when converting from another market, labelled `converted`.
  - No language model ever writes a price or a total.
- **"I don't know" is a valid answer.** An unreadable spine is logged as `unidentified` with its dimensions. A line with no sourced price is left blank, excluded from the totals and added to the review queue. Art defaults to `needs_appraisal`, as do signed, first-edition, antiquarian and high-value books.

## Reference app: kept, changed, thrown away

Reference: the Insurance Claim Live Agent Team example (`awesome-llm-apps/voice_ai_agents/insurance_claim_live_agent_team`). This project is a rewrite, not a fork.

> TODO (author): run the reference app and add one or two lines here on what you saw when running it.

- **Kept (the pattern, reimplemented):**
  - a FastAPI server bridging a browser WebSocket to `client.aio.live.connect`;
  - two concurrent loops, browser→Gemini and Gemini→browser;
  - non-blocking tool calls whose results are scheduled `SILENT` / `WHEN_IDLE`;
  - transcription in both directions;
  - barge-in interruption.
- **Changed:**
  - every frame is recorded to disk at about 3 fps, with motion-sensor data; the reference keeps only the last frame;
  - context-window compression and session resumption with GoAway reconnects are added, because without them a video session is capped at about 2 minutes;
  - sweep-direction tools replace the claim tools: `set_phase`, `log_items`, `request_recapture`, `user_override`, `set_locale`, `finish_sweep`;
  - the heavy work runs once after the sweep as a measured, multi-stage pipeline, instead of an LLM graph re-run on every turn;
  - the output is a structured JSON packet plus an HTML report, instead of Markdown.
- **Thrown away:**
  - the policy directory and coverage, fraud and routing rules;
  - incident sketches and the avatar.

## Models, APIs and tools used

| Purpose | What | Why |
|---|---|---|
| Live voice and vision | Gemini Live `gemini-3.8-live` | native video input, non-blocking tools, barge-in, cheapest live option |
| Spine counting and reading | Gemini `gemini-3.8-flash` (row crops, JSON schema) | no model training allowed; reads rotated spine text well |
| Non-book items | Gemini `gemini-3.1-flash-lite` | cheap batched detection |
| Room boundary marking | Gemini `gemini-3.8-flash`, refined to the nearest image edge with OpenCV | |
| Book identification and dimensions | Google Books API, Open Library API | free, with catalogue physical dimensions |
| Prices | eBay Browse API; Gemini 2.5 Flash-Lite Google-Search grounding (URLs only) | free official sources |
| FX | Frankfurter (ECB reference rates) | free, no key |
| Geometry | OpenCV (ORB, homographies, edges), NumPy | |
| Coding | Claude Code (Anthropic) was used as an AI pair programmer | |

No model was trained or fine-tuned. No rented GPU is needed: everything runs on a laptop CPU plus hosted APIs.

## Assumptions (where the brief is ambiguous)

- **Shelf run** is the occupied run: the sum of upright spine thicknesses, plus the longest book of each flat stack. Per-shelf values are in `shelf_run_by_shelf_m`.
- **Wall area** is gross (perimeter × ceiling height), with doors and windows not subtracted. Ground truth should be taped on the same basis.
- **Replacement cost** for a book is the median of the cheapest new listings on the local eBay marketplace, including shipping. **Used value** is the same calculation over used listings, with condition assumed "used, good". These are asking prices, not sold prices: eBay's sold-price API is restricted.
- **Edition:** an edition is claimed only when the publisher printed on the spine matches a catalogue edition. Otherwise the ISBN is a representative edition, and the `edition` field stays empty.
- **Appraisal threshold:** `APPRAISAL_THRESHOLD` in local currency, default 250. Items at or above it are routed to a human and left unpriced.
- **Supported countries:** US and GB are configured and tested. DE, AU and CA work through the same code path. India has no eBay marketplace.

## Privacy

Don't capture people or documents with personal data. Sweeps are stored locally in `runs/`, which is git-ignored. Stop ngrok when you're not using it: the tunnel URL is public.

## Repository map

```
server/     FastAPI app, Gemini Live bridge, sweep recorder
web/        phone client (camera, mic worklet, playback, camera checks, motion sensors)
pipeline/   post-sweep stages, one module each
eval/       scores a packet against ground truth (all 8 pass bars)
tests/      unit tests (geometry, book ruler, ID guard rails, pricing parsers, totals, eval)
ground_truth_template/   CSV/JSON templates for your own ground truth
```
