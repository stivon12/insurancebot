# Architecture

## How input becomes a packet

```
PHONE ── mic 16 kHz ─────────────────────────────▶ Gemini Live (voice: talks, asks, hears corrections)
  │      1 fps best-of-second frame (1080p) ────▶   ▲ app notes: [doubt] / [observe] / [capture]
  │      5 fps tracking frame (640 px)               │
  │      orientation 60 Hz + motion               SERVER (server/app.py → pipeline/live.py)
  ▼                                                  │
 every 1 fps frame ─▶ RF-DETR spine boxes + Gemini Flash spine list ─▶ merge across frames
                      ─▶ read spines, 12 masked crops per call (Flash) ─▶ catalogue match: Google Books,
                         else Open Library and the Library of Congress in turn
                      ─▶ price (eBay, shop pages, Google Shopping)                    ─▶ live inventory
 every turn / 3 s  ─▶ Flash-Lite item scan ─▶ kept if sure (≥0.9) or seen twice ─▶ eBay price ─▶ live inventory
 every 5 fps frame ─▶ optical-flow tracking + phone orientation ─▶ camera path
                      keyframes ─▶ MoGe-2 depth ─▶ wall points in one room frame
 low confidence    ─▶ [doubt] note ─▶ agent asks for another look (books, items, path lost, unseen sector)

 "I'm done" ─▶ wait for jobs (≤60 s) ─▶ room: floor grid, flood fill ─▶ identify the rest (stop at 150 s)
           ─▶ spine sizes ─▶ merge duplicate items ─▶ prices (stop at 240 s) ─▶ totals in code
           ─▶ claim_packet.json + report.html
```

Identification, pricing and measurement are separate steps. Each writes its own file (`identified.json`, `prices_<CC>.json`, `measurements.json`, `room.json`, `items.json`), and each can be re-run from a saved sweep.

## Which model does what

| Model | Job | Never does |
|---|---|---|
| `gemini-3.8-live` | talks, directs the sweep, hears corrections | write a number, price or total |
| `gemini-3.8-flash` | lists spines per frame; reads masked single-spine crops, 12 per call | name a book it can't read (unreadable = unidentified) |
| `gemini-3.1-flash-lite` | item scan per frame; checks logged items; merges duplicate items | price anything |
| RF-DETR Nano (bookshelf weights) | tight spine boxes for counting and size | — |
| MoGe-2 ViT-S | depth per tracking keyframe | set the scale on its own |

## Where metric scale comes from

- **Spines:** up to 3 identified books per frame are rulers: their height comes from catalogue dimensions. Every other spine is measured against those books as a ruler in the same frame (a leave-one-out "book ruler"). With no ruler, standard paperback and hardback heights are used. The method is stated on every line.
- **Room:** the camera path is tracked frame to frame:
  - rotation from the phone's own fused orientation, refined by the image;
  - position from feature tracks and MoGe-2 depth.

  The scale is set so the floor sits 1.40 m below the phone (±10%, flagged for a tape check). The floor is filled on a 5 cm grid out to the walls, so L-shaped and other non-rectangular rooms come out as they are.

## Where prices come from

- **Books:** eBay Browse API, one search for new and used fixed-price listings (by ISBN, else title and author): the median of the 3–5 cheapest comparable listings, with lots and outliers removed. Otherwise shop pages (Barnes & Noble, AbeBooks) or Google Shopping via Serper. Exchange rates come from Frankfurter, and converted prices are labelled.
- **Items:** eBay within the item's eBay category, with accessories filtered out, reported as a 25th–75th percentile range. A result below a third of the scanner's own estimate is rejected. Otherwise Google Shopping via Serper.
- **Every price** carries its URL, retrieval time and assumed condition. No source means no price: the line is excluded from the totals and goes to review.

## Latency, compute and cost (measured on the two sweeps, M1 Pro, 16 GB)

| | Bookstore (448 spines, 1 min 46 s) | Bedroom (10 items, 1 min 1 s) |
|---|---|---|
| Model calls | 464 spine reads, 1,459 identification calls, 30 item scans | 20 item scans |
| HTTP calls | about 5,700 catalogue and price calls | 19 |
| **Cost** | **$5.35** (about 1.2¢ per book: $2.46 spines, $2.87 identification) | **$0.02** |
| Time to packet | **580 s: failed the bar** (Google Books 429 stall). Re-streamed after the fixes: 240 s, $3.81 | 6.7 s |

| Local compute per frame (M1 Pro, warm) | Time |
|---|---|
| RF-DETR spine boxes | 0.17 s |
| MoGe-2 depth (keyframes only) | 0.51 s |
| Optical-flow tracking (640 px, 5 fps) | about 0.1 s |

Gemini spine listing takes about 23–38 s per frame (measured in the streamed re-runs) and runs in parallel during the sweep.

**Memory:** about 1.1 GB peak, of which:
- RF-DETR and its import stack: about 615 MB (the largest share);
- MoGe-2: about 200 MB;
- torch: about 145 MB;
- server and SDKs: about 110 MB.

**Disk:**
- `.venv`: 1.1 GB, of which torch is 553 MB;
- weights: 0.73 GB (RF-DETR 461 MB, MoGe-2 268 MB);
- runs: about 20 MB per minute of sweep.

## Why this way, compared with the first version

| First approach | What went wrong (measured) | Now |
|---|---|---|
| Three fixed parts (turn on the spot, then shelves, then items), with the agent steering each part | It talked over people, rushed them through the parts, and missed books shown during the "wrong" part | One free tour. Every frame is processed for everything, and the agent only asks when in doubt |
| Room from one turn on the spot: gyro heading plus marked corners, then MoGe depth per bearing | Floor 24–69% too small: one spot can't see every wall, and L-shapes are impossible | A walking tour, a tracked camera path, depth keyframes and a filled floor grid |
| COLMAP structure-from-motion | Registered 4 of 30 frames on plain walls | Phone orientation plus optical flow, which works on plain walls |
| Gemini alone listing spines | Count +16% (junk and repeated spines) | RF-DETR boxes plus readable Gemini spines: +4.7% on unseen photos |
| Google Shopping first for every price | Accessories priced as the product (a PS5 at $22) | eBay within the product's category, with accessories removed (PS5 $500–660) |
