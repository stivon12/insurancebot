# Architecture

```
 PHONE BROWSER (iOS Safari / Android Chrome)              SERVER (laptop, FastAPI)                         HOSTED APIs
 ┌──────────────────────────────┐   WebSocket   ┌────────────────────────────────────┐
 │ camera 1080p -> JPEG ~3 fps  │──frames──────▶│ Sweep recorder: runs/<id>/frames/   │
 │ blur / glare check (canvas)  │──flags───────▶│   frames.jsonl  imu.jsonl           │
 │ DeviceMotion: gravity + gyro │──imu─────────▶│   overrides.json transcript.jsonl   │
 │ mic -> 16 kHz PCM16 worklet  │──audio───────▶│                                     │  ~1 fps + audio  ┌─────────────────┐
 │ speaker <- 24 kHz PCM        │◀─audio────────│ Live bridge (2 async loops) ────────┼────────────────▶│ Gemini Live      │
 │ live inventory, transcript   │◀─events───────│ tools: set_phase log_items          │◀── tool calls ──│ gemini-3.8-live  │
 └──────────────────────────────┘               │  request_recapture user_override    │                 └─────────────────┘
                                                │  set_locale finish_sweep            │
                                                └───────────────┬────────────────────┘
                                                                │ finish_sweep
                                                                ▼
 POST-SWEEP PIPELINE (each stage = module + JSON output in runs/<id>/)
 keyframes ──▶ spines ─────▶ identify ─────▶ scale ──────┐
 (Laplacian    (Flash: rows,  (Flash reads    (book ruler:  │
  sharpness)    row lists x2;  crop; Google    leave-one-out │
                ORB merge)     Books/Open      cm/px)        ├──▶ pricing ──▶ assemble ──▶ report.html
           └──▶ items          Library match)  room ─────────┤    (eBay,       (totals in    claim_packet.json
                (Flash-Lite                    (focal from   │     JSON-LD,     code, review  metrics.json
                 batches)                       gyro; camera │     FX)          queue)
                                                height from  │
                                                books; floor │
                                                rays)        │
```

## Which model does what

| Step | Model or tool | Output it is trusted for | Never trusted for |
|---|---|---|---|
| Conversation, capture guidance, voice overrides | Gemini Live `gemini-3.8-live` | what to say; which phase is starting; what the user said | any count, size or price |
| Shelf rows; ordered spine list per row (two passes, split on disagreement) | `gemini-3.8-flash` | boxes and verbatim visible text | titles it cannot read; a pass that disagrees is re-asked |
| Spine transcription | `gemini-3.8-flash` | the characters on the spine | the identity of the book, which comes from a catalogue lookup scored in code |
| Floor-wall and wall-ceiling lines; bookcase box | `gemini-3.8-flash`, then snapped to the strongest image edge (OpenCV) | pixel positions | distances (computed by geometry) |
| Non-book items | `gemini-3.1-flash-lite`, batched with the running item list | category, material, legible brand | dimensions (geometry) and prices (eBay) |
| Retailer URL discovery (fallback) | `gemini-2.5-flash-lite` with Google Search grounding | candidate URLs | the price, which code parses from page markup |
| Duplicate merging, scale, room polygon, totals | OpenCV / NumPy / Python | everything numeric | |

## Where metric scale comes from: the books are the ruler

No markers, no LiDAR, no "the model estimated it".

**1. Spines (`pipeline/scale.py`)**
- Identified books have catalogue physical dimensions: Open Library `physical_dimensions`, or Google Books `dimensions`.
- In each frame, cm per pixel at the shelf is the median of `catalogue_height / spine_pixel_height` over the identified books visible in that frame. Rulers more than 8% from the median are rejected, and so are books whose spine aspect ratio does not match the catalogue edition.
- **Leave-one-out:** a book is measured only with the other books' scale.
- Each book's dimensions are the median across the frames it appears in.
- Fallback when a frame has fewer than 2 other rulers: the scale that snaps the frame's spines onto standard book formats (A, B, trade, Demy, Royal, hardcover). This is low confidence and goes to review.

**2. Focal length (`pipeline/room.py`)**
- During the 360° turn, the gyro gives the yaw change between two keyframes, and an ORB homography gives the image shift of the frame centre.
- `f = shift / tan(yaw)`, as a median over frame pairs.
- If it cannot be measured, a 69° field of view is assumed. This is flagged in the packet and lowers confidence.

**3. Camera height**
- In each bookcase overview frame, ruler books from the close-up shelf frames are mapped in by homography.
- Each one gives the horizontal distance `D` to the bookcase face:
  `D = H_catalogue / (tan-elevation(top) - tan-elevation(bottom))`.
  The elevations come from the gravity vector and `f`.
- The ray to the bookcase base then gives camera height `h = -D * tan-elevation(base)`. The same `D` gives each bookcase's width and height, which make up the shelved wall area.

**4. Room**
- Each floor-wall boundary pixel in the turn frames is a ray. With `h`, it lands on the floor at `(h / tan(depression), azimuth)`, with azimuth from the integrated gyro.
- **Outline:** a Manhattan polygon is fitted to those points, with any number of corners, so L-shaped and other rectilinear rooms are handled. Where furniture stands against a wall, the furthest robust point per 5° bin is used.
- **Ceiling height:** `h + wall_distance * tan(elevation of the wall-ceiling line)`, as a median.
- **Floor area:** shoelace formula. **Wall area:** perimeter × height (gross). **Shelved wall area:** sum of bookcase width × height.

## Where prices come from (`pipeline/pricing.py`)

| Figure | Source | Notes |
|---|---|---|
| Book replacement cost | eBay Browse API, `conditions:{NEW}`, by ISBN (GTIN) on the local marketplace (`EBAY_US`, `EBAY_GB`, …) | median of the 3–5 cheapest after removing lots and outliers, including shipping; labelled "active listings (asking)" |
| Book used value | eBay Browse API, `conditions:{USED}` | condition assumed: used, good |
| Fallback (no local listing) | other marketplaces converted at ECB rates (Frankfurter) | `converted: true`, with the rate and its date stored |
| Fallback (no eBay new listing) | retailer page found by grounded search | price read from schema.org JSON-LD, `product:price` meta or microdata, with the ISBN checked; free text is never parsed |
| Items with a known brand or model | eBay new listings for that model | median |
| Items with no known model | eBay image search on the item crop, then a keyword search | 25th–75th percentile of at least 5 listings, otherwise `needs_appraisal` |
| Art, signed, first edition, antiquarian, fine press, ≥ threshold | none | `needs_appraisal`, left unpriced |

Every price carries its source, URL, the evidence listing URLs and the retrieval time. Totals are sums over the lines in `assemble.compute_totals`, and are unit-tested.
