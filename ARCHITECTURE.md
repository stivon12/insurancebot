# Architecture

```
phone (1 fps JPEG, mic 16 kHz, gyro/accel) ──WS──> server/app.py ──Live API──> Gemini Live (talks, guides, calls tools)
                                                  │  tools: log_item, mark_corner, log_room, set_phase, user_override, sync_inventory
                                                  ▼
                                           pipeline/live.py  (per-sweep engine, jobs start the moment data arrives)
            shelf frame ─> books.py: Gemini rows+spines ∪ RF-DETR boxes (rule A) ─> merge across frames ─> OCR ─> catalogue ─> price
            log_item    ─> items.py: Flash-Lite check on that frame ─> merge by heading+category ─> price
            turn frame  ─> room.py: MoGe-2 depth + gravity + gyro heading ─> wall distance per 2° bearing
            Finish      ─> room outline, spine sizes, overrides, totals (code) ─> claim_packet.json + report.html
```

## Which model does what

| Model | Job | Produces |
|---|---|---|
| `gemini-3.8-live` | voice; directs the sweep; logs items, corners and layout | events only, never numbers |
| `gemini-3.8-flash` | shelf rows and spine lists; reads each spine from a masked crop | text and boxes |
| RF-DETR Nano (`Geraldine/rf-detr-nano-bookshelf`, Apache-2.0) | spine boxes | the count, together with readable Gemini spines |
| `gemini-3.1-flash-lite` | checks each logged item; picks the catalogue candidate | yes/no answers, descriptions |
| MoGe-2 ViT-S (MIT) | metric depth per room-turn frame, on CPU/MPS | room shape and focal length |

Spine count, rule A, on 15 unseen 4TU shelf photos (295 spines):

| Method | Count error | Precision | Recall |
|---|---|---|---|
| Gemini alone | +15.9% | 77% | 90% |
| RF-DETR alone | −28% | 93% | 66% |
| RF-DETR boxes + unmatched readable Gemini spines | +4.7% | 88% | 92% |

## Metric scale

- **Spines:** catalogue heights of identified books, used as a leave-one-out book ruler, with thickness fused with page count.
- **Room:** MoGe-2 gives each frame's shape, normalised by its own floor height. Scale comes from the phone height, 1.40 m held upright.
  - On three LiDAR-referenced iPhone captures, MoGe's own metric scale was 6–13% short and its floor height 18% low.
  - The 1.40 m prior was within 5%.
  - The packet states ±10% and a tape check.
- **Room bearings** come from the gyro, integrated about gravity.

## Where prices come from

- **Books:**
  - new: Barnes & Noble (exact ISBN) or AbeBooks;
  - used: AbeBooks, the cheaper half, capped at the new price;
  - otherwise Google Shopping via Serper, falling back to SerpApi if Serper fails, with at least 2 offers agreeing within 25%;
  - eBay if keys are set.
- **Items:** Google Shopping with brand + description, reported as the 25th–75th percentile of at least 5 new offers. Searches go round-robin by category value, 40 per sweep.
- **FX:** Frankfurter (ECB) when a price is converted.
- Every price carries its URL and retrieval date. No model writes a price or a total.
