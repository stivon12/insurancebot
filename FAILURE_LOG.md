# Failure log

## The three worst errors

### 1. Room outline: wrong shape, floor too small (bedroom)

**Seen:**
- the bedroom is L-shaped, but the packet gives 3.91 m² (an earlier sweep gave 4.75 m²);
- the map shows each wall as several parallel lines about 0.5 m apart;
- the narrow part of the room is missing.

**Root causes, found in turn:**

| What was wrong | Root cause | Fix → measured result |
|---|---|---|
| A third of every rotation was lost, and the gyro axes were mixed | Chrome on Android reports the gyro's axes in a different order, and frame encoding on the main thread dropped about 120 ms of motion events every 200 ms | Capture moved to a worker, plus the phone's own 60 Hz orientation stream → 0% gaps on the phone |
| Rotations came out mirrored | The orientation matrix had determinant −1 | Fixed → orientation within 2–4° of gravity on real sweeps |
| Motion and frames out of sync | Samples were stamped with arrival time, not sensor time | Sensor timestamps, motion read from a window before each frame |
| Once lost, the path never came back | No way to re-anchor | Restart from the last position (counted as a "break") → 88–89% of frames tracked, 6–10 breaks |
| Wardrobe and rack tops counted as walls | The wall band started at furniture height | Only the band under the measured ceiling → 4.08 → 4.75 m² |
| **Still open:** doubled walls and the narrow part missing | Drift at each break makes "ghost" walls. The floor fill stops at the inner ghost. Gaps under 0.9 m are closed as doorways, which seals off the narrow part | — |

**What I'd change:** merge the ghost walls by re-aligning the path at each break (loop closure on SIFT matches), and only close a doorway where the path never crossed it.

### 2. Spines: wrong titles and double counts (bookstore)

**Seen:** in a hand check of 40 identified books, 9 were wrong, all with confidence ≥0.75. Of 10 repeat pairs, about 6 were the same physical book counted twice.

**Root causes:**
1. a spine view showing only the author was matched to a record titled with the author's name ("Nora Roberts / Nora Roberts");
2. a series name with no author ("CHILDCRAFT", "THE WORLD OF SCIENCE") was matched to one work, though each volume is a different book;
3. boxes spanning 2–3 flat-stacked spines, so the read picked up a neighbour's title, and the count and thickness are off;
4. the cross-frame merge chains only the 1 fps book frames, so a revisited shelf becomes a separate segment and its books are counted again.

**Fix for 1 and 2:**
- reject author-only reads;
- reject records titled with their author;
- treat a title read with no author on 3+ spines as a series name.

**Measured on the same saved reads:** 9 wrong of 40 → 2 wrong of 33, with no correct answer lost.

**Earlier, the same kind of problem in counting:** Gemini alone over-counted by 16%; RF-DETR boxes plus readable Gemini spines gave +4.7% (4TU benchmark).

**What I'd change:**
- split boxes that cover several spines before reading;
- chain shelf frames through the 5 fps tracking frames, so revisits stay in one segment;
- merge repeats on spine text as well as title.

### 3. Packet took 9 min 40 s (bookstore, 448 books)

**Seen:** the packet arrived 580 s after the sweep, against a 300 s bar.

**Root cause:**
- Google Books ran out of quota about 90 s after the sweep ended and refused about 1,100 requests (HTTP 429) over 8 minutes;
- each refusal was retried 3 times with sleeps, while holding one of 16 request slots, so every other lookup queued behind them;
- the same volume was also fetched once per frame it appeared in;
- nothing enforced the 5-minute budget.

**Fix:**
- a host that refuses 3 times in a row is skipped for 30 s, and Open Library and the Library of Congress answer meanwhile;
- identical requests made at the same moment are sent once;
- hard deadlines: identification stops at 150 s and pricing at 240 s, and the rest go to the review queue.

**Measured:** in a mock test, 40 lookups against a refusing host took 1.5 s and 3 requests, where before they took minutes of retries. **Not yet re-run on a live sweep.** Gemini prepaid credits ran out during this work.

## Other failures fixed along the way

- **No agent voice for about 5 sweeps:** echo cancellation had put Android into call mode, so the speaker was quiet. Turned off, with the mic held back while the agent talks.
- **PS5 priced at $22 and MacBook at $45:** the shopping results were accessories. Now eBay searches within the product's category with an accessory filter → $500–660 and $1,627–3,725.
- **Junk items** (a blurred wardrobe top as a "washing machine"): a single sighting now needs confidence 0.9, otherwise two sightings. Duplicates are merged from crops at the end.
- **The agent spoke its tool JSON and reasoning** ("```json {…} Let me formulate a flag_view call…"): the `[doubt]` note had asked it to both ask and call a tool. The note is now recorded by the server and the agent only asks, and such text is hidden from the phone.
- **A Gemini outage (HTTP 402) crashed the packet build:** model errors now become "no answer" plus a review line.

## Cost and latency per stage

| Stage | Bookstore (448 spines) | Bedroom |
|---|---|---|
| Spine lists + reads | $2.46, 464 calls, about 10–20 s per frame (in parallel during the sweep) | — |
| Identification | $2.87, 1,459 model calls, 3,498 HTTP calls | — |
| Items | $0.02, 30 scans | $0.02, 20 scans |
| Pricing | free APIs, 2,183 HTTP calls | 19 calls |
| Room (local) | — | 0.1 s per tracking frame, 0.5 s per depth keyframe |
| **Sweep total** | **$5.35, 580 s to packet** | **$0.02, 6.7 s to packet** |

## With two more weeks

1. **Ground truth first:**
   - a set of real sweeps (several rooms and shelves) with tape measurements, hand-read titles, measured spines and checked prices;
   - every change scored on all of them with `eval/eval.py`, not on one room.
2. **A/B tests of models and APIs** on that set: spine readers (Flash vs Pro vs open OCR), box detectors, depth models, catalogue and price sources. Pick on accuracy, cost and latency.
3. **LiDAR for the room** on phones that have it (ARKit depth and poses). My phone has no LiDAR, so I couldn't test it on a real sweep, only through the StrayScanner captures.
4. **An LLM "context critic" for each model.** Once the models are chosen, collect the cases where each scores low, and give an LLM that context so it can clean, filter or steer the model's output. Examples:
   - for the depth model: a low ceiling, odd room shapes, beds and clutter read as walls;
   - for titles: wrong neighbours and series names.

   The deduplication and title search already do a small version of this (an LLM judges crop pairs and catalogue candidates). Done systematically, with each model's known failure modes, it could be a large gain.
5. **Fix the known open issues:** ghost walls (loop closure), multi-spine boxes, and revisit double counts.
