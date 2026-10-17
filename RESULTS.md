# Results against the pass bars

**No full ground truth was collected for the demo sweeps:**
- the books belong to a coworking-space library, so I couldn't take them out to count, hand-read and measure them;
- the bedroom wasn't tape-measured.

So no bar below is marked as passed unless it was measured. Where I have evidence, it's my own spot check on the saved frames, or a development benchmark, and labelled as such. `eval/eval.py` scores a packet against the brief's ground-truth sheet as soon as one exists (formats in `ground_truth_template/`).

Statuses:
- **Pass / Fail:** measured against the bar.
- **Estimate:** my own check on frames, not true ground truth.
- **Not measured:** no ground truth.

| Measure | Bar | What the system produced | Evidence | Status |
|---|---|---|---|---|
| Book count | within 5% | 448 spines (bookstore) | No true count. Two known biases: boxes that span 2–3 flat-stacked spines (undercount), and the same book counted twice from revisited frames (6 of 10 checked repeat pairs; overcount). Dev benchmark (4TU shelf photos, 295 labelled spines): **+4.7%** | Not measured on the demo |
| Titles correct | ≥70% of legible spines | 240 identified + 19 appraisal of 448 | Spot check of 40 random identified books against their spine crops: 30 correct, 1 plausible, 9 wrong. About 45% of all 448 detected spines got a correct title; the share of *legible* spines is unknown | Estimate: fail on all detected spines |
| Confidently wrong | ≤3% | — | Same sample: 9 of 40 wrong, all with confidence ≥0.75. That's about 13% of all detected spines, **as in the demo packet**. After the fix (author-name and series-name rules, measured on the same saved reads): 2 wrong of 33, about 3% of detected spines, with no correct answer lost | Estimate: **fail in the demo**, about at the bar after the fix |
| Spine dimensions | within 15% on 20 | heights from catalogue (77), book ruler (368), standard formats (3) | Not hand-measured. Known error: a box spanning several stacked spines gives a "thickness" of up to about 9 cm | Not measured |
| Book prices | within 25% on 15 | 227 eBay, 7 Google Shopping, 1 AbeBooks, 5 none (excluded) | Not hand-checked. Every price has its URL, retrieval time and assumed condition. 10 books are also priced for GB (`locale_comparison`) | Not measured |
| Non-book items | ≥80% found, right category | bedroom: 10 lines (PS5, MacBook, TV, router, backpack, bed, 2 clothes racks, wardrobe twice) | No true list. One duplicate (the wardrobe, seen once only by its edge). PS5 $500–660 and MacBook $561–1,106 from eBay | Not measured |
| Floor area | within 10% | bedroom: 3.91 m² (2.29 × 1.98 m) | No tape. The room is L-shaped (a narrow part at one end); the measured outline is a smaller blob, not an L. An earlier sweep of the same room gave 4.75 m². See FAILURE_LOG #1 | Not measured (likely fail) |
| Wall area | within 15% | bedroom: 22.2 m² gross, ceiling 2.71 m | No tape | Not measured |
| Time to packet | under 5 min | bedroom **6.7 s**; bookstore **580 s** in the demo | The recorded bookstore sweep re-streamed through the server after the fixes: **240 s**, with 280 books identified before the deadline. Pricing was still cut short there (209 unpriced, eBay quota); the eBay fix isn't measured yet (FAILURE_LOG #3) | Bedroom **pass**; bookstore **fail** in the demo, **pass** on time in the re-run |

## Development measurements (not the demo sweeps)

- **Spine counting:** 15 unseen 4TU shelf photos, 295 labelled spines.

  | Method | Count error |
  |---|---|
  | Gemini alone | +15.9% |
  | RF-DETR alone | −28% |
  | RF-DETR boxes + readable Gemini spines (used) | **+4.7%** (precision 88%, recall 92%) |

- **Camera path:** two StrayScanner iPhone captures with ARKit (LiDAR) poses as truth, replayed through a test rig that copies the Android phone's sensor behaviour.

  | Capture | Path error (RMS) | Scale |
  |---|---|---|
  | Office | 0.34 m | 1.05 |
  | Second capture | 0.89 m | 1.04 |

  The rig's scripts were scratch tools and aren't in the repo.

- **Old room method (one turn on the spot), same office capture against LiDAR:** floor −24%, walls −6%. That method was replaced because one spot can't see L-shaped rooms.
