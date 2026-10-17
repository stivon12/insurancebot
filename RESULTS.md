# Results against the pass bars

**Measured against:**
- **Room:** a tape measure in the bedroom (6.5 m² filmed floor, 3.0 m ceiling, L-shaped).
- **Spine heights:** catalogue heights.
- **Book count:** a rough hand count of the demo shelf (about 450).
- **Titles:** a 40-book spot check of the bookstore frames.
- **Prices:** a ballpark check of book and item prices.

`eval/eval.py` scores a packet against a full ground-truth sheet, and `eval/spine_proxy.py` runs the spine check.

| Measure | Bar | Result | Evidence | Status |
|---|---|---|---|---|
| Book count | within 5% | 448 vs about 450 | Against a rough hand count of the demo shelf (under 1% off); +4.7% on 295 labelled spines (4TU benchmark) | Pass |
| Titles correct | ≥70% of legible spines | about 65% of 332 legible | 40-book spot check after the title fix: 31 of 33 right. Unreadable and unmatched spines go to review with their sizes | Just under |
| Confidently wrong | ≤3% | about 4% | 2 of 33 wrong in the spot check, both from boxes spanning two spines | Just over |
| Spine dimensions | within 15% | 57 of 77 within 15% | Pixel-measured heights against catalogue heights, median error 7.1% | Mostly pass |
| Book prices | within 25% | ballpark right | Checked as reasonable. Every price has a source URL and date | Pass (ballpark) |
| Non-book items | ≥80% found | PS5, MacBook, TV, router, backpack, bed, racks, wardrobe | All main valuables found with the right category; one duplicate | Pass (spot check) |
| Floor area | within 10% | 3.91 m² vs 6.5 m² | −40%: doubled walls and the narrow part of the L sealed off (FAILURE_LOG #1) | Fail |
| Room shape | non-rectangular handled | L not detected | The fill closes the narrow part as a doorway | Fail |
| Ceiling | — | 2.71 m vs 3.0 m | −10%; scale from the assumed 1.4 m phone height runs short | Fail |
| Wall area | within 15% | 22.2 m² | Low, given the ceiling and the outline | Likely fail |
| Time to packet | under 5 min | bedroom 6.7 s; bookstore 240 s | The demo sweep took 580 s before the API fixes; 240 s re-run after them (FAILURE_LOG #3) | Pass after fix |

**To finish the spine bar:** measure 20 books with a ruler, film that shelf once (about $0.30 of Gemini), and run `eval/eval.py`.

## Development measurements (not the demo sweeps)

- **Spine counting:** 15 unseen 4TU shelf photos, 295 labelled spines.

  | Method | Count error |
  |---|---|
  | Gemini alone | +15.9% |
  | RF-DETR alone | −28% |
  | RF-DETR boxes + readable Gemini spines (used) | **+4.7%** (precision 88%, recall 92%) |

- **Spine sizes after the "rulers only" change (streamed re-run):** only 9 rulers had catalogue heights; 3 of 9 within 15%, median 17.6%. Fewer rulers per frame make the scale noisier: that change traded size accuracy for speed.
- **Camera path:** two StrayScanner iPhone captures with ARKit (LiDAR) poses as truth, replayed through a test rig that copies the Android phone's sensor behaviour.

  | Capture | Path error (RMS) | Scale |
  |---|---|---|
  | Office | 0.34 m | 1.05 |
  | Second capture | 0.89 m | 1.04 |

  The rig's scripts were scratch tools and aren't in the repo.

- **Old room method (one turn on the spot), same office capture against LiDAR:** floor −24%, walls −6%. That method was replaced because one spot can't see L-shaped rooms.
