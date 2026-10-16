# Failure log

Measured on test data while building the system. The demo-video sweep is still to come, and its numbers replace this section.

The test data is:
- three StrayScanner iPhone walkthroughs with LiDAR, which give the reference geometry;
- 30 photos from the 4TU book-spine dataset, with every spine labelled.

## Worst errors so far

| # | Error (measured) | Root cause | Fix tried → result | What I'd change |
|---|---|---|---|---|
| 1 | Room floor 38–69% too small (gyro + floor-line method) | The turn saw the floor line over only 120–210°, and the outline closed straight across the walls it never saw. Even with true poses, the convex hull of the marks was 4–6 m² against 8.4–9.6 m² | Refuse an outline under 200° and flag under 300°; replace the marks with MoGe-2 depth → c7d28f floor 7.26 vs 9.6 m² (−24%), wall 34.4 vs 36.6 m² (−6%) | A second depth view per bearing, so the frame-to-frame scale noise (0.75–1.35× vs LiDAR) averages out |
| 2 | Book count +16 to +18% (Gemini lists spines twice) | Short, unreadable "spines" that are junk, plus re-listed neighbours | Rule A (RF-DETR boxes + readable unmatched Gemini spines) → +4.7% on unseen photos | Flat stacks: frames of a stack filmed while walking don't line up (parallax), so one stack still counts as 7 against 3 |
| 3 | A tissue box priced at $1,600–5,625 | The shopping query was the brand alone ("Scott" matched Scott bicycles) | Query = brand + description → $5–64 | Check each offer's title against the item's category |

Other fixes made along the way:
- **Focal length was never measured on any sweep.** A sign error made every estimate negative, so it always fell back to the assumption. Fixed, with a synthetic pure-turn test.
- **Bad eBay keys crashed pricing.** Now eBay switches itself off.

## Cost and latency (replays and smoke tests)

| What | Cost | Latency |
|---|---|---|
| Room, MoGe-2 per turn frame | free, local | 0.43 s on M1 MPS (2.6 s first frame) |
| Shelf frame, spines | about $0.006 per frame | about 10–20 s per frame |
| Pricing per book | free shop pages; at most 1 Serper search (about $0.001) | — |
| Old pipeline, three captures | $0.05–0.24 per sweep | 43–125 s after Finish |
| New engine | — | packet 10–19 s after Finish |
