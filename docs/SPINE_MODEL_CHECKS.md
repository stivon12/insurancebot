# Spine counting: model checks on the Library Dataset (2026-10-16)

**Data:** 10 random annotated "split" photos from llabres/library-dataset, seed 7, 169 books, the same photos for every model.
**Mode:** count only, no title reading.

| Model | Books counted | Photos exact | Photos within ±1 | Time per photo | Cost (10 photos) |
|---|---|---|---|---|---|
| `gemini-3.8-flash`, low thinking (current) | 183 (+8.3 %) | 3/10 | 7/10 | 23.9 s | $0.14 |
| `gemini-4-argon`, low thinking | 191 (+13.0 %) | 4/10 | 7/10 | 28.1 s | not priced (similar token use) |
| RF-DETR-nano bookshelf **v1** (Apache-2.0, local, open), 3 tiles | 165 (−2.4 %) | 0/10 | 4/10 | 0.7 s | free |
| RF-DETR v1 + drop boxes shorter than 50 % of the row's tallest-quartile height | 143 (−15.4 %) | 1/10 | 5/10 | 0.3 s | free |

## Notes
- **RF-DETR v2 was not scored.** It was distilled on 285 photos of this same dataset, so its scores here would be inflated (train/test leak).
- **RF-DETR boxes are tight,** one per spine. That is what spine thickness needs, and what Gemini lacks: its boxes often span 2–4 spines.
- **RF-DETR misses thin spines on dense shelves.** It found 30 of 43, 24 of 34, 11 of 14 and 10 of 12. Its −2.4 % total looked good only because background books on distant shelves were counted too, cancelling the misses. The height filter removes those background books and exposes the misses.
- **Argon:** better on the two small photos, but on one photo its two passes over the same row disagreed (16 vs 34).
- **Conclusion:** Gemini Flash stays the counter. RF-DETR is a candidate *box refiner*: where an RF-DETR box matches a Gemini spine, use the tighter box for thickness. Not yet tested.

## Held-out check, including RF-DETR v2 (photos v2 never trained on)
- **Images:** 15 random annotated split images from the 42 library photos in v2's own validation split, checked to share no photo with its training split. 14 scored, 228 books.
- **Small bias in v2's favour:** its best checkpoint was selected on these photos.

| Model | Books counted | Photos exact | Photos within ±1 | Time per photo | Cost |
|---|---|---|---|---|---|
| `gemini-3.8-flash` | 256 (+12.3 %) | 4/14 | 9/14 | ~24 s | $0.19 |
| RF-DETR v1 (+ height filter) | 216 (−5.3 %) | 2/14 | 6/14 | 0.35 s | free |
| RF-DETR v2 (+ height filter) | 506 (+122 %) | 1/14 | 4/14 | 0.33 s | free |

**Findings**
- **v2 boxes every distant background book in the room**: 235 boxes on a 15-book shelf.
- **v2 also misses many foreground spines.** Its training labels came from a GPT model, which labelled all visible books.
- **The height filter cannot fix v2:** background boxes are the majority, so they set the reference height the filter compares against.
- **v1 is the better open model,** and still behind Flash per photo.

## Overcount re-check: the reference, not the model
- **The split images are overlapping crops of one photo,** and a book appears to be annotated in only one crop. Some visible items are also annotated `not_book`.
- **So per-crop "books" undercounts what is visible.** Real books such as Histología, Embriología clínica and "Electricidad para todos" were visible but unannotated in the crops checked.
- **Scored against the range** [books only, all annotated masks]:
  - Flash falls inside it for 9/10 photos in the first set and 10/14 in the held-out set.
  - On the held-out set, Flash's total equals the all-masks total exactly (256 vs 256).
- **Conclusion:** this dataset cannot confirm or refute the 5 % count bar. The author's own shelf, counted by hand, is the test.

## RF-DETR v1 as a box refiner (pipeline/spine_boxes.py)
- **Coverage:** on the 10 photos, 143 of 183 Gemini spines (78 %) matched a tight RF-DETR box.
- **Effect on width:** tight/loose width ratio median 0.93, 10th percentile 0.76, 90th percentile 1.02.
- **Unmatched spines** keep the Gemini box and are flagged in the review queue.
- **Thickness accuracy itself is unmeasured** until hand-measured spines exist.
