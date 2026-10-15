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
