# Failure log

> To be filled in from the real sweep in the demo video and its `eval_results.json` / `metrics.json`.
> Nothing below is a measured result yet.

## The three worst errors in the run

| # | Error (measured) | Root cause | What I tried | Measured fix (before → after) | What I would change |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

## Risks to check first (predicted before the first real sweep, not measured)

- **Floor area near the 10% bar.**
  - Floor points come from the floor-line marks multiplied by camera height. A 3% error in camera height becomes about 6% in area.
  - Furniture against walls pulls the outline inward.
  - How to test: compare against the tape measure, and check `room.json` → `overviews[].camera_height_m` spread.
- **Thin spines under 1.5 cm.** At about 1 m from the shelf, a 1 cm spine is about 20–30 px wide, so a 3 px box error is 10–15%. The planned fix, if it fails, is to snap box edges to the strongest vertical gradient.
- **Catalogue dimensions missing or for the wrong edition.** These reduce the number of rulers, so check `measurements.json` → `ruler_count`.
- **Merging duplicates across frames:** a chain of homographies breaks on a fast pan, leaving extra segments that may double-count. Check `spines.json` → number of segments.
- **eBay asking prices versus hand-checked prices.** Use the same source type (eBay listings) for the 15 hand-checked prices.

## Cost per sweep and latency per stage

From `runs/<id>/metrics.json`:

| Stage | Latency (s) | LLM calls | Cost (USD) |
|---|---|---|---|
| keyframes | | | |
| spines | | | |
| identify | | | |
| scale | | | |
| room | | | |
| items | | | |
| pricing | | | |
| **Total (wall clock)** | | | |

The live session cost is not in `metrics.json`. Read it from the Gemini usage for the session.

## With two more weeks

-
