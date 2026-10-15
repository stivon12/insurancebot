# DA3 room method: real-data checks (2026-10-15)

**Status: not validated. It failed on a held-out recording.** Keep the cross-check and review routing, and validate on a protocol-following capture with a tape measure before trusting it.

## Ground truth
- **Room:** ARKitScenes room 421337, Faro laser cloud (`roomscope/out/gt_cache/421337.npz`).
- **How the numbers were made:** `scripts/laser_gt.py` slices the vertical laser surfaces above door height and flood-fills the room on a 2 cm grid.
- **Result:** ceiling **3.055 m**, floor area **15.30 m²**, perimeter **15.92 m** (about 4.3 × 3.6 m with small notches). Roomscope's LiDAR floor area, 15.2 m², agrees with this.
- **Correction:** an earlier note compared against roomscope's LiDAR perimeter (17.88 m), which over-reads. The "+1.6 % perimeter" claimed from that comparison was wrong.

## Input
- **Images:** 7–8 eye-level photos per recording, chosen by roomscope and spread around the room. This is not the centre turn the app guides the user through.
- **Gravity:** none in these photos; "up" was found from the cloud.
- **Scale:** set from the true camera height (ARKit poses over a LiDAR floor: 1.53 m and 1.48 m). That is how the app gets scale from the book ruler.

## Results (settings tuned on 42444946)
| Recording | Floor area | Perimeter | Ceiling |
|---|---|---|---|
| 42444946, scale from the true ceiling (the earlier, optimistic number) | 16.59 m² (+8 %) | 18.16 m (+14 %) | n/a (used as the scale) |
| 42444946, scale from camera height | 18.78 m² (**+23 %**) | 19.22 m (+21 %) | 3.23 m (+6 %) |
| **42444949, held out**, scale from camera height | 28.08 m² (**+84 %**) | 22.88 m (+44 %) | not found (no ceiling peak) |

DA3 itself is fast: 3–4 s for 7–8 images on an M1.

## What failed
1. **The settings were overfitted.** The wall band, smoothing window and wall-break tolerance were chosen on the same room they were scored on, and they did not transfer.
2. **Over-reach.** Walls are read as the far edge of the points in each bearing, measured from the centroid of scattered cameras, so points seen through doorways and windows push the walls out.
3. **The camera height relative to the room is off in DA3's reconstruction.** On 42444946 the ceiling came out 6 % high even with the true camera height. Area roughly doubles that error.
4. **These photos do not follow the app's protocol.** The app asks for a centre turn plus a tilted turn, with gravity data. Whether that protocol fixes any of 1–3 is unmeasured.

## Root cause still to establish (next)
- **Separate shape error from scale error.** Compare DA3 depth with the LiDAR depth frame by frame on these captures.
- **Capture a protocol-following centre turn** in a room with a tape measure. This is the only fair test of both room methods, DA3 and gyro.
