# DA3 room method: first real-data check

**What was tested:** `pipeline/room_da3.measure` on real images. Not the full app.

**Input**
- **Capture:** ARKitScenes 42444946 (iPad Pro 2020). One room, not a simple rectangle (10-vertex outline).
- **Images:** 8 eye-level photos chosen from the recording by the author's roomscope project (`out/photo_inputs/42444946/room`).
- **What these photos are not:** they are spread around the room, not the centre turn the app guides the user through, and they include no gravity data. "Up" was therefore found from the floor and ceiling planes in the cloud, which is the fallback path.

**Reference (roomscope LiDAR tier)**
- Ceiling 3.06 m, validated against the Faro laser to within 0.8 cm.
- Floor area 15.20 m², perimeter 17.88 m.
- Wall placement on this tier: median 8.7 cm.

**Scale:** set from the reference ceiling, so this isolates the shape error. Scaling from the camera height that run produced (1.45 m) gives the same numbers, so the two routes agree with each other. In the app, scale comes from the book ruler, and its error adds on top of what is shown here.

**Speed:** DA3-BASE on 8 images took 3.5 s on an M1 (Apple GPU), after the model was loaded.

| Version | Walls seen over | Floor area | Perimeter | Corners |
|---|---|---|---|---|
| Ceiling band cut at the 99th percentile (bug) | n/a | camera came out above the ceiling (3.9 m) | n/a | n/a |
| Wall band 75–95 % of room height | 98° | 1.48 m² (unusable) | 8.76 m | 4 |
| Wall band 30–90 %, no smoothing | 272° | 17.27 m² (+14 %) | 19.82 m (+11 %) | 16 |
| **Wall band 30–90 %, 18° circular median, 0.25 m wall-break tolerance (shipped)** | **272°** | **16.59 m² (+9 %)** | **18.16 m (+1.6 %)** | 8 |

**Reading**
- **Perimeter**, which drives wall area, is within 2 %.
- **Floor area** is +9 %, just inside the 10 % bar before any error from the book-ruler scale is added. The outline over-reaches on one axis: 4.86 m against a 4.45 m bounding box. Likely cause, not yet confirmed: the room's notch is smoothed over and the far side is seen through an opening.
- **Limits:** this is one capture. Expect a centre turn with gravity data to do better. Whether it does is unmeasured until a real sweep.
