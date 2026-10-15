"""Laser ground truth for a room: ceiling height, floor area, perimeter from a z-up Faro cloud.
Walls are sliced above door height so openings do not leak; interior = flood fill from the floor's densest cell."""
import sys, numpy as np, cv2
z = np.load(sys.argv[1]); P = z["P"].astype(np.float64); N = z["N"]
h = P[:, 2]
hist, e = np.histogram(h, bins=400)
lo_half = np.arange(len(hist)) < len(hist) // 2
k_floor = np.argmax(np.where(lo_half, hist, 0)); k_ceil = np.argmax(np.where(~lo_half, hist, 0))
floor_z, ceil_z = (e[k_floor] + e[k_floor + 1]) / 2, (e[k_ceil] + e[k_ceil + 1]) / 2
H = ceil_z - floor_z
band = (h > floor_z + 2.1) & (h < ceil_z - 0.12) & (np.abs(N[:, 2]) < 0.3)   # vertical surfaces above doors
res = 0.02
xy = P[band, :2]; mn = xy.min(0) - 0.5
ij = ((xy - mn) / res).astype(int)
W, Hh = ij.max(0) + 2
occ = np.zeros((Hh, W), np.uint8); occ[ij[:, 1], ij[:, 0]] = 255
occ = cv2.dilate(occ, np.ones((5, 5), np.uint8))          # close 10 cm gaps
fl = P[np.abs(h - floor_z) < 0.03, :2]
fij = ((fl - mn) / res).astype(int); fij = fij[(fij[:, 0] < W) & (fij[:, 1] < Hh) & (fij >= 0).all(1)]
seed = tuple(np.median(fij, axis=0).astype(int))
mask = np.zeros((Hh + 2, W + 2), np.uint8); fill = occ.copy()
cv2.floodFill(fill, mask, seed, 128)
room = (fill == 128).astype(np.uint8)
leak = room[0].any() or room[-1].any() or room[:, 0].any() or room[:, -1].any()
area = room.sum() * res * res + 0.0
cnts, _ = cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key=cv2.contourArea)
poly = cv2.approxPolyDP(c, 0.05 / res, True)
perim = cv2.arcLength(poly, True) * res
# area grows by half the dilation each side: correct using the perimeter
area_corr = area + perim * (2 * res)
print(f"ceiling {H:.3f} m | floor area {area_corr:.2f} m2 (raw fill {area:.2f}) | perimeter {perim:.2f} m | corners {len(poly)} | leaked {leak}")
ys, xs = np.nonzero(room)
print("mask bbox m:", round((xs.max()-xs.min())*res,2), round((ys.max()-ys.min())*res,2))
print("poly m:", (poly.reshape(-1,2)*res).round(2).tolist())
vis = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR); vis[room > 0] = (0, 160, 0)
cv2.polylines(vis, [poly], True, (0, 0, 255), 2)
cv2.imwrite(sys.argv[2] if len(sys.argv) > 2 else "laser_room.png", vis)
