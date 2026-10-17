// Capture worker: scores, keeps and encodes the camera frames off the page's main thread, so the phone keeps
// delivering motion and orientation events (frame work on the main thread dropped ~120 ms of them every 200 ms).
"use strict";

let best = null;       // the sharpest full-size view of the current second
const hist = [];       // blur score of each second's best view: 'blurred' is relative to this session
let chain = Promise.resolve();

const qc = new OffscreenCanvas(160, 120);
const qx = qc.getContext("2d", { willReadFrequently: true });

function score(bmp) {
  const w = 160, h = Math.max(2, Math.round(160 * bmp.height / bmp.width));
  qc.width = w; qc.height = h;
  qx.drawImage(bmp, 0, 0, w, h);
  const d = qx.getImageData(0, 0, w, h).data;
  const g = new Float32Array(w * h);
  let bright = 0;
  for (let i = 0; i < w * h; i++) {
    g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
    if (g[i] > 250) bright++;
  }
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x, lap = g[i - 1] + g[i + 1] + g[i - w] + g[i + w] - 4 * g[i];
    sum += lap; sum2 += lap * lap; n++;
  }
  return { blur: sum2 / n - (sum / n) ** 2, glare: bright / (w * h) };
}

async function jpeg(bmp, quality) {
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  c.getContext("2d").drawImage(bmp, 0, 0);
  return (await c.convertToBlob({ type: "image/jpeg", quality })).arrayBuffer();
}

async function handle({ t, tSource, orient, small, full, last }) {
  const s = score(small);
  const track = await jpeg(small, 0.6);
  small.close();
  if (!best || s.blur > best.s.blur) {
    if (best) best.full.close();
    best = { full, s, t, tSource, orient };
  } else {
    full.close();
  }
  const out = { track: { t, tSource, orient, data: track } };
  if (last && best) {
    const data = await jpeg(best.full, 0.85);
    hist.push(best.s.blur);
    if (hist.length > 30) hist.shift();
    const med = hist.length >= 5 ? [...hist].sort((a, b) => a - b)[hist.length >> 1] : null;
    out.frame = { t: best.t, tSource: best.tSource, orient: best.orient, w: best.full.width, h: best.full.height,
                  blur: best.s.blur, glare: best.s.glare, glareBad: best.s.glare > 0.08,
                  blurBad: med ? best.s.blur < 0.5 * med : best.s.blur < 30, data };
    best.full.close();
    best = null;
  }
  postMessage(out, [track, ...(out.frame ? [out.frame.data] : [])]);
}

onmessage = (e) => {
  if (e.data === "ping") { postMessage({ ready: typeof OffscreenCanvas !== "undefined" && "convertToBlob" in OffscreenCanvas.prototype }); return; }
  chain = chain.then(() => handle(e.data)).catch((err) => postMessage({ error: String(err) }));
};
