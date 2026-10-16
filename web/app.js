// Phone client: camera + mic + motion sensors -> server; agent audio + live inventory <- server.
// Plain web APIs only, so it runs in iOS Safari and Android Chrome (HTTPS required for camera/mic/motion).
"use strict";

const $ = (id) => document.getElementById(id);
const FRAME_MS = 1000;         // 1 fps: every frame goes to the pipeline and the live model
const state = { ws: null, sweepId: null, frames: 0, imu: [], gravitySign: 0, playCtx: null, nextPlay: 0, sources: [], agentLine: null, userLine: null,
  stream: null, micNode: null, frameTimer: null, imuTimer: null, capturing: false };

function b64(buf) {
  const bytes = new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function setStatus(t) { $("status").textContent = t; }

function show(id) { ["welcome", "stage", "after"].forEach((x) => { $(x).hidden = x !== id; }); }

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text; // always textContent: model and catalogue text is never HTML
  return e;
}

function addTranscript(who, text) {
  const key = who === "agent" ? "agentLine" : "userLine";
  if (!state[key]) {
    state[key] = el("div", who, "");
    $("transcript").appendChild(state[key]);
  }
  state[key].textContent += text;
  const pane = $("talkPane");
  pane.scrollTop = pane.scrollHeight;
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.setAttribute("aria-selected", String(x === t)));
    ["talkPane", "invPane"].forEach((p) => { $(p).hidden = p !== t.dataset.pane; });
  };
});

// ---------- live inventory (preview), grouped by kind ----------
function renderInventory(items) {
  const box = $("inventory");
  box.textContent = "";
  $("invCount").textContent = items.length ? `(${items.length})` : "";
  if (!items.length) { box.appendChild(el("div", "empty", "Things the agent recognises appear here.")); return; }
  const groups = {};
  items.forEach((it) => { (groups[it.kind || "other"] = groups[it.kind || "other"] || []).push(it.label); });
  Object.keys(groups).sort().forEach((k) => {
    const g = el("div", "inv-group");
    g.appendChild(el("h3", "", `${k.replace(/_/g, " ")} · ${groups[k].length}`));
    const ul = el("ul");
    groups[k].forEach((label) => ul.appendChild(el("li", "", label)));
    g.appendChild(ul);
    box.appendChild(g);
  });
}

// ---------- processing screen ----------
function startProcessingView() {
  show("after");
  state.procStart = state.procStart || Date.now();
  clearInterval(state.clock);
  state.clock = setInterval(() => {
    const s = Math.round((Date.now() - state.procStart) / 1000);
    $("elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 500);
}

function money(v, cur) {
  if (v === null || v === undefined) return "–";
  return new Intl.NumberFormat(undefined, { style: "currency", currency: cur, maximumFractionDigits: v >= 100 ? 0 : 2 }).format(v);
}

function renderResult(m) {
  clearInterval(state.clock);
  $("barFill").style.width = "100%";
  $("pctText").textContent = "100%";
  $("procTitle").textContent = "Claim packet ready";
  $("procSub").textContent = "Every line links to its frame and price source in the report.";
  const t = m.totals || {}, cur = m.currency || "USD";
  const r = $("result");
  r.textContent = "";
  r.appendChild(el("h2", "", "Summary"));
  r.appendChild(el("p", "sub", m.summary));
  const stats = el("div", "stats");
  const stat = (k, v, s) => { const d = el("div", "stat"); d.appendChild(el("div", "k", k)); d.appendChild(el("div", "v", v)); if (s) d.appendChild(el("div", "s", s)); stats.appendChild(d); };
  stat("Books", String(t.book_count ?? "–"), `${t.books_identified ?? 0} identified · ${t.books_unidentified ?? 0} unreadable`);
  stat("Books, replacement", money(t.books_replacement_cost, cur), `used ${money(t.books_used_value, cur)}`);
  stat("Other contents", `${money(t.items_replacement_cost_low, cur)}–${money(t.items_replacement_cost_high, cur)}`, `${m.items ?? 0} items`);
  stat("Room floor", m.floor_area_m2 ? `${m.floor_area_m2} m²` : "not measured", `${m.review ?? 0} lines to review`);
  r.appendChild(stats);
  const actions = el("div", "actions");
  const a1 = el("a", "btn", "Open report"); a1.href = m.report; a1.target = "_blank";
  const a2 = el("a", "btn ghost", "Download claim_packet.json"); a2.href = m.json; a2.download = "claim_packet.json";
  actions.appendChild(a1); actions.appendChild(a2);
  r.appendChild(actions);
  r.hidden = false;
}

// ---------- sound (same path as the reference app: one AudioContext, playback straight to the speaker) ----------
function unlockAudio() {
  state.playCtx = state.playCtx || new (window.AudioContext || window.webkitAudioContext)();
  state.playCtx.onstatechange = checkSound;
  state.playCtx.resume().then(checkSound).catch(() => {});
}

function checkSound() {
  $("soundBar").hidden = !(state.playCtx && state.playCtx.state !== "running" && !state.ended);
}

$("soundBar").onclick = () => { unlockAudio(); setTimeout(checkSound, 300); };
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.playCtx) state.playCtx.resume().then(checkSound).catch(() => {});
});

function playPcm(base64) {
  const ctx = state.playCtx;
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const pcm = new Int16Array(bytes.buffer);
  const buf = ctx.createBuffer(1, pcm.length, 24000);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const t = Math.max(ctx.currentTime, state.nextPlay);
  src.start(t);
  state.nextPlay = t + buf.duration;
  state.sources.push(src);
  src.onended = () => { state.sources = state.sources.filter((x) => x !== src); };
}

function stopPlayback() {
  state.sources.forEach((x) => { try { x.stop(); } catch (e) {} });
  state.sources = [];
  state.nextPlay = state.playCtx ? state.playCtx.currentTime : 0;
}

function resampleToPcm16(input, inputRate, outputRate) {
  const ratio = inputRate / outputRate, n = Math.floor(input.length / ratio), pcm = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    const x = i * ratio, a = Math.floor(x), b = Math.min(a + 1, input.length - 1), w = x - a;
    const v = Math.max(-1, Math.min(1, input[a] * (1 - w) + input[b] * w));
    pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return pcm;
}

// ---------- camera-quality checks on a small grey copy ----------
const qc = document.createElement("canvas");
function quality(video) {
  const w = 160, h = Math.round(160 * video.videoHeight / video.videoWidth) || 120;
  qc.width = w; qc.height = h;
  const c = qc.getContext("2d", { willReadFrequently: true });
  c.drawImage(video, 0, 0, w, h);
  const d = c.getImageData(0, 0, w, h).data;
  const g = new Float32Array(w * h);
  let bright = 0;
  for (let i = 0; i < w * h; i++) {
    g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
    if (g[i] > 250) bright++;
  }
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const lap = g[i - 1] + g[i + 1] + g[i - w] + g[i + w] - 4 * g[i];
    sum += lap; sum2 += lap * lap; n++;
  }
  const variance = sum2 / n - (sum / n) ** 2;
  // Blur relative to this session's own frames (lighting and phone change what "sharp" scores); a fixed floor
  // only until enough frames are seen.
  const hist = (state.blurHist = state.blurHist || []);
  hist.push(variance); if (hist.length > 60) hist.shift();
  const med = hist.length >= 10 ? [...hist].sort((x, y) => x - y)[Math.floor(hist.length / 2)] : null;
  const blurBad = med ? variance < 0.4 * med : variance < 30;
  return { blur: variance, glare: bright / (w * h), blurBad, glareBad: bright / (w * h) > 0.08 };
}

// ---------- frames ----------
const fc = document.createElement("canvas");
function startFrames(video) {
  state.frameTimer = setInterval(() => {
    if (!state.capturing || !state.ws || state.ws.readyState !== 1 || !video.videoWidth) return;
    // Frames and mic audio share one socket: keep at most ~one frame queued so speech is never stuck behind a backlog
    // of frames on a slow uplink (skipped frames only lower the frame rate).
    if (state.ws.bufferedAmount > 3e5) return;
    fc.width = video.videoWidth; fc.height = video.videoHeight;
    fc.getContext("2d").drawImage(video, 0, 0);
    const q = quality(video);
    $("quality").textContent = q.blurBad ? "blurry" : q.glareBad ? "glare" : "camera ok";
    $("quality").className = "chip" + (q.blurBad || q.glareBad ? " warn" : "");
    const t = performance.now();
    const orient = (screen.orientation && screen.orientation.angle) || window.orientation || 0;
    fc.toBlob(async (blob) => {
      if (!blob || !state.capturing) return;
      state.frames++;
      $("frames").textContent = `${state.frames} frames`;
      state.ws.send(JSON.stringify({
        type: "frame", data: b64(await blob.arrayBuffer()), t, w: fc.width, h: fc.height, orient: (orient + 360) % 360,
        live: true, blur: q.blur, glare: q.glare, blur_bad: q.blurBad, glare_bad: q.glareBad,
      }));
    }, "image/jpeg", 0.85);
  }, FRAME_MS);
}

// ---------- motion sensors ----------
function onMotion(e) {
  const a = e.accelerationIncludingGravity, r = e.rotationRate;
  if (!a || a.x === null) return;
  // The spec says a phone held upright reports ay ~ +9.8; some browsers report the opposite sign.
  // Learn the sign once from the orientation sensor so the server always gets spec-direction data.
  if (!state.gravitySign && state.lastBeta !== undefined && Math.abs(state.lastBeta) > 45 && Math.abs(a.y) > 5) {
    state.gravitySign = Math.sign(a.y) === Math.sign(state.lastBeta) ? 1 : -1;
  }
  const s = state.gravitySign || 1;
  const t = performance.now();
  state.imu.push({ t_ms: t, ax: s * a.x, ay: s * a.y, az: s * a.z, ra: r ? r.alpha : 0, rb: r ? r.beta : 0, rg: r ? r.gamma : 0 });
}

function onOrientation(e) { if (e.beta !== null) state.lastBeta = e.beta; }

// Sweep over: release the camera and microphone (the phone's indicator turns off) and stop every timer and sensor.
// The AudioContext stays open only so any last words from the agent can finish playing.
function stopCapture() {
  if (!state.capturing && !state.stream) return;
  state.capturing = false;
  clearInterval(state.frameTimer);
  clearInterval(state.imuTimer);
  window.removeEventListener("devicemotion", onMotion);
  window.removeEventListener("deviceorientation", onOrientation);
  state.imu = [];
  if (state.micNode) { state.micNode.onaudioprocess = null; state.micNode.disconnect(); state.micNode = null; }
  if (state.micSource) { state.micSource.disconnect(); state.micSource = null; }
  if (state.stream) { state.stream.getTracks().forEach((t) => t.stop()); state.stream = null; }
  $("video").srcObject = null;
}

async function enableMotion() {
  try {
    if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
      await DeviceMotionEvent.requestPermission(); // iOS only; must run inside the tap handler
    }
    if (typeof DeviceOrientationEvent !== "undefined" && typeof DeviceOrientationEvent.requestPermission === "function") {
      await DeviceOrientationEvent.requestPermission();
    }
  } catch (e) { console.warn("motion permission", e); }
  window.addEventListener("deviceorientation", onOrientation);
  window.addEventListener("devicemotion", onMotion);
  state.imuTimer = setInterval(() => {
    if (state.ws && state.ws.readyState === 1 && state.imu.length) {
      state.ws.send(JSON.stringify({ type: "imu", samples: state.imu.splice(0) }));
    }
  }, 500);
}

// ---------- server messages ----------
function onMessage(ev) {
  const m = JSON.parse(ev.data);
  switch (m.type) {
    case "audio": playPcm(m.data); break;
    case "interrupted": stopPlayback(); state.agentLine = null; state.interrupts = (state.interrupts || 0) + 1; console.log("agent interrupted", state.interrupts); break;
    case "turn_complete": state.agentLine = null; state.userLine = null; break;
    case "transcript": addTranscript(m.who, m.text); if (m.who === "user") state.agentLine = null; break;
    case "locale": $("locale").textContent = `${m.country} / ${m.currency}`; break;
    case "inventory": renderInventory(m.items); break;
    case "status": setStatus(m.text); break;
    case "finished": endSweep(); break;
    case "error":
      if (state.ended) { $("errorBox").hidden = false; $("errorBox").textContent = "Error: " + m.text; }
      else setStatus("Error: " + m.text);
      break;
    case "packet": state.packetShown = true; renderResult(m); break;
  }
}

// ---------- connection ----------
// A dropped connection (Wi-Fi to mobile data, a tunnel, a weak signal) reconnects to the same sweep. The server keeps
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live?sweep_id=${state.sweepId}`);
  state.ws = ws;
  ws.onmessage = onMessage;
  ws.onclose = (ev) => {
    if (state.ws !== ws) return;
    if (ev.code === 4404) { setStatus("This sweep is no longer on the server: start a new sweep."); showError("Connection lost: start a new sweep."); return; }
    if (state.packetShown) return;
    const wait = Math.min(15000, 1000 * 2 ** (state.retries || 0));
    state.retries = (state.retries || 0) + 1;
    setStatus(`Connection lost. Reconnecting in ${Math.round(wait / 1000)} s...`);
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = setTimeout(connect, wait);
  };
  return new Promise((res) => {
    ws.onopen = () => {
      if (state.retries) {
        setStatus("Reconnected.");
        if (state.ended && !state.endSent) ws.send(JSON.stringify({ type: "end" })); // Finish was tapped while offline
      }
      state.retries = 0;
      res();
    };
  });
}

function sendEnd() {
  state.endSent = !!(state.ws && state.ws.readyState === 1);
  if (state.endSent) state.ws.send(JSON.stringify({ type: "end" }));
}

function showError(text) { $("errorBox").hidden = false; $("errorBox").textContent = text; }

// ---------- start ----------
// Camera, mic (16 kHz PCM16, as in the reference app), motion sensors and frame timer.
async function startCapture() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  state.stream = stream;
  state.capturing = true;
  const video = $("video");
  video.srcObject = stream;
  await video.play().catch(() => {});
  const ctx = state.playCtx;
  await ctx.resume().catch(() => {});
  const source = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = (e) => {
    e.outputBuffer.getChannelData(0).fill(0);
    if (!state.capturing || !state.ws || state.ws.readyState !== 1) return;
    state.ws.send(JSON.stringify({ type: "audio", data: b64(resampleToPcm16(e.inputBuffer.getChannelData(0), ctx.sampleRate, 16000).buffer) }));
  };
  source.connect(proc);
  proc.connect(ctx.destination);
  state.micNode = proc;
  state.micSource = source;
  startFrames(video);
}

fetch("/api/health").then((r) => r.json()).then((h) => { if (h.country) $("country").value = h.country === "GB" ? "GB" : "US"; }).catch(() => {});

$("start").onclick = async () => {
  unlockAudio();                        // synchronously inside the tap: phones refuse audio started later
  const motion = enableMotion();        // the iOS permission prompt must also start inside the tap
  $("start").disabled = true;
  $("welcomeNote").textContent = "Starting...";
  await motion;
  const country = $("country").value, posture = $("posture").value;
  const sw = await (await fetch("/api/sweeps", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ device: navigator.userAgent, country, posture }) })).json();
  state.sweepId = sw.id;
  $("locale").textContent = `${sw.country} / ${sw.currency}`;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  await connect();
  try {
    await startCapture();
  } catch (e) {
    $("welcomeNote").textContent = "Camera/microphone blocked: " + e.message + " (open the page over https and allow access).";
    $("start").disabled = false;
    return;
  }
  show("stage");
  $("finish").disabled = false;
  setStatus("Connected. The agent will guide you.");
};

function endSweep() {
  if (state.ended) return;
  state.ended = true;
  stopCapture();
  $("finish").disabled = true;
  state.procStart = Date.now();
  $("result").hidden = true;
  $("procTitle").textContent = "Building your claim packet";
  $("procSub").textContent = "Camera and microphone are off. This usually takes a minute or two.";
  checkSound();
  startProcessingView();
}

// Saying "I'm done" ends the tour; the button is there for when talking is awkward.
$("finish").onclick = () => { sendEnd(); endSweep(); };
