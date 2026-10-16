// Phone client: camera + mic + motion sensors -> server; agent audio + live inventory <- server.
// Plain web APIs only, so it runs in iOS Safari and Android Chrome (HTTPS required for camera/mic/motion).
"use strict";

const $ = (id) => document.getElementById(id);
const FRAME_MS = 333;          // ~3 fps saved for the pipeline
const LIVE_EVERY = 3;          // every 3rd frame (~1 fps) also goes to the live model
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

function renderProgress(m) {
  startProcessingView();
  const pct = Math.round(m.overall * 100);
  $("barFill").style.width = pct + "%";
  $("pctText").textContent = pct + "%";
  const ul = $("stages");
  ul.textContent = "";
  m.stages.forEach((st) => {
    const li = el("li", st.state);
    li.appendChild(el("span", "ico"));
    li.appendChild(el("span", "", st.label));
    const n = st.state === "done" ? (st.note || "done") : st.state === "running" && st.total ? `${st.done} / ${st.total}` : st.state === "running" ? "working" : "";
    li.appendChild(el("span", "n", n));
    ul.appendChild(li);
  });
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

// ---------- sound output ----------
// Phones only start audio from inside a tap, so this runs synchronously at the top of every tap handler.
// Agent voice goes through an <audio> element (MediaStreamDestination): media playback uses the loudspeaker even
// while the mic is open (Android otherwise switches to call mode: earpiece / very quiet) and ignores the iOS
// silent switch. If that element cannot play, sound falls back to the AudioContext output.
function unlockAudio() {
  if (!state.playCtx) {
    state.playCtx = new (window.AudioContext || window.webkitAudioContext)();
    state.playCtx.onstatechange = checkSound;
    try {
      if (navigator.audioSession) navigator.audioSession.type = "play-and-record"; // Safari 17+
    } catch (e) {}
    try {
      const dest = state.playCtx.createMediaStreamDestination();
      $("agentAudio").srcObject = dest.stream;
      state.out = dest;
    } catch (e) { state.out = null; }
  }
  const ctx = state.playCtx;
  ctx.resume().then(checkSound).catch(() => {});
  const b = ctx.createBuffer(1, 1, 22050), s = ctx.createBufferSource();
  s.buffer = b; s.connect(ctx.destination); s.start(0);
  if (state.out) $("agentAudio").play().catch(() => { state.out = null; }); // fall back to the context output
  checkSound();
}

function checkSound() {
  const off = !!state.playCtx && state.playCtx.state !== "running" && !state.ended;
  $("soundBar").hidden = !off;
}

$("soundBar").onclick = () => { unlockAudio(); setTimeout(checkSound, 300); };
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.playCtx) state.playCtx.resume().then(checkSound).catch(() => {});
});

// Half duplex: while the agent's voice is playing (and 300 ms after), the mic is not sent. On a phone speaker
// the mic hears the agent, which made it interrupt itself and fed garbled 'speech' back to the model.
function agentSpeaking() {
  return state.playCtx && state.nextPlay > 0 && state.playCtx.currentTime < state.nextPlay + 0.3;
}

// ---------- agent audio playback (24 kHz PCM16) ----------
function playPcm(base64) {
  const ctx = state.playCtx;
  const bin = atob(base64);
  const pcm = new Int16Array(bin.length / 2);
  for (let i = 0; i < pcm.length; i++) pcm[i] = (bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8)) << 16 >> 16;
  const buf = ctx.createBuffer(1, pcm.length, 24000);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(state.out || ctx.destination);
  const t = Math.max(ctx.currentTime + 0.03, state.nextPlay);
  src.start(t);
  state.nextPlay = t + buf.duration;
  state.sources.push(src);
  src.onended = () => { state.sources = state.sources.filter((s) => s !== src); };
}

function stopPlayback() {
  state.sources.forEach((s) => { try { s.stop(); } catch (e) {} });
  state.sources = [];
  state.nextPlay = 0;
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
  return { blur: variance, glare: bright / (w * h), blurBad: variance < 30, glareBad: bright / (w * h) > 0.08 };
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
        live: state.frames % LIVE_EVERY === 0, blur: q.blur, glare: q.glare, blur_bad: q.blurBad, glare_bad: q.glareBad,
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
  state.imu.push({ t_ms: performance.now(), ax: s * a.x, ay: s * a.y, az: s * a.z, ra: r ? r.alpha : 0, rb: r ? r.beta : 0, rg: r ? r.gamma : 0 });
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
  if (state.micNode) { state.micNode.port.onmessage = null; state.micNode.disconnect(); state.micNode = null; }
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
    case "phase": $("phase").textContent = m.phase.replace("_", " ") + (m.unit ? `: ${m.unit}` : ""); break;
    case "locale": $("locale").textContent = `${m.country} / ${m.currency}`; break;
    case "inventory": renderInventory(m.items); break;
    case "processed": {
      const s = $("processedStrip");
      s.hidden = false;
      s.textContent = `Processing as you go · ${m.spine_frames} shelf views read · ${m.books} books · ${m.identified} identified · ${m.priced} priced · ${m.item_frames} room views checked`;
      break;
    }
    case "override": setStatus(`noted: ${m.override.action} - ${m.override.target}`); break;
    case "status": setStatus(m.text); break;
    case "progress": renderProgress(m); break;
    case "finished": endSweep(); break;
    case "error":
      if (state.ended) { $("errorBox").hidden = false; $("errorBox").textContent = "Error: " + m.text; }
      else setStatus("Error: " + m.text);
      break;
    case "packet": state.packetShown = true; renderResult(m); break;
    case "reattached": setStatus(m.finished ? "Reconnected. The packet is being built." : "Reconnected. The agent will pick up where you left off."); break;
  }
}

// ---------- connection ----------
// A dropped connection (Wi-Fi to mobile data, a tunnel, a weak signal) reconnects to the same sweep. The server keeps
// the sweep for a while and, if the phone does not come back, builds the packet from what was recorded.
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
// Camera, mic worklet, motion sensors and frame timer; shared by Start and Continue sweep.
async function startCapture() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
    audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
  });
  state.stream = stream;
  state.capturing = true;
  const video = $("video");
  video.srcObject = stream;
  await video.play().catch(() => {});
  const ctx = state.playCtx;
  ctx.resume().then(checkSound).catch(() => {}); // the mic can re-route audio on phones: make sure it still plays
  if (!state.workletLoaded) { await ctx.audioWorklet.addModule("/static/audio-worklet.js"); state.workletLoaded = true; }
  const node = new AudioWorkletNode(ctx, "mic-processor");
  state.micNode = node;
  node.port.onmessage = (e) => {
    if (state.capturing && state.ws && state.ws.readyState === 1 && !agentSpeaking()) state.ws.send(JSON.stringify({ type: "audio", data: b64(e.data) }));
  };
  const mute = ctx.createGain();
  mute.gain.value = 0; // Safari only runs a worklet that is connected to the output
  ctx.createMediaStreamSource(stream).connect(node);
  node.connect(mute).connect(ctx.destination);
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
  $("continue").hidden = false;
  checkSound();
  startProcessingView();
}

// Two taps to finish, so a stray tap does not end the sweep.
$("finish").onclick = () => {
  const f = $("finish");
  if (!f.classList.contains("confirm")) {
    f.classList.add("confirm");
    f.textContent = "Tap again";
    clearTimeout(state.finishTimer);
    state.finishTimer = setTimeout(() => { f.classList.remove("confirm"); f.textContent = "Finish"; }, 3000);
    return;
  }
  clearTimeout(state.finishTimer);
  f.classList.remove("confirm");
  f.textContent = "Finish";
  sendEnd();
  endSweep();
};

// Finished by mistake: turn the camera and mic back on and carry on with the same sweep.
$("continue").onclick = async () => {
  unlockAudio();
  if (!state.ws || state.ws.readyState !== 1) { showError("Not connected right now: wait for the connection to come back, then try again."); return; }
  $("continue").disabled = true;
  const motion = enableMotion();
  await motion;
  try {
    await startCapture();
  } catch (e) {
    $("errorBox").hidden = false; $("errorBox").textContent = "Camera/microphone blocked: " + e.message;
    $("continue").disabled = false;
    return;
  }
  state.ws.send(JSON.stringify({ type: "resume" }));
  state.ended = false;
  state.endSent = false;
  state.packetShown = false;
  clearInterval(state.clock);
  $("continue").disabled = false;
  show("stage");
  $("finish").disabled = false;
  setStatus("Sweep continued. The agent will pick up where you left off.");
};
