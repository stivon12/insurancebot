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

function addTranscript(who, text) {
  const key = who === "agent" ? "agentLine" : "userLine";
  if (!state[key]) {
    state[key] = document.createElement("div");
    state[key].className = who;
    state[key].textContent = (who === "agent" ? "Agent: " : "You: ");
    $("transcript").appendChild(state[key]);
  }
  state[key].textContent += text;
  $("transcript").scrollTop = $("transcript").scrollHeight;
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
  src.connect(ctx.destination);
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
    if (state.ws.bufferedAmount > 4e6) return; // network is behind: skip rather than queue stale frames
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
    case "interrupted": stopPlayback(); state.agentLine = null; break;
    case "turn_complete": state.agentLine = null; state.userLine = null; break;
    case "transcript": addTranscript(m.who, m.text); if (m.who === "user") state.agentLine = null; break;
    case "phase": $("phase").textContent = m.phase.replace("_", " ") + (m.unit ? `: ${m.unit}` : ""); break;
    case "locale": $("locale").textContent = `${m.country} / ${m.currency}`; break;
    case "inventory":
      $("inventory").innerHTML = "";
      m.items.forEach((it) => { const li = document.createElement("li"); li.textContent = it.label; $("inventory").appendChild(li); });
      break;
    case "override": setStatus(`noted: ${m.override.action} - ${m.override.target}`); break;
    case "status": setStatus(m.text); break;
    case "progress": {
      const run = m.stages.find((x) => x.state === "running");
      setStatus(`${Math.round(m.overall * 100)}%` + (run ? ` - ${run.label}` + (run.total ? ` ${run.done}/${run.total}` : "") : ""));
      break;
    }
    case "finished": stopCapture(); $("finish").disabled = true; setStatus("Sweep finished. Camera and microphone are off. Preparing the claim packet..."); break;
    case "error": setStatus("Error: " + m.text); break;
    case "packet":
      setStatus("Claim packet ready.");
      $("packet").innerHTML = `<h2 style="font-size:16px">Claim packet</h2><p>${m.summary}</p>
        <p><a href="${m.report}" target="_blank">Open report</a> &middot; <a href="${m.json}" target="_blank">claim_packet.json</a></p>`;
      break;
  }
}

// ---------- start ----------
$("start").onclick = async () => {
  $("start").disabled = true;
  await enableMotion();                        // first: iOS needs the permission call inside the tap
  state.playCtx = new (window.AudioContext || window.webkitAudioContext)();
  await state.playCtx.resume();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch (e) {
    setStatus("Camera/microphone blocked: " + e.message + " (the page must be opened over https)");
    $("start").disabled = false;
    return;
  }
  state.stream = stream;
  state.capturing = true;
  const video = $("video");
  video.srcObject = stream;
  await video.play().catch(() => {});

  const health = await (await fetch("/api/health")).json();
  const sw = await (await fetch("/api/sweeps", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ device: navigator.userAgent, country: health.country }) })).json();
  state.sweepId = sw.id;
  $("locale").textContent = `${sw.country} / ${sw.currency}`;

  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/ws/live?sweep_id=${sw.id}`);
  state.ws.onmessage = onMessage;
  state.ws.onclose = () => setStatus("Connection closed.");
  await new Promise((res) => (state.ws.onopen = res));

  // Microphone -> 16 kHz PCM16 via AudioWorklet
  // One AudioContext (created inside the tap) for mic and playback: iOS keeps later-created contexts suspended.
  const ctx = state.playCtx;
  await ctx.audioWorklet.addModule("/static/audio-worklet.js");
  const node = new AudioWorkletNode(ctx, "mic-processor");
  state.micNode = node;
  node.port.onmessage = (e) => { if (state.capturing && state.ws.readyState === 1) state.ws.send(JSON.stringify({ type: "audio", data: b64(e.data) })); };
  const mute = ctx.createGain();
  mute.gain.value = 0; // Safari only runs a worklet that is connected to the output
  ctx.createMediaStreamSource(stream).connect(node);
  node.connect(mute).connect(ctx.destination);

  startFrames(video);
  $("finish").disabled = false;
  setStatus("Connected. The agent will guide you.");
};

$("finish").onclick = () => {
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify({ type: "end" }));
  stopCapture();
  $("finish").disabled = true;
  setStatus("Sweep finished. Camera and microphone are off. Preparing the claim packet...");
};
