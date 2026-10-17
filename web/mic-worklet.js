// Mic capture on the audio thread: hands the page ~43 ms chunks, so the page's main thread is not woken per buffer.
class Mic extends AudioWorkletProcessor {
  constructor() { super(); this.parts = []; this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      this.parts.push(ch.slice());
      this.n += ch.length;
      if (this.n >= 2048) {
        const out = new Float32Array(this.n);
        let o = 0;
        for (const p of this.parts) { out.set(p, o); o += p.length; }
        this.port.postMessage(out, [out.buffer]);
        this.parts = []; this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor("mic", Mic);
