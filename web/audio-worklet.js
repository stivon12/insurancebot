// Microphone capture: resample whatever the device rate is (44.1/48 kHz) to 16 kHz mono PCM16,
// posting ~100 ms chunks to the main thread.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;
    this.out = new Int16Array(1600);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    // Linear-interpolation resampler (adequate for speech).
    while (this.pos < ch.length - 1) {
      const i = Math.floor(this.pos);
      const f = this.pos - i;
      const v = ch[i] * (1 - f) + ch[i + 1] * f;
      this.out[this.n++] = Math.max(-32768, Math.min(32767, v * 32767));
      if (this.n === this.out.length) {
        this.port.postMessage(this.out.buffer.slice(0));
        this.n = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= ch.length;
    return true;
  }
}
registerProcessor("mic-processor", MicProcessor);
