// Client-side forward pass for the trained HexNet, matching alphazero_hex/net.py
// with BatchNorm folded into the preceding convolution (see export_net_js.py).
// Board size is not baked in anywhere -- same weights, any n.
'use strict';

function b64ToFloat32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

function unpack(entry) {
  return { shape: entry.shape, data: b64ToFloat32(entry.data) };
}

// Direct 3x3 (pad=1) or 1x1 convolution on a padded input buffer.
// `input` is [inC, n, n] flattened; returns a new [outC, n, n] Float32Array.
// Padding the input once (to (n+2)x(n+2)) removes all boundary branches from
// the inner loop, which is where a naive port would lose most of its speed.
function conv2d(input, inC, n, weight, bias, outC, k) {
  const out = new Float32Array(outC * n * n);
  const pad = k === 3 ? 1 : 0;
  const pn = n + 2 * pad;
  let padded = input;
  if (pad) {
    padded = new Float32Array(inC * pn * pn);
    for (let ic = 0; ic < inC; ic++) {
      const srcBase = ic * n * n, dstBase = ic * pn * pn;
      for (let r = 0; r < n; r++) {
        padded.set(input.subarray(srcBase + r * n, srcBase + r * n + n), dstBase + (r + pad) * pn + pad);
      }
    }
  }
  const kk = k * k;
  for (let oc = 0; oc < outC; oc++) {
    const outBase = oc * n * n;
    const b = bias[oc];
    for (let i = 0; i < n * n; i++) out[outBase + i] = b;
    for (let ic = 0; ic < inC; ic++) {
      const wBase = (oc * inC + ic) * kk;
      const inBase = ic * pn * pn;
      if (k === 1) {
        const w = weight[wBase];
        for (let i = 0; i < n * n; i++) out[outBase + i] += w * padded[inBase + i];
      } else {
        // 3x3: unrolled tap loop over the padded plane.
        const w00 = weight[wBase], w01 = weight[wBase + 1], w02 = weight[wBase + 2];
        const w10 = weight[wBase + 3], w11 = weight[wBase + 4], w12 = weight[wBase + 5];
        const w20 = weight[wBase + 6], w21 = weight[wBase + 7], w22 = weight[wBase + 8];
        for (let r = 0; r < n; r++) {
          const rowOut = outBase + r * n;
          const row0 = inBase + r * pn, row1 = inBase + (r + 1) * pn, row2 = inBase + (r + 2) * pn;
          for (let c = 0; c < n; c++) {
            out[rowOut + c] +=
              w00 * padded[row0 + c] + w01 * padded[row0 + c + 1] + w02 * padded[row0 + c + 2] +
              w10 * padded[row1 + c] + w11 * padded[row1 + c + 1] + w12 * padded[row1 + c + 2] +
              w20 * padded[row2 + c] + w21 * padded[row2 + c + 1] + w22 * padded[row2 + c + 2];
          }
        }
      }
    }
  }
  return out;
}

function reluInplace(a) { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; }

class HexNetJS {
  constructor(json) {
    this.cfg = json.cfg;
    const c = this.cfg.channels, h = this.cfg.head_channels;
    this.stem = { w: unpack(json.stem.w), b: unpack(json.stem.b) };
    this.blocks = json.blocks.map(bl => ({
      w1: unpack(bl.w1), b1: unpack(bl.b1), w2: unpack(bl.w2), b2: unpack(bl.b2),
    }));
    this.policy = {
      w1: unpack(json.policy_head.w1), b1: unpack(json.policy_head.b1),
      w2: unpack(json.policy_head.w2), b2: unpack(json.policy_head.b2),
    };
    this.valueConv = { w: unpack(json.value_conv.w), b: unpack(json.value_conv.b) };
    this.valueFc = {
      w1: unpack(json.value_fc.w1), b1: unpack(json.value_fc.b1),
      w2: unpack(json.value_fc.w2), b2: unpack(json.value_fc.b2),
    };
    this.c = c; this.h = h;
  }

  // planes: Float32Array[3*n*n] (own, opp, ones), as produced by HexBoard.canonicalPlanes.
  // Returns { policy: Float32Array[n*n] (softmax over legal cells masked by caller),
  //           value: number in [-1,1] }.
  forward(planes, n) {
    const c = this.c, h = this.h;
    let x = conv2d(planes, 3, n, this.stem.w.data, this.stem.b.data, c, 3);
    reluInplace(x);
    for (const blk of this.blocks) {
      let y = conv2d(x, c, n, blk.w1.data, blk.b1.data, c, 3);
      reluInplace(y);
      y = conv2d(y, c, n, blk.w2.data, blk.b2.data, c, 3);
      for (let i = 0; i < y.length; i++) { const v = x[i] + y[i]; x[i] = v > 0 ? v : 0; }
    }
    // policy head
    let p = conv2d(x, c, n, this.policy.w1.data, this.policy.b1.data, h, 1);
    reluInplace(p);
    const logits = conv2d(p, h, n, this.policy.w2.data, this.policy.b2.data, 1, 1);
    // value head
    let v = conv2d(x, c, n, this.valueConv.w.data, this.valueConv.b.data, h, 1);
    reluInplace(v);
    const pooled = new Float32Array(2 * h);
    for (let ch = 0; ch < h; ch++) {
      let sum = 0, mx = -Infinity;
      const base = ch * n * n;
      for (let i = 0; i < n * n; i++) { const val = v[base + i]; sum += val; if (val > mx) mx = val; }
      pooled[ch] = sum / (n * n);
      pooled[h + ch] = mx;
    }
    const hidden = this._linear(pooled, this.valueFc.w1, this.valueFc.b1);
    reluInplace(hidden);
    const out = this._linear(hidden, this.valueFc.w2, this.valueFc.b2);
    return { logits, value: Math.tanh(out[0]) };
  }

  _linear(x, w, b) {
    const [outF, inF] = w.shape, out = new Float32Array(outF);
    for (let o = 0; o < outF; o++) {
      let s = b.data[o];
      const base = o * inF;
      for (let i = 0; i < inF; i++) s += w.data[base + i] * x[i];
      out[o] = s;
    }
    return out;
  }
}

// Softmax over `logits`, masked to `legalCanonIdx` (indices into the n*n plane).
function maskedSoftmax(logits, legalCanonIdx) {
  let mx = -Infinity;
  for (const i of legalCanonIdx) if (logits[i] > mx) mx = logits[i];
  const exp = new Float64Array(legalCanonIdx.length);
  let sum = 0;
  for (let k = 0; k < legalCanonIdx.length; k++) {
    const e = Math.exp(logits[legalCanonIdx[k]] - mx);
    exp[k] = e; sum += e;
  }
  const out = new Float64Array(legalCanonIdx.length);
  for (let k = 0; k < legalCanonIdx.length; k++) out[k] = exp[k] / sum;
  return out;
}

if (typeof module !== 'undefined') {
  module.exports = { HexNetJS, maskedSoftmax, conv2d };
}
