/**
 * A small softmax MLP (63 → 32 → K) with hand-rolled backprop + Adam,
 * trained on the signs *you* record. A few hundred samples of 63 features
 * train in well under a second, so "✦ train" feels instant.
 *
 * Pure JS, no browser APIs — tested headlessly in test/smoke.js.
 */

const CLASSIFIER = (() => {
  'use strict';

  /* Deterministic RNG (mulberry32) so tests are reproducible. */
  function rng(seed = 1) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rand) {
    return Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  }

  class SoftmaxNet {
    constructor(inDim, hidden, K, seed = 1) {
      this.inDim = inDim; this.hidden = hidden; this.K = K;
      const rand = rng(seed);
      const init = (nin, nout) => {
        const lim = Math.sqrt(6 / (nin + nout));
        const w = new Float32Array(nin * nout);
        for (let i = 0; i < w.length; i++) w[i] = (rand() * 2 - 1) * lim;
        return w;
      };
      this.W1 = init(inDim, hidden); this.b1 = new Float32Array(hidden);
      this.W2 = init(hidden, K); this.b2 = new Float32Array(K);
    }

    /** Returns { h, probs } — hidden activations kept for backprop. */
    forward(x) {
      const { inDim, hidden, K, W1, b1, W2, b2 } = this;
      const h = new Float32Array(hidden);
      for (let j = 0; j < hidden; j++) {
        let z = b1[j];
        for (let i = 0; i < inDim; i++) z += x[i] * W1[i * hidden + j];
        h[j] = Math.tanh(z);
      }
      const logits = new Float32Array(K);
      let max = -Infinity;
      for (let k = 0; k < K; k++) {
        let z = b2[k];
        for (let j = 0; j < hidden; j++) z += h[j] * W2[j * K + k];
        logits[k] = z;
        if (z > max) max = z;
      }
      let sum = 0;
      const probs = new Float32Array(K);
      for (let k = 0; k < K; k++) { probs[k] = Math.exp(logits[k] - max); sum += probs[k]; }
      for (let k = 0; k < K; k++) probs[k] /= sum;
      return { h, probs };
    }

    predict(x) { return this.forward(x).probs; }
  }

  function accuracy(net, xs, ys) {
    if (xs.length === 0) return 0;
    let correct = 0;
    for (let i = 0; i < xs.length; i++) {
      const p = net.predict(xs[i]);
      let best = 0;
      for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
      if (best === ys[i]) correct++;
    }
    return correct / xs.length;
  }

  /**
   * Train on (xs, ys) with an 80/20 split. Gaussian feature jitter during
   * training stands in for the hand wobble the webcam will produce live.
   * Returns { net, trainAcc, valAcc }.
   */
  function train(xs, ys, K, {
    hidden = 32, steps = 500, lr = 0.01, batch = 16,
    jitter = 0.02, seed = 42, valFrac = 0.2,
  } = {}) {
    const inDim = xs[0].length;
    const rand = rng(seed);

    const order = xs.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = (rand() * (i + 1)) | 0;
      [order[i], order[j]] = [order[j], order[i]];
    }
    const nVal = Math.max(1, Math.floor(order.length * valFrac));
    const valIdx = order.slice(0, nVal), trainIdx = order.slice(nVal);

    const net = new SoftmaxNet(inDim, hidden, K, (rand() * 1e9) | 0);
    const { W1, b1, W2, b2 } = net;
    const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    const mW1 = new Float32Array(W1.length), vW1 = new Float32Array(W1.length);
    const mb1 = new Float32Array(b1.length), vb1 = new Float32Array(b1.length);
    const mW2 = new Float32Array(W2.length), vW2 = new Float32Array(W2.length);
    const mb2 = new Float32Array(b2.length), vb2 = new Float32Array(b2.length);
    const x = new Float32Array(inDim);

    for (let step = 1; step <= steps; step++) {
      const gW1 = new Float32Array(W1.length), gb1 = new Float32Array(b1.length);
      const gW2 = new Float32Array(W2.length), gb2 = new Float32Array(b2.length);

      for (let s = 0; s < batch; s++) {
        const idx = trainIdx[(rand() * trainIdx.length) | 0];
        const raw = xs[idx], y = ys[idx];
        for (let i = 0; i < inDim; i++) x[i] = raw[i] + gauss(rand) * jitter;

        const { h, probs } = net.forward(x);
        const dOut = probs; // reuse: softmax + CE gives dL/dlogit = p - onehot
        dOut[y] -= 1;
        for (let k = 0; k < net.K; k++) {
          gb2[k] += dOut[k];
          for (let j = 0; j < net.hidden; j++) gW2[j * net.K + k] += h[j] * dOut[k];
        }
        for (let j = 0; j < net.hidden; j++) {
          let sum = 0;
          for (let k = 0; k < net.K; k++) sum += W2[j * net.K + k] * dOut[k];
          const dh = sum * (1 - h[j] * h[j]);
          gb1[j] += dh;
          for (let i = 0; i < inDim; i++) gW1[i * net.hidden + j] += x[i] * dh;
        }
      }

      const c1 = 1 - Math.pow(beta1, step), c2 = 1 - Math.pow(beta2, step);
      const adam = (w, g, m, v) => {
        for (let i = 0; i < w.length; i++) {
          const gi = g[i] / batch;
          m[i] = beta1 * m[i] + (1 - beta1) * gi;
          v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
          w[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
        }
      };
      adam(W1, gW1, mW1, vW1); adam(b1, gb1, mb1, vb1);
      adam(W2, gW2, mW2, vW2); adam(b2, gb2, mb2, vb2);
    }

    const pick = idx => ({ xs: idx.map(i => xs[i]), ys: idx.map(i => ys[i]) });
    const tr = pick(trainIdx), va = pick(valIdx);
    return {
      net,
      trainAcc: accuracy(net, tr.xs, tr.ys),
      valAcc: accuracy(net, va.xs, va.ys),
    };
  }

  return { SoftmaxNet, train, accuracy, rng, gauss };
})();
