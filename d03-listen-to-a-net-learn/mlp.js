/**
 * A tiny multi-layer perceptron with hand-rolled backprop and Adam.
 *
 * No libraries — the whole point of this project is that every number in
 * training (activations, gradients, loss) is ours to sonify, so the network
 * is plain arrays: tanh hidden layers, sigmoid output, binary cross-entropy.
 *
 * Weight layout: W[l] is Float32Array[nin * nout], row-major by input
 * (W[l][i * nout + j] connects input i to output j).
 */

const MLP = (() => {
  'use strict';

  /* Deterministic RNG (mulberry32) so the smoke test is reproducible. */
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

  /* Toy two-class datasets, points roughly in [-1.2, 1.2]^2. */
  const datasets = {
    spiral(n = 240, rand = rng(7)) {
      const xs = [], ys = [];
      const m = n >> 1;
      for (let c = 0; c < 2; c++) {
        for (let i = 0; i < m; i++) {
          const r = (i + 1) / m;
          const t = r * 2.75 * Math.PI + c * Math.PI;
          xs.push([
            r * Math.sin(t) * 1.05 + gauss(rand) * 0.04,
            r * Math.cos(t) * 1.05 + gauss(rand) * 0.04,
          ]);
          ys.push(c);
        }
      }
      return { xs, ys };
    },

    circles(n = 240, rand = rng(7)) {
      const xs = [], ys = [];
      for (let i = 0; i < n; i++) {
        const c = i % 2;
        const r = c === 0 ? rand() * 0.45 : 0.75 + rand() * 0.35;
        const a = rand() * 2 * Math.PI;
        xs.push([r * Math.cos(a), r * Math.sin(a)]);
        ys.push(c);
      }
      return { xs, ys };
    },

    moons(n = 240, rand = rng(7)) {
      const xs = [], ys = [];
      for (let i = 0; i < n; i++) {
        const c = i % 2, t = rand() * Math.PI;
        const x = c === 0 ? Math.cos(t) : 1 - Math.cos(t);
        const y = c === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
        xs.push([
          (x - 0.5) * 0.85 + gauss(rand) * 0.05,
          (y - 0.25) * 0.85 + gauss(rand) * 0.05,
        ]);
        ys.push(c);
      }
      return { xs, ys };
    },

    xor(n = 240, rand = rng(7)) {
      const xs = [], ys = [];
      for (let i = 0; i < n; i++) {
        const sx = rand() < 0.5 ? -1 : 1, sy = rand() < 0.5 ? -1 : 1;
        xs.push([sx * 0.6 + gauss(rand) * 0.22, sy * 0.6 + gauss(rand) * 0.22]);
        ys.push(sx * sy > 0 ? 0 : 1);
      }
      return { xs, ys };
    },
  };

  class Net {
    constructor(sizes = [2, 24, 24, 1], seed = 1) {
      this.sizes = sizes;
      const rand = rng(seed);
      this.W = [];
      this.b = [];
      for (let l = 0; l < sizes.length - 1; l++) {
        const nin = sizes[l], nout = sizes[l + 1];
        // First layer gets a 3x hotter init: datasets like the spiral have no
        // linear signal, and standard Glorot leaves the net stuck on the
        // p=0.5 saddle for most random seeds. Hot first-layer tanh units
        // carve the plane immediately (8/8 seeds converge vs 3/8 without).
        const lim = Math.sqrt(6 / (nin + nout)) * (l === 0 ? 3 : 1);
        const w = new Float32Array(nin * nout);
        for (let i = 0; i < w.length; i++) w[i] = (rand() * 2 - 1) * lim;
        this.W.push(w);
        this.b.push(new Float32Array(nout));
      }
      // Scratch buffers so predict() (called thousands of times per boundary
      // redraw) allocates nothing.
      this._scratch = sizes.map(s => new Float32Array(s));
    }

    /** Full forward pass, keeping every layer's activations (for backprop / sonification). */
    forward(x) {
      const acts = [Float32Array.from(x)];
      for (let l = 0; l < this.W.length; l++) {
        const nin = this.sizes[l], nout = this.sizes[l + 1];
        const a = acts[l], W = this.W[l], b = this.b[l];
        const out = new Float32Array(nout);
        const last = l === this.W.length - 1;
        for (let j = 0; j < nout; j++) {
          let z = b[j];
          for (let i = 0; i < nin; i++) z += a[i] * W[i * nout + j];
          out[j] = last ? 1 / (1 + Math.exp(-z)) : Math.tanh(z);
        }
        acts.push(out);
      }
      return acts;
    }

    /** Allocation-free forward pass returning only P(class 1). */
    predict(x) {
      const s = this._scratch;
      s[0][0] = x[0]; s[0][1] = x[1];
      for (let l = 0; l < this.W.length; l++) {
        const nin = this.sizes[l], nout = this.sizes[l + 1];
        const a = s[l], W = this.W[l], b = this.b[l], out = s[l + 1];
        const last = l === this.W.length - 1;
        for (let j = 0; j < nout; j++) {
          let z = b[j];
          for (let i = 0; i < nin; i++) z += a[i] * W[i * nout + j];
          out[j] = last ? 1 / (1 + Math.exp(-z)) : Math.tanh(z);
        }
      }
      return s[s.length - 1][0];
    }
  }

  /**
   * Minibatch SGD with Adam. step() returns everything the orchestra needs:
   * batch loss, batch accuracy, and per-layer gradient RMS (how hard each
   * layer is currently being corrected — that's what detunes its voice).
   */
  class Trainer {
    constructor(net, data, { lr = 0.02, batchSize = 24, seed = 42 } = {}) {
      this.net = net;
      this.data = data;
      this.lr = lr;
      this.batchSize = batchSize;
      this.rand = rng(seed);
      this.iter = 0;
      this.beta1 = 0.9; this.beta2 = 0.999; this.eps = 1e-8;
      this.mW = net.W.map(w => new Float32Array(w.length));
      this.vW = net.W.map(w => new Float32Array(w.length));
      this.mB = net.b.map(b => new Float32Array(b.length));
      this.vB = net.b.map(b => new Float32Array(b.length));
    }

    step() {
      const net = this.net, sizes = net.sizes, L = net.W.length, B = this.batchSize;
      const gW = net.W.map(w => new Float32Array(w.length));
      const gB = net.b.map(b => new Float32Array(b.length));
      let loss = 0, correct = 0;

      for (let s = 0; s < B; s++) {
        const k = (this.rand() * this.data.xs.length) | 0;
        const x = this.data.xs[k], y = this.data.ys[k];
        const acts = net.forward(x);
        const p = acts[L][0];
        loss += -(y * Math.log(p + 1e-7) + (1 - y) * Math.log(1 - p + 1e-7));
        correct += (p > 0.5 ? 1 : 0) === y ? 1 : 0;

        // Sigmoid + BCE collapse to a clean output delta: dL/dz = p - y.
        let delta = Float32Array.of(p - y);
        for (let l = L - 1; l >= 0; l--) {
          const nin = sizes[l], nout = sizes[l + 1];
          const aPrev = acts[l];
          for (let j = 0; j < nout; j++) {
            gB[l][j] += delta[j];
            for (let i = 0; i < nin; i++) gW[l][i * nout + j] += aPrev[i] * delta[j];
          }
          if (l > 0) {
            const nd = new Float32Array(nin);
            for (let i = 0; i < nin; i++) {
              let sum = 0;
              for (let j = 0; j < nout; j++) sum += net.W[l][i * nout + j] * delta[j];
              nd[i] = sum * (1 - aPrev[i] * aPrev[i]); // tanh'
            }
            delta = nd;
          }
        }
      }

      this.iter++;
      const c1 = 1 - Math.pow(this.beta1, this.iter);
      const c2 = 1 - Math.pow(this.beta2, this.iter);
      const gradRms = [];
      for (let l = 0; l < L; l++) {
        let sq = 0;
        const w = net.W[l], gw = gW[l], mw = this.mW[l], vw = this.vW[l];
        for (let i = 0; i < w.length; i++) {
          const g = gw[i] / B;
          sq += g * g;
          mw[i] = this.beta1 * mw[i] + (1 - this.beta1) * g;
          vw[i] = this.beta2 * vw[i] + (1 - this.beta2) * g * g;
          w[i] -= this.lr * (mw[i] / c1) / (Math.sqrt(vw[i] / c2) + this.eps);
        }
        const b = net.b[l], gb = gB[l], mb = this.mB[l], vb = this.vB[l];
        for (let j = 0; j < b.length; j++) {
          const g = gb[j] / B;
          sq += g * g;
          mb[j] = this.beta1 * mb[j] + (1 - this.beta1) * g;
          vb[j] = this.beta2 * vb[j] + (1 - this.beta2) * g * g;
          b[j] -= this.lr * (mb[j] / c1) / (Math.sqrt(vb[j] / c2) + this.eps);
        }
        gradRms.push(Math.sqrt(sq / (w.length + b.length)));
      }

      return { loss: loss / B, acc: correct / B, gradRms };
    }

    /** Loss / accuracy over the whole dataset. */
    evaluate() {
      let loss = 0, correct = 0;
      const n = this.data.xs.length;
      for (let k = 0; k < n; k++) {
        const p = this.net.predict(this.data.xs[k]);
        const y = this.data.ys[k];
        loss += -(y * Math.log(p + 1e-7) + (1 - y) * Math.log(1 - p + 1e-7));
        correct += (p > 0.5 ? 1 : 0) === y ? 1 : 0;
      }
      return { loss: loss / n, acc: correct / n };
    }
  }

  return { Net, Trainer, datasets, rng };
})();
