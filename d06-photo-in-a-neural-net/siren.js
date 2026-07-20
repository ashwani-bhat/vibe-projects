/**
 * SIREN — an implicit neural representation of an image.
 *
 * Instead of storing pixels, we store a tiny network f(x, y) → (r, g, b) and
 * train it until the function IS the photo ("Implicit Neural Representations
 * with Periodic Activation Functions", Sitzmann et al., NeurIPS 2020).
 * Sine activations (with the w0=30 trick and matching init) are what let such
 * a small net capture fine detail.
 *
 * Coordinates come in [-1, 1]^2; outputs are centered colors (target − 0.5).
 * The whole photo ends up as ~5.7K weights → ~12KB quantized to 16 bits.
 */

const SIREN = (() => {
  'use strict';

  class Model {
    constructor({ hidden = 72, w0 = 30 } = {}) {
      this.sizes = [2, hidden, hidden, 3];
      this.w0 = w0;
      this.init();
    }

    init() {
      this.dispose();
      const { sizes, w0 } = this;
      this.W = []; this.b = [];
      for (let l = 0; l < sizes.length - 1; l++) {
        const nin = sizes[l];
        // SIREN init: first layer U(±1/nin), later layers U(±sqrt(6/nin)/w0).
        const lim = l === 0 ? 1 / nin : Math.sqrt(6 / nin) / w0;
        this.W.push(tf.variable(tf.randomUniform([nin, sizes[l + 1]], -lim, lim)));
        this.b.push(tf.variable(tf.zeros([sizes[l + 1]])));
      }
    }

    variables() { return [...this.W, ...this.b]; }

    paramCount() {
      const { sizes } = this;
      let n = 0;
      for (let l = 0; l < sizes.length - 1; l++) n += sizes[l] * sizes[l + 1] + sizes[l + 1];
      return n;
    }

    /** Serialized size in bytes with 16-bit weight quantization. */
    byteSize() { return this.paramCount() * 2; }

    /** coords: [N, 2] in [-1,1] → centered rgb [N, 3]. */
    forward(coords) {
      return tf.tidy(() => {
        let x = coords;
        for (let l = 0; l < this.W.length; l++) {
          x = tf.matMul(x, this.W[l]).add(this.b[l]);
          if (l < this.W.length - 1) x = tf.sin(x.mul(this.w0));
        }
        return x;
      });
    }

    /**
     * Quantize every weight to uint16 with per-tensor min/max — the "photo
     * as a file". Returns a plain JSON-able object.
     */
    exportQuantized() {
      const tensors = [];
      for (const v of this.variables()) {
        const data = v.dataSync();
        let min = Infinity, max = -Infinity;
        for (const x of data) { if (x < min) min = x; if (x > max) max = x; }
        const scale = (max - min) || 1;
        const q = new Array(data.length);
        for (let i = 0; i < data.length; i++) {
          q[i] = Math.round(((data[i] - min) / scale) * 65535);
        }
        tensors.push({ shape: v.shape, min, max, q });
      }
      return { format: 'siren-u16', sizes: this.sizes, w0: this.w0, tensors };
    }

    importQuantized(obj) {
      obj.tensors.forEach((t, i) => {
        const scale = (t.max - t.min) || 1;
        const data = new Float32Array(t.q.length);
        for (let j = 0; j < t.q.length; j++) data[j] = t.min + (t.q[j] / 65535) * scale;
        this.variables()[i].assign(tf.tensor(data, t.shape));
      });
    }

    dispose() {
      for (const v of [...(this.W || []), ...(this.b || [])]) v.dispose();
      this.W = []; this.b = [];
    }
  }

  /**
   * Adam on random minibatches of pixels. target: {w, h, data} where data is
   * Float32Array[w*h*3] in [0,1].
   */
  class Trainer {
    constructor(model, target, { lr = 5e-4, batch = 2048 } = {}) {
      this.model = model;
      this.batch = batch;
      this.optimizer = tf.train.adam(lr);
      this.target = target;
      this.iter = 0;

      const { w, h, data } = target;
      const n = w * h;
      this.coords = new Float32Array(n * 2);
      this.colors = new Float32Array(n * 3);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          this.coords[i * 2] = (x / (w - 1)) * 2 - 1;
          this.coords[i * 2 + 1] = (y / (h - 1)) * 2 - 1;
          for (let c = 0; c < 3; c++) this.colors[i * 3 + c] = data[i * 3 + c] - 0.5;
        }
      }
      this.n = n;
    }

    /** One Adam step on a random pixel batch. Returns the batch MSE. */
    step() {
      const bs = Math.min(this.batch, this.n);
      const bc = new Float32Array(bs * 2);
      const bt = new Float32Array(bs * 3);
      for (let i = 0; i < bs; i++) {
        const k = (Math.random() * this.n) | 0;
        bc[i * 2] = this.coords[k * 2];
        bc[i * 2 + 1] = this.coords[k * 2 + 1];
        bt[i * 3] = this.colors[k * 3];
        bt[i * 3 + 1] = this.colors[k * 3 + 1];
        bt[i * 3 + 2] = this.colors[k * 3 + 2];
      }
      const loss = this.optimizer.minimize(() => {
        const xs = tf.tensor2d(bc, [bs, 2]);
        const ys = tf.tensor2d(bt, [bs, 3]);
        return this.model.forward(xs).sub(ys).square().mean();
      }, true, this.model.variables());
      const v = loss.dataSync()[0];
      loss.dispose();
      this.iter++;
      return v;
    }

    /** Full-image MSE + PSNR (dB) of the current model. */
    evaluate() {
      return tf.tidy(() => {
        const xs = tf.tensor2d(this.coords, [this.n, 2]);
        const pred = this.model.forward(xs);
        const ys = tf.tensor2d(this.colors, [this.n, 3]);
        const mse = pred.sub(ys).square().mean().dataSync()[0];
        return { mse, psnr: -10 * Math.log10(Math.max(mse, 1e-10)) };
      });
    }

    dispose() { this.optimizer.dispose(); }
  }

  /** Render the model to an RGBA Uint8ClampedArray at any resolution — the
   *  net is a continuous function, so any resolution is fair game. */
  function render(model, w, h) {
    return tf.tidy(() => {
      const coords = new Float32Array(w * h * 2);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          coords[i * 2] = (x / (w - 1)) * 2 - 1;
          coords[i * 2 + 1] = (y / (h - 1)) * 2 - 1;
        }
      }
      const rgb = model.forward(tf.tensor2d(coords, [w * h, 2])).add(0.5).clipByValue(0, 1);
      const data = rgb.dataSync();
      const out = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < w * h; i++) {
        out[i * 4] = data[i * 3] * 255;
        out[i * 4 + 1] = data[i * 3 + 1] * 255;
        out[i * 4 + 2] = data[i * 3 + 2] * 255;
        out[i * 4 + 3] = 255;
      }
      return out;
    });
  }

  return { Model, Trainer, render };
})();
