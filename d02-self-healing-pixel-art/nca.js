/**
 * Growing Neural Cellular Automata — the model + in-browser trainer.
 *
 * Based on "Growing Neural Cellular Automata" (Mordvintsev et al., Distill 2020).
 *
 * Each pixel is a cell with CHANNELS values:
 *   ch 0..2  = RGB (premultiplied by alpha)
 *   ch 3     = alpha ("aliveness")
 *   ch 4..15 = hidden state the cells use to communicate
 *
 * Every tick, each cell looks at its 3x3 neighborhood (identity + Sobel
 * filters), runs a tiny 2-layer network, and adds the result to its state.
 * Training teaches this local rule to grow the target image from a single
 * seed pixel — and, because we damage samples during training, to regrow
 * it when pixels are erased.
 */

const NCA = (() => {
  'use strict';

  class Model {
    constructor({ size = 32, channels = 16, hidden = 128, fireRate = 0.5 } = {}) {
      this.size = size;
      this.channels = channels;
      this.hidden = hidden;
      this.fireRate = fireRate;

      // Fixed perception filters: identity, Sobel-x, Sobel-y per channel.
      this.perception = tf.tidy(() => {
        const ident = [[0, 0, 0], [0, 1, 0], [0, 0, 0]];
        const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]].map(r => r.map(v => v / 8));
        const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]].map(r => r.map(v => v / 8));
        // depthwiseConv2d filter shape: [3, 3, inChannels, multiplier]
        const k = tf.buffer([3, 3, channels, 3]);
        for (let y = 0; y < 3; y++) {
          for (let x = 0; x < 3; x++) {
            for (let c = 0; c < channels; c++) {
              k.set(ident[y][x], y, x, c, 0);
              k.set(sobelX[y][x], y, x, c, 1);
              k.set(sobelY[y][x], y, x, c, 2);
            }
          }
        }
        return k.toTensor();
      });

      this.initWeights();
    }

    initWeights() {
      this.dispose(true);
      const perc = this.channels * 3;
      const lim = Math.sqrt(6 / (perc + this.hidden));
      this.w1 = tf.variable(tf.randomUniform([1, 1, perc, this.hidden], -lim, lim));
      this.b1 = tf.variable(tf.zeros([this.hidden]));
      // Last layer starts at zero so the untrained CA does nothing.
      this.w2 = tf.variable(tf.zeros([1, 1, this.hidden, this.channels]));
      this.b2 = tf.variable(tf.zeros([this.channels]));
    }

    variables() {
      return [this.w1, this.b1, this.w2, this.b2];
    }

    aliveMask(x) {
      const alpha = x.slice([0, 0, 0, 3], [-1, -1, -1, 1]);
      // tf.step instead of greater(): greater has no gradient in tf.js,
      // step's is defined (zero), which is what we want for a hard mask.
      return tf.step(tf.maxPool(alpha, 3, 1, 'same').sub(0.1));
    }

    /** One CA tick. x: [B, H, W, C] -> [B, H, W, C] */
    step(x) {
      return tf.tidy(() => {
        const preAlive = this.aliveMask(x);
        const perceived = tf.depthwiseConv2d(x, this.perception, 1, 'same');
        const h = tf.relu(tf.conv2d(perceived, this.w1, 1, 'same').add(this.b1));
        const dx = tf.conv2d(h, this.w2, 1, 'same').add(this.b2);
        // step(fireRate - u) == (u < fireRate), but with a usable (zero) gradient
        const fire = tf.step(
          tf.randomUniform([x.shape[0], x.shape[1], x.shape[2], 1]).neg().add(this.fireRate));
        const x2 = x.add(dx.mul(fire));
        const alive = preAlive.mul(this.aliveMask(x2));
        return x2.mul(alive);
      });
    }

    /** A single seed cell in the middle of an empty grid (as CPU array). */
    seedArray() {
      const { size, channels } = this;
      const a = new Float32Array(size * size * channels);
      const cy = size >> 1, cx = size >> 1;
      const base = (cy * size + cx) * channels;
      for (let c = 3; c < channels; c++) a[base + c] = 1;
      return a;
    }

    seedTensor(batch = 1) {
      const one = this.seedArray();
      const all = new Float32Array(one.length * batch);
      for (let b = 0; b < batch; b++) all.set(one, b * one.length);
      return tf.tensor4d(all, [batch, this.size, this.size, this.channels]);
    }

    exportWeights() {
      return {
        size: this.size, channels: this.channels, hidden: this.hidden, fireRate: this.fireRate,
        w1: Array.from(this.w1.dataSync()),
        b1: Array.from(this.b1.dataSync()),
        w2: Array.from(this.w2.dataSync()),
        b2: Array.from(this.b2.dataSync()),
      };
    }

    importWeights(obj) {
      const perc = this.channels * 3;
      this.w1.assign(tf.tensor4d(obj.w1, [1, 1, perc, this.hidden]));
      this.b1.assign(tf.tensor1d(obj.b1));
      this.w2.assign(tf.tensor4d(obj.w2, [1, 1, this.hidden, this.channels]));
      this.b2.assign(tf.tensor1d(obj.b2));
    }

    dispose(weightsOnly = false) {
      for (const v of ['w1', 'b1', 'w2', 'b2']) {
        if (this[v]) { this[v].dispose(); this[v] = null; }
      }
      if (!weightsOnly && this.perception) { this.perception.dispose(); this.perception = null; }
    }
  }

  /**
   * Sample-pool trainer. Keeps a pool of grown states; each step it trains on
   * a batch from the pool, always re-seeding the worst sample and (after a
   * warm-up) damaging a couple of good ones so the CA learns to regenerate.
   */
  class Trainer {
    constructor(model, targetRgba, {
      poolSize = 96, batchSize = 6, stepsMin = 40, stepsMax = 60,
      damageCount = 2, damageStartIter = 300, lr = 1e-3,
    } = {}) {
      this.model = model;
      this.target = targetRgba; // Float32Array [H*W*4], premultiplied RGB
      this.poolSize = poolSize;
      this.batchSize = batchSize;
      this.stepsMin = stepsMin;
      this.stepsMax = stepsMax;
      this.damageCount = damageCount;
      this.damageStartIter = damageStartIter;
      this.optimizer = tf.train.adam(lr);
      this.iter = 0;

      const seed = model.seedArray();
      this.pool = [];
      for (let i = 0; i < poolSize; i++) this.pool.push(seed.slice());

      this.targetTensor = tf.tensor4d(targetRgba, [1, model.size, model.size, 4])
        .tile([batchSize, 1, 1, 1]);
    }

    /** MSE between a pool state's RGBA and the target, on CPU. */
    sampleLoss(state) {
      const { size, channels } = this.model;
      let sum = 0;
      for (let i = 0, n = size * size; i < n; i++) {
        for (let c = 0; c < 4; c++) {
          const d = state[i * channels + c] - this.target[i * 4 + c];
          sum += d * d;
        }
      }
      return sum / (size * size * 4);
    }

    damage(state) {
      const { size, channels } = this.model;
      const r = 3 + Math.random() * (size * 0.18);
      const cx = r + Math.random() * (size - 2 * r);
      const cy = r + Math.random() * (size - 2 * r);
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          if ((x - cx) ** 2 + (y - cy) ** 2 < r * r) {
            state.fill(0, (y * size + x) * channels, (y * size + x) * channels + channels);
          }
        }
      }
    }

    /** One optimization step. Returns {loss, sample} where sample is a grown state. */
    step() {
      const { model, batchSize } = this;
      const { size, channels } = model;
      const cellLen = size * size * channels;

      // Pick a batch from the pool, sorted worst-first by current loss.
      const idx = [];
      while (idx.length < batchSize) {
        const i = (Math.random() * this.poolSize) | 0;
        if (!idx.includes(i)) idx.push(i);
      }
      idx.sort((a, b) => this.sampleLoss(this.pool[b]) - this.sampleLoss(this.pool[a]));

      // Worst sample restarts from seed; best ones get damaged (after warm-up).
      this.pool[idx[0]] = model.seedArray();
      if (this.iter >= this.damageStartIter) {
        for (let k = 0; k < this.damageCount; k++) {
          this.damage(this.pool[idx[idx.length - 1 - k]]);
        }
      }

      const batchArr = new Float32Array(batchSize * cellLen);
      idx.forEach((p, b) => batchArr.set(this.pool[p], b * cellLen));

      const nSteps = this.stepsMin + ((Math.random() * (this.stepsMax - this.stepsMin)) | 0);
      let finalState = null;

      const lossFn = () => {
        const x0 = tf.tensor4d(batchArr, [batchSize, size, size, channels]);
        let x = x0;
        for (let i = 0; i < nSteps; i++) x = model.step(x);
        finalState = tf.keep(x.clone());
        const rgba = x.slice([0, 0, 0, 0], [-1, -1, -1, 4]);
        return rgba.sub(this.targetTensor).square().mean();
      };

      const { value, grads } = tf.variableGrads(lossFn, model.variables());

      // Per-variable gradient normalization (stabilizes long rollouts).
      const normed = {};
      for (const name of Object.keys(grads)) {
        normed[name] = tf.tidy(() => grads[name].div(grads[name].norm().add(1e-8)));
        grads[name].dispose();
      }
      this.optimizer.applyGradients(normed);
      Object.values(normed).forEach(t => t.dispose());

      // Grown states go back into the pool.
      const finalData = finalState.dataSync();
      idx.forEach((p, b) => {
        this.pool[p] = finalData.slice(b * cellLen, (b + 1) * cellLen);
      });
      const sample = this.pool[idx[idx.length - 1]];
      finalState.dispose();

      const loss = value.dataSync()[0];
      value.dispose();
      this.iter++;
      return { loss, sample };
    }

    dispose() {
      this.targetTensor.dispose();
      this.optimizer.dispose();
    }
  }

  return { Model, Trainer };
})();
