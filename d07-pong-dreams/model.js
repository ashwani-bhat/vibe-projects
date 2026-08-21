/**
 * The world model: (previous frame, current frame, action) → next frame.
 *
 * An MLP over raw pixels — two 32×32 frames give it motion, the action
 * one-hot gives it your intent, and per-pixel sigmoid cross-entropy teaches
 * it Pong's future. In dream mode we feed its own output back as input and
 * the network becomes the physics engine ("World Models", Ha & Schmidhuber
 * 2018, in miniature).
 */

const WORLD = (() => {
  'use strict';

  class Model {
    constructor({ res = 32, hidden = 384, actions = 3, lr = 1e-3 } = {}) {
      this.res = res;
      this.actions = actions;
      this.inDim = res * res * 2 + actions;
      this.outDim = res * res;
      this.hidden = hidden;
      this.optimizer = tf.train.adam(lr);

      const init = (nin, nout) => {
        const lim = Math.sqrt(6 / (nin + nout));
        return tf.variable(tf.randomUniform([nin, nout], -lim, lim));
      };
      this.W1 = init(this.inDim, hidden);
      this.b1 = tf.variable(tf.zeros([hidden]));
      this.W2 = init(hidden, this.outDim);
      this.b2 = tf.variable(tf.zeros([this.outDim]));
    }

    variables() { return [this.W1, this.b1, this.W2, this.b2]; }

    paramCount() {
      return this.inDim * this.hidden + this.hidden
        + this.hidden * this.outDim + this.outDim;
    }

    logits(x) {
      return tf.tidy(() =>
        tf.relu(tf.matMul(x, this.W1).add(this.b1)).matMul(this.W2).add(this.b2));
    }

    /** inputs: Float32Array[bs*inDim], targets: Float32Array[bs*outDim]. Returns BCE. */
    trainStep(inputs, targets, bs) {
      const loss = this.optimizer.minimize(() => {
        const xs = tf.tensor2d(inputs, [bs, this.inDim]);
        const ys = tf.tensor2d(targets, [bs, this.outDim]);
        return tf.losses.sigmoidCrossEntropy(ys, this.logits(xs));
      }, true, this.variables());
      const v = loss.dataSync()[0];
      loss.dispose();
      return v;
    }

    /** One-step prediction as pixel probabilities (Float32Array[res*res]). */
    predict(f0, f1, action) {
      return tf.tidy(() => {
        const x = buildInput(f0, f1, action, this.actions);
        const p = this.logits(tf.tensor2d(x, [1, this.inDim])).sigmoid();
        return p.dataSync();
      });
    }

    dispose() {
      for (const v of this.variables()) v.dispose();
      this.optimizer.dispose();
    }
  }

  /** Pack (f0, f1, one-hot action) into one input vector. action ∈ {−1,0,1}. */
  function buildInput(f0, f1, action, actions = 3) {
    const n = f0.length;
    const x = new Float32Array(n * 2 + actions);
    x.set(f0, 0);
    x.set(f1, n);
    x[n * 2 + (action + 1)] = 1;
    return x;
  }

  return { Model, buildInput };
})();
