/* Headless smoke test: a scaled-down NCA should learn to grow a blob.
 * Run: node test/smoke.js
 */
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');
global.tf = tf;

const NCA = eval(fs.readFileSync(path.join(__dirname, '..', 'nca.js'), 'utf8') + '; NCA');

const SIZE = 16, CH = 16;
const model = new NCA.Model({ size: SIZE, channels: CH, hidden: 32 });

// Target: a red 6x6 square in the middle (premultiplied RGBA).
const target = new Float32Array(SIZE * SIZE * 4);
for (let y = 5; y < 11; y++) {
  for (let x = 5; x < 11; x++) {
    const i = (y * SIZE + x) * 4;
    target[i] = 0.9; target[i + 3] = 1;
  }
}

const trainer = new NCA.Trainer(model, target, {
  poolSize: 16, batchSize: 2, stepsMin: 12, stepsMax: 18,
  damageStartIter: 40, damageCount: 1, lr: 2e-3,
});

const ITERS = 120;
let first = null, last = null;
for (let i = 0; i < ITERS; i++) {
  const { loss } = trainer.step();
  if (!Number.isFinite(loss)) { console.error(`FAIL: loss is ${loss} at iter ${i}`); process.exit(1); }
  if (i === 0) first = loss;
  last = loss;
  if (i % 20 === 0) console.log(`iter ${i}  loss ${loss.toExponential(3)}`);
}
console.log(`iter ${ITERS - 1}  loss ${last.toExponential(3)}`);

// Grow from seed and check the center actually fills in.
let x = model.seedTensor(1);
for (let i = 0; i < 24; i++) { const nx = model.step(x); x.dispose(); x = nx; }
const state = x.dataSync();
const center = ((SIZE / 2) * SIZE + SIZE / 2) * CH;
const alpha = state[center + 3], red = state[center];
console.log(`grown center: alpha=${alpha.toFixed(3)} red=${red.toFixed(3)}`);

// Weight round-trip.
const before = model.w1.dataSync()[0];
const exported = model.exportWeights();
model.initWeights();
model.importWeights(exported);
if (Math.abs(model.w1.dataSync()[0] - before) > 1e-6) { console.error('FAIL: weight round-trip'); process.exit(1); }

const ok = last < first * 0.7 && alpha > 0.1;
console.log(ok ? 'PASS' : `FAIL: first=${first} last=${last} alpha=${alpha}`);
process.exit(ok ? 0 : 1);
