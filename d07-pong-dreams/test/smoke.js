/* Headless smoke test: Pong physics must behave, and a scaled-down world
 * model must learn to predict frames well enough to dream without exploding.
 * Run: node test/smoke.js
 */
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');
global.tf = tf;

const load = (file, name) =>
  eval(fs.readFileSync(path.join(__dirname, '..', file), 'utf8') + '; ' + name);
const PONG = load('pong.js', 'PONG');
const WORLD = load('model.js', 'WORLD');

let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failed = true;
};

/* ---- physics ---- */
{
  const sim = new PONG.Sim(3);
  sim.by = 0.03; sim.vy = -0.02; sim.vx = 0.01; sim.bx = 0.5;
  sim.step(0);
  check(sim.vy > 0, 'ball bounces off the top wall');

  sim.bx = 0.92; sim.by = 0.5; sim.ry = 0.5; sim.vx = 0.03; sim.vy = 0;
  sim.step(0);
  check(sim.vx < 0, 'right paddle returns the ball');

  const s2 = new PONG.Sim(4);
  s2.ly = 0.5;
  s2.step(1); check(s2.ly > 0.5, 'action +1 moves the paddle down');
  s2.step(-1); s2.step(-1);
  check(s2.ly < 0.5 + 0.036, 'action −1 moves it back up');

  const f = PONG.render(new PONG.Sim(5), 32);
  const lit = f.reduce((a, b) => a + b, 0);
  check(lit >= 20 && lit <= 70, `render lights a sane pixel count (${lit})`);

  const snap = sim.snapshot();
  sim.step(1); sim.step(1);
  sim.restore(snap);
  check(sim.bx === snap.bx && sim.ly === snap.ly, 'snapshot/restore round-trips');
}

/* ---- world model learns to predict ---- */
{
  const RES = 16;
  const sim = new PONG.Sim(7);
  const rand = PONG.rng(8);
  const model = new WORLD.Model({ res: RES, hidden: 128 });

  // collect transitions
  let frames = [PONG.render(sim, RES), PONG.render(sim, RES)];
  const buffer = [];
  for (let i = 0; i < 1500; i++) {
    const a = PONG.trackerPolicy(sim, rand, 0.35);
    const x = WORLD.buildInput(frames[0], frames[1], a);
    sim.step(a);
    const next = PONG.render(sim, RES);
    buffer.push({ x, y: next });
    frames = [frames[1], next];
  }

  const BATCH = 32;
  let first = null, last = null;
  for (let s = 0; s < 500; s++) {
    const xs = new Float32Array(BATCH * model.inDim);
    const ys = new Float32Array(BATCH * model.outDim);
    for (let b = 0; b < BATCH; b++) {
      const t = buffer[(Math.random() * buffer.length) | 0];
      xs.set(t.x, b * model.inDim);
      ys.set(t.y, b * model.outDim);
    }
    const loss = model.trainStep(xs, ys, BATCH);
    if (!Number.isFinite(loss)) { console.error(`FAIL: loss ${loss} at step ${s}`); process.exit(1); }
    if (s === 0) first = loss;
    last = loss;
    if (s % 125 === 0) console.log(`step ${s}  loss ${loss.toExponential(2)}`);
  }
  console.log(`loss: ${first.toExponential(2)} → ${last.toExponential(2)}`);
  check(last < first * 0.25, 'prediction loss dropped 4x');
  check(last < 0.08, `loss low enough to dream (${last.toExponential(2)})`);

  // closed-loop dream must stay finite and bounded
  let d0 = buffer[100].x.slice(0, RES * RES);
  let d1 = buffer[100].x.slice(RES * RES, RES * RES * 2);
  let ok = true, minSum = Infinity, maxSum = 0;
  for (let s = 0; s < 80; s++) {
    const p = model.predict(d0, d1, [(-1), 0, 1][s % 3]);
    let sum = 0;
    for (const v of p) {
      if (!Number.isFinite(v) || v < 0 || v > 1) { ok = false; break; }
      sum += v;
    }
    if (!ok) break;
    minSum = Math.min(minSum, sum); maxSum = Math.max(maxSum, sum);
    const fb = new Float32Array(p.length);
    for (let i = 0; i < p.length; i++) fb[i] = p[i] > 0.5 ? 1 : 0;
    d0 = d1; d1 = fb;
  }
  console.log(`dream brightness over 80 steps: ${minSum.toFixed(1)}..${maxSum.toFixed(1)} of ${RES * RES}`);
  check(ok, 'dream stays finite with all pixels in [0,1]');
  check(maxSum < RES * RES * 0.5, 'dream does not bleach to white');
  check(minSum > 1, 'dream does not collapse to black');
}

console.log(failed ? 'FAIL' : 'PASS');
process.exit(failed ? 1 : 0);
