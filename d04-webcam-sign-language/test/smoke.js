/* Headless smoke test: landmark canonicalization must be invariant to how the
 * hand sits in frame, and the classifier must learn synthetic "signs" through
 * the full normalize → train → predict pipeline.
 * Run: node test/smoke.js   (no dependencies)
 */
const fs = require('fs');
const path = require('path');

const load = (file, name) =>
  eval(fs.readFileSync(path.join(__dirname, '..', file), 'utf8') + '; ' + name);
const LANDMARKS = load('landmarks.js', 'LANDMARKS');
const CLASSIFIER = load('classifier.js', 'CLASSIFIER');

let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failed = true;
};
const maxDiff = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

/* Synthetic hand poses: 21 seeded random points in a hand-sized blob. */
function basePose(seed) {
  const rand = CLASSIFIER.rng(seed);
  const pts = [{ x: 0.5, y: 0.8, z: 0 }]; // wrist
  for (let i = 1; i < 21; i++) {
    pts.push({ x: 0.35 + rand() * 0.3, y: 0.35 + rand() * 0.4, z: (rand() - 0.5) * 0.1 });
  }
  return pts;
}

/* Place a pose in frame: rotate about the wrist, scale, translate, jitter. */
function placed(pose, { angle = 0, scale = 1, dx = 0, dy = 0, jitter = 0, rand = CLASSIFIER.rng(1), flip = false } = {}) {
  const [w] = pose;
  const cos = Math.cos(angle), sin = Math.sin(angle);
  return pose.map(p => {
    let x = (p.x - w.x) * scale, y = (p.y - w.y) * scale;
    [x, y] = [x * cos - y * sin, x * sin + y * cos];
    x += w.x + dx + CLASSIFIER.gauss(rand) * jitter;
    y += w.y + dy + CLASSIFIER.gauss(rand) * jitter;
    const z = p.z * scale + CLASSIFIER.gauss(rand) * jitter;
    return flip ? { x: 1 - x, y, z } : { x, y, z };
  });
}

/* ---- canonicalization invariances ---- */
{
  const pose = basePose(7);
  const ref = LANDMARKS.normalize(pose);
  check(ref && ref.length === 63, 'normalize returns 63 features');
  check(Math.abs(ref[0]) < 1e-6 && Math.abs(ref[1]) < 1e-6, 'wrist sits at the origin');
  const mcp = [ref[9 * 3], ref[9 * 3 + 1]];
  check(Math.abs(mcp[0]) < 1e-5 && Math.abs(mcp[1] + 1) < 1e-5, 'wrist→middle-MCP lands on (0,-1)');

  check(maxDiff(ref, LANDMARKS.normalize(placed(pose, { dx: 0.2, dy: -0.3 }))) < 1e-5,
    'translation-invariant');
  check(maxDiff(ref, LANDMARKS.normalize(placed(pose, { scale: 2.4 }))) < 1e-5,
    'scale-invariant');
  check(maxDiff(ref, LANDMARKS.normalize(placed(pose, { angle: 0.7 }))) < 1e-5,
    'rotation-invariant (in-plane)');
  check(maxDiff(ref, LANDMARKS.normalize(placed(pose, { flip: true }), 'Left')) < 1e-5,
    'left hands mirror onto right-hand features');
  check(LANDMARKS.normalize(pose.map(() => pose[0])) === null,
    'degenerate hand (all points equal) rejected');
}

/* ---- full pipeline: normalize → train → recognize ---- */
{
  const K = 4, PER = 40;
  const poses = [1, 2, 3, 4].map(basePose);
  const rand = CLASSIFIER.rng(99);
  const sample = k => LANDMARKS.normalize(placed(poses[k], {
    angle: (rand() - 0.5) * 1.2,
    scale: 0.6 + rand() * 1.2,
    dx: (rand() - 0.5) * 0.4,
    dy: (rand() - 0.5) * 0.4,
    jitter: 0.008,
    rand,
  }));

  const xs = [], ys = [];
  for (let k = 0; k < K; k++) for (let i = 0; i < PER; i++) { xs.push(sample(k)); ys.push(k); }
  const t0 = Date.now();
  const { net, valAcc, trainAcc } = CLASSIFIER.train(xs, ys, K, { seed: 42 });
  const ms = Date.now() - t0;
  console.log(`trained ${xs.length} samples in ${ms}ms — train ${trainAcc.toFixed(3)}, holdout ${valAcc.toFixed(3)}`);
  check(valAcc >= 0.95, `holdout accuracy ≥ 0.95 (got ${valAcc.toFixed(3)})`);
  check(ms < 5000, `training fast enough for a button press (${ms}ms)`);

  // fresh unseen samples, harder placements
  let correct = 0;
  const N = 80;
  for (let i = 0; i < N; i++) {
    const k = i % K;
    const p = net.predict(sample(k));
    let best = 0;
    for (let j = 1; j < K; j++) if (p[j] > p[best]) best = j;
    if (best === k) correct++;
    if (!p.every(Number.isFinite)) check(false, 'non-finite probability');
  }
  console.log(`fresh-sample accuracy: ${(correct / N).toFixed(3)}`);
  check(correct / N >= 0.95, `fresh-sample accuracy ≥ 0.95 (got ${(correct / N).toFixed(3)})`);

  const probs = net.predict(xs[0]);
  const sum = probs.reduce((a, b) => a + b, 0);
  check(Math.abs(sum - 1) < 1e-5, 'probabilities sum to 1');
}

console.log(failed ? 'FAIL' : 'PASS');
process.exit(failed ? 1 : 0);
