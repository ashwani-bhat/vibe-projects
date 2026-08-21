/* Headless smoke test: physics must be stable and deterministic, and
 * evolution must actually produce better walkers.
 * Run: node test/smoke.js   (no dependencies)
 */
const fs = require('fs');
const path = require('path');

const load = (file, name) =>
  eval(fs.readFileSync(path.join(__dirname, '..', file), 'utf8') + '; ' + name);
const PHYS = load('physics.js', 'PHYS');
global.PHYS = PHYS;
const EVOLVE = load('evolve.js', 'EVOLVE');

let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failed = true;
};

/* ---- physics ---- */
{
  const g = EVOLVE.randomGenome(EVOLVE.rng(5));
  const sim = new PHYS.Sim(g);
  for (let i = 0; i < 600; i++) sim.step();
  check(sim.ok(), 'sim stays finite for 20 simulated seconds');
  check(sim.nodes.every(n => n.y <= PHYS.GROUND_Y + 1e-6), 'no node sinks through the ground');

  const f1 = PHYS.evaluate(g, 4);
  const f2 = PHYS.evaluate(g, 4);
  check(f1 === f2, 'evaluation is deterministic');
  check(Number.isFinite(f1), `fitness is finite (${f1.toFixed(3)})`);
}

/* ---- mutation keeps genomes legal ---- */
{
  const rand = EVOLVE.rng(11);
  let g = EVOLVE.randomGenome(rand);
  for (let i = 0; i < 200; i++) g = EVOLVE.mutate(g, rand, 1.5);
  check(g.nodes.length >= 3 && g.nodes.length <= 7, `node count stays in [3,7] (${g.nodes.length})`);
  check(g.muscles.every(m => m.a < g.nodes.length && m.b < g.nodes.length),
    'muscles always reference real nodes');
  check(g.nodes.every(n => n.grip >= 0 && n.grip <= 1), 'grip stays in [0,1]');
  check(Number.isFinite(PHYS.evaluate(g, 2)), '200-fold mutant still simulates');
}

/* ---- evolution improves fitness ---- */
{
  const pop = new EVOLVE.Population({ size: 40, seed: 7, evalSeconds: 6 });
  const start = pop.generation();
  let last = start;
  for (let gen = 0; gen < 40; gen++) last = pop.generation();
  console.log(`gen 1 best ${start.best.toFixed(3)} → gen ${pop.gen} best ${last.best.toFixed(3)}`);
  check(last.best > start.best * 1.5 && last.best > 0.5,
    `champion walks much further than gen-1 best (${start.best.toFixed(2)} → ${last.best.toFixed(2)})`);
  check(pop.history.length === pop.gen, 'history records every generation');
  const bests = pop.history.map(h => h.best);
  const monotonicish = bests[bests.length - 1] >= Math.max(...bests) - 1e-9;
  check(monotonicish, 'elitism: final best equals all-time best');
}

console.log(failed ? 'FAIL' : 'PASS');
process.exit(failed ? 1 : 0);
