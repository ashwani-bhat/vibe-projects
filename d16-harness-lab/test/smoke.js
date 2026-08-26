/* Headless smoke test: the harness knobs must produce the trade-offs the
 * simulation claims — determinism, context rot, doom loops, verifier
 * economics, and a reachable ship gate.
 * Run: node test/smoke.js   (no dependencies)
 */
const fs = require('fs');
const path = require('path');

const HARNESS = eval(
  fs.readFileSync(path.join(__dirname, '..', 'sim.js'), 'utf8') + '; HARNESS');

let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failed = true;
};
const pct = v => (v * 100).toFixed(1) + '%';
const batch = (over, seed = 7) =>
  HARNESS.runBatch({ ...HARNESS.DEFAULT_CFG, ...over }, 400, seed);

/* ---- determinism ---- */
{
  const a = batch({}, 42), b = batch({}, 42);
  check(JSON.stringify(a) === JSON.stringify(b), 'same seed → identical batch');
}

/* ---- the naive harness is bad ---- */
const naive = batch({});
console.log(`naive: pass ${pct(naive.passRate)} · false ${pct(naive.falseRate)} · doom ${pct(naive.doomRate)} · cost/solve ${(naive.costPerSolve / 1000).toFixed(0)}k`);
check(naive.passRate < 0.60, `naive harness passes < 60% (${pct(naive.passRate)})`);
check(naive.falseRate > 0.03, `naive harness ships wrong fixes (${pct(naive.falseRate)})`);
check(naive.doomRate > 0.05, `naive harness doom-loops (${pct(naive.doomRate)})`);

/* ---- a tuned harness clears the ship gate — same model ---- */
const tuned = batch({
  stepBudget: 26, verifier: 'strict', shaping: true,
  compaction: true, structuredErrors: true, noProgress: true,
});
console.log(`tuned: pass ${pct(tuned.passRate)} · false ${pct(tuned.falseRate)} · doom ${pct(tuned.doomRate)} · cost/solve ${(tuned.costPerSolve / 1000).toFixed(0)}k`);
const gate = HARNESS.gateCheck(tuned);
check(gate.pass && gate.honest && gate.cheap,
  `tuned harness clears the ship gate (pass ${pct(tuned.passRate)}, false ${pct(tuned.falseRate)}, ${(tuned.costPerSolve / 1000).toFixed(0)}k/solve)`);

/* ---- individual knobs do what they claim ---- */
{
  const noVerify = batch({ stepBudget: 26, shaping: true, compaction: true, structuredErrors: true, noProgress: true, verifier: 'none' });
  check(noVerify.falseRate > tuned.falseRate + 0.02,
    `dropping the verifier ships more wrong fixes (${pct(noVerify.falseRate)} vs ${pct(tuned.falseRate)})`);

  // isolate observation shaping: no compaction in either arm, so nothing caps the window
  const shapedNC = batch({ stepBudget: 26, verifier: 'strict', compaction: false, structuredErrors: true, noProgress: true, shaping: true });
  const rawNC = batch({ stepBudget: 26, verifier: 'strict', compaction: false, structuredErrors: true, noProgress: true, shaping: false });
  check(rawNC.avgCost > shapedNC.avgCost * 1.3,
    `raw logs cost >30% more without compaction (${(rawNC.avgCost / 1000).toFixed(1)}k vs ${(shapedNC.avgCost / 1000).toFixed(1)}k avg)`);
  check(rawNC.passRate < shapedNC.passRate,
    `raw logs also hurt accuracy via context rot (${pct(rawNC.passRate)} vs ${pct(shapedNC.passRate)})`);
  // with compaction on, raw logs still cost more — the knobs interact, they don't cancel
  const rawLogs = batch({ stepBudget: 26, verifier: 'strict', compaction: true, structuredErrors: true, noProgress: true, shaping: false });
  check(rawLogs.avgCost > tuned.avgCost * 1.1,
    `even compacted, raw logs cost >10% more (${(rawLogs.avgCost / 1000).toFixed(1)}k vs ${(tuned.avgCost / 1000).toFixed(1)}k avg)`);

  const noTyped = batch({ stepBudget: 26, verifier: 'strict', shaping: true, compaction: true, noProgress: false, structuredErrors: false });
  check(noTyped.doomRate > 0.03,
    `bare tool errors reintroduce doom loops (${pct(noTyped.doomRate)})`);

  const detectorOn = batch({ stepBudget: 26, verifier: 'strict', shaping: true, compaction: true, structuredErrors: false, noProgress: true });
  check(detectorOn.avgCost < noTyped.avgCost,
    `no-progress detector cuts wasted spend (${(detectorOn.avgCost / 1000).toFixed(1)}k vs ${(noTyped.avgCost / 1000).toFixed(1)}k)`);
}

/* ---- step budget is a real trade-off ---- */
{
  const tiny = batch({ stepBudget: 8, verifier: 'strict', shaping: true, compaction: true, structuredErrors: true, noProgress: true });
  check(tiny.passRate < tuned.passRate - 0.1,
    `a starved step budget truncates real work (${pct(tiny.passRate)})`);
}

/* ---- trajectory log is coherent ---- */
{
  const log = [];
  const rand = HARNESS.rng(5);
  const t = HARNESS.runTrajectory({ ...HARNESS.DEFAULT_CFG, stepBudget: 26, shaping: true, structuredErrors: true, noProgress: true, verifier: 'weak' }, rand, log);
  check(log.length >= 4 && log.every(e => e.type && e.text), `log has ${log.length} typed events`);
  check(Number.isFinite(t.cost) && t.cost > 0, 'trajectory cost is finite and positive');
}

console.log(failed ? 'FAIL' : 'PASS');
process.exit(failed ? 1 : 0);
