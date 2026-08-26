/* Headless smoke test: the judge biases must be real and measurable, the
 * debiasing must actually debias, and a biased protocol must be capable of
 * producing a wrong leaderboard.
 * Run: node test/smoke.js   (no dependencies)
 */
const fs = require('fs');
const path = require('path');

const JURY = eval(
  fs.readFileSync(path.join(__dirname, '..', 'judges.js'), 'utf8') + '; JURY');

let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failed = true;
};
const pct = v => (v * 100).toFixed(1) + '%';
const byId = id => JURY.JUDGES.find(j => j.id === id);

const cfg = (over = {}) => ({
  panel: (over.panel || ['judge-A1']).map(byId),
  swap: false, aggregate: 'majority', excludeSelf: false, lengthNormalize: false,
  ...over,
  panel: (over.panel || ['judge-A1']).map(byId),
});

/* ---- determinism ---- */
{
  const a = JURY.runEval(cfg(), { n: 200, seed: 3 });
  const b = JURY.runEval(cfg(), { n: 200, seed: 3 });
  check(a.acc === b.acc && a.kappa === b.kappa, 'same seed → identical result');
}

/* ---- a single biased judge is mediocre and its errors are attributable ---- */
const solo = JURY.runEval(cfg(), { n: 400, seed: 5 });
console.log(`solo judge-A1: acc ${pct(solo.acc)} · κ ${solo.kappa.toFixed(2)} · close-call acc ${pct(solo.closeAcc)}`);
console.log(`  attribution: position ${(solo.attribution.position * 100).toFixed(1)}pp · verbosity ${(solo.attribution.verbosity * 100).toFixed(1)}pp · self-pref ${(solo.attribution.selfPref * 100).toFixed(1)}pp`);
check(solo.acc > 0.5 && solo.acc < 0.85, `solo judge is better than chance but far from reliable (${pct(solo.acc)})`);
check(solo.closeAcc < solo.acc, `close calls are harder than average (${pct(solo.closeAcc)} vs ${pct(solo.acc)})`);

/* Verbosity bias is insidious precisely because it barely moves aggregate
 * accuracy while corrupting the ranking — the metric people actually report. */
{
  const wordyJudge = JURY.runEval(cfg({ panel: ['judge-B1'] }), { n: 500, seed: 5 });
  check(Math.abs(wordyJudge.attribution.verbosity) < 0.05,
    `verbosity bias is nearly invisible in aggregate accuracy (${(wordyJudge.attribution.verbosity * 100).toFixed(1)}pp)`);
  check(!wordyJudge.topCorrect || wordyJudge.spearman < 1,
    `…yet it corrupts the leaderboard (top = ${wordyJudge.judged[0].id}, ρ = ${wordyJudge.spearman.toFixed(2)})`);
}

/* ---- position-swap debiasing removes position bias ---- */
{
  const swapped = JURY.runEval(cfg({ swap: true }), { n: 400, seed: 5 });
  check(Math.abs(swapped.attribution.position) < Math.abs(solo.attribution.position) + 1e-9,
    `swap shrinks position-bias attribution (${(swapped.attribution.position * 100).toFixed(1)}pp vs ${(solo.attribution.position * 100).toFixed(1)}pp)`);
  check(swapped.cost > solo.cost * 1.9, `swap roughly doubles cost (${swapped.cost} vs ${solo.cost})`);
}

/* ---- length normalization blunts verbosity bias ---- */
{
  const norm = JURY.runEval(cfg({ lengthNormalize: true }), { n: 400, seed: 5 });
  check(norm.attribution.verbosity < solo.attribution.verbosity,
    `rubric length cap leaves less verbosity error on the table (${(norm.attribution.verbosity * 100).toFixed(1)}pp vs ${(solo.attribution.verbosity * 100).toFixed(1)}pp)`);
  check(norm.acc > solo.acc, `and improves accuracy (${pct(norm.acc)} vs ${pct(solo.acc)})`);
}

/* ---- a monoculture council inflates its own family's win rate ---- */
{
  const monoculture = JURY.runEval(cfg({ panel: ['judge-A1', 'judge-A2'] }), { n: 400, seed: 5 });
  const diverse = JURY.runEval(cfg({ panel: ['judge-A1', 'judge-B1', 'judge-C1'] }), { n: 400, seed: 5 });
  const familyAWinRate = r => {
    const a = r.judged.filter(x => JURY.CANDIDATES.find(c => c.id === x.id).family === 'A');
    return a.reduce((s, x) => s + x.v, 0) / a.length;
  };
  const mono = familyAWinRate(monoculture), div = familyAWinRate(diverse);
  console.log(`monoculture(A,A): acc ${pct(monoculture.acc)} · family-A win rate ${pct(mono)}`);
  console.log(`diverse(A,B,C):   acc ${pct(diverse.acc)} · family-A win rate ${pct(div)}`);
  check(mono > div,
    `an all-family-A panel inflates family-A win rate (${pct(mono)} vs ${pct(div)}) — the bias shows up in the ranking, not the accuracy`);
  check(diverse.acc > monoculture.acc, `diverse council is also more accurate (${pct(diverse.acc)} vs ${pct(monoculture.acc)})`);
}

/* ---- the full protocol is good, and costs more ---- */
const full = JURY.runEval(cfg({
  panel: ['judge-A1', 'judge-B1', 'judge-C1', 'judge-H'],
  swap: true, aggregate: 'confidence', excludeSelf: true, lengthNormalize: true,
}), { n: 400, seed: 5 });
console.log(`full protocol: acc ${pct(full.acc)} · κ ${full.kappa.toFixed(2)} · ${Math.round(full.costPerItem)} tok/item · top-1 ${full.topCorrect ? 'correct' : 'WRONG'} · ρ ${full.spearman.toFixed(2)}`);
check(full.acc > solo.acc + 0.05, `full protocol beats the solo judge by >5pp (${pct(full.acc)} vs ${pct(solo.acc)})`);
check(full.kappa > solo.kappa, `and by kappa too (${full.kappa.toFixed(2)} vs ${solo.kappa.toFixed(2)})`);
check(full.costPerItem > solo.costPerItem * 3, `at >3x the cost per item (${Math.round(full.costPerItem)} vs ${Math.round(solo.costPerItem)})`);
check(full.topCorrect && full.spearman > 0.5, 'full protocol recovers the true top model');

/* ---- the payoff: a biased protocol can rank a worse model first ---- */
{
  // boreal is worse than atlas but much wordier; a verbosity-loving judge prefers it
  const wordy = JURY.runEval(cfg({ panel: ['judge-B1'] }), { n: 500, seed: 5 });
  const top = wordy.judged[0].id;
  console.log(`verbosity-loving solo judge ranks: ${wordy.judged.map(x => x.id).join(' > ')}`);
  console.log(`ground truth ranks:              ${wordy.truth.map(x => x.id).join(' > ')}`);
  check(wordy.spearman < full.spearman || top !== 'atlas',
    `a biased protocol distorts the leaderboard (ρ ${wordy.spearman.toFixed(2)} vs ${full.spearman.toFixed(2)}, top=${top})`);
}

/* ---- kappa sanity ---- */
{
  const perfect = [{ pick: 'left', truth: 'left' }, { pick: 'right', truth: 'right' }];
  check(Math.abs(JURY.kappa(perfect) - 1) < 1e-9, 'kappa = 1 for perfect agreement');
  const ids = ['a', 'b', 'c'];
  check(Math.abs(JURY.spearman({ a: 1, b: 2, c: 3 }, { a: 1, b: 2, c: 3 }, ids) - 1) < 1e-9,
    'spearman = 1 for identical rankings');
}

console.log(failed ? 'FAIL' : 'PASS');
process.exit(failed ? 1 : 0);
