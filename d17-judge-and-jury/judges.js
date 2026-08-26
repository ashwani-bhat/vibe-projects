/**
 * LLM-as-a-Judge, simulated honestly.
 *
 * Judges are not noisy oracles — they are *biased* oracles, and the biases are
 * documented and reproducible:
 *
 *   position bias      — prefers whichever answer is shown first (or second)
 *   verbosity bias     — prefers the longer answer regardless of quality
 *   self-preference    — prefers output from its own model family
 *   severity           — a constant offset, harmless in pairwise, fatal in absolute
 *   noise              — everything else
 *
 * The interesting part is not that judges are wrong. It is that their errors are
 * *correlated*, so a council of similar judges is confidently wrong together,
 * and a leaderboard built on them can rank a worse model first.
 *
 * Pure JS, seeded, deterministic — testable headlessly.
 */

const JURY = (() => {
  'use strict';

  function rng(seed = 1) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gauss = rand =>
    Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());

  /* The four systems under evaluation. `quality` is the ground truth the judges
   * are trying to recover; `verbosity` is how long their answers tend to run. */
  // Verbosity is deliberately uncorrelated with quality: the best model is terse
  // and the second-best is the wordiest, so length is pure noise to a good judge
  // and a leaderboard-flipping trap to a length-loving one.
  const CANDIDATES = [
    { id: 'atlas',  family: 'A', quality: 0.74, verbosity: 0.30 }, // best, terse
    { id: 'boreal', family: 'B', quality: 0.70, verbosity: 0.92 }, // worse, wordy
    { id: 'cirrus', family: 'C', quality: 0.62, verbosity: 0.48 },
    { id: 'delta',  family: 'A', quality: 0.51, verbosity: 0.62 }, // worst, chatty
  ];

  /* Judge roster. The default council is deliberately family-A-heavy — the most
   * common real mistake, because family A is usually the strongest model. */
  const JUDGES = [
    { id: 'judge-A1', family: 'A', position: 0.16, verbosity: 0.30, selfPref: 0.22, noise: 0.11, cost: 900 },
    { id: 'judge-A2', family: 'A', position: 0.12, verbosity: 0.26, selfPref: 0.20, noise: 0.10, cost: 900 },
    { id: 'judge-B1', family: 'B', position: 0.10, verbosity: 0.34, selfPref: 0.24, noise: 0.13, cost: 700 },
    { id: 'judge-C1', family: 'C', position: 0.18, verbosity: 0.12, selfPref: 0.18, noise: 0.16, cost: 400 },
    { id: 'judge-H',  family: 'H', position: 0.06, verbosity: 0.08, selfPref: 0.00, noise: 0.08, cost: 2600 },
  ];

  /**
   * One evaluation item: two answers to the same prompt, from two candidates.
   * `advantage` is the ground-truth quality gap (positive → left answer better).
   */
  function makeItems(n, rand) {
    const items = [];
    for (let i = 0; i < n; i++) {
      let a = (rand() * CANDIDATES.length) | 0;
      let b = (rand() * CANDIDATES.length) | 0;
      while (b === a) b = (rand() * CANDIDATES.length) | 0;
      const A = CANDIDATES[a], B = CANDIDATES[b];
      // per-item draw around each candidate's true quality
      const qa = A.quality + gauss(rand) * 0.10;
      const qb = B.quality + gauss(rand) * 0.10;
      const la = A.verbosity + gauss(rand) * 0.12;
      const lb = B.verbosity + gauss(rand) * 0.12;
      items.push({
        left: A, right: B,
        qa, qb, la, lb,
        truth: qa > qb ? 'left' : 'right',
        margin: Math.abs(qa - qb),
      });
    }
    return items;
  }

  /**
   * A judge scores one item. `flip` presents the answers in swapped order.
   * `mask` can zero out individual biases — that's how the lab measures how much
   * error each bias is actually responsible for (counterfactual, not guesswork).
   */
  function judgeItem(judge, item, rand, { flip = false, mask = {} } = {}) {
    const pos = mask.position ? 0 : judge.position;
    const verb = mask.verbosity ? 0 : judge.verbosity;
    const self = mask.selfPref ? 0 : judge.selfPref;

    const score = (q, len, cand, isFirst) =>
      q
      + verb * (len - 0.5)
      + (cand.family === judge.family ? self : 0)
      + (isFirst ? pos : 0)
      + gauss(rand) * judge.noise;

    // presentation order
    const leftFirst = !flip;
    const sL = score(item.qa, item.la, item.left, leftFirst);
    const sR = score(item.qb, item.lb, item.right, !leftFirst);
    return { pick: sL > sR ? 'left' : 'right', conf: Math.abs(sL - sR) };
  }

  /**
   * Run one item through the configured evaluation protocol.
   * cfg: { panel: [judgeIds], swap: bool, aggregate: 'majority'|'confidence',
   *        excludeSelf: bool, lengthNormalize: bool }
   */
  function evaluateItem(cfg, item, rand, mask = {}) {
    const votes = [];
    let cost = 0;
    for (const j of cfg.panel) {
      // self-preference mitigation: a judge does not vote on its own family
      if (cfg.excludeSelf && (item.left.family === j.family || item.right.family === j.family)) continue;
      const effective = cfg.lengthNormalize
        ? { ...j, verbosity: j.verbosity * 0.25 }   // rubric caps length credit
        : j;
      const r1 = judgeItem(effective, item, rand, { mask });
      cost += j.cost;
      if (cfg.swap) {
        const r2 = judgeItem(effective, item, rand, { flip: true, mask });
        cost += j.cost;
        // position-swap debiasing: agree → keep, disagree → judge is undecided
        if (r1.pick === r2.pick) votes.push({ pick: r1.pick, conf: (r1.conf + r2.conf) / 2 });
        else votes.push({ pick: r1.conf >= r2.conf ? r1.pick : r2.pick, conf: Math.abs(r1.conf - r2.conf) * 0.5 });
      } else {
        votes.push(r1);
      }
    }
    if (votes.length === 0) return { pick: rand() < 0.5 ? 'left' : 'right', cost, abstained: true };

    if (cfg.aggregate === 'confidence') {
      let l = 0, r = 0;
      for (const v of votes) (v.pick === 'left' ? (l += v.conf) : (r += v.conf));
      return { pick: l >= r ? 'left' : 'right', cost };
    }
    const l = votes.filter(v => v.pick === 'left').length;
    return { pick: l * 2 >= votes.length ? 'left' : 'right', cost };
  }

  /** Cohen's kappa against ground truth for a binary decision. */
  function kappa(rows) {
    const n = rows.length;
    if (!n) return 0;
    let agree = 0, jLeft = 0, tLeft = 0;
    for (const r of rows) {
      if (r.pick === r.truth) agree++;
      if (r.pick === 'left') jLeft++;
      if (r.truth === 'left') tLeft++;
    }
    const po = agree / n;
    const pl = (jLeft / n) * (tLeft / n);
    const pr = ((n - jLeft) / n) * ((n - tLeft) / n);
    const pe = pl + pr;
    return pe === 1 ? 0 : (po - pe) / (1 - pe);
  }

  /** Spearman rank correlation between two id→rank maps over the same ids. */
  function spearman(rankA, rankB, ids) {
    const n = ids.length;
    let d2 = 0;
    for (const id of ids) d2 += Math.pow(rankA[id] - rankB[id], 2);
    return 1 - (6 * d2) / (n * (n * n - 1));
  }

  /**
   * The whole evaluation: agreement with truth, kappa, cost, per-bias error
   * attribution (by counterfactually masking each bias), and the leaderboard
   * the protocol produces versus the true one.
   */
  function runEval(cfg, { n = 300, seed = 11 } = {}) {
    const items = makeItems(n, rng(seed));

    const pass = (mask) => {
      const rand = rng(seed + 999); // same judge noise draws across masks
      const rows = [];
      let cost = 0;
      const wins = {}, plays = {};
      for (const c of CANDIDATES) { wins[c.id] = 0; plays[c.id] = 0; }
      for (const it of items) {
        const r = evaluateItem(cfg, it, rand, mask);
        cost += r.cost;
        rows.push({ pick: r.pick, truth: it.truth, margin: it.margin });
        plays[it.left.id]++; plays[it.right.id]++;
        wins[(r.pick === 'left' ? it.left : it.right).id]++;
      }
      const acc = rows.filter(r => r.pick === r.truth).length / rows.length;
      return { acc, k: kappa(rows), cost, rows, wins, plays };
    };

    const base = pass({});
    // counterfactual attribution: how much accuracy does removing each bias buy?
    const attribution = {
      position: pass({ position: true }).acc - base.acc,
      verbosity: pass({ verbosity: true }).acc - base.acc,
      selfPref: pass({ selfPref: true }).acc - base.acc,
    };

    // leaderboards: judged win-rate vs true quality
    const ids = CANDIDATES.map(c => c.id);
    const judged = ids.map(id => ({ id, v: base.plays[id] ? base.wins[id] / base.plays[id] : 0 }))
      .sort((a, b) => b.v - a.v);
    const truth = [...CANDIDATES].sort((a, b) => b.quality - a.quality).map(c => ({ id: c.id, v: c.quality }));
    const rJ = {}, rT = {};
    judged.forEach((x, i) => { rJ[x.id] = i + 1; });
    truth.forEach((x, i) => { rT[x.id] = i + 1; });

    // accuracy on the hard cases — where an eval actually earns its keep
    const close = base.rows.filter(r => r.margin < 0.08);
    const closeAcc = close.length ? close.filter(r => r.pick === r.truth).length / close.length : 0;

    return {
      n,
      acc: base.acc,
      kappa: base.k,
      closeAcc,
      closeN: close.length,
      cost: base.cost,
      costPerItem: base.cost / n,
      attribution,
      judged,
      truth,
      topCorrect: judged[0].id === truth[0].id,
      spearman: spearman(rJ, rT, ids),
    };
  }

  return { JUDGES, CANDIDATES, runEval, makeItems, judgeItem, evaluateItem, kappa, spearman, rng };
})();
