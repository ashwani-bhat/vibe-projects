/**
 * The simulation: a fixed, imperfect "model" works bug-fix tickets inside a
 * harness that YOU configure. Every mechanic is a real agentic-harness
 * phenomenon with the same shape it has in production:
 *
 *  - context rot: action accuracy decays as the window fills
 *  - quadratic cost: every turn re-reads the whole context
 *  - observation shaping: raw logs balloon the window, shaped ones don't
 *  - compaction: caps the window but can drop a load-bearing fact
 *  - structured tool errors: a failure that carries information stops loops
 *  - doom loops: same failure, same retry, until something breaks the cycle
 *  - verification: catches bad edits at a price — including false rejects
 *  - false success: the worst outcome is shipping a wrong fix that LOOKS done
 *
 * The model's competence parameters are constants. Only the harness knobs
 * move. That's the point.
 *
 * Pure JS, seeded, deterministic — testable headlessly.
 */

const HARNESS = (() => {
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

  /* ---- the model (fixed; not a knob, on purpose) ---- */
  const MODEL = {
    baseActionAcc: 0.88,   // chance of the right next action with a clean window
    rotExponent: 1.1,      // how hard context quality hits action accuracy
    editAcc: 0.78,         // chance an edit is actually correct, at q = 1
    editRotExponent: 0.8,
    structuredBoost: 0.25, // a typed error is new information
    lostFactPenalty: 0.6,
  };

  const ENV = {
    testsCatchBadEdit: 0.65, // the test suite is incomplete, like all test suites
    doomChance: 0.5,         // chance repeated uninformative failure wedges the loop
    rotScale: 14000,         // tokens at which the window is fully rotten
    compactAt: 6500,
    compactTo: 1800,
    compactLossChance: 0.12, // compaction drops a load-bearing fact
    falseRejectStrict: 0.05, // strict verifiers reject some good edits too
  };

  const DEFAULT_CFG = {
    stepBudget: 20,          // 6..40
    verifier: 'none',        // 'none' | 'weak' | 'strict'
    shaping: false,          // shape observations vs raw logs
    compaction: false,
    structuredErrors: false,
    noProgress: false,       // stop after 3 consecutive uninformative failures
  };

  const PHASES = ['locate', 'read', 'edit', 'test'];

  /**
   * One ticket, end to end. Returns outcome + cost accounting.
   * If `log` is provided, human-readable events are pushed into it.
   */
  function runTrajectory(cfg, rand, log = null) {
    const say = (type, text) => { if (log) log.push({ type, text }); };
    // A ticket needs 1–3 separate fixes (avg ≈ 1.75) — real tickets are rarely one edit.
    const r0 = rand();
    const nFixes = r0 < 0.45 ? 1 : r0 < 0.8 ? 2 : 3;
    let fixesDone = 0;
    let badFixes = 0;          // wrong fixes currently believed to be done
    let phase = 0;
    let ctx = 900;             // system prompt + ticket
    let cost = 0;
    let steps = 0;
    let editGood = null;
    let consecFails = 0;
    let lostFact = false;

    const q = () => Math.max(0.25, 1 - ctx / ENV.rotScale);
    const turnCost = () => { cost += Math.round(ctx * 0.25 + 300); };

    say('info', `ticket opened · ${nFixes} underlying bug${nFixes > 1 ? 's' : ''} · context ${ctx} tok`);

    while (steps < cfg.stepBudget) {
      steps++;
      turnCost();

      if (cfg.noProgress && consecFails >= 3) {
        say('stop', `no-progress detector fired after ${consecFails} dead turns — stopping with findings`);
        return { solved: false, falseSuccess: false, steps, cost, doom: false, stopped: true };
      }

      let pAct = MODEL.baseActionAcc * Math.pow(q(), MODEL.rotExponent);
      if (consecFails > 0 && cfg.structuredErrors) pAct = Math.min(0.95, pAct + MODEL.structuredBoost);
      if (lostFact) pAct *= MODEL.lostFactPenalty;

      if (rand() < pAct) {
        consecFails = 0;
        switch (PHASES[phase]) {
          case 'locate':
            ctx += 400;
            say('act', `turn ${steps} · search("stack trace") → src/billing/refund.js:412`);
            phase++;
            break;
          case 'read':
            ctx += cfg.shaping ? 350 : 1400;
            lostFact = false;
            say('act', `turn ${steps} · read_file(refund.js) → ${cfg.shaping ? 'relevant region, 350 tok' : 'whole file, 1400 tok'}`);
            phase++;
            break;
          case 'edit':
            editGood = rand() < MODEL.editAcc * Math.pow(q(), MODEL.editRotExponent);
            ctx += 250;
            say('act', `turn ${steps} · edit(refund.js) — patch applied`);
            phase++;
            break;
          case 'test': {
            const obs = cfg.shaping ? 200 : 800 + ((rand() * 2200) | 0);
            ctx += obs;
            if (editGood) {
              say('ok', `turn ${steps} · run_tests → PASS · fix ${fixesDone + 1}/${nFixes} done (${obs} tok of output)`);
              fixesDone++;
              phase = fixesDone < nFixes ? 1 : PHASES.length; // next bug or finish
            } else if (rand() < ENV.testsCatchBadEdit) {
              say('err', `turn ${steps} · run_tests → FAIL: test_refund_rounding (${obs} tok)`);
              phase = 2; // informative failure: back to edit
              consecFails++;
            } else {
              say('warn', `turn ${steps} · run_tests → PASS — but the suite never covered this path`);
              fixesDone++;
              badFixes++;
              phase = fixesDone < nFixes ? 1 : PHASES.length;
            }
            break;
          }
        }

        if (phase >= PHASES.length) {
          if (cfg.verifier !== 'none') {
            const strict = cfg.verifier === 'strict';
            cost += strict ? 2600 : 900;
            const caught = badFixes > 0 && rand() < (strict ? 0.95 : 0.6);
            if (caught) {
              say('verf', `verifier: REJECT — a patch does not address its root cause. back to work.`);
              badFixes--;
              fixesDone--;
              phase = 1;
              consecFails++;
              continue;
            }
            if (badFixes === 0 && strict && rand() < ENV.falseRejectStrict) {
              say('verf', `verifier: REJECT (false alarm) — re-doing a correct fix`);
              fixesDone--;
              phase = 1;
              continue;
            }
            say('verf', `verifier: APPROVE`);
          }
          const clean = badFixes === 0;
          if (clean) say('done', `ticket closed · genuinely fixed · ${steps} turns · ${(cost / 1000).toFixed(1)}k tok`);
          else say('bad', `ticket closed · ${badFixes} WRONG FIX${badFixes > 1 ? 'ES' : ''} SHIPPED — nothing in the loop could tell`);
          return { solved: clean, falseSuccess: !clean, steps, cost, doom: false };
        }
      } else {
        consecFails++;
        const r = rand();
        if (r < 0.5) {
          ctx += cfg.shaping ? 250 : 900;
          say('err', `turn ${steps} · grep(entire repo) → 3,000 irrelevant matches${cfg.shaping ? ' (shaped to 250 tok)' : ' dumped into context'}`);
        } else if (r < 0.8) {
          ctx += 500;
          say('err', `turn ${steps} · read_file(utils/refund_helper.js) → ${cfg.structuredErrors ? 'typed error: no such file; nearest: refund.js' : 'Error'}`);
        } else {
          ctx += 200;
          say('err', `turn ${steps} · model wrote prose instead of a tool call`);
        }

        if (!cfg.structuredErrors && consecFails >= 3 && rand() < ENV.doomChance) {
          say('doom', `doom loop: the same bare error, retried identically…`);
          if (!cfg.noProgress) {
            while (steps < cfg.stepBudget) { steps++; turnCost(); }
            say('stop', `step budget exhausted inside the loop · ${(cost / 1000).toFixed(1)}k tok burned`);
            return { solved: false, falseSuccess: false, steps, cost, doom: true };
          }
        }
      }

      if (cfg.compaction && ctx > ENV.compactAt) {
        ctx = ENV.compactTo;
        cost += 1200;
        if (rand() < ENV.compactLossChance) {
          lostFact = true;
          if (phase > 1) phase = 1;
          say('warn', `compaction dropped a load-bearing fact — re-reading`);
        } else {
          say('info', `compacted context → ${ENV.compactTo} tok`);
        }
      }
    }

    say('stop', `step budget exhausted · unfinished`);
    return { solved: false, falseSuccess: false, steps, cost, doom: false, truncated: true };
  }

  /** N trajectories → the numbers a harness engineer actually watches. */
  function runBatch(cfg, n = 200, seed = 1) {
    const rand = rng(seed);
    let solved = 0, falseSucc = 0, doom = 0, truncated = 0, stopped = 0;
    let totalCost = 0;
    const stepsArr = [];
    for (let i = 0; i < n; i++) {
      const t = runTrajectory(cfg, rand);
      if (t.solved) solved++;
      if (t.falseSuccess) falseSucc++;
      if (t.doom) doom++;
      if (t.truncated) truncated++;
      if (t.stopped) stopped++;
      totalCost += t.cost;
      stepsArr.push(t.steps);
    }
    stepsArr.sort((a, b) => a - b);
    return {
      n,
      passRate: solved / n,
      falseRate: falseSucc / n,
      doomRate: doom / n,
      truncRate: truncated / n,
      stopRate: stopped / n,
      avgCost: totalCost / n,
      costPerSolve: solved ? totalCost / solved : Infinity,
      p95Steps: stepsArr[Math.min(n - 1, Math.floor(n * 0.95))],
    };
  }

  /** The ship gate: what "good enough to deploy" means here. */
  const GATE = { passRate: 0.80, falseRate: 0.02, costPerSolve: 45000 };

  function gateCheck(m) {
    return {
      pass: m.passRate >= GATE.passRate,
      honest: m.falseRate <= GATE.falseRate,
      cheap: m.costPerSolve <= GATE.costPerSolve,
    };
  }

  return { runTrajectory, runBatch, gateCheck, rng, DEFAULT_CFG, GATE, MODEL, ENV };
})();
