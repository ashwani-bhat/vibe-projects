/**
 * UI: harness knobs → batch metrics + ship gates + a Pareto frontier of every
 * config you've tried, plus a live terminal replay of single tickets.
 */

(() => {
  'use strict';

  const $ = id => document.getElementById(id);

  const state = {
    cfg: { ...HARNESS.DEFAULT_CFG },
    seed: 7,
    runs: [],          // { cost, pass, label, gates } — the Pareto history
    last: null,
    animTimer: null,
  };

  /* ---------- knobs ---------- */

  function readCfg() {
    state.cfg = {
      stepBudget: parseInt($('k-budget').value, 10),
      verifier: document.querySelector('#k-verifier .on').dataset.v,
      shaping: $('k-shaping').checked,
      compaction: $('k-compaction').checked,
      structuredErrors: $('k-typed').checked,
      noProgress: $('k-noprog').checked,
    };
    $('k-budget-val').textContent = state.cfg.stepBudget;
  }

  function writeCfg(cfg) {
    $('k-budget').value = cfg.stepBudget;
    for (const b of document.querySelectorAll('#k-verifier button')) {
      b.classList.toggle('on', b.dataset.v === cfg.verifier);
    }
    $('k-shaping').checked = cfg.shaping;
    $('k-compaction').checked = cfg.compaction;
    $('k-typed').checked = cfg.structuredErrors;
    $('k-noprog').checked = cfg.noProgress;
    readCfg();
  }

  const PRESETS = {
    naive: { ...HARNESS.DEFAULT_CFG },
    kitchen: {
      stepBudget: 40, verifier: 'strict', shaping: true,
      compaction: true, structuredErrors: true, noProgress: true,
    },
  };

  /* ---------- batch runs ---------- */

  function cfgLabel(c) {
    const bits = [`${c.stepBudget}st`, c.verifier];
    if (c.shaping) bits.push('shape');
    if (c.compaction) bits.push('compact');
    if (c.structuredErrors) bits.push('typed');
    if (c.noProgress) bits.push('detect');
    return bits.join('·');
  }

  function runBatch() {
    readCfg();
    const m = HARNESS.runBatch(state.cfg, 200, state.seed);
    state.seed = (state.seed * 1664525 + 1013904223) >>> 0; // fresh tickets next run
    state.last = m;
    const gates = HARNESS.gateCheck(m);
    state.runs.push({ cost: m.costPerSolve, pass: m.passRate, label: cfgLabel(state.cfg), gates });
    renderMetrics(m, gates);
    drawPareto();
  }

  function fmtK(v) { return Number.isFinite(v) ? (v / 1000).toFixed(1) + 'k' : '∞'; }
  const pct = v => (v * 100).toFixed(1) + '%';

  function renderMetrics(m, gates) {
    $('m-pass').textContent = pct(m.passRate);
    $('m-false').textContent = pct(m.falseRate);
    $('m-doom').textContent = pct(m.doomRate + m.truncRate + m.stopRate);
    $('m-cost').textContent = fmtK(m.avgCost);
    $('m-solve').textContent = fmtK(m.costPerSolve);
    $('m-p95').textContent = m.p95Steps;

    const gateEls = [['g-pass', gates.pass, `pass ≥ ${HARNESS.GATE.passRate * 100}%`],
      ['g-honest', gates.honest, `false success ≤ ${HARNESS.GATE.falseRate * 100}%`],
      ['g-cheap', gates.cheap, `≤ ${HARNESS.GATE.costPerSolve / 1000}k/solve`]];
    for (const [id, ok, label] of gateEls) {
      const el = $(id);
      el.textContent = `${ok ? '✓' : '✗'} ${label}`;
      el.className = `gate ${ok ? 'gate--ok' : 'gate--bad'}`;
    }
    const ship = gates.pass && gates.honest && gates.cheap;
    $('ship').textContent = ship
      ? '🚢 SHIP IT — this harness clears the gate'
      : 'not shippable yet — same model, keep tuning the harness';
    $('ship').className = `ship ${ship ? 'ship--yes' : ''}`;
  }

  /* ---------- Pareto chart ---------- */

  const chart = $('pareto-canvas'), pctx = chart.getContext('2d');

  function drawPareto() {
    const w = chart.width, h = chart.height;
    const L = 56, B = 30, T = 22;               // plot margins
    const X = c => L + Math.min(1, c / 120000) * (w - L - 14);
    const Y = p => (h - B) - p * (h - B - T);
    pctx.clearRect(0, 0, w, h);

    // ship-gate region
    pctx.fillStyle = 'rgba(115, 239, 247, 0.06)';
    pctx.fillRect(L, Y(1), X(HARNESS.GATE.costPerSolve) - L, Y(HARNESS.GATE.passRate) - Y(1));
    pctx.strokeStyle = '#2c3047';
    pctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    pctx.fillStyle = '#566c86';
    pctx.font = '10px ui-monospace, monospace';

    // axis titles sit outside the plot: x along the bottom-left, y rotated at the edge
    pctx.fillText('cost per solved ticket →', L, h - 6);
    pctx.save();
    pctx.translate(11, h - B);
    pctx.rotate(-Math.PI / 2);
    pctx.fillText('pass rate →', 0, 0);
    pctx.restore();

    pctx.textAlign = 'center';
    for (const c of [30000, 60000, 90000]) pctx.fillText(`${c / 1000}k`, X(c), h - 18);
    pctx.textAlign = 'right';
    for (const p of [0.5, 1]) pctx.fillText(pct(p), L - 6, Y(p) + 3);
    pctx.textAlign = 'left';

    state.runs.forEach((r, i) => {
      const isLast = i === state.runs.length - 1;
      const ship = r.gates.pass && r.gates.honest && r.gates.cheap;
      pctx.beginPath();
      pctx.arc(X(r.cost), Y(r.pass), isLast ? 6 : 4, 0, 2 * Math.PI);
      pctx.fillStyle = isLast ? '#ffcd75' : ship ? '#73eff7' : '#3a3f5c';
      pctx.fill();
      if (isLast) { pctx.strokeStyle = '#e6e8f2'; pctx.stroke(); }
    });
    if (state.runs.length) {
      const r = state.runs[state.runs.length - 1];
      $('pareto-note').textContent =
        `latest: ${r.label} → ${pct(r.pass)} at ${fmtK(r.cost)}/solve · ${state.runs.length} config${state.runs.length > 1 ? 's' : ''} tried`;
    }
  }

  /* ---------- single-ticket replay ---------- */

  function watchOne() {
    readCfg();
    if (state.animTimer) clearInterval(state.animTimer);
    const log = [];
    HARNESS.runTrajectory(state.cfg, HARNESS.rng((Math.random() * 1e9) | 0), log);
    const term = $('term');
    term.innerHTML = '';
    let i = 0;
    state.animTimer = setInterval(() => {
      if (i >= log.length) { clearInterval(state.animTimer); state.animTimer = null; return; }
      const e = log[i++];
      const div = document.createElement('div');
      div.className = `tl tl--${e.type}`;
      div.textContent = e.text;
      term.appendChild(div);
      term.scrollTop = term.scrollHeight;
    }, 240);
  }

  /* ---------- wiring ---------- */

  $('k-budget').addEventListener('input', readCfg);
  for (const b of document.querySelectorAll('#k-verifier button')) {
    b.addEventListener('click', () => {
      document.querySelector('#k-verifier .on').classList.remove('on');
      b.classList.add('on');
      readCfg();
    });
  }
  for (const id of ['k-shaping', 'k-compaction', 'k-typed', 'k-noprog']) {
    $(id).addEventListener('change', readCfg);
  }
  $('btn-run').addEventListener('click', runBatch);
  $('btn-watch').addEventListener('click', watchOne);
  $('btn-naive').addEventListener('click', () => writeCfg(PRESETS.naive));
  $('btn-kitchen').addEventListener('click', () => writeCfg(PRESETS.kitchen));
  $('btn-clear').addEventListener('click', () => { state.runs = []; drawPareto(); $('pareto-note').textContent = ''; });

  writeCfg(PRESETS.naive);
  drawPareto();
  runBatch(); // show the naive baseline immediately

  // Test hook for the headless browser check.
  window.LAB = {
    run: () => { runBatch(); return state.last; },
    setCfg: writeCfg,
    cfg: () => state.cfg,
    runs: () => state.runs,
    watchOne,
  };
})();
