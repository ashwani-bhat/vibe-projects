/**
 * UI: evolution runs in fast-forward (whole generations between frames)
 * while the reigning champion walks in real time on the main canvas.
 */

(() => {
  'use strict';

  const $ = id => document.getElementById(id);

  const REPLAY_SECONDS = 12;

  let pop = null;
  let running = false;
  let champSim = null;
  let champFitness = -Infinity;
  let camX = 0;

  const canvas = $('walk-canvas'), ctx = canvas.getContext('2d');
  const chart = $('chart-canvas'), chartCtx = chart.getContext('2d');

  const SCALE = 170;                       // world units → px
  const GROUND_PX = canvas.height - 60;

  function reset() {
    pop = new EVOLVE.Population({ size: 60, seed: (Math.random() * 1e9) | 0 });
    champSim = null;
    champFitness = -Infinity;
    camX = 0;
    setStatus('generation 0 — a primordial soup of random wigglers');
    drawChart();
  }

  function setStatus(t) { $('status').textContent = t; }

  /* ---------- drawing ---------- */

  function worldToScreen(x, y) {
    return [
      canvas.width * 0.35 + (x - camX) * SCALE,
      GROUND_PX - (PHYS.GROUND_Y - y) * SCALE,
    ];
  }

  function drawWalker() {
    ctx.fillStyle = '#0f1119';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (champSim) {
      const cx = champSim.centroidX();
      camX += (cx - camX) * 0.35;
    }

    // ground + distance markers every 0.5 units
    ctx.strokeStyle = '#2c3047';
    ctx.beginPath();
    ctx.moveTo(0, GROUND_PX); ctx.lineTo(canvas.width, GROUND_PX);
    ctx.stroke();
    ctx.fillStyle = '#566c86';
    ctx.font = '10px ui-monospace, monospace';
    const first = Math.floor(camX - 2);
    for (let m = first; m < camX + 3; m += 1) {
      const [sx] = worldToScreen(m, PHYS.GROUND_Y);
      ctx.strokeStyle = '#232739';
      ctx.beginPath();
      ctx.moveTo(sx, GROUND_PX); ctx.lineTo(sx, GROUND_PX + 6);
      ctx.stroke();
      ctx.fillText(`${m.toFixed(1)}`, sx - 8, GROUND_PX + 18);
    }

    if (!champSim) {
      ctx.fillStyle = '#9aa0b8';
      ctx.font = '13px ui-monospace, monospace';
      ctx.fillText('press ▶ evolve — the first champion appears after generation 1', 40, 120);
      return;
    }

    // muscles: line thickness pulses with contraction
    for (const m of champSim.genome.muscles) {
      const a = champSim.nodes[m.a], b = champSim.nodes[m.b];
      const [ax, ay] = worldToScreen(a.x, a.y);
      const [bx, by] = worldToScreen(b.x, b.y);
      const squeeze = champSim.muscleLen(m) / m.rest;
      ctx.strokeStyle = squeeze < 1 ? '#ffcd75' : '#3a3f5c';
      ctx.lineWidth = 2 + (1.15 - Math.min(squeeze, 1.15)) * 12;
      ctx.beginPath();
      ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
      ctx.stroke();
    }
    // nodes: brighter = grippier
    for (const n of champSim.nodes) {
      const [sx, sy] = worldToScreen(n.x, n.y);
      ctx.fillStyle = `rgba(115, 239, 247, ${0.35 + n.grip * 0.65})`;
      ctx.beginPath();
      ctx.arc(sx, sy, 7, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#0f1119';
      ctx.stroke();
    }

    ctx.fillStyle = '#9aa0b8';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(`champion replay · ${champSim.distance().toFixed(2)} units walked`, 12, 18);
  }

  function drawChart() {
    const w = chart.width, h = chart.height;
    chartCtx.clearRect(0, 0, w, h);
    chartCtx.strokeStyle = '#2c3047';
    chartCtx.strokeRect(0.5, 0.5, w - 1, h - 1);
    const hist = pop.history;
    if (hist.length < 2) return;
    const max = Math.max(...hist.map(p => p.best), 0.1);
    const min = Math.min(...hist.map(p => p.mean), 0);
    const px = i => (i / (hist.length - 1)) * (w - 8) + 4;
    const py = v => h - 4 - ((v - min) / (max - min || 1)) * (h - 8);
    for (const [key, color] of [['mean', '#3a3f5c'], ['best', '#73eff7']]) {
      chartCtx.beginPath();
      hist.forEach((p, i) => i ? chartCtx.lineTo(px(i), py(p[key])) : chartCtx.moveTo(px(i), py(p[key])));
      chartCtx.strokeStyle = color;
      chartCtx.stroke();
    }
    chartCtx.fillStyle = '#9aa0b8';
    chartCtx.font = '10px ui-monospace, monospace';
    chartCtx.fillText(`fitness · best ${hist[hist.length - 1].best.toFixed(2)} · mean ${hist[hist.length - 1].mean.toFixed(2)}`, 8, 13);
  }

  /* ---------- main loop ---------- */

  function runGenerations(count) {
    const rate = parseFloat($('mut-rate').value);
    for (let i = 0; i < count; i++) pop.generation(rate);
    const champ = pop.champion();
    if (champ.fitness > champFitness + 1e-9) {
      champFitness = champ.fitness;
      champSim = new PHYS.Sim(champ.genome);
      camX = champSim.centroidX();
    }
    const { best, mean } = pop.history[pop.history.length - 1];
    setStatus(`generation ${pop.gen} · best walked ${best.toFixed(2)} · population mean ${mean.toFixed(2)}`);
    drawChart();
  }

  function tick() {
    if (running) runGenerations(1);
    if (champSim) {
      champSim.step();
      if (champSim.t > REPLAY_SECONDS) {
        champSim = new PHYS.Sim(champSim.genome);
        camX = champSim.centroidX();
      }
    }
    drawWalker();
    requestAnimationFrame(tick);
  }

  /* ---------- controls ---------- */

  $('btn-run').addEventListener('click', () => {
    running = !running;
    $('btn-run').textContent = running ? '⏸ pause evolution' : '▶ evolve';
    $('btn-run').classList.toggle('active', running);
  });
  $('btn-ff').addEventListener('click', () => runGenerations(10));
  $('btn-reset').addEventListener('click', reset);

  reset();
  tick();

  // Test hook for the headless browser check.
  window.WALK_APP = {
    gen: () => pop.gen,
    best: () => (pop.history.length ? pop.history[pop.history.length - 1].best : null),
    champDistance: () => (champSim ? champSim.distance() : null),
    runGenerations,
  };
})();
