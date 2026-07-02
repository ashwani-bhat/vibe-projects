/**
 * UI glue: training loop, decision-boundary canvas, loss curve, tuner
 * needles, and note scheduling against the Web Audio clock.
 */

(() => {
  'use strict';

  const $ = id => document.getElementById(id);

  const SIZES = [2, 24, 24, 1];
  const STEPS_PER_FRAME = 3;
  const DOMAIN = 1.2;          // data lives in [-DOMAIN, DOMAIN]^2
  const GRID = 96;             // boundary raster resolution
  const C0 = [115, 239, 247];  // class 0 — accent cyan
  const C1 = [255, 205, 117];  // class 1 — accent amber
  const DARK = [29, 32, 48];   // panel background

  let net, trainer, data;
  let orchestra = null;        // created on first ▶ (browsers require a user gesture for audio)
  let running = false;
  let lossHistory = [];
  let lossEma = 0.7, accEma = 0.5, evalAcc = 0;
  let gradEma = [0, 0, 0];
  let gradRef = 0;             // slowly-decaying running max of gradient RMS
  let milestones = {};
  let nextNote = 0;
  let flashes = [];            // recently-sounded samples, drawn as rings
  let frame = 0;

  const bCanvas = $('boundary-canvas'), bCtx = bCanvas.getContext('2d');
  const lCanvas = $('loss-canvas'), lCtx = lCanvas.getContext('2d');
  const tCanvas = $('tuner-canvas'), tCtx = tCanvas.getContext('2d');
  const off = document.createElement('canvas');
  off.width = GRID; off.height = GRID;
  const offCtx = off.getContext('2d');
  const img = offCtx.createImageData(GRID, GRID);

  function reset() {
    data = MLP.datasets[$('sel-dataset').value](240, MLP.rng((Math.random() * 1e9) | 0));
    net = new MLP.Net(SIZES, (Math.random() * 1e9) | 0);
    trainer = new MLP.Trainer(net, data, {
      lr: parseFloat($('sel-lr').value),
      seed: (Math.random() * 1e9) | 0,
    });
    lossHistory = [];
    lossEma = 0.7; accEma = 0.5; evalAcc = 0;
    gradEma = [0, 0, 0]; gradRef = 0;
    milestones = { 75: false, 90: false, 97: false };
    flashes = [];
    setStatus('fresh model — silence before the music');
  }

  function setStatus(text) { $('status').textContent = text; }

  /* ---------- training ---------- */

  function trainSome() {
    for (let k = 0; k < STEPS_PER_FRAME; k++) {
      const r = trainer.step();
      lossEma = lossEma * 0.95 + r.loss * 0.05;
      accEma = accEma * 0.95 + r.acc * 0.05;
      for (let l = 0; l < 3; l++) {
        gradEma[l] = gradEma[l] * 0.9 + r.gradRms[l] * 0.1;
        // Reference decays very slowly: fast decay tracks the shrinking
        // gradients and the chord never audibly tunes up.
        gradRef = Math.max(gradRef * 0.99998, gradEma[l]);
      }
      lossHistory.push(r.loss);
      if (lossHistory.length > 4000) lossHistory.shift();
    }
    if (trainer.iter % 30 < STEPS_PER_FRAME) evalAcc = trainer.evaluate().acc;
    setStatus(`step ${trainer.iter} · loss ${lossEma.toFixed(3)} · accuracy ${(evalAcc * 100).toFixed(0)}%`);
  }

  /* ---------- audio ---------- */

  function updateDrone() {
    const h = SONIFY.harshness(lossEma);
    orchestra.setHarshness(h);
    orchestra.setVoice(0, 0, h); // root voice: the reference pitch, always in tune
    for (let l = 0; l < 3; l++) {
      orchestra.setVoice(l + 1, SONIFY.detuneCents(gradEma[l], gradRef), h);
    }
  }

  function scheduleNotes() {
    const t0 = orchestra.now();
    if (nextNote < t0) nextNote = t0 + 0.02;
    const rate = SONIFY.tempo(lossEma);
    while (nextNote < t0 + 0.15) {
      const k = (Math.random() * data.xs.length) | 0;
      const x = data.xs[k], y = data.ys[k];
      const acts = net.forward(x);
      const p = acts[acts.length - 1][0];
      const correct = (p > 0.5 ? 1 : 0) === y;
      const h = acts[acts.length - 2];
      let mean = 0;
      for (const v of h) mean += v;
      mean /= h.length;
      orchestra.note(SONIFY.noteFreq(mean, correct), SONIFY.noteVel(p - y), nextNote);
      flashes.push({ x, correct, at: performance.now() + (nextNote - t0) * 1000 });
      nextNote += 1 / rate;
    }
  }

  function maybeChime() {
    const pct = evalAcc * 100;
    for (const m of [75, 90, 97]) {
      if (!milestones[m] && pct >= m) {
        milestones[m] = true;
        const chord = SONIFY.chordFreqs(220);
        orchestra.chime(m === 97 ? [...chord, 440 * (5 / 4), 440 * (3 / 2)] : chord.slice(0, m === 90 ? 4 : 3));
        if (m === 97) setStatus(`step ${trainer.iter} · the orchestra is in tune 🎉 (${pct.toFixed(0)}%)`);
      }
    }
  }

  /* ---------- drawing ---------- */

  function px(v) { return ((v + DOMAIN) / (2 * DOMAIN)) * bCanvas.width; }
  function py(v) { return ((DOMAIN - v) / (2 * DOMAIN)) * bCanvas.height; }

  function drawBoundary() {
    for (let gy = 0; gy < GRID; gy++) {
      const y = DOMAIN - (gy / (GRID - 1)) * 2 * DOMAIN;
      for (let gx = 0; gx < GRID; gx++) {
        const x = (gx / (GRID - 1)) * 2 * DOMAIN - DOMAIN;
        const p = net.predict([x, y]);
        const conf = Math.abs(p - 0.5) * 2 * 0.8; // never fully saturate
        const c = p > 0.5 ? C1 : C0;
        const i = (gy * GRID + gx) * 4;
        img.data[i] = DARK[0] + (c[0] - DARK[0]) * conf;
        img.data[i + 1] = DARK[1] + (c[1] - DARK[1]) * conf;
        img.data[i + 2] = DARK[2] + (c[2] - DARK[2]) * conf;
        img.data[i + 3] = 255;
      }
    }
    offCtx.putImageData(img, 0, 0);
    bCtx.imageSmoothingEnabled = false;
    bCtx.drawImage(off, 0, 0, bCanvas.width, bCanvas.height);

    for (let k = 0; k < data.xs.length; k++) {
      const [x, y] = data.xs[k];
      bCtx.beginPath();
      bCtx.arc(px(x), py(y), 3.2, 0, 2 * Math.PI);
      bCtx.fillStyle = data.ys[k] === 1 ? '#ffcd75' : '#73eff7';
      bCtx.fill();
      bCtx.strokeStyle = '#0f1119';
      bCtx.lineWidth = 1;
      bCtx.stroke();
    }

    // Rings on the samples the melody just played.
    const now = performance.now();
    flashes = flashes.filter(f => now - f.at < 350 && f.at - now < 400);
    for (const f of flashes) {
      const age = now - f.at;
      if (age < 0) continue;
      const r = 5 + age / 22;
      bCtx.beginPath();
      bCtx.arc(px(f.x[0]), py(f.x[1]), r, 0, 2 * Math.PI);
      bCtx.strokeStyle = f.correct
        ? `rgba(230, 232, 242, ${1 - age / 350})`
        : `rgba(255, 90, 90, ${1 - age / 350})`;
      bCtx.lineWidth = 2;
      bCtx.stroke();
    }
  }

  function drawLoss() {
    const w = lCanvas.width, h = lCanvas.height;
    lCtx.clearRect(0, 0, w, h);
    lCtx.strokeStyle = '#2c3047';
    lCtx.strokeRect(0.5, 0.5, w - 1, h - 1);
    if (lossHistory.length < 2) return;
    const max = Math.max(0.75, ...lossHistory);
    lCtx.beginPath();
    for (let i = 0; i < lossHistory.length; i++) {
      const x = (i / (lossHistory.length - 1)) * (w - 8) + 4;
      const y = h - 4 - (lossHistory[i] / max) * (h - 8);
      i ? lCtx.lineTo(x, y) : lCtx.moveTo(x, y);
    }
    lCtx.strokeStyle = '#73eff7';
    lCtx.lineWidth = 1;
    lCtx.stroke();
    lCtx.fillStyle = '#9aa0b8';
    lCtx.font = '10px ui-monospace, monospace';
    lCtx.fillText('loss', 8, 13);
  }

  function drawTuner() {
    const w = tCanvas.width, h = tCanvas.height;
    const rows = ['layer 1', 'layer 2', 'layer 3'];
    tCtx.clearRect(0, 0, w, h);
    tCtx.strokeStyle = '#2c3047';
    tCtx.strokeRect(0.5, 0.5, w - 1, h - 1);
    tCtx.font = '10px ui-monospace, monospace';
    const cx = w * 0.62;
    for (let l = 0; l < rows.length; l++) {
      const y = 20 + l * 28;
      tCtx.fillStyle = '#9aa0b8';
      tCtx.fillText(rows[l], 10, y + 4);
      tCtx.strokeStyle = '#2c3047';
      tCtx.beginPath();
      tCtx.moveTo(cx - 80, y); tCtx.lineTo(cx + 80, y);
      tCtx.stroke();
      tCtx.strokeStyle = '#566c86';
      tCtx.beginPath();
      tCtx.moveTo(cx, y - 6); tCtx.lineTo(cx, y + 6);
      tCtx.stroke();
      const cents = SONIFY.detuneCents(gradEma[l], gradRef);
      const dx = (cents / 85) * 76;
      tCtx.fillStyle = cents < 8 ? '#73eff7' : '#ffcd75';
      tCtx.beginPath();
      tCtx.arc(cx + dx, y, 4, 0, 2 * Math.PI);
      tCtx.fill();
    }
    tCtx.fillStyle = '#566c86';
    tCtx.fillText('in tune ↑', cx - 24, h - 6);
  }

  /* ---------- main loop ---------- */

  function tick() {
    if (running) {
      trainSome();
      updateDrone();
      scheduleNotes();
      maybeChime();
    }
    if (frame % 4 === 0 || !running) {
      drawBoundary();
      drawLoss();
      drawTuner();
    }
    frame++;
    requestAnimationFrame(tick);
  }

  /* ---------- controls ---------- */

  $('btn-play').addEventListener('click', () => {
    if (!orchestra) {
      orchestra = new ORCHESTRA.Orchestra();
      orchestra.setVolume(parseFloat($('vol').value));
      orchestra.startDrone(SONIFY.chordFreqs(110));
    }
    running = !running;
    if (running) orchestra.resume(); else orchestra.suspend();
    $('btn-play').textContent = running ? '⏸ pause' : '▶ listen & train';
    $('btn-play').classList.toggle('active', running);
  });

  $('btn-reset').addEventListener('click', reset);
  $('sel-dataset').addEventListener('change', reset);
  $('sel-lr').addEventListener('change', () => { trainer.lr = parseFloat($('sel-lr').value); });
  $('vol').addEventListener('input', () => { if (orchestra) orchestra.setVolume(parseFloat($('vol').value)); });

  reset();
  tick();
})();
