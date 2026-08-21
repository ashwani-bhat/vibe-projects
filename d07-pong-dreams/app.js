/**
 * UI: learn mode (real Pong + live one-step prediction, model training on the
 * stream) and dream mode (the net hallucinates every frame closed-loop while
 * you steer the left paddle; a parallel real sim shows how far the dream has
 * drifted from actual physics).
 */

(() => {
  'use strict';

  const $ = id => document.getElementById(id);

  const RES = 32;
  const BUFFER_CAP = 3000;
  const BATCH = 32;
  const TRAIN_PER_FRAME = 2;
  const MIN_BUFFER = 300;

  const sim = new PONG.Sim((Math.random() * 1e9) | 0);
  const rand = PONG.rng((Math.random() * 1e9) | 0);
  const model = new WORLD.Model({ res: RES });

  let mode = 'learn';           // 'learn' | 'dream'
  let running = false;
  let frames = [PONG.render(sim, RES), PONG.render(sim, RES)]; // [f_{t-1}, f_t]
  let buffer = [];              // { x: Float32Array, y: Float32Array }
  let bufPos = 0;
  let lossEma = null;
  let lossHistory = [];
  let steps = 0;

  let dream = null;             // { d0, d1, realSim, step }
  let action = 0;               // current player action −1/0/+1
  const keys = new Set();

  /* ---------- canvases ---------- */

  const canvases = {};
  for (const id of ['real', 'pred', 'dream', 'reality']) {
    const c = $(`${id}-canvas`);
    canvases[id] = { c, ctx: c.getContext('2d') };
  }
  const off = document.createElement('canvas');
  off.width = RES; off.height = RES;
  const offCtx = off.getContext('2d');
  const offImg = offCtx.createImageData(RES, RES);

  function drawFrame(name, f) {
    const { c, ctx } = canvases[name];
    for (let i = 0; i < RES * RES; i++) {
      const v = Math.max(0, Math.min(1, f[i])) * 255;
      offImg.data[i * 4] = v * 0.85;
      offImg.data[i * 4 + 1] = v;
      offImg.data[i * 4 + 2] = v;
      offImg.data[i * 4 + 3] = 255;
    }
    offCtx.putImageData(offImg, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, c.width, c.height);
  }

  function drawLoss() {
    const c = $('loss-canvas'), ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.strokeStyle = '#2c3047';
    ctx.strokeRect(0.5, 0.5, c.width - 1, c.height - 1);
    if (lossHistory.length < 2) return;
    const max = Math.max(...lossHistory);
    ctx.beginPath();
    for (let i = 0; i < lossHistory.length; i++) {
      const x = (i / (lossHistory.length - 1)) * (c.width - 8) + 4;
      const y = c.height - 4 - (lossHistory[i] / max) * (c.height - 8);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.strokeStyle = '#73eff7';
    ctx.stroke();
    ctx.fillStyle = '#9aa0b8';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText('prediction loss', 8, 13);
  }

  function setStatus(t) { $('status').textContent = t; }

  /* ---------- learn mode ---------- */

  function learnTick() {
    // advance the real game, collecting transitions
    for (let k = 0; k < 3; k++) {
      const a = PONG.trackerPolicy(sim, rand, 0.35);
      const x = WORLD.buildInput(frames[0], frames[1], a);
      sim.step(a);
      const next = PONG.render(sim, RES);
      const t = { x, y: next };
      if (buffer.length < BUFFER_CAP) buffer.push(t);
      else { buffer[bufPos] = t; bufPos = (bufPos + 1) % BUFFER_CAP; }
      frames = [frames[1], next];
    }

    if (buffer.length >= MIN_BUFFER) {
      for (let k = 0; k < TRAIN_PER_FRAME; k++) {
        const xs = new Float32Array(BATCH * model.inDim);
        const ys = new Float32Array(BATCH * model.outDim);
        for (let b = 0; b < BATCH; b++) {
          const t = buffer[(Math.random() * buffer.length) | 0];
          xs.set(t.x, b * model.inDim);
          ys.set(t.y, b * model.outDim);
        }
        const loss = model.trainStep(xs, ys, BATCH);
        lossEma = lossEma === null ? loss : lossEma * 0.97 + loss * 0.03;
        steps++;
        if (steps % 10 === 0) {
          lossHistory.push(lossEma);
          if (lossHistory.length > 500) lossHistory.shift();
        }
      }
      setStatus(`step ${steps} · loss ${lossEma.toExponential(2)} · ` +
        (lossEma < 0.025 ? 'the dream is ready ✨' : 'learning how Pong works…'));
    } else {
      setStatus(`watching the game… ${buffer.length}/${MIN_BUFFER} transitions`);
    }

    drawFrame('real', frames[1]);
    if (steps > 0 && (steps % 8 === 0 || !running)) {
      const a = PONG.trackerPolicy(sim, rand, 0);
      drawFrame('pred', model.predict(frames[0], frames[1], a));
    }
    drawLoss();
  }

  /* ---------- dream mode ---------- */

  function enterDream() {
    const realSim = new PONG.Sim(1);
    realSim.restore(sim.snapshot());
    dream = {
      d0: Float32Array.from(frames[0]),
      d1: Float32Array.from(frames[1]),
      realSim,
      step: 0,
    };
    mode = 'dream';
    $('learn-view').hidden = true;
    $('dream-view').hidden = false;
    $('btn-mode').textContent = '← back to training';
  }

  function exitDream() {
    mode = 'learn';
    dream = null;
    $('learn-view').hidden = false;
    $('dream-view').hidden = true;
    $('btn-mode').textContent = '💤 enter the dream';
  }

  function dreamTick() {
    const probs = model.predict(dream.d0, dream.d1, action);
    const crisp = $('chk-crisp').checked;
    let fb = probs;
    if (crisp) {
      fb = new Float32Array(probs.length);
      for (let i = 0; i < probs.length; i++) fb[i] = probs[i] > 0.5 ? 1 : 0;
    }
    dream.d0 = dream.d1;
    dream.d1 = fb;
    dream.step++;

    dream.realSim.step(action);

    drawFrame('dream', probs);
    if ($('chk-reality').checked) {
      $('reality-wrap').hidden = false;
      drawFrame('reality', PONG.render(dream.realSim, RES));
    } else {
      $('reality-wrap').hidden = true;
    }
    setStatus(`dream step ${dream.step} — every pixel is the network's imagination`);
  }

  /* ---------- main loop ---------- */

  let frame = 0;
  function tick() {
    action = keys.has('up') ? -1 : keys.has('down') ? 1 : 0;
    if (running) {
      if (mode === 'learn') learnTick();
      else if (frame % 2 === 0) dreamTick(); // ~30fps: dreams are playable, not frantic
    }
    frame++;
    requestAnimationFrame(tick);
  }

  /* ---------- controls ---------- */

  $('btn-run').addEventListener('click', () => {
    running = !running;
    $('btn-run').textContent = running ? '⏸ pause' : '▶ start';
    $('btn-run').classList.toggle('active', running);
  });

  $('btn-mode').addEventListener('click', () => {
    if (mode === 'learn') {
      if (steps === 0) { setStatus('let it train first — the dream needs a brain'); return; }
      enterDream();
    } else exitDream();
  });

  $('btn-reseed').addEventListener('click', () => {
    if (mode === 'dream') {
      dream.realSim.restore(sim.snapshot());
      dream.d0 = Float32Array.from(frames[0]);
      dream.d1 = Float32Array.from(frames[1]);
      dream.step = 0;
    }
  });

  const keyMap = { ArrowUp: 'up', ArrowDown: 'down', w: 'up', s: 'down', W: 'up', S: 'down' };
  window.addEventListener('keydown', e => {
    if (keyMap[e.key]) { keys.add(keyMap[e.key]); e.preventDefault(); }
  });
  window.addEventListener('keyup', e => {
    if (keyMap[e.key]) keys.delete(keyMap[e.key]);
  });
  for (const [id, dir] of [['btn-up', 'up'], ['btn-down', 'down']]) {
    const el = $(id);
    el.addEventListener('pointerdown', e => { e.preventDefault(); keys.add(dir); });
    el.addEventListener('pointerup', () => keys.delete(dir));
    el.addEventListener('pointerleave', () => keys.delete(dir));
  }

  tf.ready().then(() => {
    $('backend').textContent =
      `backend: ${tf.getBackend()} · world model: ${model.paramCount().toLocaleString()} parameters`;
  });

  drawFrame('real', frames[1]);
  drawLoss();
  tick();

  // Test hook for the headless browser check.
  window.PONG_APP = {
    steps: () => steps,
    loss: () => lossEma,
    bufferSize: () => buffer.length,
    mode: () => mode,
    dreamStep: () => (dream ? dream.step : -1),
    dreamFrame: () => (dream ? dream.d1 : null),
    enterDream, exitDream,
  };
})();
