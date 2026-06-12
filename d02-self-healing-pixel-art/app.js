/* Self-healing pixel art — UI: draw → train → play. */
(() => {
  'use strict';

  // ---------- constants ----------
  const GRID = 32;          // CA grid (model size)
  const SPRITE = 24;        // drawable sprite area, centered in the grid
  const PAD = (GRID - SPRITE) >> 1;
  const CHANNELS = 16;
  const STORAGE_KEY = 'd02-self-healing-art';

  const PALETTE = [
    '#1a1c2c', '#5d275d', '#b13e53', '#ef7d57', '#ffcd75', '#a7f070',
    '#38b764', '#257179', '#29366f', '#3b5dc9', '#41a6f6', '#73eff7',
    '#f4f4f4', '#94b0c2', '#566c86', '#333c57',
  ]; // Sweetie-16 palette

  const PRESETS = {
    heart: {
      map: { r: '#b13e53', w: '#f4a4b0' },
      rows: [
        '..rrr....rrr..',
        '.rrrrr..rrrrr.',
        'rrwwrrrrrrrrrr',
        'rwwrrrrrrrrrrr',
        'rrrrrrrrrrrrrr',
        '.rrrrrrrrrrrr.',
        '..rrrrrrrrrr..',
        '...rrrrrrrr...',
        '....rrrrrr....',
        '.....rrrr.....',
        '......rr......',
      ],
    },
    mushroom: {
      map: { r: '#b13e53', w: '#f4f4f4', f: '#ffcd75', e: '#29366f' },
      rows: [
        '....rrrrrrrr....',
        '..rrrwwrrrrrrr..',
        '.rrrwwwwrrwwrrr.',
        '.rrrwwwwrrwwwwr.',
        'rrrrrwwrrrwwwwrr',
        'rwwrrrrrrrrwwrrr',
        'rwwwrrrrrrrrrrrr',
        'rwwrrrrrrrrrrrrr',
        '.rrrrrrrrrrrrrr.',
        '..ffffffffffff..',
        '..fffeffffefff..',
        '..fffeffffefff..',
        '...ffffffffff...',
        '....ffffffff....',
      ],
    },
    ghost: {
      map: { c: '#73eff7', w: '#f4f4f4', p: '#29366f' },
      rows: [
        '.....cccccc.....',
        '...cccccccccc...',
        '..cccccccccccc..',
        '.ccwwccccwwcccc.',
        '.cwwwwccwwwwccc.',
        '.cwwppccwwppccc.',
        'ccwwppccwwppcccc',
        'cccwwccccwwccccc',
        'cccccccccccccccc',
        'cccccccccccccccc',
        'cccccccccccccccc',
        'cccccccccccccccc',
        'cc.ccc.cc.ccc.cc',
        'c...cc..c.cc...c',
      ],
    },
  };

  // ---------- state ----------
  const art = new Uint8ClampedArray(SPRITE * SPRITE * 4); // editor RGBA
  let currentColor = PALETTE[2];
  let tool = 'paint';

  let model = null;
  let trainer = null;
  let trainedArtHash = null;
  let training = false;
  const losses = [];

  let playState = null;      // tf tensor [1, GRID, GRID, CHANNELS]
  let playRunning = false;
  let playSpeed = 1;
  let pendingDamage = null;  // Float32Array [GRID*GRID] multiplier, or null

  // ---------- dom ----------
  const $ = id => document.getElementById(id);
  const drawCanvas = $('draw-canvas');
  const trainCanvas = $('train-canvas');
  const lossCanvas = $('loss-canvas');
  const playCanvas = $('play-canvas');

  // ---------- editor ----------
  function setArtPixel(x, y, color) {
    const i = (y * SPRITE + x) * 4;
    if (color === null) {
      art[i] = art[i + 1] = art[i + 2] = art[i + 3] = 0;
    } else {
      const v = parseInt(color.slice(1), 16);
      art[i] = (v >> 16) & 255; art[i + 1] = (v >> 8) & 255; art[i + 2] = v & 255;
      art[i + 3] = 255;
    }
  }

  function renderEditor() {
    const ctx = drawCanvas.getContext('2d');
    const s = drawCanvas.width / SPRITE;
    for (let y = 0; y < SPRITE; y++) {
      for (let x = 0; x < SPRITE; x++) {
        const i = (y * SPRITE + x) * 4;
        if (art[i + 3] > 0) {
          ctx.fillStyle = `rgb(${art[i]},${art[i + 1]},${art[i + 2]})`;
        } else {
          ctx.fillStyle = (x + y) % 2 ? '#23263a' : '#1d2030';
        }
        ctx.fillRect(x * s, y * s, s, s);
      }
    }
  }

  function artHash() {
    let h = 0;
    for (let i = 0; i < art.length; i++) h = (h * 31 + art[i]) | 0;
    return h;
  }

  function artIsEmpty() {
    for (let i = 3; i < art.length; i += 4) if (art[i] > 0) return false;
    return true;
  }

  function loadPreset(name) {
    const p = PRESETS[name];
    art.fill(0);
    const h = p.rows.length, w = p.rows[0].length;
    const ox = (SPRITE - w) >> 1, oy = (SPRITE - h) >> 1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const c = p.map[p.rows[y][x]];
        if (c) setArtPixel(ox + x, oy + y, c);
      }
    }
    renderEditor();
    saveArt();
  }

  function saveArt() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(art))); } catch (e) { /* ignore */ }
  }

  function restoreArt() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) { art.set(JSON.parse(saved)); return true; }
    } catch (e) { /* ignore */ }
    return false;
  }

  function attachPaint(canvas, cells, onCell) {
    let down = false;
    const cellAt = e => {
      const r = canvas.getBoundingClientRect();
      return [
        Math.floor((e.clientX - r.left) / r.width * cells),
        Math.floor((e.clientY - r.top) / r.height * cells),
      ];
    };
    canvas.addEventListener('pointerdown', e => {
      down = true; canvas.setPointerCapture(e.pointerId);
      onCell(...cellAt(e)); e.preventDefault();
    });
    canvas.addEventListener('pointermove', e => { if (down) onCell(...cellAt(e)); });
    canvas.addEventListener('pointerup', () => { down = false; });
  }

  // ---------- target / rendering ----------
  /** Editor art -> premultiplied RGBA float target on the full grid. */
  function buildTarget() {
    const t = new Float32Array(GRID * GRID * 4);
    for (let y = 0; y < SPRITE; y++) {
      for (let x = 0; x < SPRITE; x++) {
        const src = (y * SPRITE + x) * 4;
        const dst = ((y + PAD) * GRID + (x + PAD)) * 4;
        const a = art[src + 3] / 255;
        t[dst] = (art[src] / 255) * a;
        t[dst + 1] = (art[src + 1] / 255) * a;
        t[dst + 2] = (art[src + 2] / 255) * a;
        t[dst + 3] = a;
      }
    }
    return t;
  }

  /** Render a CA state (Float32Array, [GRID*GRID*CHANNELS]) onto a canvas. */
  const stateImage = new ImageData(GRID, GRID);
  function renderState(state, canvas) {
    const d = stateImage.data;
    for (let i = 0, n = GRID * GRID; i < n; i++) {
      const a = Math.min(Math.max(state[i * CHANNELS + 3], 0), 1);
      for (let c = 0; c < 3; c++) {
        // premultiplied color composited on white
        const v = 1 - a + state[i * CHANNELS + c];
        d[i * 4 + c] = Math.min(Math.max(v, 0), 1) * 255;
      }
      d[i * 4 + 3] = 255;
    }
    const ctx = canvas.getContext('2d');
    const off = renderState.off || (renderState.off = new OffscreenCanvas(GRID, GRID));
    off.getContext('2d').putImageData(stateImage, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }

  function renderLossChart() {
    const ctx = lossCanvas.getContext('2d');
    const W = lossCanvas.width, H = lossCanvas.height;
    ctx.fillStyle = '#14172499';
    ctx.fillRect(0, 0, W, H);
    if (losses.length < 2) return;
    const logs = losses.map(l => Math.log10(Math.max(l, 1e-8)));
    const lo = Math.min(...logs), hi = Math.max(...logs);
    ctx.strokeStyle = '#73eff7';
    ctx.beginPath();
    logs.forEach((v, i) => {
      const x = i / (logs.length - 1) * W;
      const y = H - ((v - lo) / (hi - lo + 1e-9)) * (H - 8) - 4;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }

  // ---------- training ----------
  function ensureTrainer() {
    const h = artHash();
    if (trainer && trainedArtHash === h) return;
    stopPlay();
    if (trainer) trainer.dispose();
    if (!model) model = new NCA.Model({ size: GRID, channels: CHANNELS });
    else model.initWeights();
    trainer = new NCA.Trainer(model, buildTarget());
    trainedArtHash = h;
    losses.length = 0;
    $('train-status').textContent = 'fresh model — not trained yet';
  }

  async function trainLoop() {
    while (training) {
      const { loss, sample } = trainer.step();
      losses.push(loss);
      if (trainer.iter % 2 === 0) renderState(sample, trainCanvas);
      if (trainer.iter % 5 === 0) renderLossChart();
      const phase = trainer.iter < trainer.damageStartIter
        ? 'learning to grow' : 'learning to heal';
      $('train-status').textContent =
        `step ${trainer.iter} · loss ${loss.toExponential(2)} · ${phase}`;
      await tf.nextFrame();
    }
  }

  function setTraining(on) {
    if (on && artIsEmpty()) {
      $('train-status').textContent = 'draw something first!';
      switchTab('draw');
      return;
    }
    if (on) ensureTrainer();
    training = on;
    $('btn-train').textContent = on ? '⏸ pause training' : '▶ train';
    $('btn-train').classList.toggle('active', on);
    if (on) trainLoop();
  }

  // ---------- play ----------
  function resetPlayState() {
    if (playState) playState.dispose();
    playState = model.seedTensor(1);
    pendingDamage = null;
  }

  function startPlay() {
    if (!model) return;
    if (!playState) resetPlayState();
    if (playRunning) return;
    playRunning = true;
    const tick = () => {
      if (!playRunning) return;
      const old = playState;
      playState = tf.tidy(() => {
        let x = old;
        if (pendingDamage) {
          x = x.mul(tf.tensor4d(pendingDamage, [1, GRID, GRID, 1]));
          pendingDamage = null;
        }
        for (let i = 0; i < playSpeed; i++) x = model.step(x);
        return x;
      });
      old.dispose();
      renderState(playState.dataSync(), playCanvas);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function stopPlay() {
    playRunning = false;
    if (playState) { playState.dispose(); playState = null; }
  }

  function eraseAt(cx, cy) {
    const r = +$('brush-size').value;
    if (!pendingDamage) pendingDamage = new Float32Array(GRID * GRID).fill(1);
    for (let y = 0; y < GRID; y++) {
      for (let x = 0; x < GRID; x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) pendingDamage[y * GRID + x] = 0;
      }
    }
  }

  // ---------- weights export/import ----------
  function exportWeights() {
    if (!model || !trainer) return;
    const blob = new Blob(
      [JSON.stringify({ art: Array.from(art), iter: trainer.iter, ...model.exportWeights() })],
      { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'self-healing-pixel-art-weights.json';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function importWeights(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const obj = JSON.parse(reader.result);
        art.set(obj.art);
        renderEditor(); saveArt();
        stopPlay();
        if (trainer) trainer.dispose();
        if (!model) model = new NCA.Model({ size: GRID, channels: CHANNELS });
        else model.initWeights();
        model.importWeights(obj);
        trainer = new NCA.Trainer(model, buildTarget());
        trainer.iter = obj.iter || 0;
        trainedArtHash = artHash();
        losses.length = 0;
        $('train-status').textContent = `imported model (trained ${trainer.iter} steps)`;
        switchTab('play');
      } catch (e) {
        $('train-status').textContent = 'could not read that file';
      }
    };
    reader.readAsText(file);
  }

  // ---------- tabs ----------
  function switchTab(name) {
    for (const t of ['draw', 'train', 'play']) {
      $(`tab-${t}`).classList.toggle('active', t === name);
      $(`panel-${t}`).hidden = t !== name;
    }
    if (name !== 'train' && training) setTraining(false);
    if (name === 'play') {
      if (!model || !trainer || trainer.iter === 0) {
        $('play-hint').textContent = 'the model is untrained — go train it first, then come back';
      } else {
        $('play-hint').textContent = 'drag on the canvas to erase — watch it grow back';
      }
      if (model) { resetPlayState(); startPlay(); }
    } else {
      stopPlay();
    }
  }

  // ---------- wiring ----------
  function init() {
    // palette
    const pal = $('palette');
    PALETTE.forEach(c => {
      const b = document.createElement('button');
      b.className = 'swatch'; b.style.background = c; b.title = c;
      b.onclick = () => {
        currentColor = c; tool = 'paint';
        document.querySelectorAll('.swatch, #btn-eraser').forEach(el => el.classList.remove('selected'));
        b.classList.add('selected');
      };
      pal.appendChild(b);
    });
    pal.children[2].classList.add('selected');

    $('btn-eraser').onclick = () => {
      tool = 'erase';
      document.querySelectorAll('.swatch').forEach(el => el.classList.remove('selected'));
      $('btn-eraser').classList.add('selected');
    };
    $('btn-clear').onclick = () => { art.fill(0); renderEditor(); saveArt(); };
    document.querySelectorAll('[data-preset]').forEach(b => {
      b.onclick = () => loadPreset(b.dataset.preset);
    });

    attachPaint(drawCanvas, SPRITE, (x, y) => {
      if (x < 0 || y < 0 || x >= SPRITE || y >= SPRITE) return;
      setArtPixel(x, y, tool === 'erase' ? null : currentColor);
      renderEditor(); saveArt();
    });

    attachPaint(playCanvas, GRID, (x, y) => eraseAt(x, y));

    $('tab-draw').onclick = () => switchTab('draw');
    $('tab-train').onclick = () => switchTab('train');
    $('tab-play').onclick = () => switchTab('play');

    $('btn-train').onclick = () => setTraining(!training);
    $('btn-reset-model').onclick = () => {
      setTraining(false);
      trainedArtHash = null;
      ensureTrainer();
      trainCanvas.getContext('2d').clearRect(0, 0, trainCanvas.width, trainCanvas.height);
      renderLossChart();
    };
    $('btn-export').onclick = exportWeights;
    $('file-import').onchange = e => { if (e.target.files[0]) importWeights(e.target.files[0]); };

    $('btn-reseed').onclick = () => { if (model) resetPlayState(); };
    $('btn-meteor').onclick = () => {
      // wipe a random half of the grid
      pendingDamage = new Float32Array(GRID * GRID).fill(1);
      const vertical = Math.random() < 0.5, side = Math.random() < 0.5;
      for (let y = 0; y < GRID; y++) {
        for (let x = 0; x < GRID; x++) {
          const half = vertical ? (x < GRID / 2) : (y < GRID / 2);
          if (half === side) pendingDamage[y * GRID + x] = 0;
        }
      }
    };
    $('play-speed').onchange = e => { playSpeed = +e.target.value; };

    if (!restoreArt()) loadPreset('heart');
    renderEditor();
    switchTab('draw');

    tf.setBackend('webgl').catch(() => {}).then(() => {
      $('backend').textContent = `tf.js backend: ${tf.getBackend()}`;
    });
  }

  window.addEventListener('DOMContentLoaded', init);
})();
