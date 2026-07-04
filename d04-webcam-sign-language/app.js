/**
 * UI + camera glue. The recognition pipeline (handleLandmarks → record /
 * predict / spell) is camera-independent and exposed on window.SIGN_APP so
 * the browser test can drive it with synthetic landmarks — MediaPipe and
 * getUserMedia only enter in enableCamera().
 */

(() => {
  'use strict';

  const $ = id => document.getElementById(id);

  const MIN_SAMPLES = 10;      // per sign, before training is allowed
  const MAX_SAMPLES = 150;     // per sign, so one long hold can't drown the rest
  const STABLE_FRAMES = 22;    // ~1s of agreement before a letter is spelled
  const SPELL_CONF = 0.85;
  const COOLDOWN_FRAMES = 40;

  const state = {
    classes: [],        // { name, samples: Float32Array[] }
    recordingIdx: -1,
    net: null,
    netLabels: [],      // class names frozen at train time
    probsEma: null,
    stableLabel: null,
    stableCount: 0,
    cooldown: 0,
    noHandFrames: 0,
  };

  /* ---------- pipeline (no camera needed) ---------- */

  function addClass(name) {
    name = (name || '').trim();
    if (!name) return false;
    if (state.classes.some(c => c.name === name)) return false;
    state.classes.push({ name, samples: [] });
    renderClasses();
    return true;
  }

  function removeClass(idx) {
    state.classes.splice(idx, 1);
    if (state.recordingIdx === idx) state.recordingIdx = -1;
    state.net = null; state.netLabels = []; state.probsEma = null;
    renderClasses(); renderBars();
    setTrainStatus('signs changed — retrain when ready');
  }

  function setRecording(idx) {
    state.recordingIdx = idx;
    renderClasses();
  }

  function train() {
    const usable = state.classes.filter(c => c.samples.length >= MIN_SAMPLES);
    if (usable.length < 2) {
      setTrainStatus(`need ≥2 signs with ≥${MIN_SAMPLES} samples each`);
      return null;
    }
    const xs = [], ys = [];
    usable.forEach((c, k) => c.samples.forEach(s => { xs.push(s); ys.push(k); }));
    const t0 = performance.now();
    const { net, valAcc } = CLASSIFIER.train(xs, ys, usable.length, {
      seed: (Math.random() * 1e9) | 0,
    });
    state.net = net;
    state.netLabels = usable.map(c => c.name);
    state.probsEma = null;
    state.stableLabel = null; state.stableCount = 0; state.cooldown = 0;
    const ms = (performance.now() - t0).toFixed(0);
    setTrainStatus(`trained on ${xs.length} samples in ${ms}ms — holdout accuracy ${(valAcc * 100).toFixed(0)}%`);
    renderBars();
    return valAcc;
  }

  /** One frame's landmarks (or null if no hand). The heart of the app. */
  function handleLandmarks(lm, handed) {
    if (!lm) {
      if (++state.noHandFrames > 15) {
        state.probsEma = null;
        state.stableLabel = null; state.stableCount = 0;
        renderBars();
      }
      return null;
    }
    state.noHandFrames = 0;

    const feat = LANDMARKS.normalize(lm, handed);
    if (!feat) return null;

    if (state.recordingIdx >= 0) {
      const cls = state.classes[state.recordingIdx];
      if (cls && cls.samples.length < MAX_SAMPLES) {
        cls.samples.push(feat);
        renderClasses();
      }
    }

    if (state.net) {
      const probs = state.net.predict(feat);
      if (!state.probsEma) state.probsEma = Float32Array.from(probs);
      else for (let k = 0; k < probs.length; k++) {
        state.probsEma[k] = state.probsEma[k] * 0.8 + probs[k] * 0.2;
      }
      spellTick();
      renderBars();
    }
    return feat;
  }

  function topPrediction() {
    if (!state.probsEma) return null;
    let best = 0;
    for (let k = 1; k < state.probsEma.length; k++) {
      if (state.probsEma[k] > state.probsEma[best]) best = k;
    }
    return { label: state.netLabels[best], prob: state.probsEma[best] };
  }

  function spellTick() {
    if (state.cooldown > 0) { state.cooldown--; return; }
    const top = topPrediction();
    if (top && top.prob >= SPELL_CONF) {
      if (top.label === state.stableLabel) {
        if (++state.stableCount >= STABLE_FRAMES) {
          $('transcript').textContent += top.label.length === 1 ? top.label : ` ${top.label} `;
          state.stableCount = 0;
          state.cooldown = COOLDOWN_FRAMES;
        }
      } else {
        state.stableLabel = top.label;
        state.stableCount = 1;
      }
    } else {
      state.stableLabel = null;
      state.stableCount = 0;
    }
  }

  /* ---------- rendering ---------- */

  function setStatus(text) { $('status').textContent = text; }
  function setTrainStatus(text) { $('train-status').textContent = text; }

  function renderClasses() {
    const box = $('classes');
    box.innerHTML = '';
    state.classes.forEach((c, i) => {
      const row = document.createElement('div');
      row.className = 'class-row';
      const rec = state.recordingIdx === i;
      row.innerHTML =
        `<span class="class-name">${c.name}</span>` +
        `<span class="class-count">${c.samples.length}</span>` +
        `<button class="rec${rec ? ' recording' : ''}" data-i="${i}">` +
        `${rec ? '◉ recording…' : '● hold to record'}</button>` +
        `<button class="del" data-i="${i}" title="delete">✕</button>`;
      box.appendChild(row);
    });
    for (const btn of box.querySelectorAll('.rec')) {
      const i = +btn.dataset.i;
      btn.addEventListener('pointerdown', e => { e.preventDefault(); setRecording(i); });
      btn.addEventListener('pointerup', () => setRecording(-1));
      btn.addEventListener('pointerleave', () => { if (state.recordingIdx === i) setRecording(-1); });
    }
    for (const btn of box.querySelectorAll('.del')) {
      btn.addEventListener('click', () => removeClass(+btn.dataset.i));
    }
  }

  function renderBars() {
    const box = $('bars');
    box.innerHTML = '';
    const top = topPrediction();
    $('prediction').textContent =
      state.net ? (top && top.prob > 0.6 ? top.label : '…') : '—';
    if (!state.net) return;
    state.netLabels.forEach((name, k) => {
      const p = state.probsEma ? state.probsEma[k] : 0;
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML =
        `<span class="bar-label">${name}</span>` +
        `<span class="bar-track"><span class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>` +
        `<span class="bar-pct">${(p * 100).toFixed(0)}%</span>`;
      box.appendChild(row);
    });
  }

  /* ---------- camera / MediaPipe glue ---------- */

  let started = false;

  function enableCamera() {
    if (started) return;
    if (typeof Hands === 'undefined' || typeof Camera === 'undefined') {
      setStatus('⚠ hand-tracking library failed to load — check your connection and reload');
      return;
    }
    started = true;
    setStatus('starting camera…');

    const video = $('video');
    const canvas = $('overlay');
    const ctx = canvas.getContext('2d');

    const hands = new Hands({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}`,
    });
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });

    hands.onResults(results => {
      ctx.save();
      ctx.scale(-1, 1); // selfie mirror
      ctx.translate(-canvas.width, 0);
      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

      const lm = results.multiHandLandmarks && results.multiHandLandmarks[0];
      const handed = results.multiHandedness && results.multiHandedness[0]
        ? results.multiHandedness[0].label : 'Right';

      if (lm && typeof drawConnectors !== 'undefined') {
        drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: '#73eff7', lineWidth: 2 });
        drawLandmarks(ctx, lm, { color: '#ffcd75', lineWidth: 1, radius: 2.5 });
      }
      ctx.restore();

      if (state.recordingIdx >= 0) {
        ctx.strokeStyle = '#ff5a5a';
        ctx.lineWidth = 6;
        ctx.strokeRect(0, 0, canvas.width, canvas.height);
      }

      handleLandmarks(lm || null, handed);
      if (!lm) setStatus('show one hand to the camera');
      else if (!state.net) setStatus('hand tracked — record samples, then train');
      else setStatus('recognizing live');
    });

    const camera = new Camera(video, {
      onFrame: async () => { await hands.send({ image: video }); },
      width: 640,
      height: 480,
    });
    camera.start().catch(err => {
      started = false;
      setStatus(`⚠ camera failed: ${err.message || err}`);
    });
    $('btn-camera').textContent = '🎥 camera on';
  }

  /* ---------- wiring ---------- */

  $('btn-camera').addEventListener('click', enableCamera);
  $('btn-add').addEventListener('click', () => {
    if (addClass($('new-name').value)) $('new-name').value = '';
  });
  $('new-name').addEventListener('keydown', e => {
    if (e.key === 'Enter') $('btn-add').click();
  });
  for (const btn of document.querySelectorAll('[data-sign]')) {
    btn.addEventListener('click', () => addClass(btn.dataset.sign));
  }
  $('btn-train').addEventListener('click', train);
  $('btn-clear').addEventListener('click', () => { $('transcript').textContent = ''; });
  $('btn-backspace').addEventListener('click', () => {
    const t = $('transcript');
    t.textContent = t.textContent.replace(/\s*\S\s*$/, '');
  });

  renderClasses();
  renderBars();

  // Test hook: lets the headless browser check drive the full pipeline
  // (add signs → record synthetic landmarks → train → recognize) without a camera.
  window.SIGN_APP = { state, addClass, setRecording, handleLandmarks, train, topPrediction };
})();
