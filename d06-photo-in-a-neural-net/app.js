/**
 * UI glue: target image management, the training loop, side-by-side
 * comparison (neural net vs JPEG squeezed to the same byte size), and the
 * continuous-zoom render that shows why a function beats a pixel grid.
 */

(() => {
  'use strict';

  const $ = id => document.getElementById(id);

  const RES = 112;          // training resolution (RES × RES pixels)
  const DISPLAY = 320;      // canvas display size
  const STEPS_PER_FRAME = 4;

  let model = null, trainer = null;
  let running = false;
  let target = null;        // { w, h, data: Float32Array[w*h*3] in [0,1] }
  let lossEma = null;
  let frame = 0;
  let lastEval = { psnr: 0 };

  const tCanvas = $('target-canvas'), tCtx = tCanvas.getContext('2d');
  const rCanvas = $('recon-canvas'), rCtx = rCanvas.getContext('2d');
  const jCanvas = $('jpeg-canvas'), jCtx = jCanvas.getContext('2d');

  /* ---------- target images ---------- */

  /** Procedural demo photo: sunset over hills — smooth + sharp edges. */
  function demoTarget() {
    const c = document.createElement('canvas');
    c.width = RES; c.height = RES;
    const ctx = c.getContext('2d');
    const sky = ctx.createLinearGradient(0, 0, 0, RES * 0.72);
    sky.addColorStop(0, '#2b2d64');
    sky.addColorStop(0.55, '#c4547a');
    sky.addColorStop(1, '#ffb45e');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, RES, RES);
    const sun = ctx.createRadialGradient(RES * 0.62, RES * 0.55, 2, RES * 0.62, RES * 0.55, RES * 0.2);
    sun.addColorStop(0, '#fff3c8');
    sun.addColorStop(0.5, '#ffd27a');
    sun.addColorStop(1, 'rgba(255, 210, 122, 0)');
    ctx.fillStyle = sun;
    ctx.fillRect(0, 0, RES, RES);
    ctx.fillStyle = '#2a1e3f';
    ctx.beginPath();
    ctx.moveTo(0, RES * 0.78);
    for (let x = 0; x <= RES; x++) {
      ctx.lineTo(x, RES * 0.74 + Math.sin(x * 0.08) * 4 + Math.sin(x * 0.021 + 2) * 7);
    }
    ctx.lineTo(RES, RES); ctx.lineTo(0, RES);
    ctx.fill();
    ctx.fillStyle = '#171226';
    ctx.beginPath();
    ctx.moveTo(0, RES * 0.9);
    for (let x = 0; x <= RES; x++) {
      ctx.lineTo(x, RES * 0.88 + Math.sin(x * 0.05 + 5) * 3);
    }
    ctx.lineTo(RES, RES); ctx.lineTo(0, RES);
    ctx.fill();
    return c;
  }

  function setTargetFromCanvas(src) {
    tCtx.imageSmoothingEnabled = true;
    tCtx.clearRect(0, 0, DISPLAY, DISPLAY);
    tCtx.drawImage(src, 0, 0, DISPLAY, DISPLAY);
    const small = document.createElement('canvas');
    small.width = RES; small.height = RES;
    small.getContext('2d').drawImage(src, 0, 0, RES, RES);
    const img = small.getContext('2d').getImageData(0, 0, RES, RES).data;
    const data = new Float32Array(RES * RES * 3);
    for (let i = 0; i < RES * RES; i++) {
      data[i * 3] = img[i * 4] / 255;
      data[i * 3 + 1] = img[i * 4 + 1] / 255;
      data[i * 3 + 2] = img[i * 4 + 2] / 255;
    }
    target = { w: RES, h: RES, data };
    resetModel();
    renderJpegRival();
  }

  function loadFile(file) {
    const img = new Image();
    img.onload = () => {
      // center-crop to square
      const s = Math.min(img.width, img.height);
      const c = document.createElement('canvas');
      c.width = RES; c.height = RES;
      c.getContext('2d').drawImage(
        img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, RES, RES);
      setTargetFromCanvas(c);
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  }

  /* ---------- model / training ---------- */

  function resetModel() {
    running = false;
    $('btn-train').textContent = '▶ compress';
    if (trainer) { trainer.dispose(); trainer = null; }
    if (model) model.dispose();
    model = new SIREN.Model({ hidden: 72 });
    trainer = new SIREN.Trainer(model, target);
    lossEma = null;
    lastEval = { psnr: 0 };
    $('size-note').textContent =
      `network: 2→72→72→3, ${model.paramCount().toLocaleString()} weights ≈ ${(model.byteSize() / 1024).toFixed(1)}KB at 16-bit`;
    setStatus('untrained — the net knows nothing about your photo yet');
    drawRecon();
  }

  function setStatus(t) { $('status').textContent = t; }

  function drawRecon(scale = 1) {
    const res = RES * scale;
    const rgba = SIREN.render(model, res, res);
    const off = document.createElement('canvas');
    off.width = res; off.height = res;
    off.getContext('2d').putImageData(new ImageData(rgba, res, res), 0, 0);
    rCtx.imageSmoothingEnabled = scale > 1;
    rCtx.clearRect(0, 0, DISPLAY, DISPLAY);
    rCtx.drawImage(off, 0, 0, DISPLAY, DISPLAY);
  }

  /** Binary-search JPEG quality so the file matches the net's byte budget. */
  function renderJpegRival() {
    const budget = model.byteSize();
    const src = document.createElement('canvas');
    src.width = RES; src.height = RES;
    const img = new ImageData(new Uint8ClampedArray(RES * RES * 4), RES, RES);
    for (let i = 0; i < RES * RES; i++) {
      img.data[i * 4] = target.data[i * 3] * 255;
      img.data[i * 4 + 1] = target.data[i * 3 + 1] * 255;
      img.data[i * 4 + 2] = target.data[i * 3 + 2] * 255;
      img.data[i * 4 + 3] = 255;
    }
    src.getContext('2d').putImageData(img, 0, 0);

    const bytesAt = q => {
      const b64 = src.toDataURL('image/jpeg', q).split(',')[1];
      return { bytes: Math.ceil(b64.length * 3 / 4), b64 };
    };
    let lo = 0.01, hi = 1, best = bytesAt(lo);
    for (let i = 0; i < 12; i++) {
      const mid = (lo + hi) / 2;
      const r = bytesAt(mid);
      if (r.bytes <= budget) { best = r; lo = mid; } else { hi = mid; }
    }
    const pic = new Image();
    pic.onload = () => {
      jCtx.imageSmoothingEnabled = true;
      jCtx.clearRect(0, 0, DISPLAY, DISPLAY);
      jCtx.drawImage(pic, 0, 0, DISPLAY, DISPLAY);
      $('jpeg-note').textContent =
        `JPEG squeezed to ${(best.bytes / 1024).toFixed(1)}KB (same budget)`;
    };
    pic.src = 'data:image/jpeg;base64,' + best.b64;
  }

  /* ---------- main loop ---------- */

  function tick() {
    if (running && trainer) {
      for (let k = 0; k < STEPS_PER_FRAME; k++) {
        const loss = trainer.step();
        lossEma = lossEma === null ? loss : lossEma * 0.95 + loss * 0.05;
      }
      if (frame % 8 === 0) drawRecon();
      if (trainer.iter % 100 < STEPS_PER_FRAME) lastEval = trainer.evaluate();
      setStatus(`step ${trainer.iter} · loss ${lossEma.toExponential(2)} · PSNR ${lastEval.psnr.toFixed(1)}dB`);
    }
    frame++;
    requestAnimationFrame(tick);
  }

  /* ---------- controls ---------- */

  $('btn-train').addEventListener('click', () => {
    running = !running;
    $('btn-train').textContent = running ? '⏸ pause' : '▶ compress';
    $('btn-train').classList.toggle('active', running);
  });

  $('btn-reset').addEventListener('click', resetModel);
  $('btn-demo').addEventListener('click', () => setTargetFromCanvas(demoTarget()));
  $('file-photo').addEventListener('change', e => {
    if (e.target.files[0]) loadFile(e.target.files[0]);
  });

  $('sel-zoom').addEventListener('change', () => drawRecon(parseInt($('sel-zoom').value, 10)));

  $('btn-export').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(model.exportQuantized())], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'photo-as-a-brain.json';
    a.click();
    URL.revokeObjectURL(a.href);
  });

  $('file-import').addEventListener('change', e => {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        model.importQuantized(JSON.parse(reader.result));
        drawRecon();
        setStatus('imported a brain — this photo was never sent as pixels');
      } catch (err) {
        setStatus(`⚠ import failed: ${err.message}`);
      }
    };
    reader.readAsText(f);
  });

  tf.ready().then(() => { $('backend').textContent = `backend: ${tf.getBackend()}`; });

  setTargetFromCanvas(demoTarget());
  tick();

  // Test hook for the headless browser check.
  window.INR_APP = {
    step: () => trainer.step(),
    evaluate: () => trainer.evaluate(),
    iters: () => trainer.iter,
    byteSize: () => model.byteSize(),
  };
})();
