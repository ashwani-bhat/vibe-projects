/* Headless smoke test: a scaled-down SIREN should learn a procedural image,
 * and the quantized export must round-trip without wrecking the picture.
 * Run: node test/smoke.js
 */
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');
global.tf = tf;

const SIREN = eval(fs.readFileSync(path.join(__dirname, '..', 'siren.js'), 'utf8') + '; SIREN');

let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failed = true;
};

// Target: 32×32 smooth gradient with a hard-edged disc — both regimes SIREN
// must handle.
const RES = 32;
const data = new Float32Array(RES * RES * 3);
for (let y = 0; y < RES; y++) {
  for (let x = 0; x < RES; x++) {
    const i = (y * RES + x) * 3;
    data[i] = x / (RES - 1);
    data[i + 1] = y / (RES - 1);
    data[i + 2] = 0.6;
    const dx = x - RES * 0.6, dy = y - RES * 0.4;
    if (dx * dx + dy * dy < 36) { data[i] = 0.95; data[i + 1] = 0.85; data[i + 2] = 0.2; }
  }
}

const model = new SIREN.Model({ hidden: 32 });
const trainer = new SIREN.Trainer(model, { w: RES, h: RES, data }, { batch: 512, lr: 1e-3 });

const first = trainer.evaluate();
console.log(`start: mse ${first.mse.toExponential(2)}, psnr ${first.psnr.toFixed(1)}dB`);
for (let i = 0; i < 600; i++) {
  const loss = trainer.step();
  if (!Number.isFinite(loss)) { console.error(`FAIL: loss is ${loss} at step ${i}`); process.exit(1); }
  if (i % 150 === 0) console.log(`step ${i}  loss ${loss.toExponential(2)}`);
}
const last = trainer.evaluate();
console.log(`end: mse ${last.mse.toExponential(2)}, psnr ${last.psnr.toFixed(1)}dB`);

check(last.mse < first.mse * 0.1, 'MSE dropped by 10x');
check(last.psnr > first.psnr + 8, `PSNR improved by >8dB (${first.psnr.toFixed(1)} → ${last.psnr.toFixed(1)})`);

// Quantized export/import round trip: the picture must survive 16-bit weights.
const exported = model.exportQuantized();
const json = JSON.stringify(exported);
console.log(`exported size: ${(json.length / 1024).toFixed(1)}KB JSON, ${(model.byteSize() / 1024).toFixed(1)}KB binary-equivalent`);
check(exported.tensors.length === 6, 'export has all 6 tensors');

const clone = new SIREN.Model({ hidden: 32 });
clone.importQuantized(JSON.parse(json));
const t2 = new SIREN.Trainer(clone, { w: RES, h: RES, data }, {});
const rt = t2.evaluate();
console.log(`round-trip psnr: ${rt.psnr.toFixed(1)}dB (was ${last.psnr.toFixed(1)}dB)`);
check(Math.abs(rt.psnr - last.psnr) < 1, 'quantization costs <1dB PSNR');

// Continuous rendering at 2x the training resolution must produce sane pixels.
const up = SIREN.render(clone, RES * 2, RES * 2);
check(up.length === RES * 2 * RES * 2 * 4, 'renders at 2x resolution');
let sane = true;
for (let i = 0; i < up.length; i++) if (!(up[i] >= 0 && up[i] <= 255)) { sane = false; break; }
check(sane, 'all rendered values in [0,255]');

console.log(failed ? 'FAIL' : 'PASS');
process.exit(failed ? 1 : 0);
