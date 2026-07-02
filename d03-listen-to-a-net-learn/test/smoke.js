/* Headless smoke test: the hand-rolled MLP should actually learn, and the
 * sonification mappings should stay in musical range.
 * Run: node test/smoke.js   (no dependencies)
 */
const fs = require('fs');
const path = require('path');

const load = (file, name) =>
  eval(fs.readFileSync(path.join(__dirname, '..', file), 'utf8') + '; ' + name);
const MLP = load('mlp.js', 'MLP');
const SONIFY = load('sonify.js', 'SONIFY');

let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failed = true;
};

/* ---- the net learns an easy dataset nearly perfectly ---- */
{
  const data = MLP.datasets.circles(240, MLP.rng(3));
  const net = new MLP.Net(undefined, 5);
  const trainer = new MLP.Trainer(net, data, { seed: 11 });
  const first = trainer.evaluate();
  for (let i = 0; i < 500; i++) {
    const { loss } = trainer.step();
    if (!Number.isFinite(loss)) { console.error(`FAIL: loss is ${loss} at step ${i}`); process.exit(1); }
    if (i % 100 === 0) console.log(`circles step ${i}  loss ${loss.toFixed(4)}`);
  }
  const last = trainer.evaluate();
  console.log(`circles: loss ${first.loss.toFixed(3)} -> ${last.loss.toFixed(3)}, acc ${last.acc.toFixed(3)}`);
  check(last.acc > 0.95, `circles accuracy > 0.95 (got ${last.acc.toFixed(3)})`);
  check(last.loss < first.loss * 0.5, 'circles loss halved');
}

/* ---- the hard dataset (spirals) is learnable too ---- */
{
  const data = MLP.datasets.spiral(240, MLP.rng(3));
  const net = new MLP.Net(undefined, 5);
  const trainer = new MLP.Trainer(net, data, { seed: 11 });
  for (let i = 0; i < 2500; i++) trainer.step();
  const { loss, acc } = trainer.evaluate();
  console.log(`spiral: loss ${loss.toFixed(3)}, acc ${acc.toFixed(3)}`);
  check(acc > 0.9, `spiral accuracy > 0.9 (got ${acc.toFixed(3)})`);
  check(trainer.step().gradRms.every(Number.isFinite), 'gradient RMS finite for all 3 layers');
}

/* ---- sonification mappings stay in range ---- */
{
  for (let a = -1; a <= 1.001; a += 0.1) {
    for (const correct of [true, false]) {
      const f = SONIFY.noteFreq(a, correct);
      if (!(f >= 200 && f <= 950)) { check(false, `noteFreq(${a.toFixed(1)}, ${correct}) = ${f} out of range`); }
    }
  }
  check(true, 'noteFreq stays within [200, 950] Hz across activations');
  check(SONIFY.noteFreq(0, false) > SONIFY.noteFreq(0, true), 'wrong answers sit sharp of right ones');
  check(SONIFY.detuneCents(0, 0.5) === 0, 'zero gradient = perfectly in tune');
  check(SONIFY.detuneCents(0.5, 0.5) === 85, 'gradient at reference = max sourness (85 cents)');
  check(SONIFY.detuneCents(99, 0.5) === 85, 'detune clamps above reference');
  check(SONIFY.detuneCents(0.1, 0) === 0, 'no reference yet = no detune');
  check(SONIFY.tempo(0.7) > SONIFY.tempo(0.05), 'high loss = faster melody');
  check(SONIFY.harshness(0) === 0 && SONIFY.harshness(5) === 1, 'harshness clamps to [0, 1]');
  const chord = SONIFY.chordFreqs(110);
  check(chord.length === 4 && chord[0] === 110 && chord[3] === 220, 'chord is root..octave, 4 voices');
  check(SONIFY.noteVel(0) > 0 && SONIFY.noteVel(3) <= 0.58, 'note velocity bounded');
}

console.log(failed ? 'FAIL' : 'PASS');
process.exit(failed ? 1 : 0);
