/**
 * Training state → music. Pure functions only (no Web Audio here), so the
 * whole mapping is unit-testable in Node.
 *
 * The score has two parts:
 *  - a DRONE: a just-intonation chord with one voice per network layer.
 *    Each voice is detuned by its layer's gradient magnitude — layers being
 *    corrected hard play sour, layers at rest play pure. Convergence sounds
 *    like an orchestra tuning up.
 *  - a MELODY: one note per classified sample. Pitch comes from the hidden
 *    activations, correct answers land on a pentatonic scale, mistakes sit a
 *    sour semitone off it, and loudness is the size of the error.
 */

const SONIFY = (() => {
  'use strict';

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  /** Just-intonation major chord (root, third, fifth, octave) on `root` Hz. */
  function chordFreqs(root = 110) {
    return [1, 5 / 4, 3 / 2, 2].map(r => root * r);
  }

  /**
   * How far off-pitch a layer's voice sits, in cents.
   * `ref` is a slowly-decaying running max of gradient RMS maintained by the
   * caller, so early training (grads at their peak) maps to maximum sourness.
   */
  function detuneCents(gradRms, ref, maxCents = 85) {
    if (!(ref > 0)) return 0;
    return clamp(gradRms / ref, 0, 1) * maxCents;
  }

  /** 1 = untrained (bright sawtooth rasp), 0 = converged (warm triangle hum). */
  function harshness(loss) {
    return clamp(loss / 0.7, 0, 1); // 0.7 ≈ BCE of a coin-flip model
  }

  /** Melody rate in notes/sec: frantic while learning, calm once converged. */
  function tempo(loss) {
    return 1.5 + 6.5 * harshness(loss);
  }

  const PENTA = [0, 3, 5, 7, 10]; // A minor pentatonic (semitones above root)

  /**
   * Pitch for one classified sample. `actMean` is the mean of the last hidden
   * layer (tanh, so in [-1, 1]) — the net's internal "posture" for this input.
   * Wrong answers get pushed one semitone up, off the scale, which reads as
   * unmistakably sour against the pentatonic drone.
   */
  function noteFreq(actMean, correct, base = 220) {
    const NOTES = PENTA.length * 2; // two octaves
    const d = clamp(Math.floor((actMean + 1) / 2 * NOTES), 0, NOTES - 1);
    let semis = PENTA[d % PENTA.length] + 12 * Math.floor(d / PENTA.length);
    if (!correct) semis += 1;
    return base * Math.pow(2, semis / 12);
  }

  /** Loudness from prediction error: confident-correct is a whisper, blunders ring out. */
  function noteVel(err) {
    return 0.08 + 0.5 * clamp(Math.abs(err), 0, 1);
  }

  return { chordFreqs, detuneCents, harshness, tempo, noteFreq, noteVel, PENTA };
})();
