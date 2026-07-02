/**
 * The Web Audio side: a small synth "orchestra" the training loop drives.
 * Browser-only (needs AudioContext); all the mapping logic lives in sonify.js.
 */

const ORCHESTRA = (() => {
  'use strict';

  class Orchestra {
    constructor() {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();

      this.master = this.ctx.createGain();
      this.master.gain.value = 0.7;
      const comp = this.ctx.createDynamicsCompressor();
      this.master.connect(comp);
      comp.connect(this.ctx.destination);

      // Drone bus: voices → shared lowpass (its cutoff tracks harshness) → fade-in gain.
      this.droneFilter = this.ctx.createBiquadFilter();
      this.droneFilter.type = 'lowpass';
      this.droneFilter.frequency.value = 2200;
      this.droneBus = this.ctx.createGain();
      this.droneBus.gain.value = 0;
      this.droneFilter.connect(this.droneBus);
      this.droneBus.connect(this.master);

      this.voices = [];
    }

    /** Start one sustained voice per frequency. Each voice mixes a sawtooth
     *  (the "untrained" rasp) and a triangle (the "converged" hum). */
    startDrone(freqs) {
      const t = this.ctx.currentTime;
      for (const f of freqs) {
        const saw = this.ctx.createOscillator();
        saw.type = 'sawtooth';
        saw.frequency.value = f;
        const tri = this.ctx.createOscillator();
        tri.type = 'triangle';
        tri.frequency.value = f;
        const sawG = this.ctx.createGain();
        sawG.gain.value = 0.05;
        const triG = this.ctx.createGain();
        triG.gain.value = 0.02;
        saw.connect(sawG); sawG.connect(this.droneFilter);
        tri.connect(triG); triG.connect(this.droneFilter);
        saw.start(); tri.start();
        this.voices.push({ saw, tri, sawG, triG });
      }
      this.droneBus.gain.linearRampToValueAtTime(1, t + 1.5);
    }

    /** Detune voice i by `cents` and set its saw/triangle balance from harshness. */
    setVoice(i, cents, harsh) {
      const v = this.voices[i];
      if (!v) return;
      const t = this.ctx.currentTime;
      v.saw.detune.setTargetAtTime(cents, t, 0.1);
      v.tri.detune.setTargetAtTime(cents, t, 0.1);
      v.sawG.gain.setTargetAtTime(0.055 * harsh, t, 0.2);
      v.triG.gain.setTargetAtTime(0.015 + 0.05 * (1 - harsh), t, 0.2);
    }

    setHarshness(h) {
      this.droneFilter.frequency.setTargetAtTime(500 + 2600 * h, this.ctx.currentTime, 0.25);
    }

    /** One melody note: a short plucked triangle. `when` is on the audio clock. */
    note(freq, vel, when, dur = 0.35) {
      const o = this.ctx.createOscillator();
      o.type = 'triangle';
      o.frequency.value = freq;
      const g = this.ctx.createGain();
      g.gain.setValueAtTime(0, when);
      g.gain.linearRampToValueAtTime(vel, when + 0.015);
      g.gain.exponentialRampToValueAtTime(0.001, when + dur);
      o.connect(g);
      g.connect(this.master);
      o.start(when);
      o.stop(when + dur + 0.05);
    }

    /** Rising arpeggio for accuracy milestones. */
    chime(freqs, vel = 0.22) {
      let t = this.ctx.currentTime + 0.02;
      for (const f of freqs) {
        this.note(f, vel, t, 0.7);
        t += 0.09;
      }
    }

    setVolume(v) {
      this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05);
    }

    now() { return this.ctx.currentTime; }
    resume() { return this.ctx.resume(); }
    suspend() { return this.ctx.suspend(); }
  }

  return { Orchestra };
})();
