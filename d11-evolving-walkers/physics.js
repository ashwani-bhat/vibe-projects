/**
 * Soft-body creature physics: verlet nodes + oscillating muscle constraints.
 *
 * A creature is a genome: nodes (position, grip) and muscles (node pair,
 * amplitude, period, phase). Muscles rhythmically change their rest length;
 * nodes with high grip stick to the ground while low-grip nodes slide — that
 * asymmetry is all a body needs to turn wiggling into walking.
 *
 * Pure JS, deterministic (no randomness inside the sim) — testable headlessly.
 */

const PHYS = (() => {
  'use strict';

  const DT = 1 / 30;
  const GRAVITY = 2.2;        // units/s² (creatures are ~0.5 units tall)
  const DAMPING = 0.992;
  const GROUND_Y = 1;         // y grows downward; ground plane at y = 1
  const ITERS = 4;            // constraint relaxation passes
  const VMAX = 0.05;          // per-step speed cap: animated constraints can
                              // pump unbounded energy, and evolution WILL find
                              // that exploit (it did) — cap it at ~1.5 units/s

  class Sim {
    constructor(genome) {
      this.genome = genome;
      this.t = 0;
      this.nodes = genome.nodes.map(n => ({
        x: n.x, y: n.y, px: n.x, py: n.y, grip: n.grip,
      }));
      this.startX = this.centroidX();
    }

    centroidX() {
      let s = 0;
      for (const n of this.nodes) s += n.x;
      return s / this.nodes.length;
    }

    /** Current rest length of muscle m at time t. */
    muscleLen(m) {
      return m.rest * (1 + m.amp * Math.sin((2 * Math.PI * this.t) / m.period + m.phase));
    }

    step() {
      this.t += DT;
      // verlet integration
      for (const n of this.nodes) {
        const vx = Math.max(-VMAX, Math.min(VMAX, (n.x - n.px) * DAMPING));
        const vy = Math.max(-VMAX, Math.min(VMAX, (n.y - n.py) * DAMPING));
        n.px = n.x; n.py = n.y;
        n.x += vx;
        n.y += vy + GRAVITY * DT * DT;
      }
      // muscles as stiff distance constraints with animated rest length
      for (let it = 0; it < ITERS; it++) {
        for (const m of this.genome.muscles) {
          const a = this.nodes[m.a], b = this.nodes[m.b];
          const dx = b.x - a.x, dy = b.y - a.y;
          const d = Math.hypot(dx, dy) || 1e-9;
          const want = this.muscleLen(m);
          const diff = ((d - want) / d) * 0.5 * 0.9;
          a.x += dx * diff; a.y += dy * diff;
          b.x -= dx * diff; b.y -= dy * diff;
        }
        // ground
        for (const n of this.nodes) {
          if (n.y > GROUND_Y) {
            n.y = GROUND_Y;
            // grip: how much horizontal motion the ground eats
            n.x -= (n.x - n.px) * n.grip;
          }
        }
      }
    }

    /** Distance walked so far (centroid displacement, rightward positive). */
    distance() { return this.centroidX() - this.startX; }

    ok() {
      return this.nodes.every(n => Number.isFinite(n.x) && Number.isFinite(n.y));
    }
  }

  /** Run a genome for `seconds` and return the distance walked (fitness). */
  function evaluate(genome, seconds = 8) {
    const sim = new Sim(genome);
    const steps = Math.round(seconds / DT);
    for (let i = 0; i < steps; i++) sim.step();
    return sim.ok() ? sim.distance() : -Infinity;
  }

  return { Sim, evaluate, DT, GROUND_Y };
})();
