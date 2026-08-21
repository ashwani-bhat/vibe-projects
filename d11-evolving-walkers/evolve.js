/**
 * Neuroevolution without the neuro: genomes are body plans (nodes + muscles
 * with rhythmic parameters) and the only learning signal is distance walked.
 * Truncation selection + gaussian mutation + occasional body-plan surgery +
 * a trickle of random immigrants. No gradients anywhere.
 *
 * Pure JS, seeded RNG — testable headlessly.
 */

const EVOLVE = (() => {
  'use strict';

  function rng(seed = 1) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gauss = rand =>
    Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

  /** Fully-connected small body: 3–6 nodes hovering just above the ground. */
  function randomGenome(rand) {
    const n = 3 + ((rand() * 4) | 0);
    const nodes = [];
    for (let i = 0; i < n; i++) {
      nodes.push({
        x: rand() * 0.6,
        y: 0.45 + rand() * 0.5,
        grip: rand(),
      });
    }
    const muscles = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        muscles.push({
          a: i, b: j,
          rest: clamp(dist(nodes[i], nodes[j]), 0.1, 0.8),
          amp: 0.05 + rand() * 0.3,
          period: 0.5 + rand() * 1.2,
          phase: rand() * 2 * Math.PI,
        });
      }
    }
    return { nodes, muscles };
  }

  function clone(g) {
    return {
      nodes: g.nodes.map(n => ({ ...n })),
      muscles: g.muscles.map(m => ({ ...m })),
    };
  }

  function mutate(g, rand, rate = 1) {
    const c = clone(g);
    const s = 0.08 * rate;
    for (const n of c.nodes) {
      n.x = clamp(n.x + gauss(rand) * s, -0.2, 0.8);
      n.y = clamp(n.y + gauss(rand) * s, 0.1, 0.98);
      n.grip = clamp(n.grip + gauss(rand) * s * 2, 0, 1);
    }
    for (const m of c.muscles) {
      m.rest = clamp(m.rest + gauss(rand) * s, 0.08, 0.9);
      m.amp = clamp(m.amp + gauss(rand) * s, 0, 0.45);
      m.period = clamp(m.period + gauss(rand) * s, 0.3, 2.5);
      m.phase = m.phase + gauss(rand) * s * 4;
    }
    // rare body-plan surgery: grow a limb node wired to its two nearest kin
    if (rand() < 0.06 && c.nodes.length < 7) {
      const nn = { x: rand() * 0.6, y: 0.45 + rand() * 0.5, grip: rand() };
      const near = c.nodes
        .map((n, i) => ({ i, d: dist(n, nn) }))
        .sort((p, q) => p.d - q.d)
        .slice(0, 2);
      const idx = c.nodes.length;
      c.nodes.push(nn);
      for (const { i } of near) {
        c.muscles.push({
          a: i, b: idx,
          rest: clamp(dist(c.nodes[i], nn), 0.1, 0.8),
          amp: 0.05 + rand() * 0.3,
          period: 0.5 + rand() * 1.2,
          phase: rand() * 2 * Math.PI,
        });
      }
    }
    return c;
  }

  class Population {
    constructor({ size = 60, seed = 1, evalSeconds = 8 } = {}) {
      this.rand = rng(seed);
      this.size = size;
      this.evalSeconds = evalSeconds;
      this.gen = 0;
      this.members = [];
      for (let i = 0; i < size; i++) {
        this.members.push({ genome: randomGenome(this.rand), fitness: null });
      }
      this.history = []; // { best, mean } per generation
    }

    /** Evaluate everyone, select, breed. Returns { best, mean }. */
    generation(mutationRate = 1) {
      for (const m of this.members) {
        if (m.fitness === null) m.fitness = PHYS.evaluate(m.genome, this.evalSeconds);
      }
      this.members.sort((a, b) => b.fitness - a.fitness);
      const best = this.members[0].fitness;
      const mean = this.members.reduce((s, m) => s + m.fitness, 0) / this.members.length;
      this.history.push({ best, mean });

      // top half survives (fitness kept); bottom half replaced
      const half = this.size >> 1;
      const next = this.members.slice(0, half);
      while (next.length < this.size - 3) {
        const parent = next[(this.rand() * half) | 0];
        next.push({ genome: mutate(parent.genome, this.rand, mutationRate), fitness: null });
      }
      while (next.length < this.size) {
        next.push({ genome: randomGenome(this.rand), fitness: null }); // immigrants
      }
      this.members = next;
      this.gen++;
      return { best, mean };
    }

    champion() { return this.members[0]; }
  }

  return { Population, randomGenome, mutate, rng };
})();
