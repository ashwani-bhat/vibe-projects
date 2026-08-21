/**
 * Pong itself — a tiny continuous-coordinate simulation plus a rasterizer.
 * Pure JS (no browser APIs, no tf) so the physics is testable headlessly.
 *
 * Coordinates live in [0,1]^2. The LEFT paddle is the controllable one
 * (action −1/0/+1); the right paddle always runs the tracker policy, so the
 * world model ends up learning the opponent's behavior as part of "physics".
 */

const PONG = (() => {
  'use strict';

  const BALL_SPEED = 0.025;
  const PADDLE_SPEED = 0.035;
  const PH = 0.10;             // paddle half-height
  const LX = 0.06, RX = 0.94;  // paddle x planes

  function rng(seed = 1) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class Sim {
    constructor(seed = 1) {
      this.rand = rng(seed);
      this.ly = 0.5;
      this.ry = 0.5;
      this.serve(this.rand() < 0.5 ? 1 : -1);
    }

    serve(dir) {
      this.bx = 0.5; this.by = 0.2 + this.rand() * 0.6;
      const ang = (this.rand() - 0.5) * 1.0;
      this.vx = Math.cos(ang) * BALL_SPEED * dir;
      this.vy = Math.sin(ang) * BALL_SPEED;
    }

    snapshot() {
      const { bx, by, vx, vy, ly, ry } = this;
      return { bx, by, vx, vy, ly, ry };
    }
    restore(s) { Object.assign(this, s); }

    /** aLeft ∈ {−1, 0, 1}. Right paddle plays by itself. Returns miss info. */
    step(aLeft) {
      this.ly = Math.min(1 - PH, Math.max(PH, this.ly + aLeft * PADDLE_SPEED));

      // Opponent: imperfect tracker (deliberately beatable).
      const want = this.by - this.ry;
      if (Math.abs(want) > 0.02) {
        this.ry += Math.sign(want) * PADDLE_SPEED * 0.75;
        this.ry = Math.min(1 - PH, Math.max(PH, this.ry));
      }

      const px = this.bx;
      this.bx += this.vx;
      this.by += this.vy;

      if (this.by < 0.02) { this.by = 0.02; this.vy = Math.abs(this.vy); }
      if (this.by > 0.98) { this.by = 0.98; this.vy = -Math.abs(this.vy); }

      const hit = (py) => {
        this.vy += (this.by - py) * 0.06;
        const s = Math.hypot(this.vx, this.vy);
        this.vx *= BALL_SPEED / s; this.vy *= BALL_SPEED / s;
      };
      if (px >= LX && this.bx < LX && Math.abs(this.by - this.ly) < PH + 0.03) {
        this.bx = LX; this.vx = Math.abs(this.vx); hit(this.ly);
      }
      if (px <= RX && this.bx > RX && Math.abs(this.by - this.ry) < PH + 0.03) {
        this.bx = RX; this.vx = -Math.abs(this.vx); hit(this.ry);
      }

      let missLeft = false, missRight = false;
      if (this.bx < -0.02) { missLeft = true; this.serve(1); }
      if (this.bx > 1.02) { missRight = true; this.serve(-1); }
      return { missLeft, missRight };
    }
  }

  /** Rasterize to a res×res grayscale Float32Array (0 = black, 1 = white). */
  function render(sim, res) {
    const f = new Float32Array(res * res);
    const put = (x, y) => {
      if (x >= 0 && x < res && y >= 0 && y < res) f[y * res + x] = 1;
    };
    // ball: 2×2
    const bx = Math.round(sim.bx * (res - 1)), by = Math.round(sim.by * (res - 1));
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) put(bx + dx - 1, by + dy - 1);
    // paddles: 2-wide columns
    const h = Math.round(PH * res);
    const lyc = Math.round(sim.ly * (res - 1)), ryc = Math.round(sim.ry * (res - 1));
    for (let dy = -h; dy <= h; dy++) {
      put(1, lyc + dy); put(2, lyc + dy);
      put(res - 3, ryc + dy); put(res - 2, ryc + dy);
    }
    return f;
  }

  /** Follow the ball, imperfectly. noise ∈ [0,1] adds exploration for data collection. */
  function trackerPolicy(sim, rand, noise = 0) {
    if (rand() < noise) return [(-1), 0, 1][(rand() * 3) | 0];
    const d = sim.by - sim.ly;
    return Math.abs(d) < 0.03 ? 0 : Math.sign(d);
  }

  return { Sim, render, trackerPolicy, rng, PH };
})();
