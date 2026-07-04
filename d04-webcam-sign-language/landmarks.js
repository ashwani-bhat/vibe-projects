/**
 * Hand-landmark geometry: turn MediaPipe's 21 landmarks into a feature vector
 * a classifier can trust.
 *
 * Raw landmarks change with where you stand, how big your hand looks, and how
 * you tilt your wrist — none of which changes the *sign*. So we canonicalize:
 *   1. mirror left hands so every hand looks right-handed
 *   2. move the wrist to the origin            (translation invariance)
 *   3. rotate so wrist → middle-finger-MCP points straight up
 *                                              (in-plane rotation invariance)
 *   4. scale so that bone has length 1         (distance-to-camera invariance)
 *
 * Pure functions, no browser APIs — tested headlessly in test/smoke.js.
 */

const LANDMARKS = (() => {
  'use strict';

  const WRIST = 0, MIDDLE_MCP = 9;
  const DIM = 21 * 3;

  /**
   * lm: array of 21 {x, y, z} (MediaPipe normalized image coords).
   * handed: 'Right' | 'Left' — MediaPipe's handedness label.
   * Returns Float32Array(63) or null if the hand is degenerate.
   */
  function normalize(lm, handed = 'Right') {
    if (!lm || lm.length !== 21) return null;
    const sx = handed === 'Left' ? -1 : 1;

    const wx = lm[WRIST].x * sx, wy = lm[WRIST].y, wz = lm[WRIST].z;
    const vx = lm[MIDDLE_MCP].x * sx - wx, vy = lm[MIDDLE_MCP].y - wy;
    const len = Math.hypot(vx, vy);
    if (len < 1e-6) return null;

    // Rotate wrist→middleMCP onto (0, -1) ("up" in image coords).
    const a = -Math.PI / 2 - Math.atan2(vy, vx);
    const cos = Math.cos(a), sin = Math.sin(a);

    const out = new Float32Array(DIM);
    for (let i = 0; i < 21; i++) {
      const x = lm[i].x * sx - wx, y = lm[i].y - wy, z = lm[i].z - wz;
      out[i * 3] = (x * cos - y * sin) / len;
      out[i * 3 + 1] = (x * sin + y * cos) / len;
      out[i * 3 + 2] = z / len;
    }
    return out;
  }

  return { normalize, DIM };
})();
