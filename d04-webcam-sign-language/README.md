# 🤟 Webcam Sign-Language Recognizer

Teach your browser **your** signs. Record a handful of samples per sign, train a
tiny network right in the page (a few hundred milliseconds), then fingerspell at
your webcam and watch it type. Nothing leaves your machine — no backend, no upload.

A zero-build, static web page. MediaPipe Hands loads from a CDN; the classifier
is hand-rolled plain JS.

## Run it

```bash
cd d04-webcam-sign-language
python3 -m http.server 8000
# open http://localhost:8000 and allow camera access
```

## How to use

1. **Enable the camera** and show one hand — you'll see the landmark skeleton.
2. **Add signs** — type a name or grab an ASL preset (A, B, C, L, Y, 👍).
3. **Hold ● record** on a sign while holding that handshape steady. Vary the
   angle and distance a little; 30–60 samples per sign works well.
4. **✦ train** — instant, and it reports holdout accuracy so you know if two of
   your signs look too similar.
5. Sign at the camera. Hold a recognized sign steady for about a second and it
   gets **typed into the transcript** — fingerspell your name!

## How it actually works

- **Hand tracking:** [MediaPipe Hands](https://developers.google.com/mediapipe)
  finds 21 3D hand landmarks per webcam frame, in the browser.
- **Canonicalization** (`landmarks.js`): raw landmarks change with where you
  stand, how big your hand looks, and wrist tilt — none of which changes the
  sign. So every frame is normalized: left hands mirrored to right, wrist moved
  to the origin, rotated so the wrist→middle-knuckle bone points straight up,
  and scaled so that bone has length 1. The classifier sees the *shape* of your
  hand, nothing else. All pure functions, unit-tested for translation / scale /
  rotation / mirror invariance.
- **Classifier** (`classifier.js`): a 63→32→K softmax MLP with hand-rolled
  backprop and Adam — no ML library. A few hundred samples train in
  ~200ms. Gaussian feature jitter during training stands in for webcam wobble,
  and a 20% holdout split gives the honest accuracy shown after training.
- **Recognition UX** (`app.js`): per-frame probabilities are EMA-smoothed; a
  sign above 85% confidence for ~22 consecutive frames gets typed, then a
  cooldown prevents repeats. Losing the hand resets the smoother.

## Files

| file | what |
|------|------|
| `index.html` | the page |
| `landmarks.js` | landmark canonicalization (pure, tested) |
| `classifier.js` | softmax MLP + Adam trainer (pure, tested) |
| `app.js` | UI, recording, spell-out logic, MediaPipe/camera glue |
| `test/smoke.js` | headless sanity check (`npm test`) — invariance properties + full normalize→train→recognize pipeline on synthetic hands |

## Things to try

- Teach it two signs that differ only by thumb position and check the holdout
  accuracy — you're measuring how separable they are in landmark space.
- Record a sign with your right hand, then show it with your left: the mirror
  normalization means it should still be recognized.
- Walk backwards across the room while signing — scale invariance at work.
- It's not limited to ASL: any static handshape works. Invent a secret alphabet.

## Honest limitations

Static handshapes only — real sign language is also motion, orientation, and
face/body grammar, which a per-frame classifier can't capture. Signs that differ
mainly in dynamics (like ASL J or Z) won't work. This is a toy that shows the
teachable-machine pattern, not an accessibility tool.
