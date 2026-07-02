# 🎧 Listen to a Neural Net Learn

A tiny neural network learns to untangle two spirals — and **its training is the
score**. It starts as an out-of-tune orchestra and tunes itself as it learns.

A zero-build, static web page. Everything — the network, backprop, the synth —
runs in your browser. No libraries at all, not even for the math.

## Run it

```bash
cd d03-listen-to-a-net-learn
python3 -m http.server 8000
# open http://localhost:8000
```

(Any static file server works; it's just HTML + JS.)

## How to use

1. Put on headphones.
2. Hit **▶ listen & train**. A 2-24-24-1 network starts learning the two-spirals
   problem live, and the page turns its training signals into sound.
3. Watch the decision boundary sharpen while the chord tunes itself. Chimes mark
   75% / 90% / 97% accuracy. Try the other datasets (circles, moons, XOR) and the
   **reckless** learning rate.

## What you're hearing

Every element of the sound is a real training signal — nothing is decorative:

| sound | signal |
|-------|--------|
| drone chord (4 voices, just-intonation A major) | one voice per weight layer; each is **detuned in cents proportional to that layer's gradient RMS** — backprop literally plays the orchestra out of tune until learning stops |
| timbre (sawtooth rasp → warm triangle) + filter cutoff | current loss: harsh at coin-flip loss (~0.69 BCE), warm as it converges |
| melody notes | one training point classified per beat: mean activation of the last hidden layer picks the pitch on a **pentatonic scale**; wrong answers land a **sour semitone off-scale** |
| note loudness | size of the prediction error — confident-correct is a whisper, blunders ring out |
| tempo | tracks the loss: frantic early (~8 notes/s), calm at convergence (~1.5 notes/s) |
| chimes | accuracy milestones (75% / 90% / 97%) |

So the arc of a training run has a musical shape: it opens fast, loud, and sour;
mistakes get rarer and quieter; the chord's voices drift to pure fifths and
thirds; and it ends as a slow, consonant hum. Bump the learning rate to
*reckless* and you can hear Adam scramble — the detune needles jump with every
gradient spike.

## How it actually works

- `mlp.js` — a hand-rolled MLP (tanh hidden layers, sigmoid + BCE output) with
  per-sample backprop and Adam, all in plain `Float32Array`s. The first layer
  gets a 3× hotter init than Glorot: spiral-like datasets have no linear signal,
  and a standard init leaves the net stuck predicting 0.5 on most seeds.
  `step()` returns loss, batch accuracy, and per-layer gradient RMS — the
  orchestra's entire input.
- `sonify.js` — pure functions mapping training state → musical parameters
  (chord frequencies, detune cents, harshness, tempo, note pitch/velocity).
  No Web Audio in here, so it's unit-testable in Node.
- `audio.js` — the synth: 4 sustained voices (sawtooth + triangle pairs through
  a shared lowpass) for the drone, plucked triangles for the melody, a
  compressor so nothing clips.
- `app.js` — training loop (~180 steps/s), decision-boundary raster, loss
  sparkline, tuner needles, and note scheduling against the audio clock.

## Files

| file | what |
|------|------|
| `index.html` | the page |
| `mlp.js` | network + datasets + Adam trainer (pure JS) |
| `sonify.js` | training state → music mapping (pure, tested) |
| `audio.js` | Web Audio synth engine |
| `app.js` | UI, render loops, note scheduler |
| `test/smoke.js` | headless sanity check (`npm test`) — trains on circles + spirals, asserts accuracy, checks every sonification mapping stays in musical range |

## Things to try

- **Careful vs reckless learning rate** — same net, same data, totally different
  music. Reckless converges faster but you can hear it overshoot: the detune
  needles spike long after accuracy looks fine.
- **XOR blobs** — so easy it's over in seconds: a single sour phrase, then calm.
- Hit **↺ new random brain** mid-listen and hear the whole journey restart from
  noise — same piece, different performance every time.
- Listen for the *order* in which layers tune up. The output layer usually
  settles last — its gradients stay live as long as any mistake remains.
