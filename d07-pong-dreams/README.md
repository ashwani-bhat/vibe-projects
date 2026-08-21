# 💤 Pong Dreams

A neural net watches Pong until it learns how the world works — then the game
switches off and **you play inside the network's dream**, where every frame is
hallucinated and physics is whatever the net believes physics to be.

A zero-build, static web page. Training runs in your browser via TensorFlow.js.

## Run it

```bash
cd d07-pong-dreams
python3 -m http.server 8000
# open http://localhost:8000
```

## How to use

1. **▶ start** — the real game plays itself (with exploration noise) while a
   world model trains on the stream. Watch the right panel: the net's one-step
   prediction of the next frame sharpens from static to Pong. Train until the
   loss flattens (~a minute on a GPU-backed browser).
2. **💤 enter the dream** — the simulator stops feeding the screen. From here
   every frame is the model predicting from its own previous output, with your
   paddle input (↑/↓, W/S, or the buttons) injected each step.
3. Watch the **reality panel**: true physics running from the same seed with
   the same inputs. The gap between the two screens is exactly the model's
   error, compounding at 30fps.
4. Dreams decay — when the ball dissolves into fog, **🌱 re-seed** from reality.
   Toggle **crisp dream** off to watch the blur death-spiral in its full glory.

## What's actually happening

This is a **world model** in miniature (Ha & Schmidhuber,
[worldmodels.github.io](https://worldmodels.github.io/), 2018):

- While reality runs, transitions (frame t−1, frame t, action) → frame t+1
  fill a 3,000-entry replay buffer. Frames are 32×32 grayscale.
- An MLP — two stacked frames + a 3-way action one-hot → 384 ReLU → 1024
  pixel logits, ~1.2M parameters — trains with per-pixel sigmoid cross-entropy
  (Adam, minibatches of 32 sampled from the buffer).
- **Dream mode is closed-loop:** the model's own output becomes its next
  input. Nothing checks it against reality, so errors compound — that slow
  melt *is* the honest signature of imperfect world models, not a bug.
- The "crisp dream" toggle binarizes each fed-back frame, which kills
  accumulated pixel uncertainty and makes dreams last dramatically longer.
- Two details worth savoring: the **opponent paddle in the dream isn't
  programmed** — it's the net's learned memory of the tracker policy it
  watched. And your own paddle obeys you in the dream only because the model
  learned what your action does to the world.

## Files

| file | what |
|------|------|
| `index.html` | the page |
| `pong.js` | the real game: continuous-coordinate sim + 32×32 rasterizer (pure, tested) |
| `model.js` | the world model: MLP + trainer + one-step predictor |
| `app.js` | learn/dream modes, replay buffer, reality-drift panel, controls |
| `test/smoke.js` | headless sanity check (`npm test`) — physics invariants, prediction loss convergence, and an 80-step closed-loop dream that must stay bounded |

## Things to try

- Enter the dream *early* (loss still high) — physics is drunk: balls curve,
  teleport, split in two.
- Hold your paddle against the top wall for a while. Some dreams learn you're
  part of the furniture and stop obeying you — action conditioning fades if
  the recent context never varies.
- Turn off crisp dream and count how many steps until heat death.
- In the dream, the ball sometimes bounces off a paddle *that isn't there
  anymore*. The model learned correlations, not objects.
