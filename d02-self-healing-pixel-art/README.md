# 🩹 Self-Healing Pixel Art

Draw a pixel-art sprite, let a tiny neural network learn it, then **erase chunks of it
and watch it grow back** — like a starfish regrowing an arm.

A zero-build, static web page. Everything — including training — runs in your browser.

## Run it

```bash
cd projects/d02-self-healing-pixel-art
python3 -m http.server 8000
# open http://localhost:8000
```

(Any static file server works; it's just HTML + JS. TensorFlow.js loads from a CDN.)

## How to use

1. **Draw** — paint a 24×24 sprite (or grab a preset: heart / mushroom / ghost).
2. **Train** — hit ▶ and watch the model learn to *grow* your sprite from a single
   seed pixel. After ~300 steps it starts practicing on deliberately damaged copies,
   which is what teaches it to heal. ~1500–3000 steps gives solid regeneration
   (a few minutes on a laptop GPU via WebGL).
3. **Destroy & Heal** — drag across the live sprite to erase pixels, or nuke half the
   grid with ☄️. The wound closes itself.

You can export trained weights to a JSON file and re-import them later — no retraining.

## How it actually works

This is an implementation of **Growing Neural Cellular Automata**
([Mordvintsev et al., Distill 2020](https://distill.pub/2020/growing-ca/)):

- Every pixel is a cell carrying 16 numbers: RGB, alpha ("am I alive?"), and 12
  hidden channels the cells use to talk to each other.
- Each tick, every cell perceives its 3×3 neighborhood through fixed identity + Sobel
  filters, feeds that through a 2-layer network (~8K parameters, shared by all cells),
  and adds the output to its own state. Updates fire stochastically (50%), and cells
  with no living neighbors are zeroed out.
- Training backpropagates through 40–60 CA steps: start from a seed, run the CA,
  MSE against the target image. A **sample pool** keeps grown states around so the
  CA also learns to *persist*, and a fraction of pool samples get circular holes
  punched in them so it learns to *regenerate*.
- No cell ever sees the whole image. The sprite is a stable attractor of thousands
  of purely local interactions — which is exactly why damage repairs itself.

## Files

| file | what |
|------|------|
| `index.html` | the page |
| `nca.js` | the CA model + sample-pool trainer (TensorFlow.js) |
| `app.js` | pixel editor, training loop UI, destroy-and-heal mode |
| `test/smoke.js` | headless sanity check (`npm test`) — trains a tiny CA and asserts the loss drops, no tensor leaks |

## Things to try

- Train only ~400 steps and then erase: it heals sloppily — regeneration is a skill
  it acquires, not a property it has.
- Erase *everything* except one edge. Often it regrows the whole sprite from the stump.
- Draw something asymmetric: the CA must invent its own coordinate system from
  purely local signals to know *which* side the detail goes on.
