# 🦠 Evolving Walkers

Blobs of springs and joints learn to walk with **no brain and no backprop** —
just mutation, and the brutal rule that whoever crawls furthest gets to
reproduce.

A zero-build, zero-dependency static page. Not even a physics library: the
whole simulation is ~60 lines of verlet integration.

## Run it

```bash
cd d11-evolving-walkers
python3 -m http.server 8000
# open http://localhost:8000
```

## How to use

- **▶ evolve** runs generations continuously (about 40/second) while the
  current champion replays in real time on the big canvas. **⏩ +10** jumps
  ahead. The **mutation** slider trades exploration against stability.
- Nodes are feet — brighter cyan means grippier. Muscles flash yellow while
  contracting. The fitness chart shows best (cyan) and population mean (grey):
  the staircase jumps are new gaits being discovered.
- **↺ new primordial soup** starts over — you'll get a different champion
  bodyplan almost every run.

## How it actually works

- **Creature = genome:** 3–7 nodes (position + a `grip` gene in [0,1]) and a
  muscle per node pair. Each muscle rhythmically oscillates its rest length:
  `len(t) = rest · (1 + amp · sin(2πt/period + phase))` with evolved amplitude,
  period, and phase.
- **Physics:** verlet integration, muscles as stiff distance constraints
  (4 relaxation passes), ground contact where a node's grip decides how much
  horizontal motion the floor eats. High-grip feet anchor while low-grip parts
  slide — that asymmetry, phased right, is locomotion.
- **Evolution:** population 60; everyone gets 8 simulated seconds and fitness =
  centroid distance walked. Top half survives untouched (elitism), the rest are
  replaced by gaussian-mutated copies of survivors, plus rare "grow a limb"
  surgery and 3 random immigrants per generation to keep the gene pool weird.
- **No gradients anywhere.** The only signal is who walked furthest.

### The exploit evolution found (true story)

The first version had no speed cap. Within 200 generations, evolution
discovered that rapidly oscillating a muscle inside the constraint solver
*injects energy from nothing*, and bred vibrating blobs that slid at 16
units/second — 30 body-lengths per second — without anything resembling a
gait. Reward hacking in sixty lines of physics. The fix is the `VMAX` per-step
velocity clamp in `physics.js`, with a comment marking the crime scene: give
an optimizer a loophole, and it will not walk, it will *buzz*.

## Files

| file | what |
|------|------|
| `index.html` | the page |
| `physics.js` | verlet soft-body sim + fitness evaluation (pure, deterministic, tested) |
| `evolve.js` | genome, mutation, population loop (pure, seeded, tested) |
| `app.js` | champion replay with camera follow, fitness chart, controls |
| `test/smoke.js` | headless sanity check (`npm test`) — sim stability, mutation legality, and 40 generations of measurable improvement |

## Things to try

- Crank mutation to 3 and watch the mean crater while the best survives —
  elitism is doing the remembering.
- Restart a few times and compare champions: tripods, inchworms, and
  face-planting rowers all show up. Same rules, different history.
- Watch the grey mean line: its jumps lag the cyan best line by a few
  generations — that's a discovery spreading through the population.
