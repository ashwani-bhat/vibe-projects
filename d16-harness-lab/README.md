# 🔧 Harness Lab

**The model is fixed. Everything you can change is the harness around it.**

Tune step budgets, verifiers, observation shaping, compaction, typed tool errors
and a no-progress detector; run 200 simulated bug tickets; watch pass rate,
wrong-fixes-shipped and cost-per-solve move. Same model throughout — from ~34%
of tickets solved to >92%, purely from harness engineering.

A zero-build, zero-dependency static page.

## Run it

```bash
cd d16-harness-lab
python3 -m http.server 8000
# open http://localhost:8000
```

## The point

The most common failure in agentic systems is diagnosing a harness problem as a
model problem. "The agent is failing, let's swap in a bigger model" is the
mid-level instinct; "show me twenty failed traces, bucketed" is the senior one.
This lab makes that concrete: model competence is a **constant** in `sim.js`
(`MODEL.baseActionAcc`, `MODEL.editAcc`, …) and never changes. Only the harness
knobs move — and they move the outcome by 60 percentage points.

## What's simulated (and why each one is real)

| mechanic | shape in the sim | why it's there |
|----------|------------------|----------------|
| **context rot** | action accuracy = `base · q^1.1`, where `q` falls as the window fills | long contexts don't just cost more, they make the model worse |
| **quadratic cost** | every turn bills the whole context again | why a 40-turn loop with raw logs is a budget event |
| **observation shaping** | shaped tool output ≈ 200 tok, raw ≈ 800–3000 | the single highest-leverage knob here |
| **compaction** | caps the window, 12% chance of dropping a load-bearing fact | summarisation is lossy state estimation, not free |
| **typed tool errors** | a structured failure raises next-action accuracy | an error that carries information ends the retry loop |
| **doom loops** | 3 uninformative failures can wedge the trajectory | the budget burns while nothing happens |
| **verification** | weak/strict catch wrong patches at a token cost; strict also false-rejects | the only thing between a plausible patch and a bad deploy |
| **false success** | tests miss 35% of bad edits — ticket "closes" wrong | the worst outcome isn't failure, it's undetected failure |

Tickets carry 1–3 real bugs (avg 1.75), so a single lucky edit doesn't close them.

## What to try

- **Start naive, hit run.** ~34% pass, ~9% wrong fixes shipped, most trajectories
  burning out in doom loops. Everything looks like a model problem.
- **Turn on typed errors alone.** Doom loops collapse. You didn't touch the model.
- **Turn off compaction and compare raw vs shaped logs.** Pass rate goes from ~91%
  to ~45% — pure context rot, no other change. This is the most dramatic single
  knob in the lab.
- **Run strict vs no verifier at high pass rate.** Wrong-fixes-shipped goes from
  ~1% to ~19%. The verifier isn't buying accuracy, it's buying *honesty* — and
  the metric it protects is the one that doesn't show up in a demo.
- **Starve the step budget to 8.** Watch truncation eat genuinely-solvable work.
- **Explore the frontier.** Every run leaves a dot on the pass-rate/cost plot.
  There's no single "best" config, only a frontier and a gate.
- **Watch one ticket.** The terminal replays a single trajectory turn by turn —
  the same events, but legible as a story.

## Files

| file | what |
|------|------|
| `index.html` | the page |
| `sim.js` | the model constants, environment, and trajectory simulation (pure, seeded, tested) |
| `app.js` | knobs, batch metrics, ship gates, Pareto chart, trajectory replay |
| `test/smoke.js` | headless check (`npm test`) — determinism plus one assertion per claimed trade-off: verifier↔false-success, shaping↔cost and rot, typed errors↔doom loops, detector↔wasted spend, budget↔truncation |

The smoke test is the interesting file: it asserts the *economics*, not just that
the code runs. If a future change breaks the claim that dropping the verifier
ships more wrong fixes, the test fails.

## Honest limitations

This is a simulation with hand-chosen constants, not a benchmark. The numbers are
illustrative — chosen to be specific enough to argue with, in the same spirit as
the [Agentic Harness field manual](https://ashwani-bhat.github.io/interview-blogs/agentic-harness-field-manual.html)
this accompanies. What's meant to transfer is the *shape* of the trade-offs and
the habit of measuring false-success separately from failure.
