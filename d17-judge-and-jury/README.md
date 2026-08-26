# ⚖️ Judge & Jury

**An LLM-as-a-Judge eval can look fine on aggregate accuracy and still crown the
wrong model.** This lab lets you build the evaluation protocol — panel, ordering,
aggregation, recusal, rubric — and shows you both numbers: the one you'd report,
and the leaderboard it actually produces.

A zero-build, zero-dependency static page.

## Run it

```bash
cd d17-judge-and-jury
python3 -m http.server 8000
# open http://localhost:8000
```

## The setup

Four systems are compared pairwise over 400 items. Ground truth quality is known
to the simulation and hidden from the judges. Crucially, **verbosity is
uncorrelated with quality**: the best model (`atlas`) is terse, the second-best
(`boreal`) is the wordiest, and the worst (`delta`) is chatty. That's the trap
real evals fall into.

Each judge carries documented biases, all reproducible:

| bias | effect |
|------|--------|
| **position** | prefers whichever answer was shown first |
| **verbosity** | prefers the longer answer regardless of content |
| **self-preference** | prefers output from its own model family |
| **noise** | everything else |

## What you'll find

- **One cheap judge** → 66% agreement, κ 0.33, and it ranks `cirrus` — the
  *third*-best model — first.
- **The wordy judge** → 73.8% agreement, which most teams would ship. It still
  crowns `boreal` over `atlas`. Aggregate accuracy hides this completely; only
  the leaderboard shows it.
- **The monoculture** (two same-family judges) → its self-preference *cancels
  within an item* when both answers come from that family, so accuracy looks
  unremarkable — while family-A win rate inflates from 46.8% to 58.5%. The bias
  lands in the ranking, not the metric.
- **The full council** (4 diverse judges, position-swap, confidence weighting,
  recusal, length-capped rubric) → 89.8% agreement, κ 0.80, ranking intact — at
  **16× the token cost** of the cheap judge. That's the real trade, and it's why
  you sample rather than judge everything.

The bias attribution bars are computed by **counterfactual**: each bias is
switched off using the same random draws, and the accuracy delta is what that
bias was costing you. Not attribution by vibes.

## Why κ and not just accuracy

Two classes, imbalanced-ish judges, and a metric that rewards guessing the
majority side: raw agreement flatters everyone. Cohen's κ corrects for
agreement-by-chance, and the "close calls" column (items where the true quality
gap is under 0.08) shows what happens on the only items where an eval earns its
keep — a solo judge scores ~56% there, barely better than a coin flip.

## Files

| file | what |
|------|------|
| `index.html` | the page |
| `judges.js` | candidates, judge roster, biased scoring, protocols, κ and Spearman (pure, seeded, tested) |
| `app.js` | panel builder, metrics, bias attribution, leaderboard diff |
| `test/smoke.js` | headless check (`npm test`) — 18 assertions on the claimed effects: swap removes position bias, rubric caps verbosity error, monoculture inflates own-family win rate, full protocol beats solo on κ at >3× cost, and a biased protocol demonstrably distorts the ranking |

## Honest limitations

Simulated judges with hand-chosen bias magnitudes, not measured ones — the point
is the *structure* of the failure, not the constants. Real judge bias is messier
(prompt-sensitivity, formatting effects, refusal asymmetries) and the real fix
always includes human-labelled calibration data, which no simulation substitutes
for. Treat the numbers as illustrative and the habits as transferable: report κ
alongside accuracy, always check the ranking separately, and never let a model
family judge itself.
