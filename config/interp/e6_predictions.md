# E6, magnitude as a number vs magnitude as stakes, in Gemma 4 31B: predictions written before the run

Written 2026-10-15, before any E6 forward pass on the real model. Committed in its own commit; the paid launcher refuses
to start unless this file is committed and unchanged, and the run manifest records its sha256. E6 is **exploratory**:
the predictions make the result an honest test of a hypothesis, not a registered study. Every prediction is reported
with its value, hit, miss or n.a. (`evaluate()` in `scripts/interp_e6_laptop.py` scores them exactly as written here).

## Question

Do models comprehend magnitude, and how? Study 3 shows that larger amounts make all five models open more modestly,
and E5 shows that in Gemma the effect becomes decisive at layer ~41 on the tokens that write the price, while the
limit's magnitude is readable from layer ~5 (X52). Two hypotheses about what "magnitude" is inside the model:

- **Magnitude as a number** is computed early and does not depend on how the amount is written.
- **Magnitude as stakes** ("this is a lot of money") is a separate, later signal: the negotiation scale direction.
  If it is a general stakes variable, adding it should change other money decisions (risk, scale-free valuations)
  but not arithmetic, comparisons or the size of numbers the model writes.

## Design (fixed before the run; code in `scripts/interp_e6_laptop.py`, `scripts/interp/tasks_e6.py`)

- **Part A, the number.** 9 mantissas (12 … 96) × 10^0 … 10^7, in five notations:
  - digits ("81000000 dollars");
  - words ("81 million dollars");
  - short ("$81M");
  - commas ("$81,000,000");
  - a non-money count ("81000000 people").

  That gives 360 sentences ending in " in total.". For each, the residual at the final token is taken at every layer,
  and MLP neuron activations (the input of `mlp.down_proj`) at layers 4, 10, 20, 30 and 41.

  For each notation and layer, the magnitude direction is the covariance direction of the states with log10(amount).
  Measures:
  - **within r**: held-out-mantissa Pearson r;
  - **transfer r**: the direction from one notation, projections of another notation's states, Pearson r with log10;
  - **neuron census**: dials and tuned units, descriptive only.
- **Part B, the stakes.**
  - **Direction.** E5's seller build (Study 3's A0/A1 prompts, 19 fixtures) gives vN = ×10⁶ − ×1 per aligned token.
    vR is the mean of vN's reply-token rows at layers 42 and 44, rescaled to their mean per-token norm.
  - **Controls, all at vR's norm:** vS (the S rows), u (the non-money contrast) and a random vector.
  - **Steering.** sign × α × vector is added at every token of the model-turn header and a forced JSON answer
    prefix, in the prompt's forward pass only; generated tokens are not steered. Signs: +1 on $100 prompts, −1 on
    $100M prompts; Fermi prompts have no scale and use +1.
  - **Rows per prompt:** 4 plain rows, plus vR@42 (α ±1, ±2), vR@44 (±2), and vS, u, rand at layer 42 (±2).
  - **Families:**

    | Family | Prompts | Readout |
    |---|---|---|
    | nego (positive control) | Study 3's A0 seller prompts, 7 fixtures × $100 and $100M | the JSON opening price, leave-one-fixture-out vectors |
    | risk | sure X vs a 50% chance of 2.0/2.4/3.0 × X, five X, both letter orders, $100 and $100M | log-odds of the safe option |
    | compare | which of two amounts is larger, 10 pairs × 2 orders × 2 scales | log-odds of the correct letter |
    | percent | finder's reward, contingency reserve, charity donation, five amounts × 2 scales | the percentage written |
    | sum | a + b dollars, 12 pairs × 2 scales | exact match |
    | fermi | 10 estimates with no money | log10 of the estimate |

  - **Natural gap:** the plain rows' $100M − $100 difference, paired by item. **Share:** the steered minus plain
    change on $100 prompts, divided by the natural gap.

## Predictions

| ID | Prediction | Basis | Scored as |
|---|---|---|---|
| E6-1 | **Positive control.** vR@42, α +2, moves $100 seller openings ≥ 40% of the natural $100 → $100M gap, and the random vector ≤ 10% | E5: per-token vN@42 α 2 moved A0 openings 118%; one collapsed vector on fewer tokens should keep part of it | share(vR) ≥ 0.40 and \|share(rand)\| ≤ 0.10 |
| E6-2 | **A format-independent number.** At some layer in 10–40, the digits direction transfers to words and to short notation, r ≥ 0.8 for both | X52: magnitude readable early; digit count alone cannot explain words or "$81M" | max over L in 10–40 of min(r_words, r_short) ≥ 0.8 |
| E6-3 | **Computed, not lexical.** Transfer to words at layers 0–2 is ≥ 0.2 below its peak in 10–40 | Digits and words share no magnitude tokens; a shared code must be built | peak(10–40) − max(0–2) ≥ 0.2 |
| E6-4 | **Not money-specific.** At some layer in 10–40, the money (digits) direction transfers to people counts, r ≥ 0.8 | Number magnitude should not need "dollars" | max over L in 10–40 of r_count ≥ 0.8 |
| E6-5 | **Natural stakes risk aversion.** Plain risk choices are safer at $100M than at $100 by ≥ 0.5 nats | Human stakes effects; Study 3's modesty at scale | natural gap ≥ 0.5 |
| E6-6 | **Stakes transfers to risk.** vR@42, α +2, on $100 choices moves ≥ 25% of the natural gap toward $100M, more than 3× the u and random controls | The hypothesis: vR carries "large stakes" | n.a. if gap < 0.5; else share ≥ 0.25 and \|share\| > 3 × max(\|u\|, \|rand\|) |
| E6-7 | **Stakes transfers to scale-free valuations.** The same, for the percentages | As E6-6 | n.a. if \|gap\| < 1 point; else as E6-6 |
| E6-8 | **Number intact.** Under vR@42, α +2, comparison margins move ≤ 10% of their plain value and sum exact-match ≤ 5 points, at both scales | The negotiation steering left prices on their own scale (E4: 0–6 of 1,080 off-scale) | all four conditions |
| E6-9 | **No number-size leak.** Fermi estimates move ≤ 0.15 in log10 under vR@42, α +2 and −2 | As E6-8; vR should not mean "bigger numbers" | max \|Δ log10\| ≤ 0.15 |
| E6-10 | **Separate geometry.** \|cos\| between the digits magnitude direction and vR at layer 42 ≤ 0.2 | Different positions and functions; a shared direction would argue against two signals | \|cos\| ≤ 0.2 |

## What would count against the two-signal account

- **E6-1 misses.** The collapsed vector doesn't steer its own task, so Part B's transfer results are uninterpretable
  and are reported as such.
- **E6-1 hits but E6-6 and E6-7 miss.** The negotiation direction is task-specific: a "price-writing" signal, not a
  general stakes variable.
- **E6-8 or E6-9 miss.** vR carries number magnitude, which argues against a dissociation.
- **E6-2 to E6-4 miss.** Magnitude is not represented in a format-independent way at a single readout position.
  The neuron census and the per-notation within-r values are reported either way.

Every outcome is reported as found.
