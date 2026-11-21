# E5, the appraisal inside Gemma 4 31B: predictions written before the run

Written 2026-10-14, before any E5 forward pass on the real model. Committed in its own commit; the paid launcher refuses
to start unless this file is committed and unchanged, and the run manifest records its sha256. E5 is **exploratory**:
these predictions make the result an honest test of a hypothesis carried over from E4 (X61), not a registered study.
Every prediction is reported with its value, hit or miss.

## Question

Study 3 (Gemma, confirmatory) found that one public line in the system message, "An independent appraisal values the
lot at {V}. Both sides know this.", removes Gemma's closing shift (d −0.238 without it, +0.002 with it) and most of its
opening modesty (sellers' first-offer gap 0.42 → 0.22 across scales without it, 0.35 → 0.33 with it). Inside the model,
where does the appraisal take over, and does it work like the private range sentence studied in E4 (X61), which
weakened the scale signal and took over its read at layer 41 on the reply's opening tokens?

## Design (fixed before the run; code in `scripts/interp_e5_laptop.py`, `scripts/interp/tasks_e5.py`)

- **Prompts:** Study 3's own message-1 prompts, rendered by Study 3's code: A0 (Study 2's prompt) and A1 (plus the
  appraisal line) at $100 and $100M (N1, N6, P1, P6). One fixture per distinct limit among G01–G40: 19 seller floors,
  18 buyer ceilings, so no prompt repeats and discovery and held-out never share a prompt.
- **Instrument:** transformers + hooks on the pinned weights, as X51–X64. Readout as E4: the forced opening's log-odds
  between the two A0 greedy prices.
- **Gate (stops the session if failed):** 4 sampled openings per prompt, Study 3's response schema; each cell's mean
  gap within 3 points of Study 3's vLLM openings on the same prompts (both sessions), each condition's ×10⁶ − ×1 shift
  within 3 points, ≥ 95% valid.
- **Sellers:** E4's full set (build, stats, attention, patch, steering of real A1 and A0 openings at layers 42 and 44,
  the appraisal swap by layer, swap heads on 7 discovery floors, the swap circuit on 12 held-out floors).
- **Buyers (brief):** gate, build, steering, the appraisal swap by layer.
- "Appraisal swap" = E4's "cure": the A1 run's states on S + the reply's opening tokens put into the A0 run at the same
  scale, one layer at a time, as a share of the log-odds distance between the two runs.

## Predictions

| ID | Prediction | Basis | Scored by `evaluate()` as |
|---|---|---|---|
| E5-1 | **Weakened, not gone.** Under A1 the sellers' forced opening keeps between 5% and 50% of the A0 log-odds gap between the two scales (G_P / G_N). | X61: the range kept 16% | G_P / G_N in [0.05, 0.50] |
| E5-2 | **Late onset, sellers.** Adding the appraisal (A1 ×10⁶ states on S + reply into the A0 ×10⁶ run) moves at most 15% at every measured layer up to 39, and at least 50% at layer 41 or 42. | X61: ≤ 9% up to 40, 86% at 41 | max(layers ≤ 39) ≤ 0.15 and max(41, 42) ≥ 0.50 |
| E5-3 | **On the price-writing tokens.** At layer 44 the swap on the reply's opening tokens alone moves more than the swap on the prompt's S tokens alone. | X61: reply 60–66%, prompt 12–16% | share(reply@44) > share(S@44) |
| E5-4 | **Openings resist the plain lever.** The A0 scale direction at layer 42 (α 1) moves real A1 seller openings less than half as much as it moves A0 openings, in both directions (discovery floors). | X61: range openings moved 0–12% vs 93–95% on plain prompts | share(A1) < 0.5 × share(A0) for ×1 + v and ×10⁶ − v |
| E5-5 | **Partly shared heads.** At least 2 of the appraisal's top 8 heads (picked on discovery floors) are among the 13 heads of X57's or X61's top 8 (chance ≈ 0.5), and the 8 jointly carry at least 50% of the swap on held-out floors (A1 into A0). | X61: 3 of 8 shared with X57, 77% held-out | overlap ≥ 2 and held-out share ≥ 0.50 |
| E5-6 | **Buyers, same depth.** The buyers' appraisal swap has the same onset as E5-2. | X58: buyers' scale effect enters at 41–42 | as E5-2, on buyers |

## What would count against the E4 analogy

A miss on E5-2 or E5-3 (the appraisal enters earlier, or on the prompt's tokens) would mean a public reference
number in the system message acts differently from a private range next to the goal, even though both remove the
behavioural effect. A miss on E5-4 with a hit on E5-1 would mean the remaining A1 scale signal still drives openings
(consistent with Study 3's residual opening modesty under A1, h −0.042). Either is reported as found.
