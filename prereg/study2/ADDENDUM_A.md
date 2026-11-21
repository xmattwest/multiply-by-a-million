# Study 2, Addendum A: opener modesty at high stakes (HA, HB)

**Status: REGISTERED 2026-10-03, approved by the user (18:41), committed before any Gemma game.**
The commit date is the timestamp. This is a separate document. The frozen
Study 2 pre-registration (`PREREGISTRATION.md`, §10: no edits after its
first real game) is not changed. This addendum only adds tests. The
primary analysis, equivalence test, S1–S6 and the Holm family of
`PREREGISTRATION.md` §4–5 stay exactly as registered.

## 1. Where the hypotheses come from (all exploratory, all Qwen)

They come from looking at Qwen data many ways, so the Qwen data can't
confirm them. See STORY.md §12, ledger X05, X06 and X10.

- **Study 2, Qwen** (30 fixtures). Without the appraisal, the opener's
  first offer sat closer to its own limit at $100M than at $100:
  - seller markup 24.5% → 12.7% (smaller in 26 of 30 fixtures)
  - buyer discount 36.6% → 22.3% (25 of 30)
- **The same Qwen data** showed the opener's advantage reversing at
  $100M: an opener × stakes interaction of −0.33.
- **The probe** (`data/explore/X-q1q3-…`: 12 fresh fixtures, appraisal
  on vs off, randomized in one session):
  - The opening shrink replicated without the appraisal, but weaker (8
    of 12 fixtures per side), and was absent with it.
  - The reversal did not replicate (−0.04, p 0.68).

## 2. Data this addendum is tested on

- **Only:** the Study 2 real runs (180 games each, `config/fixtures.json`,
  Study 2 prompt, settings and gates exactly as registered) of
  **gemma-4-12b-it** and **gemma-4-31b-it**.
- **Not:**
  - Qwen (the source of the hypotheses)
  - any pilot
  - any `data/explore/` probe
- If a Gemma run doesn't complete, the tests use its scored games, and
  the shortfall is reported. A model that never runs is reported as not
  run.

## 3. Hypotheses and tests

Notation: P₁ is the opener's first offer, F is the seller's floor, C is
the buyer's ceiling, s is the seller's share. Both games use the same
fixture and opener; only the scale differs.

**HA (confirmatory): without a reference price, the opener's first offer
sits closer to its own limit at $100M than at $100.**
- **Gap:**
  - seller opens: g = P₁/F − 1
  - buyer opens: g = 1 − P₁/C
  - Defined only if the opener's first move is an OFFER.
- **Per fixture:** h_f is the mean, over the openers where both the $100
  and $100M gaps are defined, of g($100M) − g($100). Fixtures with no
  such opener are left out.
- **Test:**
  - one-sided Wilcoxon signed-rank on the h_f, alternative median < 0
  - 200,000 random sign flips
  - zeros dropped, average ranks for ties
  - seed: the `analyze.py` rule, what = `wilcoxon-HA`
- **Family:** Holm across the two Gemma models, α = 0.05.
- **Reading:** HA is supported on a model if its Holm-adjusted p < 0.05.
- **Also reported:**
  - the mean h_f, with a 95% bootstrap CI (10,000; what =
    `bootstrap-HA`)
  - the number of fixtures where h_f < 0
  - the seller-opened and buyer-opened gaps separately (descriptive)

**HB (secondary): the opener's advantage falls at $100M.**
- **Per fixture:** I_f = [s(seller opens, $100M) − s(seller opens, $100)]
  − [s(buyer opens, $100M) − s(buyer opens, $100)]. Only fixtures whose
  four games all reached a deal.
- **Test:**
  - **two-sided** Wilcoxon, sign flips as above, what = `wilcoxon-HB`
  - two-sided because it failed to replicate in the probe
  - Holm across the two Gemma models, separately from HA
- **Also reported:** the mean I_f.

## 4. What counts as what

- **HA supported on both Gemma models.** The finding generalizes beyond
  the model it was found in.
- **HA supported on one model.** Reported as model-dependent.
- **HA supported on neither.** It goes down as a Qwen-specific
  observation. Reported as such, not dropped.
- **Either way,** none of these change the Study 2 primary reading.
  "The mean split is invariant" and "openers move with stakes" can both
  be true.

## 5. Code

- `python3 scripts/analyze.py <run_dir> --addendum`, committed in the same
  commit as this file and before any Gemma game.
- It was only checked to run on the Qwen Study 2 data. Those numbers are
  not part of this test.
