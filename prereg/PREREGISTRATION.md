# Pre-registration: does multiplying every number change how an AI splits a deal?

**Status: PRE-REGISTRATION, frozen 2026-10-02** in the project's first git
commit (hash and time: `git log --reverse | head -3`). §0 was approved by the
user on 2026-10-02, all as recommended. Changes after this point follow §10
only.

**No experimental game may run before that commit.** Pilot games on the
separate pilot fixtures (E4) may run before it. They're exploratory, and
nothing from them enters the confirmatory analysis.

Every detail below cites the file that holds it.

---

## 0. Decisions (approved by the user 2026-10-02, all as recommended)

Earlier items keep their IDs; G-items are new in Phase 6.

| ID | Question | Decision | Where it's written |
|---|---|---|---|
| **G1** | Prompt wording | Approve `prereg/prompts/` as written (read it: `python3 scripts/render_prompt.py F12-x1e8-S seller`). | §3 |
| **G2 (C18)** | What's being sold | **An unnamed "fixed lot of goods" appraised at V**, the same words at every scale. A named item ("a used car") would be absurd at $100 or at $100M. Naming a scale-appropriate quantity would change the words. The cost: the scenario is abstract and a bit odd at both extremes, and that oddness is the same at every scale. | `system.txt` |
| **G3** | Primary test | Per model: are the $100M and $100 seller shares different? Two-sided Wilcoxon signed-rank on 30 fixture-level differences. | §5.1 |
| **G4** | Multiplicity | Holm correction across the models that run (2 or 3). Unadjusted p-values are reported alongside. | §5.1 |
| **G5** | Claiming "invariant" | Only with an equivalence check: the 90% CI of the mean difference lies inside **±0.10** seller share. A non-significant test alone is never called "invariant". | §5.2 |
| **G6** | Sample size | **30 fixtures, fixed** (your design). Power (§7) depends on an unknown spread. The pilot measures it, and if it's large, you decide before the real runs whether to add fixtures. | §7 |
| **G7** | P7 (thinking on) | **Not tested.** This design runs with thinking off; STORY §3 P7 is recorded as untested. Alternative: a labeled exploratory thinking-on arm (extra cost). | §6 |
| **G8** | Quality gates | Gates are based on format and infrastructure only, **never on outcomes**. See §8. | §8 |
| **G9** | Amendments | Dated amendments are allowed until the first real game. After that only reported deviations, never silent changes. | §10 |
| **C1/C2** | Sampling | Each family's published non-thinking values, with presence penalty 0 (design): Qwen T 0.7 / top_p 0.8 / top_k 20; Gemma T 1.0 / 0.95 / 64. **Because the temperatures differ, cross-family comparisons are descriptive only.** | `prereg/settings.json` |
| **C14** | Coarse prices at $100 | Pre-registered sensitivity analysis (§5.3, S5). | §5.3 |
| **C15** | Reply-length cap | Message ≤ 500 characters, `max_tokens` 512. Checked against real token counts in Phase 7b; an amendment if they don't fit. | `settings.json` |
| **C16** | No-deal games | Option (c): seller share among agreements is primary (§5.1); agreement rate and ZOPA utilities are pre-registered secondary outcomes. | §4, §5.3 |
| **C17** | Whitespace in JSON | Keep vLLM's default, and watch for runaway whitespace in the pilot. An amendment if it shows up. | §8 |
| **E1–E10** | Data plan | As recommended in `docs/data_plan.md` §1 | data plan |
| **D1–D11** | Infra | As recommended in `docs/infra_plan.md` §1. The pre-reg relies on D1 (one GPU type), D5 (locked settings) and D11 (concurrency). **D11 proposed: 1 game at a time** for the confirmatory runs; slower but simplest. | infra plan |
| **P4, P5** | Your predictions | Written in STORY §3 on 2026-10-02, before the commit: scale shifts the split, and the buyer is squeezed at $100M (as in the preliminary data). | STORY §3 |
| **Budget** | Spending limit | $15 total; stop and ask before any paid step that could pass it | `docs/infra_plan.md` §6 |

---

## 1. Question and hypotheses

**Question.** When every dollar figure in a single-issue price negotiation is
multiplied by the same factor, does the split of the surplus between the two
AI negotiators change? A perfectly scale-invariant negotiator's split would
not move.

**For each model m** (the same model plays both sides):

- **H1 (primary): shift.** The seller's share of the surplus at $100,000,000 differs from that at $100.
  - Null: the fixture-level differences are symmetric around 0.
- **H2 (primary): equivalence.** The mean difference is inside ±0.10 seller share (G5).
  - H1 and H2 are reported together. The four possible combinations each have a pre-written reading (§5.4).
- **Direction is not predicted.** STORY §3 records Claude's predictions P1–P3, P6 and P7, and your P4 and P5 (sha256 prefix `6a2ad6fac7efdd5a` for the section as committed). They are on record but are **not** the hypotheses tested here.

---

## 2. Design

| | | Source |
|---|---|---|
| Game | Single-issue price bargaining. The seller's floor F and the buyer's ceiling C are private; the reference V is public; a deal is always possible (F < C). | ZOPA §2.1 |
| Scales | V = $100, $100,000, $100,000,000. The same words; only the digits change (tested, §3). | design |
| Fixtures | 30 pairs: f ∈ 75–95 %, c ∈ 105–125 % of V, the same percentages at every scale | `config/fixtures.json` (sha256 `cf9d83288b10ef9f`), `docs/data_plan.md` §2 |
| Per model | 180 games = 30 fixtures × 3 scales × 2 openers, played in a fixed, blocked order | `config/fixtures.json` `run_order`; data plan §4 |
| Moves | OFFER (whole-dollar price), ACCEPT, WALKAWAY; strict JSON; at most 16 messages | ZOPA §4; data plan §7 |
| Models | Qwen3.8-27B-FP8 and Gemma 4 12B (primary); Gemma 4 31B (third). Pinned commits. | `config/pins.json`, `config/manifests/` |
| Serving | vLLM v0.30.0 pinned by image digest, one GPU type for all models; thinking off | `config/pins.json`; infra plan D1, D2, D5 |
| Each call | Built from scratch: a system message with the rules, plus a user message with the role, the private limit, the public history and whose turn it is. No memory carries between calls. | `scripts/render_prompt.py` |
| Sampling | Per family (C1/C2), presence penalty 0, every field sent explicitly, a per-call seed shared across the three scales | `prereg/settings.json`; data plan E5 |

---

## 3. The prompt (G1)

- **Exact text:** `prereg/prompts/system.txt`, `user.txt` and `fragments.txt`.
- **Filled in by:** `scripts/render_prompt.py`.
- **Money format:** always "`<digits> dollars`", with no commas and no "$". The JSON price is a bare integer.
- **Each side is told:**
  - V;
  - both sampling ranges (ZOPA §2.2);
  - its own limit;
  - the rules;
  - the message number and cap;
  - the public history, with the other side's messages quoted as JSON strings so they can't pose as prompt text.
- **Never shown:** the other side's limit. `render()` has no input for it.

**Verified offline** by `tests/test_render_prompt.py` (15 tests) and
`tests/mutation_check_render_prompt.sh` (9 of 9 planted flaws caught):
- **Same words at every scale:** for every fixture × opener × history length (0, 1, 2, 7 and 15 earlier messages), deleting the digits leaves the three scales' prompts identical.
- **Every number scales exactly:** each money figure at $100,000 or $100M is exactly 1,000 or 1,000,000 times the $100 one.
- **Every number is accounted for:** each number in a prompt is public, the side's own limit, or a price already in the history.
- **No digits in the templates themselves**, and they're plain ASCII.
- **Worst-case length:** the longest possible prompt is 10,957 characters, about 3.7k tokens, which fits the planned 8,192-token window (infra plan §2).

**Not yet verified** (Phase 7b, before the commit if possible, or by amendment):
- the rendered chat-template text for each model, including thinking off;
- real token counts.

---

## 4. Outcomes

**Per game** (the formulas are in `docs/data_plan.md` §6.3 and §8):

| Outcome | Definition |
|---|---|
| Terminal state | AGREEMENT, WALKAWAY, TIMEOUT or PROTOCOL_FORFEIT (void and interrupted attempts are replayed, never scored) |
| Seller share s | For agreements: (P − F)/(C − F), an exact fraction, not clamped |
| Agreement | 1 if AGREEMENT, else 0 |
| ZOPA utility | Seller (P−F)/(C−F) and buyer (C−P)/(C−F) for a deal; 0 / 0 for walk-away or timeout; −1 for the side that forfeits, 0 for the other (ZOPA §2.4, §4.4) |
| Opening offer | (first price − F)/(C − F) |

**Per fixture i, model m** (the primary unit):
- For each opener o, d_{i,o} = s(i, $100M, o) − s(i, $100, o). It is defined only when both games agreed.
- **d_i = mean of the d_{i,o} that are defined.** A fixture with neither defined is left out of the primary test, and the count left out is reported.

---

## 5. Analysis

### 5.1 Primary test (H1)
- **Test:** for each model, a two-sided Wilcoxon signed-rank test on the d_i.
  - Zero differences are dropped (Wilcoxon's original rule).
  - Tied magnitudes get average ranks.
  - The p-value comes from 200,000 random sign flips with a fixed seed. That handles ties exactly; Monte Carlo error is about ±0.0005.
- **Multiplicity:** Holm across the models that ran (G4). Each model's family-wise α is 0.05.
- **Reported:** the mean and median of d_i, and a 95% percentile bootstrap CI of the mean (10,000 resamples of fixtures, fixed seed).

### 5.2 Equivalence (H2)
- **"Shift smaller than 0.10"** is declared only if the 90% bootstrap CI of the mean d_i lies entirely inside (−0.10, +0.10). That is the standard "two one-sided tests" logic at 5%.

### 5.3 Secondary (pre-registered, Holm within each family of tests per model)
- **S1. Other scale pairs.** The same test for $100,000 vs $100, and for $100M vs $100,000.
- **S2. Agreement rate.**
  - Agreements per fixture, out of 2, compared between $100M and $100.
  - Test: Wilcoxon on the paired counts, same method as 5.1.
- **S3. Dispersion (STORY P3).**
  - Per fixture and scale, the disagreement between the two opener games, |s_S − s_B|, compared between $100M and $100 (Wilcoxon).
  - Also the across-fixture SD of shares at each scale, compared by bootstrap.
- **S4. Opening offer.** The opener's first offer as a share of the ZOPA, $100M vs $100 (Wilcoxon on fixture means).
- **S5. Coarse-price check (C14).**
  - Repeat 5.1 after rounding every $100M price to the $100 grid: P′ = round-half-even(P / 1,000,000), F′ = F / 1,000,000, C′ = C / 1,000,000.
  - Recompute shares on that grid.
  - If H1's conclusion changes, the headline reports both versions.
- **S6. Utility version (C16).** Repeat 5.1 using ZOPA seller utility for every game, so no-deal games count too.

### 5.4 Reading the results (written before any data)

| H1 | H2 | Reading |
|---|---|---|
| significant | not equivalent | **Scale shifts this model's split.** Report the direction and size. |
| not significant | equivalent | **Invariant to within ±0.10** for this model |
| not significant | not equivalent | **Inconclusive.** The data can't tell a shift from no shift at this sample size. Never called "invariant". |
| significant | equivalent | **A real but small shift** (below 0.10) |

- **Comparisons between models** (P1, P6) are descriptive: the two families' different temperatures (C2) confound them.
- **Anything not listed above is exploratory** and labeled as such.

---

## 6. What this design does not test
- **Thinking on (STORY P7):** untested (G7).
- **Whether real people would negotiate differently:** out of scope.
- **Generality:** other currencies, number formats ("100 million"), or multi-issue deals aren't covered. The prompt's fixed format, digits with no commas, is one choice among several and may itself matter. The result applies to this format.

---

## 7. Power (Assessment; `scripts/power_sim.py`, `logs/power_sim_20261003T021146Z.txt`)

**Simulation setup:**
- 10,000 simulated experiments per point;
- 30 fixtures;
- an exact Wilcoxon test, cross-checked against brute force for n = 5 and 8;
- normal differences, no ties.

The false-positive rate at α 0.05 came out at 0.045.

**Smallest shift detectable with 80% power,** by the unknown spread σ of the per-fixture differences:

| σ | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|
| α 0.05 | 0.027 | 0.054 | 0.081 | 0.108 | 0.135 | 0.162 |
| α 0.0167 (Holm, worst case of 3) | 0.032 | 0.064 | 0.096 | 0.128 | 0.160 | 0.192 |
| Chance of showing equivalence ±0.10 when the true shift is 0 | 1.00 | 1.00 | 0.95 | 0.72 | 0.42 | 0.17 |

**How to read it:**
- If the paired differences spread by about 0.10, the experiment detects shifts of about 6 points of surplus.
- If they spread by 0.30, only shifts of about 16–19 points, and "invariant" becomes nearly impossible to show.
- **The pilot's morgen EXP-004 shift was 0.19 → 0.46**, about 27 points. Its 95% CIs for 5 sessions spanned up to 0.23–0.77, so session-level spread there was large. It was a different model and setup, so it's only a hint.
- The pilot run (Phase 7, pilot fixtures) estimates σ before any real game (G6).

---

## 8. Quality gates (G8)

They are set in advance, based only on format and infrastructure, never on outcomes.

**Before each model's real run:**
1. The weights manifest matches.
2. `/version` reports 0.30.0.
3. The logged effective sampling settings equal `settings.json`.
4. The rendered prompt shows thinking off (pass criteria in api_map §2.3).
5. **Pilot:** at least 95% of pilot replies are valid and `message.reasoning` is null in all of them.
   - If a model fails the pilot, a dated amendment may change the prompt *format instructions* or C15/C17 settings, then the pilot reruns.
   - The pilot's outcomes, seller shares included, are never looked at for this purpose.

**During a real run:**
- **Stop only for infrastructure reasons** (the server is down or the attempt limits are exhausted, data plan E8).
- **Never stop because of the outcomes.**
- If more than 25% of a model's games end in PROTOCOL_FORFEIT, its results are reported with a "format-limited" flag, and S6 is shown next to the primary.

---

## 9. Frozen inputs

- `prereg/inputs.sha256` lists the sha256 of every file that defines the experiment:
  - pins, manifests and fixtures;
  - prompts and settings;
  - the renderer and the fixture generator.
- `shasum -a 256 -c prereg/inputs.sha256` checks them.
- Each run's manifest records this file's own hash and the git commit (data plan §6.1).
- **Regenerated when §0 is settled, right before the commit.**

---

## 10. Amendments and deviations (G9)
- **Until the first real game:** changes go in as dated entries in this section, each in its own commit, explaining why. Pilot results may motivate them only through the format gates in §8.
- **After that:** this document isn't edited. Every deviation is listed in the results with its reason.

### Amendment A1 (2026-10-03, before any real game): sample size fixed; pilot and real run may share a session

**Why:** two rules conflicted.
- §0 G6 says the pilot measures σ, and "if it's large, you decide before the real runs whether to add fixtures."
- §10 says pilot results "may motivate [changes] only through the format gates in §8."
- Adding fixtures because of the pilot's σ would be a change motivated by pilot *outcomes*, which §10 forbids. It would also require a human look at pilot outcomes between the pilot and each real run.

**Changes:**
1. **The sample size is fixed at 30 fixtures × 3 scales × 2 openers = 180 games per model, whatever the pilot shows.** G6's option to add fixtures is withdrawn.
   - The pilot's σ is still computed and reported, next to the §7 power table, as context for reading the results.
   - It can't change the design.
2. **The pilot and the model's real run may run in the same pod session.**
   - The real run starts only if gates 1–5 pass.
   - Gate 5 is evaluated automatically from format counts only.
   - No pilot outcome is shown or used before the real run.
3. **The pilot is the first 60 games of `config/pilot_fixtures.json` in run order**, at all 3 scales (pilot phrase `scale-invariance/fixtures/pilot`, IDs `P-…`; data plan E4).
   - Before, the pilot's size was not fixed anywhere.
   - If the pilot fails gate 5, §8 applies unchanged: an amendment may change format instructions or C15/C17 settings, then the pilot reruns, in a new session.
4. **Clarifies E9 for long runs:** if a real run ends early for an infrastructure reason (for example the pod's lifetime cap), a later session resumes it.
   - The unfinished game is marked interrupted and replayed from the start.
   - Games already finished are kept.
   - This is not a stop for outcomes (§8).

Nothing in `prereg/inputs.sha256` changes.

