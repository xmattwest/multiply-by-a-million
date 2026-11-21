# Pre-registration, Study 2: the same question with no public appraisal

**Status: PRE-REGISTRATION, approved by the user and frozen 2026-10-03.** It is frozen by its
own commit, which must come before any Study 2 game (pilot games
included). Hash and time: `git log -- prereg/study2/PREREGISTRATION.md`.

Everything not stated here is exactly as in `prereg/PREREGISTRATION.md`
(Study 1), including amendment A1. Each section below says what differs.

---

## 0. Why there is a Study 2, and what happens to Study 1

- **Study 1** ran its first model, Qwen3.8-27B-FP8, on 2026-10-03 (real run
  `qwen3.8-27b-fp8-20261003T180835Z`). Its result was invariant within
  ±0.10.
- Reading those games showed that **deals anchor on the public appraisal**:
  - median deal price 1.015 × V;
  - the middle 80% within 0.945–1.060 × V.

  The appraisal, and the public ranges centered on it, may be what keeps the
  split constant across scales.
- **Decision (user, 2026-10-03): Study 1 stops at Qwen.**
  - Its Gemma runs are not run. That is a scope change, reported in Study
    1's results as a deviation, with this reason.
  - Study 1's Qwen result stands and is reported in full.
- **Study 2 asks the same question under a prompt with no reference
  value:** each side knows only its own limit, and that a deal is possible.

**Reporting commitment:**
- Both studies are reported in full, whatever they show.
- Neither is chosen, dropped or re-labeled because of its outcomes.

## 1. Question and hypotheses

These are H1 and H2 of Study 1, tested under the Study 2 prompt:
- Does multiplying every number by 1,000,000 change how the surplus is
  split, when no public reference value is given?
- Same primary contrast: $100M vs $100.
- Same equivalence margin: ±0.10.

## 2. Design (unchanged, except the prompt)

- **Fixtures and scales:** the same fixtures and schedule, `config/fixtures.json`, 180 games per model.
  - The pilot uses the same pilot fixtures, `config/pilot_fixtures.json`: the first 60 in run order (A1).
  - V still sets each fixture's F and C (F = f% of V, C = c% of V), but V is never shown to the models.
- **Models:** Qwen3.8-27B-FP8, Gemma 4 12B, Gemma 4 31B, in that order. A model that can't be run within the project budget is reported as not run.
- **Sampling, server, rig, gates and record formats:** unchanged (`prereg/settings.json`; Study 1 §8; data plan).
  - One session per model: the pilot, then gate 5 decided from format counts, then the real run (A1).

## 3. The prompt (the only change)

- **System message:** `prereg/study2/prompts/system.txt`. It is Study 1's system message with three lines removed:
  - "An independent appraisal values the lot at {{V}}. Both sides know this."
  - the floor's line ends at "…the seller will accept.", without "It was drawn at random from the range …";
  - the ceiling's line ends at "…the buyer will pay.", without "It was drawn at random from the range …".

  Everything else is word for word the same, including "Each side knows only its own limit. The floor is always below the ceiling, so a deal is always possible."
- **User message and fragments:** identical to Study 1 (`prereg/study2/prompts/user.txt` and `fragments.txt` are copies). The scale reaches the model only through:
  - its own limit ("Your private floor is 88000000 dollars");
  - the prices in the history.
- **Renderer:** `scripts/render_prompt_s2.py`. It reuses Study 1's frozen renderer unchanged and only replaces the system message.

## 4–5. Outcomes and analysis

- Identical to Study 1 §4 and §5: the primary test, equivalence, secondary S1–S6, and the reading table.
- Seeds and code: `scripts/analyze.py`.
- **Holm across the Study 2 models that ran.** Study 1's family is separate.

**Exploratory, labeled as such:**
- Study 1 vs Study 2 on Qwen, the one model in both: what removing the appraisal does to shares, deal prices, agreement rates and the spread of d.
- The runs were sequential, not interleaved, so this isn't a randomized comparison.

## 6. What this design does not test

- Study 1 §6 applies.
- Also: Study 2 removes the appraisal and the public ranges together, so it can't say which of the two mattered.

## 7. Risks named in advance

- **Without a reference point, more games may end without a deal.**
  - The share is defined only for agreements. If agreement rates differ by scale, the primary contrast compares different sets of games.
  - S2 (agreement rate) and S6 (the utility version, which counts every game) are the pre-registered checks on this, and are reported next to H1.
- **Spread may be larger,** which lowers power. The pilot's σ is reported only, as in A1.
- The 25% forfeit flag (Study 1 §8) applies unchanged.

## 8–9. Gates and frozen inputs

- **Gates:** as Study 1 §8 and A1. The pilot judges format only.
- **Frozen inputs:**
  - `prereg/study2/inputs.sha256` holds the Study 2 prompts and renderer.
  - Study 1's `prereg/inputs.sha256` covers everything else (fixtures, settings, base renderer).
  - The job checks both before launching.

## 10. Amendments

- The same rule as Study 1: until the first Study 2 real game, dated entries in this section, each in its own commit.
- After that, no edits; deviations are reported.
