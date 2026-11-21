# Study 3 data access log

Every look at Study 3 outcome data before all registered sessions are complete is recorded here, in its own
commit, before the look. This file is not part of `inputs.sha256` and changes no registered rule.

## 2026-10-14 (MDT), interim look at the two open-weight models

- **State of data collection.** Complete: Gemma 4 31B sessions 1 and 2, Qwen3.8 27B sessions 1 and 2, API
  session 1. Not yet run: API session 2 (must be on a different day from API session 1, §3.3).
- **Before this look, only format and quality data had been examined:** game counts, terminal states, forfeits,
  reasoning tokens and text, ignored trailing text, returned model versions and the replay audit. No split,
  opening, midpoint or test statistic had been computed for any model.
- **What is examined.** Gemma 4 31B and Qwen3.8 27B only, both sessions, through the frozen
  `scripts/analyze_study3.py` (commit `c8feea1`), unchanged, written to
  `derived/study3/interim_vllm_20261014.json`.
- **What is not examined.** No outcome statistic of gpt-6-luna, claude-haiku-5.5 or gemini-3.5-flash-lite is
  computed or viewed until API session 2 is complete.
- **Status of the interim numbers.** Estimates and confidence intervals for the two models use all their
  registered data and are final. Holm-adjusted p-values here use a two-model family and are not the registered
  ones; the registered family is all five models (§5), computed once all six sessions exist.
- **Commitment.** API session 2 runs as registered, from the frozen commit, regardless of these numbers. Nothing
  in the protocol, analysis or reporting changes because of this look. The paper reports this access.
- **External registration.** OSF timestamp relative to the first registered game: to be recorded here by the
  author.

## 2026-10-15 (MDT), all sessions complete: the registered analysis

- **State of data collection.** All six registered sessions are complete: Gemma 4 31B and Qwen3.8 27B sessions 1
  and 2 (vLLM), API sessions 1 (10-14) and 2 (10-15, a different day, §3.3). Each passed the replay audit
  (every game record reproduces from the raw calls).
- **API session 2 stalled and was resumed.** It started 12:13 UTC; the Mac slept at 1,266 finished games (the
  runner keeps the machine awake only on the vLLM backend), and the run was stopped and resumed with `--resume`
  from the same frozen commit, under `caffeinate`. The resume replays no finished game and replayed no cut-off
  attempt; 0 games were given up after void attempts. Every game in the run is the registered game for its cell,
  fixture and sample.
- **Before this look, only format and quality data of the API models had been examined** (both sessions): game
  counts and terminal states, forfeits, finish reasons, reasoning tokens and text, refusals, ignored trailing text,
  returned providers and versions.
  - Luna: 35 protocol forfeits (s1 20, s2 15; 34 truncated, 1 not JSON), 23 timeouts, trailing text ignored on 346
    (s1) and 270 (s2) replies. Haiku and Gemini: no forfeits, no trailing text.
  - All three: 0 reasoning tokens and 0 reasoning text (Gemini included), 0 refusals; returned versions
    `openai/gpt-6-luna-20260922`, `anthropic/claude-haiku-5.5-20261007`, `google/gemini-3.5-flash-lite-20260721`
    and providers OpenAI, Anthropic, Google AI Studio, unchanged across sessions.
- **What is examined.** All five models, all six sessions, through the frozen `scripts/analyze_study3.py`
  (commit `c8feea1`), unchanged, run once, written to `derived/study3/study3.json`. This is the registered analysis;
  its Holm families are the registered five-model families and supersede the two-model values of the interim look.
