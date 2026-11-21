# Study 2, Addendum B: opener modesty on three API models (HA)

**Status: REGISTERED 2026-10-10, approved by the user (13:10). Committed before the registered run.**
The format pilot (§9) passed for all three models (§12, B1). This commit freezes the addendum together
with `prereg/addendum_B/inputs.sha256` and the files listed there. The commit date is the timestamp.
`scripts/run_addendum_b.py` refuses the registered run unless that file is committed and every file
it lists matches.

This is a separate document. It changes nothing in `PREREGISTRATION.md` (Study 2) or in
`ADDENDUM_A.md`. It only adds one test.

## 1. Where the hypothesis comes from

- **HA was found in Qwen.** STORY X06 and X10: without an appraisal, Qwen's openers sat closer to their
  own limit at $100M.
- **It was registered in Addendum A** and tested on Gemma 4 31B, where it was supported: mean h −0.161
  (95% CI −0.181 to −0.140), 30 of 30 fixtures below 0, one-sided p < 0.00001
  (`derived/s2-gemma-4-31b-it-20261004T012932Z/primary.json`, `addendum_A`).
- **Addendum B asks whether HA holds beyond open-weight models:** on one model from each of three API
  vendors, with thinking off, as in Studies 1–2.
- **None of these three models has been run on this task before.** Nothing in this project's data comes
  from them.

## 2. Data this addendum is tested on

- **Only** the registered run of `scripts/run_addendum_b.py`: `data/runs/addB-<stamp>/`, fixtures
  F01–F30.
- **Not** included:
  - the format pilot (X01–X05, §9)
  - any `--preview` run
  - any other run
- If the run doesn't complete, the test uses the scored items and the shortfall is reported. A model
  that never runs is reported as not run.

## 3. Models (all reached through OpenRouter)

| Arm | Requested model | Pinned provider | Thinking off | Dated version (catalogue; logged per call) |
|---|---|---|---|---|
| gpt-6-luna | `openai/gpt-6-luna` | `openai` | `reasoning.effort: "none"` | `openai/gpt-6-luna-20260922` |
| claude-haiku-5.5 | `anthropic/claude-haiku-5.5` | `anthropic` | `reasoning.enabled: false` (vendor: `thinking: disabled`) | `anthropic/claude-haiku-5.5-20261007` |
| gemini-3.5-flash-lite | `google/gemini-3.5-flash-lite` | `google-ai-studio` | `reasoning.effort: "minimal"`, the lowest level; Google: "minimal does not guarantee that thinking is off" | `google/gemini-3.5-flash-lite-20260721` |

- **Routing:** every request sets `provider: {only: [<vendor>], allow_fallbacks: false,
  require_parameters: true, data_collection: "deny"}`. A call that can't be served by the vendor's own
  endpoint fails; it never goes elsewhere. The flex and fast tiers are not used.
- **Records:** every response's model ID, provider, dated model version, system fingerprint, token
  counts (incl. reasoning tokens) and cost go into `games.jsonl`. The response's `model` field carries
  only the alias; the dated version comes from OpenRouter's routing metadata (seen in the pilot). The raw response text goes into `calls.jsonl`.
- **Pinning:** Anthropic pins its model IDs. OpenAI and Google serve an alias that they may re-point.
  This is reported as a limitation, with the returned IDs and the run dates.
- **Sources:** vendor facts in `logs/api_panel_facts_20261010.txt` (pages read 2026-10-10).

## 4. Design

- **One call = the opener's first move.** Study 2 prompt (`scripts/render_prompt_s2.py`, no appraisal),
  message 1, no history.
  - Fixture f from `config/fixtures.json` (F01–F30).
  - Role: seller or buyer.
  - Scale: V = $100 (x1e2) or $100M (x1e8). Own limit = f% (seller) or c% (buyer) of V.
  - The prompts are those of the digits probe's D2/D8 conditions, i.e. Study 2's message 1 at those
    scales. `tests/test_addendum_b.py` checks this word for word.
- **Samples:** k = 4 per (model, fixture, role, scale) (§7), i.e. 480 calls per model and 1,440 in total.
- **Order:** one shuffle with a fixed public label, so models, scales and roles interleave in time.
- **Seeds:** first 8 hex digits of SHA-256(`scale-invariance/addendum-b/v1|fixture|role|sample`), masked
  to 31 bits (amendment B1).
  - The same at both scales (common random numbers).
  - Sent only where the endpoint accepts a seed (OpenAI, Google). Anthropic's doesn't. Vendors don't
    promise determinism; the seed only pairs the draws where it's honoured.
- **Sampling:** each vendor's defaults. No temperature, top_p or top_k is sent. Anthropic rejects
  non-default values. OpenAI accepts them only with reasoning off. Google recommends its default of 1.0.
- **Reply format:** strict JSON schema (action enum; price integer or null; message string;
  `additionalProperties: false`).
  - Without `minimum` and `maxLength`: Anthropic's and Google's structured outputs don't enforce them.
  - `game_engine.validate_reply` is the unchanged ground truth (price ≥ 1, message ≤ 500 characters,
    no repair).
  - `max_tokens` 1,024 (Study 2: 512), so that a little thinking on Gemini can't truncate a reply.

**Deviations from `prereg/settings.json`, stated here so none is silent:**
- vendor sampling defaults
- no seed on Anthropic
- vendor-specific thinking switches
- `max_tokens` 1,024
- schema bounds checked client-side only

## 5. Hypothesis and test

**HA (confirmatory): without a reference price, the opener's first offer sits closer to its own limit
at $100M than at $100.**

- **Gap of one valid OFFER:**
  - seller: g = P/F − 1
  - buyer: g = 1 − P/C
- **Cell mean:** the mean g over the cell's valid OFFERs (up to k = 4).
- **Per fixture:** h_f = the mean, over the roles where both scales have a cell mean, of
  [cell mean at $100M − cell mean at $100]. Fixtures with no such role are left out (counted).
- **Test, per model:** exactly as in Addendum A (`scripts/analyze.py`):
  - one-sided Wilcoxon signed-rank on the h_f, alternative median < 0
  - 200,000 random sign flips
  - zeros dropped, average ranks for ties
  - seed: the `analyze.py` rule with what = `addendumB-wilcoxon-HA`, model = the arm name
- **Family:** Holm across the three models, α = 0.05.
- **Reading:** HA is supported on a model if its Holm-adjusted p < 0.05.
- **Also reported:**
  - mean h_f with a 95% bootstrap CI (10,000 resamples of fixtures; what = `addendumB-bootstrap-HA`)
  - the number of fixtures with h_f < 0
  - seller-opened and buyer-opened gaps separately
- **HB (Addendum A's opener-advantage test) is not tested.** It needs full games.

**Code:** `python3 scripts/analyze_addendum_b.py data/runs/addB-<stamp>`, writing
`derived/addB-<stamp>/addendum_B.json`.

## 6. Invalid replies, refusals and infrastructure failures

- **Each item gets one completed reply, which is never re-asked,** whatever it says. Retries happen only
  when no completed reply came back: a transport error, a non-200 status, an error object, or
  finish_reason "error".
  - Up to 3 tries with the identical request, each recorded in `calls.jsonl` with its own call ID.
  - After 3 failed tries the item is missing (INFRASTRUCTURE).
  - A 4xx rejection other than 408/429 means our request is wrong. It is not retried; the run stops so
    the request can be fixed (amendment B1).
- **Excluded from the gap:**
  - invalid replies (any `validate_reply` error, incl. TRUNCATED)
  - refusals (a refusal field, or finish_reason `content_filter`; code REFUSAL)
  - WALKAWAYs
  - missing items

  This is the same rule as Addendum A, where the gap is defined only for an OFFER.
- **Reported per model and scale:**
  - replies
  - valid OFFERs
  - WALKAWAYs
  - invalid replies by error code
  - refusals
  - missing items
  - offers on the wrong side of the own limit
- **Flags:**
  - "format-limited" if more than 25% of a model's completed replies are invalid or refused (Study 1
    §8's 25% rule)
  - "scale-dependent format failures" if that share differs by more than 10 points between the two
    scales, because exclusions could then bias h

  A flag qualifies the reading. It doesn't change the test.

## 7. Thinking, fixtures, samples: the choices and why

- **Thinking: one family, all off.** This mirrors Studies 1–2 and Addendum A (Gemma and Qwen ran with
  thinking off). Thinking-on arms are not run, so the addendum makes no claim about thinking.
- **Gemini's residual thinking:**
  - It is part of that model's "off". The primary test keeps those replies.
  - A sensitivity analysis without replies that used any reasoning tokens is reported next to it.
  - For gpt-6-luna and claude-haiku-5.5, any reply with reasoning tokens > 0 is reported as a deviation
    and kept.
- **Fixtures: F01–F30, the confirmatory set.** Addendum A's result is on these fixtures, so the
  comparison is like for like.
  - These models never generated any hypothesis here, so reusing the fixtures raises no circularity.
  - Fresh fixtures would add sampling noise without guarding against anything.
- **Samples: k = 4.** From `scripts/power_addendum_b.py` (`logs/power_addendum_b_20261010.txt`), with
  the exact Wilcoxon at α 0.05/3. Spreads are taken from Gemma 31B (digits probe, k = 4) and from Gemma
  and Qwen Study 2 (k = 1). Power to detect a true mean h of −0.04 (a quarter of Gemma's):

  | Scenario | Power |
  |---|---|
  | Gemma | 1.00 |
  | Qwen | 0.98 |
  | Pessimistic (both spread parts doubled) | 0.51 |

  At −0.08 even the pessimistic case gives 0.98. Moving from k = 4 to k = 8 adds little except in the
  pessimistic case (0.60). These numbers assume normal effects and no ties.

## 8. What counts as what

- **HA supported on all three:** HA generalises beyond open-weight models, across three vendors.
- **On one or two:** reported as model-dependent, naming which.
- **On none:** HA is reported as a property of the open models tested (Qwen, Gemma 31B), not of LLM
  negotiators in general. It is reported, not dropped.
- **In every case:**
  - the claim is limited to these three models, at the tested setting, accessed on the run dates
  - none of this changes the Study 2 primary reading or Addendum A

## 9. Format pilot (before freezing; excluded from every test)

- **What runs:** `python3 scripts/run_addendum_b.py --pilot` on fixtures **X01–X05** only
  (`make_fixtures.build("scale-invariance/fixtures/explore-v1")`). That is 5 fixtures × 2 roles × 2
  scales × 2 samples × 3 models = 120 calls.
- **What it checks, per model:**
  - every item got a completed reply
  - at least 95% of replies are valid
  - the returned provider is the vendor's own
  - no reasoning tokens (gpt-6-luna, claude-haiku-5.5); Gemini's count is recorded
  - the request was accepted with all its parameters (`require_parameters`)
- **What is looked at:** the pilot prints format counts only. No HA number is computed from pilot data
  before this addendum is frozen.
- **If a model fails:** its format settings may be changed only by an amendment in §12, before the
  freeze, and the pilot reruns for that model. The thinking switch, `max_tokens` and the schema are the
  only things that may change. A model that can't pass is reported as not run.

## 10. Cost, limits, data handling

- **Worst cases** (`run_addendum_b.py` dry runs; every reply billed at 2,000 input + 1,024 output
  tokens, and every retry billed too):
  - pilot $0.56
  - registered run $6.61
  - expected, from the token counts seen on Gemma: about $0.30
- **Spend controls:**
  - a hard cap of $10 in the client (no call is sent if it could cross the cap)
  - the OpenRouter key's own $10 credit limit, checked before any prompt is sent
- **API costs are separate from the GPU budget.**
- **Data handling:**
  - Only the prompts in §4 are sent, to OpenRouter and on to the three vendors.
  - Paid API traffic isn't used for training by any of them.
  - Abuse-monitoring retention: Google 55 days, OpenAI 30 days, Anthropic deleted within 30 days.
  - OpenRouter's prompt logging is off.

## 11. Not in this addendum

- **Full games on these models** (which would let HB and the split be tested) are out of scope. If run,
  they will be labelled exploratory.
- **Thinking-on arms are not run.**
- **Flagship models are not run.** The three vendors' flagships can't turn thinking off.

## 12. Amendments

- Until the freeze commit: dated entries here, each in its own commit.
- After it: no edits. Deviations are reported.

**B1 (2026-10-10, after format pilot 1; committed with the freeze).**
- **Pilot 1:** `data/pilot/addB-pilot-20261010T184737Z`, 120 items, $0.0138.
  - gpt-6-luna and claude-haiku-5.5 passed: 40 of 40 valid, 0 reasoning tokens, the vendor's own
    provider.
  - gemini-3.5-flash-lite failed: 24 of 40 items got HTTP 400 on all 3 tries, "Invalid value at
    'generation_config.seed' (TYPE_INT32)". Google accepts only seeds below 2³¹, and the seed rule gave
    values up to 2³². Its 16 completed replies were all valid, with 0 reasoning tokens.
- **Changes:**
  - (a) Seeds are masked to 31 bits, for all arms, so the rule stays common.
  - (b) A 4xx rejection other than 408/429 stops the run instead of being retried.
  - (c) The dated model version is logged from the routing metadata.
- **No setting in §3–§5 changed.**
- **Pilot 2, gemini-3.5-flash-lite only, with B1:** `data/pilot/addB-pilot-20261010T185839Z`, 40 items,
  $0.0116. Passed: 40 of 40 valid, 0 reasoning tokens, provider Google AI Studio, version
  `google/gemini-3.5-flash-lite-20260721`.
- **Pilot spend in total:** $0.0254. Only format counts were looked at; no HA number was computed.

## 13. Code (listed with SHA-256 in `prereg/addendum_B/inputs.sha256`)

- `scripts/run_addendum_b.py`, `scripts/openrouter_client.py`, `scripts/analyze_addendum_b.py`,
  `scripts/analyze.py`
- `scripts/render_prompt_s2.py` and the Study 2 prompt files
- `scripts/game_engine.py`, `scripts/run_records.py`, `config/fixtures.json`
- tests: `tests/test_addendum_b.py` (42 offline tests) and `tests/mutation_check_addendum_b.py`
  (26 of 26 planted flaws caught), both run 2026-10-10 after B1
- real API calls so far: format pilot 1 only (§12)
