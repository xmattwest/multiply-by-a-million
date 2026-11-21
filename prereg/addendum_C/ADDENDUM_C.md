# Study 2, Addendum C: full games on the three API models (the split, and the midpoint account)

**Status: REGISTERED 2026-10-11. The user approved this freeze commit before any registered game.**
The authorised format pilot passed for all three models (§13); only format and pooled cost checks were
inspected. This commit freezes this file, `prereg/addendum_C/inputs.sha256` and the files listed there.
The commit date is the timestamp. `scripts/run_addendum_c.py` requires the committed freeze and matching
inputs before the registered run.

This is a separate document. It changes nothing in `PREREGISTRATION.md` (Study 2), `ADDENDUM_A.md` or
`ADDENDUM_B.md`. C1–C5 were approved on 2026-10-11 (§14). After the authorised $1-capped format pilot,
the user revised C4 to use the OpenRouter key's cap alone for the registered run (§13).

## 1. Why there is an Addendum C

- **The split result rests on two open models.** Study 2 found Qwen3.8 27B's split invariant within ±0.10
  and Gemma 4 31B's seller share falling by 0.256 at $100M. Addendum B showed the opening shift (HA) on three
  API models from three vendors, but tested first moves only, so it says nothing about their splits.
- **The midpoint account is post hoc.** In Study 2's games, deals landed near the midpoint of the two sides'
  first offers, so the split moved when the two openings shrank by different amounts (paper §5.3). That was
  found by looking at the data (exploratory). Addendum C tests it on games that have not been played yet.
- **Addendum B §11 said full games on these models, "if run, will be labelled exploratory".** That sentence
  stays as written. **C1 decided (2026-10-11):** register the full games as confirmatory under Addendum C,
  with the hypothesis hierarchy in §5. This is an explicit change from B §11's stated plan. Addendum C is
  a separate registration whose hypotheses were agreed before the format pilot; the paper will disclose
  both facts.
- **What is already known about these models:** their first moves (Addendum B, registered run
  `addB-20261010T191121Z`). That data shapes the stated expectations in §5.4, which are therefore not tests.
  No full-game bargaining outcomes have been inspected; the pilot checks are limited to §9.

## 2. Data this addendum is tested on

- **Only** the registered run of `scripts/run_addendum_c.py`: `data/runs/addC-<stamp>/`, fixtures F01–F30.
- **Not:** the format pilot (X01–X05, §9), any `--preview` run, any other run.
- If the run doesn't complete, the tests use the scored games and the shortfall is reported. A model that
  never runs is reported as not run.

## 3. Models

Exactly Addendum B's three arms (`scripts/run_addendum_b.py` `ARMS`, imported unchanged):

| Arm | Requested model | Pinned provider | Thinking off |
|---|---|---|---|
| gpt-6-luna | `openai/gpt-6-luna` | `openai` | `reasoning.effort: "none"` |
| claude-haiku-5.5 | `anthropic/claude-haiku-5.5` | `anthropic` | `reasoning.enabled: false` |
| gemini-3.5-flash-lite | `google/gemini-3.5-flash-lite` | `google-ai-studio` | `reasoning.effort: "minimal"` (lowest level; not guaranteed zero) |

- Routing, records and pinning as Addendum B §3: the vendor's own endpoint, no fallbacks,
  `require_parameters`, `data_collection: "deny"`; every call's returned model ID, dated version, provider,
  fingerprint, token counts and cost are logged. OpenAI's and Google's IDs are aliases the vendor may
  re-point; the dated versions returned are reported. **C5 decided (2026-10-11): same three models.**

## 4. Design

- **One game = exactly Study 2's game.** `scripts/game_engine.play_game` (unchanged) with the Study 2 renderer
  (`scripts/render_prompt_s2.py`, no appraisal): two copies of the same model, at most 16 messages, messages
  of at most 500 characters, strict JSON moves (OFFER with an integer price, ACCEPT, WALKAWAY), checked by
  `game_engine.validate_reply` with no repair. An invalid reply forfeits the game (PROTOCOL_FORFEIT), as in
  Studies 1–2.
- **Cells:** fixture F01–F30 (`config/fixtures.json`) × scale ($100, $100K, $100M) × opener (seller,
  buyer), with **k = 4 games per cell** (§7): 720 games per model, 2,160 in total. Own limits are f% / c% of
  V, as in Study 2. **C3 decided (2026-10-11): all three scales, k = 4.**
- **Order:** one shuffle with a fixed public label (`scale-invariance/addendum-c/order/v1`), so models,
  scales and openers interleave in time.
- **Seeds:** first 8 hex digits of SHA-256(`scale-invariance/addendum-c/v1|fixture|opener|sample|message|attempt`),
  masked to 31 bits. The same at all three scales (common random numbers). Sent only where the endpoint
  accepts a seed (OpenAI, Google); vendors don't promise determinism.
- **Requests:** each call sends only the Study 2 messages (one system, one user, rebuilt from scratch) plus
  Addendum B's arm settings. The engine's vLLM sampling fields and seed are never sent.
- **Infrastructure:** a call with no completed reply is retried up to 3 times with the identical request. If
  it still fails, the game attempt is void and is replayed from the start as attempt 2, then 3 (the attempt
  enters the seed), as in Studies 1–2 (data plan E8). After 3 void attempts the game is missing. A completed
  reply is never re-asked. A 4xx rejection other than 408/429 stops the run.

**Deviations from Study 2 and `prereg/settings.json`, stated so none is silent:**
- vendor sampling defaults; no seed on Anthropic; vendor-specific thinking switches; `max_tokens` 1,024
  (Study 2: 512), all as Addendum B;
- the schema's `minimum` and `maxLength` are checked client-side only (Addendum B §4). On vLLM the grammar
  prevented a message over 500 characters; here such a reply is generated and forfeits the game;
- k = 4 games per cell (Study 2: 1), so every per-fixture quantity uses cell means;
- interleaved order instead of Study 2's blocked order.

## 5. Hypotheses and tests

Notation: s is the seller share (P − F)/(C − F) of a game that reached a deal. For a game, m is the midpoint
of the first seller price and the first buyer price, as a share of the zone: ((P_S1 + P_B1)/2 − F)/(C − F).

### 5.1 H1 and H2: the split (primary; Study 2 §4–5 with cell means)

- **Per fixture:** d_io = mean s at $100M − mean s at $100 (deals only, cell means over up to k games);
  d_i = mean of the d_io that are defined. Fixtures with none are left out (counted).
- **H1:** two-sided Wilcoxon signed-rank on the d_i, 200,000 random sign flips, zeros dropped, average ranks
  (`analyze.py` method; seed what = `addendumC-wilcoxon-H1`). **Holm across the three models.**
- **H2:** "shift smaller than 0.10" if the 90% bootstrap CI of the mean d_i (10,000 resamples of fixtures;
  what = `addendumC-bootstrap-H1`) lies inside (−0.10, +0.10).
- **Reading:** Study 2 §5.4, unchanged: shift / invariant within ±0.10 / inconclusive / real but small.

### 5.2 HM: the split moves with the first-offer midpoint (confirmatory, new)

- **Per fixture**, on deals in which both sides made an offer (a deal at message 2 has no midpoint; counted):
  e_i = mean over openers of [(mean s at $100M − mean s at $100) − (mean m at $100M − mean m at $100)],
  both terms on the same games.
- **HM1 (calibration):** the 90% bootstrap CI of the mean e_i lies inside **(−0.05, +0.05)** (what =
  `addendumC-bootstrap-HM1`). **C2 decided (2026-10-11): margin ±0.05**, half the split's ±0.10, so the
  account explains a shift that matters with an error well below it.
- **HM2 (tracking):** Spearman correlation across fixtures between the share shift and the midpoint shift
  (same games) > 0; one-sided permutation test, 20,000 permutations (what = `addendumC-perm-HM2`); **Holm
  across the three models.**
- **HM is supported on a model if HM1 and HM2 both hold.**
- **What HM can and can't show.** A deal price lies between the standing offers, so HM2 is partly mechanical;
  HM1 is the substantive part (concessions are symmetric enough that the deal lands on the midpoint, on
  average, at both scales). HM is still an accounting identity's test, not a causal one: it doesn't show that
  the openings cause the split. The causal test (clamping the openings) stays in Study 3.

### 5.3 Secondary (registered, labelled secondary)

- **HA in full games:** Addendum B's gap of the opener's first offer, cell means, h_f as Addendum B §5;
  one-sided Wilcoxon (what = `addendumC-wilcoxon-HA`), Holm across three. A near-replication of Addendum B
  in a new run (message 1 is the same prompt; the seeds differ).
- **HB (opener advantage), as Addendum A:** I_f = [s̄(seller opens, $100M) − s̄(seller opens, $100)] −
  [s̄(buyer opens, $100M) − s̄(buyer opens, $100)], fixtures whose four cells have a deal; two-sided Wilcoxon
  (what = `addendumC-wilcoxon-HB`), Holm across three.
- **Study 2's S1, S2, S5, S6** with cell means: S1 ($100K vs $100; $100M vs $100K), S2 (agreement rate),
  S5 (coarse-price check), S6 (seller utility, every game). Holm within each model, as Study 2.
- **Not carried over:** S3 (opener disagreement) and S4 (replaced by HA above).

### 5.4 Stated expectations (not tests)

Written before any game, from Addendum B's frozen first moves, so they can be compared with the result but
can't be confirmed by it. If both sides opened at their Addendum B first-move gaps and the deal landed at the
midpoint, the expected d is:

| Model | Expected mean d | Fixtures expected below 0 |
|---|---|---|
| gpt-6-luna | −0.066 | 18 of 30 |
| claude-haiku-5.5 | −0.089 | 16 of 30 |
| gemini-3.5-flash-lite | −0.333 | 25 of 30 |

These are weak guides. On Study 2's own games the same opener-only calculation gave Gemma −0.155 against the
observed −0.256 (fixture correlation 0.06) and Qwen +0.095 against +0.009 (0.61), because a responder's first
offer differs from an opener's (Gemma's buyer gap as responder 0.18 → 0.24; as opener 0.45 → 0.36). HM uses
the in-game first offers for that reason. On Study 2 (exploratory, the source of the account): Gemma mean e
+0.000, 90% CI [−0.024, 0.026]; Qwen −0.035 [−0.064, −0.007], which would fail HM1 at ±0.05 and pass at ±0.10
(both printed by `scripts/power_addendum_c.py`).

## 6. Invalid replies, refusals and quality flags

- **As Studies 1–2:** an invalid reply (any `validate_reply` error, incl. TRUNCATED and MESSAGE_TOO_LONG), a
  refusal (no content) or a content-filter stop forfeits the game for the side that sent it. Forfeits count
  in S6 (utility −1 for that side) and are excluded from s, d and e.
- **Reported per model and scale:** games, terminal states, forfeits by error code, refusals, calls with
  reasoning tokens, mean messages, ZOPA violations, deals outside the zone, void attempts, missing games.
- **Flags:** "format-limited" if more than 25% of a model's games end in a forfeit (Study 1 §8); "scale-
  dependent forfeits" if the forfeit share differs by more than 10 points between $100 and $100M. A flag
  qualifies the reading; it doesn't change the test.
- **Gemini's residual thinking:** kept in the primary test; a sensitivity analysis without games in which any
  call used reasoning tokens is reported next to it. For the other two, any reasoning tokens are a reported
  deviation.

## 7. Samples and power

From `scripts/power_addendum_c.py` (`logs/power_addendum_c_20261011.txt`), 30 fixtures, exact Wilcoxon at
α 0.05/3. Study 2 has k = 1 only, so the split of d_i's spread into a fixture part and a game part (which
shrinks with k) is bounded, not measured: the game part is at most half the SD of the two openers' d_io
difference.

| Spread like… | k | H1 power at a shift of 0.08 | 0.10 | 0.15 | Chance of H2 (±0.10) if the true shift is 0 |
|---|---|---|---|---|---|
| Gemma, game part at the bound | 1 / 4 | 0.82 / 1.00 | 0.95 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 |
| Qwen, game part at the bound | 1 / 4 | 0.16 / 0.29 | 0.26 / 0.46 | 0.60 / 0.85 | 0.22 / 0.58 |
| Pessimistic (2 × Qwen) | 1 / 4 | 0.04 / 0.08 | 0.07 / 0.10 | 0.14 / 0.25 | 0.00 / 0.00 |

- **Reading:** if these models bargain as consistently as Gemma, k = 4 detects shifts well under 0.08; if as
  noisily as Qwen, only shifts of about 0.15, and "invariant" is about a coin flip. The study is not
  well powered against a Qwen-like spread at any affordable k: k = 4 is where the gain per dollar flattens.
- **HM1 at the decided margin ±0.05:** chances of showing equivalence below use Study 2's spread of e_i
  (0.085–0.094), or twice that spread. True e = −0.035 is Qwen's Study 2 mean (rounded).

  | Spread of e_i like… | k | Chance of HM1 if true e = 0 | If true e = −0.035 (Qwen) |
  |---|---|---|---|
  | Gemma | 1 / 4 | 0.88 / 1.00 | 0.27 / 0.64 |
  | Qwen | 1 / 4 | 0.80 / 1.00 | 0.23 / 0.56 |
  | 2 × Gemma | 1 / 4 | 0.06 / 0.88 | 0.03 / 0.26 |
  | 2 × Qwen | 1 / 4 | 0.01 / 0.79 | 0.01 / 0.24 |

  These HM1 scenarios treat all spread as game-level, so the gains from k are optimistic; a fixture-level
  part would not shrink with k. The log also gives ±0.10 for comparison only.
- These assume normal effects and no ties; H2 and HM1 use normal-theory intervals as a stand-in for the
  registered bootstrap. Read them as optimistic.

## 8. What counts as what

- **The split, per model:** read by Study 2 §5.4. Across the panel: "the split moves on m of the five models"
  is reported together with Studies 1–2, with Qwen and Gemma from Study 2 and these three from Addendum C.
- **HM supported on all three:** the midpoint account holds on new models and new games; the paper's §5.3
  becomes a registered result on three models (and stays exploratory on Gemma and Qwen).
  **On one or two:** reported as model-dependent, naming which. **On none:** the account is reported as a
  description of the two open models only.
- **In every case:** the claims are limited to these three models, at the tested setting, accessed on the
  run dates; none of this changes the Study 2 primary reading or Addenda A and B.

## 9. Format pilot (before freezing; excluded from every test)

- **What runs:** `python3 scripts/run_addendum_c.py --pilot --confirm-cost 1.00` on fixtures **X01–X05**
  (`make_fixtures.build("scale-invariance/fixtures/explore-v1")`), $100 and $100M, both openers, one game per
  cell: 60 games, about 450 calls. Spend cap $1.00.
- **Gate, per model:** every call got a completed reply (no void attempt); at least 95% of replies valid
  (Study 1 gate 5); the vendor's own provider; no reasoning tokens (gpt-6-luna, claude-haiku-5.5; Gemini's
  count recorded).
- **What is looked at:** format counts, plus the pooled mean calls and cost per game (for the cost estimate
  in §10). No share, opening, midpoint or per-scale number is computed from the pilot. The analysis script
  refuses a real pilot run.
- **If a model fails:** only the thinking switch, `max_tokens` and the schema may change, by an amendment in
  §13 before the freeze, and the pilot reruns for that model. The prompt doesn't change. A model that can't
  pass is reported as not run.

## 10. Cost, limits, data handling

- **Expected from the format pilot:** about **$4.02** for the registered run, from 720 games per model ×
  that model's pooled mean cost per pilot game. The pilot cost **$0.111585** for 60 games and 572 replies
  (§13; `logs/pilot_addendum_c_20261011.txt`). This replaces the pre-pilot estimates of $4.5 for the run
  and $0.15 for the pilot.

  | Model | Pilot games | Mean calls per game (pooled) | Mean cost per game | Estimate for 720 games |
  |---|---|---|---|---|
  | gpt-6-luna | 20 | 13.65 | $0.001666 | $1.20 |
  | claude-haiku-5.5 | 20 | 8.10 | $0.001499 | $1.08 |
  | gemini-3.5-flash-lite | 20 | 6.85 | $0.002415 | $1.74 |

  The same extrapolation gives about 20,592 calls. It assumes the pooled pilot averages carry over to
  F01–F30 and all three scales; the pilot uses X01–X05 and the two extreme scales. It is an estimate, not
  a spending bound.
- **Worst cases** (dry runs, every call billed at 6,000 input + 1,024 output tokens): pilot $2.11, registered
  run $75.85 if every game ran all 16 messages, before retries. These are token-cost calculations; spending
  is limited as below.
- **Spend controls:**
  - **Pilot:** a hard $1 client cap; each try reserves its worst-case cost first. The key's remaining credit
    must cover the pilot cap before any prompt is sent.
  - **Registered run:** the OpenRouter key's cap is the sole spending limit. There is no registered
    dollar cap in the client and no requirement to have $15, or the theoretical worst-case cost, available.
    Before starting, the runner checks that the key has a finite positive limit and positive remaining
    credit, and records its limit, usage and remaining credit in the run manifest. Costs are still logged.
    The last observed key limit was $10 at the pilot's preflight check ($0.495041 used before the pilot).
  - **Command after the freeze:** `python3 scripts/run_addendum_c.py --confirm-cost key-limit`.
    `key-limit` explicitly acknowledges that the main run relies on the key's spending limit.
- **A key-limit rejection (HTTP 402) stops the run** under §4's existing 4xx rule; it is not retried.
  The run is incomplete and §2 applies (the scored games are tested, the shortfall reported).
- **API costs are separate from the GPU budget.** Data handling as Addendum B §10.

## 11. Not in this addendum

- Thinking-on arms; flagship models; other vendors.
- A second prompt (Study 1's appraisal prompt): Gemma's split moved under Study 2's prompt, so that is the
  like-for-like comparison.
- The causal test of the midpoint account (clamping the openings) and any interpretability work: Study 3.

## 12. Code (listed with SHA-256 in `prereg/addendum_C/inputs.sha256`)

- `scripts/run_addendum_c.py` (new), `scripts/analyze_addendum_c.py` (new), `scripts/power_addendum_c.py` (new)
- reused unchanged: `scripts/run_addendum_b.py` (arms, response parsing, spend cap), `scripts/openrouter_client.py`,
  `scripts/analyze.py`, `scripts/game_engine.py`, `scripts/run_records.py`, `scripts/render_prompt.py`,
  `scripts/render_prompt_s2.py`, `scripts/make_fixtures.py`, the Study 2 prompt files, `config/fixtures.json`,
  `prereg/settings.json`
- data read by the analysis: `data/runs/addB-20261010T191121Z/games.jsonl` (§5.4 expectations only)
- tests: `tests/test_addendum_c.py` (31 offline tests) and `tests/mutation_check_addendum_c.py` (20 of 20
  planted flaws caught), both run 2026-10-11 after the budget amendment; `tests/test_addendum_b.py` still passes
- power evidence: `logs/power_addendum_c_20261011.txt` (§7), included in this freeze
- format and cost evidence: `logs/pilot_addendum_c_20261011.txt` (§13), included in this freeze
- real API calls so far for this addendum: the format pilot only (§13); no registered games

## 13. Amendments

- Pre-freeze amendments are dated below and committed with this freeze, as authorised by the user.
- After it: no edits. Deviations are reported.

**Pilot 1 (2026-10-11; committed with the freeze).**
- User-authorised command: `python3 scripts/run_addendum_c.py --pilot --confirm-cost 1.00`.
- Run: `data/pilot/addC-pilot-20261011T180113Z`, 60 of 60 games completed, 572 replies, $0.111585 spent;
  no missing games, no void attempts, no unpriced replies.
- All three models passed §9's format gate on the vendor's own provider and returned the dated versions
  listed in the pilot log. No reply had reasoning tokens.
  - gpt-6-luna: 271 of 273 replies valid (99.27%); two `TEXT_AFTER_JSON` forfeits.
  - claude-haiku-5.5: 162 of 162 replies valid (100%).
  - gemini-3.5-flash-lite: 137 of 137 replies valid (100%).
- Only format counts and pooled calls/cost per game were inspected. No share, opening, midpoint or
  per-scale result was inspected. All pilot games remain excluded from the registered tests.
- The pilot passed without any adjustment to the prompt or model settings. Its initial cost report used
  the then-proposed $15 registered-run cap; the subsequent budget amendment below supersedes that policy.

**Budget amendment (2026-10-11, after the format pilot; before the freeze).**
- At the user's instruction, remove the proposed $15 registered-run client cap and the requirement to
  raise the key's remaining credit to $15. The registered run relies on the existing OpenRouter key cap.
- The runner now uses `--confirm-cost key-limit` for the main run, checks and records the capped key's
  available credit, and keeps a cost counter without a client-side stop threshold. HTTP 402 still stops
  the run. The pilot's $1 client cap and numeric confirmation remain in place.
- Only spending controls, their records, the CLI and associated offline tests changed. The design,
  prompts, model settings, scoring and hypothesis tests are unchanged. No further API call was authorised
  by the request for a command; no registered game has run and no bargaining outcome was inspected.
- Verification: 31 offline tests passed; 20 of 20 mutation checks caught their planted flaws. Addendum B's
  tests passed (41 passed, one expected skip), and the paper-build tests passed (8). Frozen B inputs match.

## 14. Decisions and authorisation

- **C1. DECIDED 2026-10-11 — Addendum B §11.** Register the full games as confirmatory under Addendum C,
  with the hierarchy in §5, and openly disclose the change from B §11's exploratory designation in the paper.
- **C2. DECIDED 2026-10-11 — HM1 margin.** ±0.05. Study 2's Qwen would fail it, Gemma pass.
- **C3. DECIDED 2026-10-11 — Scales and k.** All three scales with k = 4: 720 games per model, 2,160 total.
- **C4. REVISED 2026-10-11 — Key limit.** Use the OpenRouter key's existing cap as the sole registered-run
  spending limit. Remove the $15 client cap and the required key-limit increase (§13).
- **C5. DECIDED 2026-10-11 — Models.** The same three as Addendum B, with its routing and settings.
- **Pilot authorisation (2026-10-11):** the user approved the paid format pilot if its estimated cost is
  below $10. The estimate is about $0.15; the command in §9 keeps the hard cap at $1.
