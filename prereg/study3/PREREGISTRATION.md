# Pre-registration, Study 3: one protocol, five models, the appraisal, and an opening clamp (v3)

**Status: FINAL TEXT for approval, 2026-10-14 (draft v3). NOT registered, NOT frozen.** All three format
pilots passed (§6).
Supersedes the 10-04 draft (commit `066d136`; still in git history). Nothing in Study 3 runs until the author
approves this text, it is frozen with its own commit and `prereg/study3/inputs.sha256`, and an external
timestamp on OSF Registries is recorded.

---

## 0. Why there is a Study 3

Studies 1–2 and Addenda A–C answered their registered questions, but piecemeal: different models saw
different conditions (Gemma 4 31B and the API models never saw the appraisal), and the protocol changed
between runs (one game per cell vs four, blocked vs interleaved order, 512 vs 1,024 output tokens, grammar
constraints). Their results, and the exploratory probes, generated the hypotheses below. Study 3 tests them
**once, under one protocol, on all five models, on fixtures no run has used**. In the paper, Studies 1–2,
Addenda A–C and the probes become the discovery phase; Study 3 is the confirmatory evidence.

**What earlier data motivate (not tested here):**
- Opening modesty at $100M on all five models (Qwen exploratory; Gemma, Luna, Haiku, Gemini confirmed).
- Without the appraisal, closing splits shift on Gemma, Luna and Gemini and meet ±0.10 equivalence on Qwen
  and Haiku.
- With the appraisal and the limits' ranges shown together, Qwen showed neither opening shrinkage nor a
  split shift (Study 1; probe X10). Study 3's A1 shows the appraisal alone (§3.1), so this is motivation,
  not a like-for-like precedent.
- Deals close near the midpoint of the two first offers; the midpoint account had prospective support on Luna
  only (Addendum C), and Gemma's information probe left a concession residual (X26–X27).

**Reporting commitment (unchanged).** Every registered test is reported in full, nulls and failures
included. No test, model, cell or session is dropped, relabelled or reweighted because of its outcome.
Discrepancies with earlier studies are reported as such.

---

## 1. Models (pinned before data)

| Arm | Backend | Weights / ID | Sampling | Thinking |
|---|---|---|---|---|
| Qwen3.8 27B | vLLM 0.30.0 (image digest as Studies 1–2) | `Qwen/Qwen3.8-27B-FP8`, `config/pins.json` | family values (T 0.7, top-p 0.8, top-k 20) | off (`enable_thinking false`) |
| Gemma 4 31B | vLLM 0.30.0 | `google/gemma-4-31B-it` bf16, `config/pins.json` | family values (T 1.0, top-p 0.95, top-k 64) | off |
| gpt-6-luna | OpenRouter, OpenAI endpoint only | requested alias; returned `openai/gpt-6-luna-20260922` expected | vendor default | effort "none" |
| claude-haiku-5.5 | OpenRouter, Anthropic only | returned `anthropic/claude-haiku-5.5-20261007` expected | vendor default | disabled |
| gemini-3.5-flash-lite | OpenRouter, Google only | returned `google/gemini-3.5-flash-lite-20260721` expected | vendor default | "minimal" |

Presence/frequency penalties 0 and repetition penalty 1 on vLLM. If an API alias returns a dated version other
than the one expected, the run continues and the returned version is reported (it is the tested model).
Sampling stays model-native: "one protocol" means every procedural rule below is identical, not that
sampling is.

---

## 2. Fixtures

- **Fresh set:** 40 fixtures from the public phrase `scale-invariance/fixtures/study3-v1` via
  `scripts/make_fixtures.py` (n raised to 40), IDs `G01–G40`. No earlier run uses this phrase.
  n = 40 (decided 10-13).
- **Geometry unchanged:** floor at 75–95% and ceiling at 105–125% of V (whole percentages), so the results are
  comparable with Studies 1–2.
- **Scales:** V = $100, $100K, $100M (`x1e2`, `x1e5`, `x1e8`). Money is always "`<digits> dollars`", the JSON
  price a bare integer; masking the digits leaves every scale's templates identical.

---

## 3. Design

### 3.1 Core grid (all five models)

Self-play, fully crossed: **appraisal (A1 shown / A0 not shown) × 3 scales × 2 openers × 40 fixtures × k = 4
games**, 1,920 games per model.

- **A0** uses Study 2's frozen templates (`prereg/study2/prompts/`): each side knows its own limit and that a
  deal is possible.
- **A1** adds one line and nothing else: "An independent appraisal values the lot at {{V}}. Both sides know
  this." (`prereg/study3/prompts/system_a1.txt`; Study 2's system message with that line inserted where
  Study 1 had it; user message and fragments are Study 2's). **No limit ranges are shown in either
  condition** (decided 10-13), so A1 − A0 isolates a single public, scale-matched reference number.
- Where the appraisal falls in the zone, (V − F)/(C − F), follows from each fixture's limits (0.21–0.79 on
  G01–G40, median 0.48); it is the same in A0 and A1, so the contrast is within fixture. It is reported per fixture and
  used as a pre-declared moderator in S9.
- Consequence: A1 is not Study 1's prompt (Study 1 showed the appraisal and both ranges). Comparisons with
  Study 1 are descriptive only (S8).

### 3.2 Opening clamp (all five models, A0 only)

Both first offers are scripted into the public history and the model plays both sides from message 3 on.
The design is a within-fixture 2 × 2: **game scale G ∈ {$100, $100M} × opening source O ∈ {$100, $100M}**,
× 2 openers × 40 fixtures × k = 4, 1,280 games per model.

- **Scripted offers.** Per model, role of the opener and source scale, the positions of the seller's and the
  buyer's first offers in zone units, x = (P − F)/(C − F) (floor 0, ceiling 1), are fixed in advance from that
  model's earlier no-appraisal games (Study 2 for Qwen and Gemma; Addendum C for the API models), as means over
  games containing both first offers. Scripted price = F + x·(C − F), rounded half up to a multiple of V/100
  (whole dollars at $100, $1M at $100M), so a fixture's scripted $100M prices are its $100 prices with six zeros
  appended, like every other amount. Values: `prereg/study3/clamp_values.json`, computed by
  `scripts/study3_clamp_values.py` (Appendix T); they are inputs, not results.
- **Why zone units (changed 10-13, before any Study 3 game).** Own-limit gaps (seller P/F − 1, buyer 1 − P/C),
  transplanted onto narrow fixtures, made the buyer's scripted counter exceed the seller's ask in 42 of the
  clamp games (for example Qwen seller-opened at $100M). Zone positions keep every scripted pair ordered
  (checked for every arm, fixture, cell and opener of the pilot and main sets) and are the units of the split
  and of the midpoint account.
- **Scripted text.** One neutral template per role at every scale and cell: seller "I can offer the lot at
  {P}.", buyer "I can pay {P} for the lot." (`prereg/study3/prompts/clamp.txt`, with {P} written "N dollars").
  The rendered history line is identical in form to a model move ("Message 1, you: OFFER at …"); the opener is
  attributed message 1. No request is sent for messages 1–2; the record lists them as scripted.
- **Interpretation.** Cells (G, O) = ($100, $100) and ($100M, $100M) reproduce each scale's own openings;
  the off-diagonal cells transplant one scale's openings into the other scale's game.
  - *Opening channel* ω_i: mean over G of s(G, O = $100M) − s(G, O = $100).
  - *Concession channel* κ_i: mean over O of s(G = $100M, O) − s(G = $100, O): the scale effect with both
    openings held at the same relative positions.
  - *Scripting artefact*: the diagonal clamp cells against the natural A0 core games at the same scale.
- Source of the scripted values: earlier data (decided 10-13), so every scripted value is frozen in this
  document before any Study 3 game and clamp cells can be interleaved with the core grid.

### 3.3 Run organisation

- **Two independent sessions** per model: samples 1–2 in session 1, samples 3–4 in session 2. Different
  pods for the open-weight models; different days for the API models. Each session interleaves every core and
  clamp cell of its models in one fixed shuffle (`scale-invariance/study3/order/v1|<backend>|<session>`).
- **Format pilot** per model before freezing: fixtures Q01–Q05 from `scale-invariance/fixtures/study3-pilot`,
  k = 1, cells A0 and A1 at $100 and $100M plus K-$100M-o$100 (50 games per model). Judged on format only
  (§6); excluded from every test. The vLLM pilots also measured throughput at 16 concurrent games (Qwen
  2,189 games/h, Gemma 1,579 games/h, 10-14), which sets the sessions' pod lifetime (§8).

---

## 4. Protocol (identical for every arm)

| Item | Study 3 rule |
|---|---|
| Engine | `scripts/game_engine.py` unchanged. The clamp's two scripted moves are answered by the call function itself (`scripts/study3.py`); the replay audit (`study3.py audit`) re-derives them from the frozen values and rebuilds every request byte for byte |
| Messages | at most 16 per game, each call rebuilt from the rules, own limit and public history |
| Reply | one JSON object; `max_tokens` 1,024; message ≤ 500 characters |
| Response grammar | The same schema for every arm (Addendum C's, without `minimum`/`maxLength`), with identical client-side validation (decided 10-13). Both open-weight models passed the pilot format gate under it (§6), so no backend-specific fallback grammar is used |
| Invalid replies | forfeit, never repaired (`validate_reply`, unchanged), with one exception (decided 10-14, after the format pilot, §6): non-whitespace text after the first JSON object is ignored (`study3.first_json_only`), and the first object is checked by `validate_reply` in full. Nothing in the object is edited; a reply that doesn't start with a complete object, or is truncated, still forfeits. Each ignored tail is flagged on its call and counted per game (`trailing_text_calls`); the raw response is kept whole. The other side sees only action, price and message, as always |
| Seeds | `sha256('scale-invariance/study3/v1\|fixture\|opener\|sample\|msg_no\|attempt')[:8] & 0x7fffffff`. Scale, appraisal and clamp cell are not inputs, so every within-fixture contrast shares seeds. Not sent to Anthropic |
| Concurrency | a fixed number of concurrent games per backend: vLLM 16, API 8 (confirmed by the pilots, 10-14), recorded in the manifest. Studies 1–2 ran one game at a time; batching makes vLLM outputs non-bit-identical to sequential runs, which no registered result depends on |
| Retries, voids | 3 tries per call; a void game is replayed up to 3 attempts; non-retryable 4xx stops the run (as Addendum C) |
| Records | hash-chained calls/games/events, run manifest with commit hash and `inputs_sha256`, replay audit after every session |

---

## 5. Hypotheses and tests

Analysis unit: the fixture. With k = 4, every cell's outcome is the mean over its scored games (Addendum C).
Seller share s = (P − F)/(C − F), unclipped. Bootstrap CIs resample fixtures 10,000 times; Wilcoxon p-values
come from 200,000 random sign flips; Holm correction runs across the five models within each test below.
α = 0.05.

### 5.1 Primary

**P1. The appraisal attenuates opening modesty (openings, appraisal × scale).**
- Statistic: per fixture, h_f(A) = mean over openers of g($100M) − g($100), with g the opener's first-offer
  gap from its own limit (Addendum A's HA); then I_f = h_f(A1) − h_f(A0).
- Test: one-sided Wilcoxon, I > 0, per model, Holm across models.
- Prediction: I > 0 on all five models.

**P2. Closing shift or equivalence, per model and appraisal condition.**
- Statistic: d_i = s($100M) − s($100), mean over openers with agreements at both scales (Studies 1–2, C).
- H1: two-sided Wilcoxon, Holm across models within each condition. H2: equivalence if the 90% CI of mean d
  lies within ±0.10. A non-significant H1 is never read as equivalence.
- Predictions: A0 replicates the earlier pattern (shift on Gemma, Luna, Gemini; equivalence on Qwen and
  Haiku). A1: equivalence on all five.

**P3. The opening clamp decomposes the closing shift (A0).**
- P3a, concession channel: κ_i (§3.2). Two-sided Wilcoxon (Holm across models) and equivalence within
  ±0.05 (90% CI). "Openings suffice" predicts κ equivalent to 0.
- P3b, opening channel: ω_i compared with the scripted midpoint difference Δm_i, the mean over openers and
  game scales of m(O = $100M) − m(O = $100), m = ((P_S1 + P_B1)/2 − F)/(C − F) on the scripted prices
  (Appendix T). Equivalence of mean (ω − Δm) within ±0.05 (90% CI). "Deals follow the scripted midpoint"
  predicts equivalence.
- Per opener, κ and ω use only fixtures whose four clamp cells all have agreements; the fixture value averages
  the openers that qualify.
- Either outcome is informative; a non-zero κ locates part of the scale effect in the concessions.

### 5.2 Registered secondaries (labelled secondary in the paper)

- **S1** HA replication: h(A0) < 0, one-sided Wilcoxon, Holm across models; h(A1) equivalence within ±3
  points.
- **S2** Adjacent scales ($100K − $100, $100M − $100K), both conditions, Holm within model.
- **S3** Agreement-rate contrast; all-game seller utility (failed deals 0, forfeits −1); coarse-grid prices at
  $100 (as S5/S6 before).
- **S4** HB: opener advantage at $100M vs $100, two-sided, Holm across models.
- **S5** HM in the natural A0 games: tracking (Spearman, 20,000 permutations, Holm) and calibration (±0.05),
  as in Addendum C.
- **S6** Scripting artefact: diagonal clamp cells vs natural A0 core games, per scale.
- **S7** Session consistency: every primary statistic estimated per session, with the between-session
  difference and its CI. Descriptive; a difference is reported, not resolved.
- **S8** Consistency with earlier studies: Study 3 A0 estimates vs Study 2 (Qwen, Gemma) and Addendum C (API),
  where the prompt is identical; A1 vs Study 1 (Qwen) is reported with the caveat that Study 1 also showed the
  ranges. Unpaired fixture bootstrap of the difference. Descriptive.
- **S9** Appraisal position: in A1, the association across fixtures between (V − F)/(C − F) and seller share,
  per scale (Spearman, 20,000 permutations), and whether A1's closing shift varies with it. Exploratory-labelled
  moderator, pre-declared.

---

## 6. Gates, invalid replies and quality

- **Pilot format gate** per model: ≥ 95% valid replies, no reasoning text, no reasoning tokens, returned
  providers and versions as expected, logged sampling settings equal to the request (vLLM). A model that fails
  is reported as not run, not amended away. One exception, as in Addenda B–C: gemini-3.5-flash-lite runs at
  `reasoning.effort "minimal"`, its lowest level, which Google does not guarantee to be zero, so its reasoning
  tokens are counted and reported (per cell, in every session) but do not fail the gate. Reasoning text fails
  the gate for every model.
- **Format pilots (10-14), all passed.** API (rerun under the trailing-text rule below): 150 games, 0 forfeits;
  valid replies Luna 100%, Haiku 100%, Gemini 100%; reasoning tokens and text 0 on all three, Gemini included;
  returned versions as in §1; trailing text ignored on 11 Luna replies (10 games), none on Haiku or Gemini.
  Qwen: 50 games, 100% valid, 0 forfeits. Gemma: 50 games, 100% valid, 0 forfeits. Replay audit exact on all
  runs.
- **Registered quality flags** (reported, never used to drop data): more than 25% forfeits for a model, or a
  difference of more than 10 percentage points in forfeits between extreme scales. Ignored trailing text
  (§4) is reported per model and cell.
- **Why trailing text is ignored** (decided 10-14). In the API format pilot, 13 of Luna's 50 games forfeited
  (none on Haiku or Gemini): 12 replies were a complete, valid move followed by junk (chat-template tags, text
  in other scripts) and then the same move again, and one was truncated. In every one of the 12, and in 93 of
  94 such replies in Addendum C, the repeated move had the same action and price as the first, so the first
  object is the intended move. All 12 pilot replies, and all 87 Addendum C replies that failed only on
  trailing text, pass `validate_reply` once the tail is ignored. The prompt already asks for exactly one JSON
  object; the tails are not a response to it, so the prompt is unchanged. Under the strict rule, one such reply
  ends a game, and Luna's long games (12.6 calls on average in the pilot) would lose roughly a quarter of the
  closings P2 and P3a need, possibly unevenly across scales (pilot: 3/20 at $100, 10/30 at $100M).
  Luna's Study 3 forfeit rate is therefore not comparable with Addendum C's, and no comparison is made.
- Out-of-zone agreements are retained and scored unclipped, as in Addendum C.

---

## 7. Samples and power

`scripts/power_study3.py` (run 10-13; `logs/power_study3.json`) takes each arm's per-fixture SDs of the earlier
analogue of every primary statistic and projects the 90% CI half-width at n = 40 fixtures, 1.645 · SD / √40.
P1's interaction is taken as √2 · SD(h); P3a's κ as SD(d); P3b as SD(e), Addendum C's closing-minus-midpoint
shift. Open-model SDs come from one game per cell and overstate Study 3's spread at k = 4.

| Arm | SD d | P2 half-width | P1 half-width | P3a half-width | SD e | P3b half-width |
|---|---|---|---|---|---|---|
| Qwen3.8 27B | 0.290 | 0.075 | 0.026 | 0.075 | 0.094 | 0.024 |
| Gemma 4 31B | 0.124 | 0.032 | 0.022 | 0.032 | 0.085 | 0.022 |
| gpt-6-luna | 0.070 | 0.018 | 0.020 | 0.018 | 0.057 | 0.015 |
| claude-haiku-5.5 | 0.076 | 0.020 | 0.027 | 0.020 | 0.117 | 0.030 |
| gemini-3.5-flash-lite | 0.200 | 0.052 | 0.020 | 0.052 | 0.099 | 0.026 |

H2's ±0.10 has room on every arm. P3a's ±0.05 is within reach except, on these projections, Qwen (0.075,
from k = 1 data) and Gemini (0.052): for them a failure to establish κ-equivalence would be expected even at
κ = 0 and is reported as inconclusive, not as a concession channel. n = 40 was fixed by the author on 10-13.

---

## 8. Size and cost (estimates; the worst cases are printed by the runners)

- Games per model: 1,920 core + 1,280 clamp = 3,200; 16,000 in all (9,600 API, 6,400 vLLM); plus the pilots
  (150 API games, 50 per vLLM model).
- API: the re-pilot cost $0.286 for 150 games ($0.0019 per game; Luna's games run longer once trailing text no
  longer ends them), so about $9–10 per session and about $20 in all. Sessions are capped by the OpenRouter key
  only; the pilot has a $2 client cap.
- GPU: 1,600 games per model per session. The pilots measured 2,189 (Qwen) and 1,579 (Gemma) games/h at 16
  concurrent (10-14), so a session plays in about 45 and 61 minutes. The
  runner sets each pod's lifetime from a setup allowance sized to the model's weights at 20 MB/s plus the games
  at 1,000/h (Qwen 132 min, Gemma 159 min per session; worst cases about $6.3 and $7.4; expected cost about $2.5
  and $3.5). Pods are placed in North American data centers that stock the GPU (launcher `--dc us`; same GPU
  type, image digest and weight hashes everywhere; the data center and download speed are logged per launch).
  Every launch passes the launcher's budget gate (`config/budget.json`) and prints its worst case, which the
  author types back.

---

## 9. What Study 3 does not test

- Thinking or reasoning modes; models outside the five.
- Mechanisms: the interpretability analyses stay exploratory and on Gemma 4 31B; Study 3 supplies the matched
  behavioural contrast (A1 vs A0) for that model.
- The limits' ranges: they are shown in no condition. Whether information about the other side (as in
  probes X23–X24) acts like the appraisal is out of scope.
- Prompt families or games other than this one.

---

## 10. Code and frozen inputs (`prereg/study3/inputs.sha256`)

Written and tested offline before freezing (`tests/test_study3.py`, 20 tests: fake servers only, no network,
no money). Every file below is hashed in `prereg/study3/inputs.sha256`:
- `scripts/study3.py`: fixtures, cells, games, seeds, the clamp prefix, request bodies, one game, the trailing-text
  rule, the replay audit;
- `scripts/run_study3.py`: both backends, sessions, interleaving, concurrency, resume, pilot gate, pod lifetime;
- `scripts/analyze_study3.py`: every test in §5 (built on `analyze.py` and `analyze_addendum_c.py`);
- `scripts/render_prompt_s3.py`, `prereg/study3/prompts/system_a1.txt`, `prereg/study3/prompts/clamp.txt`;
- `scripts/study3_clamp_values.py`, `prereg/study3/clamp_values.json`, `scripts/power_study3.py`;
- `tests/test_study3.py`; this document.

Every other file a Study 3 game, launch or analysis executes or reads is hashed in the same list, so the frozen
state does not depend on earlier hash lists (which do not cover all of them):
- unchanged since earlier studies: `game_engine.py`, `run_records.py`, `render_prompt.py`, `render_prompt_s2.py`,
  `make_fixtures.py`, `openrouter_client.py`, `vllm_client.py`, `run_addendum_b.py`, `run_addendum_c.py`,
  `replay_audit.py`, `analyze.py`, `analyze_addendum_c.py`, `lib/pod_ssh.sh`, `lib/pack_interp.sh`,
  `pod/boot.sh`, `connections/03_pod.sh`, `connections/04_vllm.sh`, Study 2's prompts, `prereg/settings.json`,
  `config/pins.json` and the two open-weight manifests;
- changed on 10-14, before freezing, for infrastructure only: `launch_pod.sh` (the `--dc` data-center option and
  the download-speed log) and `run_job.py` (shows those lines in the progress panel). Neither change touches a
  request, a reply, a score or a seed.

The runner refuses to start a registered session unless the hash list is committed and clean (and, on vLLM,
scripts/, config/ and prereg/ are committed).

---

## 11. Amendments

Dated entries, each in its own commit, until the first registered game. After that, no edits; deviations are
reported.

---

## Appendix T. Scripted first-offer positions (`scripts/study3_clamp_values.py`, 10-13)

Positions in zone units (floor 0, ceiling 1); midpoint = (x_S + x_B)/2. Means over each arm's earlier
no-appraisal games with both first offers. `prereg/study3/clamp_values.json` is the frozen source; this table
is a copy.

| Model | Opener | Source scale | Seller's position x_S | Buyer's position x_B | Scripted midpoint | n games |
|---|---|---|---|---|---|---|
| qwen3.8-27b-fp8 | seller | $100 | 0.7467 | 0.0126 | 0.3797 | 30 |
| qwen3.8-27b-fp8 | seller | $100M | 0.3981 | -0.1021 | 0.1480 | 29 |
| qwen3.8-27b-fp8 | buyer | $100 | 0.7040 | -0.5278 | 0.0881 | 30 |
| qwen3.8-27b-fp8 | buyer | $100M | 0.5891 | 0.0613 | 0.3252 | 30 |
| gemma-4-31b-it | seller | $100 | 1.3084 | 0.2420 | 0.7752 | 30 |
| gemma-4-31b-it | seller | $100M | 0.6352 | 0.0856 | 0.3604 | 30 |
| gemma-4-31b-it | buyer | $100 | 1.1235 | -0.8802 | 0.1216 | 30 |
| gemma-4-31b-it | buyer | $100M | 0.5649 | -0.5168 | 0.0240 | 30 |
| gpt-6-luna | seller | $100 | 1.5181 | -0.0151 | 0.7515 | 120 |
| gpt-6-luna | seller | $100M | 1.0955 | 0.0773 | 0.5864 | 120 |
| gpt-6-luna | buyer | $100 | 1.0508 | -0.8088 | 0.1210 | 120 |
| gpt-6-luna | buyer | $100M | 0.7523 | -0.5938 | 0.0792 | 119 |
| claude-haiku-5.5 | seller | $100 | 2.5091 | 0.2316 | 1.3703 | 120 |
| claude-haiku-5.5 | seller | $100M | 2.1405 | 0.4198 | 1.2801 | 120 |
| claude-haiku-5.5 | buyer | $100 | 0.9076 | -0.4641 | 0.2217 | 120 |
| claude-haiku-5.5 | buyer | $100M | 0.6720 | -0.1332 | 0.2694 | 120 |
| gemini-3.5-flash-lite | seller | $100 | 1.6703 | -0.1087 | 0.7808 | 115 |
| gemini-3.5-flash-lite | seller | $100M | 0.5638 | -0.0493 | 0.2572 | 114 |
| gemini-3.5-flash-lite | buyer | $100 | 0.8846 | -0.7889 | 0.0479 | 120 |
| gemini-3.5-flash-lite | buyer | $100M | 0.3415 | -0.4370 | -0.0477 | 90 |

Positions below 0 or above 1 are first offers outside the zone (sellers asking above the buyer's ceiling, buyers
offering below the seller's floor), as the models actually played. They are scripted as given. In particular,
Haiku's seller-opened midpoints (1.37 at $100, 1.28 at $100M) lie above the ceiling in levels. P3b uses only the
difference Δm between source scales, and every scripted pair is ordered (seller above buyer), so the clamp is
well defined for every arm; the levels are reported, not interpreted as targets inside the zone.

Mean scripted midpoint shift Δm over openers (zone units): Qwen +0.00, Haiku −0.02, Luna −0.10, Gemma −0.26,
Gemini −0.31. Where Δm is near 0 (Qwen, Haiku), P3b cannot separate "deals follow the scripted midpoint" from
"openings have no effect"; this is stated with the P3b result.

---

## Decisions log (all closed)

1. §3.1 appraisal: decided 10-13, the appraisal alone; no ranges in any condition.
2. §3.2 clamp source: decided 10-13, earlier data, in zone units.
3. §4 grammar: decided 10-13, identical schema; every model passed the pilot under it (10-14), no fallback.
4. §2/§7 fixtures: decided 10-13, 40.
5. External registry: decided 10-13, OSF Registries (full document and `inputs.sha256` attached).
6. §4 trailing text after the first JSON object: decided 10-14, ignored and counted (§6).
7. §8 data centers: decided 10-14, North American data centers that stock the GPU.
