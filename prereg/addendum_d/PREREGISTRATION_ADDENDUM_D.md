# Preregistration — Addendum D (Post-Study 3 Robustness Suite)

**Status:** Frozen before any Addendum D game is executed (`prereg/addendum_d/inputs.sha256`).
**Hard invariant:** Zero files in `prereg/study3/inputs.sha256` (37 files) are modified. Addendum D reuses Study 3's exact fixture bank (`G01–G40` from `scripts/study3.py`), game engine (`scripts/game_engine.py`), and prompt templates (`prereg/study2/prompts/`, `prereg/study3/prompts/system_a1.txt`) read-only so paired within-fixture Wilcoxon and equivalence tests link directly to Study 3.

---

## 1. Motivation & Blocks

Addendum D addresses the four robustness questions raised in TMLR pre-submission review:

1. **Block D1 — Thinking-On Contrast (`thinking_budget > 0`):**
   - **Question:** Does enabling native chain-of-thought reasoning eliminate, preserve, or alter opening modesty ($h_f$) and closing shifts ($d_i$) relative to Study 3's thinking-off baseline?
   - **Grid:** Conditions `A0` and `A1` × scales `{$100, $100M}` × 2 openers × 40 fixtures (`G01–G40`) × `k = 2` replicates (`320` games per condition; `640` games per model).
   - **Models:** `gemini-3.5-flash-lite` (native Gemini API `thinkingConfig: {thinkingBudget: 1024, includeThoughts: true}`, `maxOutputTokens: 4096`), plus optional `gemma-4-31b-it`, `qwen3.8-27b-fp8` (vLLM thinking enabled), and `claude-haiku-5.5` (OpenRouter extended thinking) when those backends are available.
   - **Primary Tests (within-fixture paired across `G01–G40`, 200,000 sign flips, 10,000 bootstrap resamples):**
     - `D1-H1`: Two-sided Wilcoxon on $d_i(\text{A0, Thinking-On})$ and 90% CI equivalence within $\pm 0.10$.
     - `D1-H2`: Paired contrast $\Delta d_i = d_i(\text{Thinking-On}) - d_i(\text{Study 3 Thinking-Off})$ on `A0` and `A1`, plus paired opening-modesty contrast $\Delta h_f$.
     - `D1-Audit`: Mean thinking tokens per call and descriptive audit of whether reasoning traces mention dollar stakes/caution.

2. **Block D2 — Cross-Model Asymmetric Matrix (`960` games, requires OpenRouter/vLLM):**
   - **Grid:** All 6 directed off-diagonal `(Seller, Buyer)` pairs among `{Gemma 4 31B, Qwen3.8 27B, claude-haiku-5.5}` × `A0` × `{$100, $100M}` × 2 openers × 40 fixtures × `k = 1`.
   - **Primary Test:** Two-sided Wilcoxon on $d_i = s(\$100\text{M}) - s(\$100)$ per directed pair; tests whether high-shrinkage seller vs. low-shrinkage buyer yields negative $d_i$ while the reverse seat holds steady or rises.

3. **Block D3 — `$1,000` Integer-Discretization Anchor Check (`160` games per model):**
   - **Question:** Does replacing the `$100` whole-dollar baseline (where `$1` is 3–10% of the zone) with `$1,000` (`V = 1000`, where `$1` is `< 0.5%` of the zone) materially change the scale shift to `$100M`?
   - **Grid:** `A0` × `$1,000` (`x1e3`) × 2 openers × 40 fixtures (`G01–G40`) × `k = 2` (`160` games per model), paired against Study 3's `A0` `$100M` and `$100` games on `G01–G40`.
   - **Primary Test:** $d_i(\$100\text{M} - \$1\text{K})$ (two-sided Wilcoxon) and paired difference $d_i(\$100\text{M} - \$1\text{K}) - d_i(\$100\text{M} - \$100)$ tested for equivalence within $\pm 0.05$ (90% CI).

4. **Block E6b — BOS/Covariance-Matched Random Steering Control (`scripts/interp_e6b_controls.py`, GPU pod):**
   - **Question:** In E6 Part B on Gemma 4 31B, isotropic Gaussian random vectors at $\alpha = \pm 2$ broke short-prompt formatting (`0.006` valid option mass; `15/15` parse failures). Does a layer-covariance-matched random direction orthogonalized against the BOS/format subspace maintain $\ge 95\%$ valid-option mass while remaining inert on risky choices and percentages?

---

## 2. Scoring & Quality Rules

- Identical to Study 3: `game_engine.validate_reply` with `first_json_only=True`, no repair on invalid JSON, unclipped seller share $s = (P - F)/(C - F)$ on agreements, fixture-level means over `G01–G40`.
- All raw calls, thinking token counts, game records, and run manifests are stored under `data/runs/addD-<timestamp>/` and analyzed by `scripts/analyze_addendum_d.py` into `derived/addendum_d/addendum_d.json`.
