# TMLR Submission Cover Note (Draft)

**Title:** *Multiply by a Million: Scale Invariance in LLM Negotiators*

## Fit with TMLR Acceptance Criteria

TMLR evaluates submissions on whether the claims are supported by accurate, convincing, and clear evidence. This manuscript is designed around that standard:

1. **Confirmatory SHA-256 pre-freeze across 5 models and 16,000 games (Study 3):** Hypotheses, equivalence margins ($\pm 0.10$ for closing shifts, $\pm 0.05$ for opening-clamp decomposition), scripted opening values, and analysis code (37 files) were cryptographically hashed before the first registered game. All 25 primary tests and all 9 secondary families (S1–S9) are reported without selection.
2. **Equivalence testing + causal opening-clamp decomposition:** Rather than reading non-significant $p$-values as invariance, we apply Lakens's equivalence testing (90% fixture-bootstrap CIs) and separate opening-driven ($\omega$) from concession-driven ($\kappa$) scale shifts by scripting both first offers.
3. **Transparent reporting of falsified predictions:** Four of 25 pre-committed predictions fail (notably P1/P2/P3a on `claude-haiku-5.5` and P2 A1 on `gemini-3.5-flash-lite`), demonstrating that adding a public reference appraisal is an anchor, not a universal remedy.
4. **Exploratory boundary conditions & mechanistic localisation:** Section 5.4 surfaces 10,000+ exploratory control games (explicit scale-neutral prompting failure, digits vs. unit words and cents granularity, cross-model asymmetric play, and zone geometry), and Section 6 localises Gemma 4 31B's appraisal takeover to layer 41/60 while cleanly distinguishing numerical magnitude from cross-task risk preferences.

## Suggested Action Editor Areas

- LLM evaluation, robustness, and statistical methodology
- Multi-agent systems and behavioural game theory with language models
- Mechanistic interpretability of numerical representations and decision-making
