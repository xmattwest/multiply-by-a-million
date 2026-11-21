# Amendment WP2d — two more model families (2026-10-04)

**What changes.** Two models are added to the frozen inputs, for the WP2d
family-generalisation probes and Study 3. Nothing about Study 1 or Study 2
changes: their three models, commits, manifests and sampling values are
byte-for-byte the same.

| Name | Repo | Commit | Licence | Sampling |
|---|---|---|---|---|
| `mistral-small-24b-it` | `mistralai/Mistral-Small-24B-Instruct-2501` | `9527884be6e5616bdd54de542f9ae13384489724` | Apache-2.0 | T 0.15 (card's recommendation), top_p 1.0, top_k off |
| `phi-4-14b` | `microsoft/phi-4` | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` | MIT | T 0.3, top_p 1.0, top_k off |

**Files touched.** `config/pins.json` (two entries appended),
`config/manifests/{mistral-small-24b-it,phi-4-14b}.tsv` (new, from
`scripts/make_manifests.sh`), `prereg/settings.json` (two sampling entries + a
note), `prereg/inputs.sha256` (re-hashed; new manifests added).
`scripts/run_job.py` gains the two prompt endings (`TAILS`) and `family_of()`,
which now refuses an unknown model instead of treating it as Gemma.

**Why these sampling values.** The rule (C1/C2) is each family's published
values. Mistral publishes T 0.15. Phi-4 publishes none (its
`generation_config.json` has only token ids; values seen online are for the
separate Phi-4-reasoning repos). T 0.3 is a user decision: low, but not greedy,
so a retry after an invalid reply can produce a different reply. Temperatures
differ across families, so cross-family comparisons stay descriptive (as in C1/C2).

**Prompt endings checked against the pinned templates.** Mistral's
`tokenizer_config.json` ends the last user turn with `[/INST]` and has no
reasoning mode; Phi-4's adds `<|im_start|>assistant<|im_sep|>`. The pod check
still confirms this on the rendered prompt before any game.

**Tokenisation.** Mistral's Tekken pre-tokeniser splits numbers on `\p{N}`
(one digit per token, like the current three). Phi-4 uses a cl100k-style
tokeniser that groups up to three digits per token. Phi-4 is therefore the
digit-encoding contrast; Mistral is a new family with the encoding held fixed.

**Disk.** The Mistral repo holds the weights twice (`consolidated.safetensors`
and 10 shards, ~94 GB total); vLLM serves the ~47 GB shards.

**Update 2026-10-04 (after the first Mistral pilot).** The first pod failed
its preflight with no games run ($0.92): with `tekken.json` present, vLLM's
`tokenizer_mode=auto` loaded Mistral's own tokenizer, which rejects requests
that carry `chat_template_kwargs` ("chat_template is not supported for Mistral
tokenizers"). Fix, in `config/pins.json`: Mistral now has
`"exclude": ["consolidated.safetensors", "params.json", "tekken.json"]`, so the
manifest (and the pod) holds only the Hugging Face format: config.json, the 10
shards, tokenizer.json and tokenizer_config.json (whose chat template ends the
prompt with `[/INST]`). vLLM then uses the standard tokenizer and template, the
same path as the other models, and the download halves to ~47 GB.
Tokenisation is unchanged: the HF `tokenizer.json` pre-splits numbers into runs
of up to 3 digits, but its 131,072-token vocabulary has **no multi-digit
tokens** (only the 10 single digits), so every digit is still its own token.
