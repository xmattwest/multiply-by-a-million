# Interp pod mode (mechanistic interpretability track)

Plan: `docs/plan_mechinterp.md`. Target model: see STORY X50 (corrected
2026-10-04). Gemma 4 12B's ×10⁶ "collapse" is mostly digit slips and its buyer
reverses, so 31B is the recommended target. Everything here is exploratory.

## What this mode is

A pod that loads the pinned Gemma 4 weights with **transformers + PyTorch
hooks** (not vLLM serving), so we can read and perturb the residual stream.
It reuses the serving pods' safety machinery unchanged: lifetime watchdog,
self-delete, one-key SSH, and the **per-file manifest hash check** on the
weights. HF is offline after download.

- **Image:** the pinned `vllm/vllm-openai:v0.30.0` (torch 2.13). Its
  `transformers >= 5.10.4` is newer than the version that built the Gemma 4
  weights (`transformers_version 5.10.0.dev0` in the model's `config.json`),
  so the `Gemma4Unified*` classes are present. No pip install needed.
- **Text-only:** Gemma 4 12B is a *unified, encoder-free* multimodal model
  (`architectures: ["Gemma4UnifiedForConditionalGeneration"]`,
  `model_type: "gemma4_unified"`, with `text_config`/`vision_config`/
  `audio_config`). We load the whole checkpoint but run only the text path
  (no pixel/audio inputs), and hook the 48 decoder layers under
  `model.language_model.layers.*`. The serving pods do the equivalent with
  vLLM's `--language-model-only`.

## Pieces (owned by the MI track)

- `scripts/pod/boot_interp.sh` — pod boot: watchdog + SSH + weights check
  (same as `boot.sh`), then unpacks the interp code (shipped in the launch as
  a hash-checked tar) and runs `run_interp.py`.
- `scripts/interp/model_io.py` — text-only load, manifest re-check, layer map,
  chat-template render (thinking off).
- `scripts/interp/hooks.py` — residual-stream capture, activation patching,
  steering (PyTorch forward hooks; no extra frameworks).
- `scripts/interp/tasks.py` — the experiments. E0 (fidelity) now; E1–E3 added
  after E0 passes.
- `scripts/interp/run_interp.py` — pod entry: load, self-check, serve a tiny
  localhost control API (`/health`, `/run`, `/stop`) on `127.0.0.1:8000`.
- `scripts/launch_interp.py` — laptop driver (stdlib only): render the D2/D8
  minimal-pair prompts, drive the pod over the SSH tunnel, write
  `data/interp/<run_id>/`.
- `scripts/launch_pod.sh` — `--mode interp` added **additively** (serve's model
  plumbing + the interp boot script + the interp-code env). `tests/
  test_launch_pod.sh` covers it; `tests/test_interp_offline.py` covers the
  laptop pieces.

## Running

Free preview (fake machine, canned numbers; shows the live panel):

    python3 scripts/launch_interp.py --preview --model gemma-4-31b-it --roles seller

Paid, one terminal. It prints the plan and the worst case, you type that number, then it launches the
pod (slot 8), shows a live panel (stages, progress and ETA, speed, GPU memory, what the pod loaded,
reply validity and digit slips, live E0 numbers next to the vLLM run's on the same items, time left,
cost so far), runs E0, and deletes the machine. Ctrl-C stops after the current chunk and deletes it.

    python3 scripts/launch_interp.py --model gemma-4-31b-it --roles seller

`progress.json` in the run folder is updated after every chunk. `--attach --slot N` drives a session
launched separately with `bash scripts/launch_pod.sh --mode interp ...` (then end it yourself with
`touch logs/pod_session.N.stop`). That launch needs `--interp-plugins` (`all`, `none` or `a,b,...`). Worst case = (lifetime + 20) / 60 × live `$/hr` + disk.

Heavy data stays on the pod (decided 2026-10-04): activations are analysed there in PyTorch and only
results come back (kilobytes). Activations are recomputable from the commit, prompts and seeds.

The loader accepts both Gemma 4 configs: 12B (`gemma4_unified`, encoder-free,
48 layers, verified on a pod 2026-10-05 with transformers 5.17.0) and 31B
(`gemma4`, `Gemma4ForConditionalGeneration` with a vision tower, 60 layers,
hidden 5376; not yet loaded on a pod). The pod self-check fails loudly if the
weights look wrong (mean loss on a plain English sentence must be < 5 nats).

## Results

`data/interp/<run_id>/`: `run_manifest.json`, `inputs.jsonl` (every rendered
prompt), `results.jsonl` (R1/R2 readouts), `summary.json`, `hashes.json`.
Only scalars and small per-layer vectors leave the pod; full activation caches
stay on the pod and die with it.

## E0 fidelity gate (plan §2)

Pass if the transformers setup reproduces the vLLM first-move gaps on the same
prompts: mean seller gap within 3 points of X50 (D2 +44.3%, D8 −19.2%) and the
shift within 25%. R2 (teacher-forced price digits) is kept only if it tracks
R1 at r ≥ 0.8. On pass, proceed to E1 (magnitude probe), E2 (patching), E3
(steering), then the go/no-go in plan §6.

## Plug-ins (for new experiments, and for chats working in parallel)

New experiments go in their own files, so two chats never edit the same file:

- `scripts/interp/tasks_<name>.py` (pod): defines `TASKS = {"<task>": fn}`, with the same signature as
  `tasks.py`. It may `import tasks` to reuse its helpers. It is loaded automatically, and a task name
  that is already taken is an error. The code ships inside the create request, and RunPod rejects
  requests over 102,400 bytes (the launcher stops at 100,000). So the packed `scripts/interp` leaves
  out the laptop-only `exp_*` and `mi_figures*` files (since 2026-10-07). It also leaves out every
  plug-in the session doesn't need (since 2026-10-09; `scripts/lib/pack_interp.sh`):
  - `launch_interp.py` passes `--interp-plugins`: `tasks_<experiment>.py` plus the `tasks_*` modules
    it imports, or `none` for the built-in experiments.
  - Before any machine exists, it refuses a session whose steps call a task the shipped code lacks.
  - The dry run builds the request and prints its size. If a plug-in needs another plug-in's tasks,
    import that module, and it ships too.
  - `python3 scripts/launch_interp.py --experiment <name> --print-pod-plugins` shows the list.
- `scripts/interp/exp_<name>.py` (laptop; standard library only): selected with `--experiment <name>`.
  It provides:
  - `setup(a, settings, schema, li)`, which returns `{"exp", "plan", "inputs", "ctx", "steps", ...}`.
    `steps(ctx)` returns `[(label, task, items, kw, chunk), ...]`. Optional keys:
    - `live(got)`: lines for the panel
    - `after_step(label, got, ctx)`: return a reason string to stop the session (a gate), or edit a
      later step's `kw` dict in place (an adaptive choice)
  - `stub(task, items, ctx)`: canned results for `--preview`
  - `summarize(got, ctx)` and `report(summary, preview)`
  - Optional: `add_args(ap)` (its own flags, prefixed with its name), `configure(a)` (for example
    `a.roles = "buyer"`), `on_pod_info(ctx, pod_info)`, `manifest(ctx)`
- `scripts/interp/mi_figures_<name>.py`: provides `cards()`, which returns figure-card HTML for the
  website. It reads its own tracked `config/interp/mi_figures_<name>.json`. A broken plug-in is
  skipped, so it never breaks the site build.
- Tests and rehearsal:
  - Tests go in a new `tests/test_interp_<name>.py`.
  - Rehearse with `REHEARSE_SLOT=<1-9> bash scripts/interp/rehearse_interp.sh <name>`, using a
    different slot for each chat; extra flags go in `REHEARSE_ARGS`.
  - If the rehearsal venv or tiny model is missing, run
    `bash scripts/interp/make_rehearsal_model.sh`.
