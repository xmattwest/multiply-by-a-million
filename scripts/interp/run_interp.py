#!/usr/bin/env python3
"""run_interp.py (POD-SIDE): load Gemma 4 text-only, check it against vLLM, serve a control API.

Started by scripts/pod/boot_interp.sh after the weights pass their manifest check:
  1. re-checks a few small files against the manifest;
  2. (--vllm-ref) runs vllm_ref.py in its own process: vLLM's loss on a plain sentence and its
     greedy replies to two fixed negotiation prompts (diag_prompts.json) = the ground truth;
  3. loads the model with transformers (model_io);
  4. diagnosis: the same sentence's loss under the default attention and under each SDPA
     kernel and eager attention; keeps the setting with the lowest loss if it is close to
     vLLM's; then greedy replies to the same prompts, compared token for token with vLLM's;
  5. serves 127.0.0.1:<port> in every case (GET /health, POST /run, POST /stop), so the laptop
     can read the diagnosis; /run refuses work unless the pod passed (info.usable).
Single-threaded on purpose: one request at a time on the GPU. Exploratory.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # scripts/interp
sys.path.insert(0, os.path.dirname(HERE))     # scripts/ (game_engine)
import model_io                               # noqa: E402
import tasks                                  # noqa: E402

NLL_MAX = 5.0             # used only without a vLLM reference
NLL_VS_VLLM = 0.5         # transformers' plain-sentence loss must be within this of vLLM's
REPLY_DIFF_MAX = 0.1      # mean |loss difference| per token on vLLM's own replies (bf16 kernels differ a little)


def log(msg):
    print(f"[interp] {msg}", flush=True)


def code_hashes():
    files = sorted(glob.glob(os.path.join(HERE, "*.py")) + glob.glob(os.path.join(HERE, "*.json"))) \
        + [os.path.join(os.path.dirname(HERE), "game_engine.py")]
    return {os.path.relpath(f, os.path.dirname(os.path.dirname(HERE))): model_io.sha256_file(f) for f in files}


class State:
    model = tok = info = model_name = None


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            return

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "info": state.info})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                self._send(400, {"error": "bad json"})
                return
            if self.path == "/stop":
                self.server.stop = True
                self._send(200, {"ok": True})
                return
            if self.path != "/run":
                self._send(404, {"error": "not found"})
                return
            if not state.info.get("usable"):
                self._send(409, {"error": "pod failed its checks: " + str(state.info.get("unusable_why"))})
                return
            fn = tasks.TASKS.get(req.get("task"))
            if fn is None:
                self._send(400, {"error": f"unknown task {req.get('task')!r}", "known": sorted(tasks.TASKS)})
                return
            kw = {k: v for k, v in req.items() if k not in ("task", "items", "settings", "model_name")}
            t0 = time.time()
            try:
                res = fn(state.model, state.tok, req["items"], req["settings"],
                         req.get("model_name") or state.model_name, model_io, **kw)
                log(f"task {req['task']}: {len(req['items'])} items in {time.time() - t0:.0f} s")
                self._send(200, {"results": res, "secs": round(time.time() - t0, 1), "gpu": model_io.gpu_memory()})
            except Exception as e:            # noqa: BLE001  report, keep serving
                log(f"task {req.get('task')} failed: {type(e).__name__}: {e}")
                self._send(500, {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]})
    return H


def vllm_reference(model_dir, timeout_s=1200):
    out = "/tmp/si_vllm_ref.json"
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "vllm_ref.py"), model_dir,
                            os.path.join(HERE, "diag_prompts.json"), out],
                           capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0 or not os.path.exists(out):
            tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
            return {"error": f"vllm_ref exited {r.returncode}: " + " | ".join(tail)[-800:]}
        ref = json.load(open(out))
        ref["wall_secs"] = round(time.time() - t0, 1)
        return ref
    except subprocess.TimeoutExpired:
        return {"error": f"vllm_ref timed out after {timeout_s} s"}


def nll_of(state, ids):
    import torch
    with torch.no_grad(), model_io.attn_ctx():
        lg = state.model(input_ids=ids).logits[0, :-1].float()
    return torch.nn.functional.cross_entropy(lg, ids[0, 1:]).item()


def reply_nll_of(state, prompt_ids, reply_ids):
    """transformers' mean loss on vLLM's greedy reply tokens, given the identical prompt ids."""
    import torch
    seq = torch.tensor([list(prompt_ids) + list(reply_ids)], device=state.model.device)
    with torch.no_grad(), model_io.attn_ctx():
        lg = state.model(input_ids=seq).logits[0, len(prompt_ids) - 1:-1].float()
    return torch.nn.functional.cross_entropy(lg, seq[0, len(prompt_ids):]).item()


def plain_ids(state, text):
    """Gemma needs <bos>; the tokenizer doesn't add it to plain text (see vllm_ref.py)."""
    import torch
    bos = state.tok.bos_token_id if state.tok.bos_token_id is not None else 2
    ids = [bos] + state.tok(text, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids], device=state.model.device)


def diagnose(state, plain_text, ref):
    """Per attention setting: plain-sentence loss and the loss of vLLM's own greedy replies.
    Keeps the setting whose reply loss is closest to vLLM's; usable if it is close enough."""
    import torch
    from torch.nn.attention import SDPBackend
    ref = ref if isinstance(ref, dict) and "error" not in ref else {}
    ref_plain = ref.get("plain_nll")
    chats = [c for c in ref.get("chat", []) if c.get("reply_ids") and c.get("reply_nll") is not None]
    ids = plain_ids(state, plain_text)
    impl = getattr(state.model.config, "_attn_implementation", None)
    options = [("default", None, None)]
    for name in ("MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION"):
        if getattr(SDPBackend, name, None) is not None:
            options.append((f"sdpa_{name.lower()}", getattr(SDPBackend, name), None))
    options.append(("eager", None, "eager"))
    table = {"attn_implementation": impl, "vllm_plain": ref_plain,
             "vllm_reply": [round(c["reply_nll"], 4) for c in chats]}
    for name, backend, new_impl in options:
        row = {}
        try:
            if new_impl:
                state.model.set_attn_implementation(new_impl)
            model_io.SDPA_BACKENDS = [backend] if backend is not None else None
            row["plain"] = round(nll_of(state, ids), 3)
            if chats:
                mine = [reply_nll_of(state, c["prompt_ids"], c["reply_ids"]) for c in chats]
                row["reply"] = [round(m, 4) for m in mine]
                row["reply_diff"] = round(st_mean([abs(m - c["reply_nll"]) for m, c in zip(mine, chats)]), 4)
        except Exception as e:                # noqa: BLE001
            row = {"error": f"{type(e).__name__}: {e}"[:160]}
        finally:
            model_io.SDPA_BACKENDS = None
            if new_impl:
                try:
                    state.model.set_attn_implementation(impl)
                except Exception:             # noqa: BLE001
                    pass
            torch.cuda.empty_cache()
        table[name] = row
    good = {k: v for k, v in table.items() if isinstance(v, dict) and "plain" in v}
    if not good:
        return table, None, "every attention setting raised an error"
    key = (lambda k: good[k]["reply_diff"]) if chats else \
        ((lambda k: abs(good[k]["plain"] - ref_plain)) if ref_plain is not None else (lambda k: good[k]["plain"]))
    best = min(good, key=key)
    b = good[best]
    why = None
    if chats and b["reply_diff"] > REPLY_DIFF_MAX:
        why = f"loss on vLLM's replies differs by {b['reply_diff']:.3f} nats/token (max {REPLY_DIFF_MAX})"
    elif ref_plain is not None and abs(b["plain"] - ref_plain) > NLL_VS_VLLM:
        why = f"plain-sentence loss {b['plain']:.2f} vs vLLM {ref_plain:.2f}"
    elif not chats and ref_plain is None and b["plain"] >= NLL_MAX:
        why = f"no vLLM reference and plain-sentence loss {b['plain']:.2f} nats"
    if why is None:
        opt = dict((n, (bk, im)) for n, bk, im in options)[best]
        if opt[1]:
            state.model.set_attn_implementation(opt[1])
        model_io.SDPA_BACKENDS = [opt[0]] if opt[0] is not None else None
    return table, (best if why is None else None), why


def st_mean(xs):
    return sum(xs) / len(xs)


def greedy_match(state, spec, ref):
    """transformers' greedy replies vs vLLM's, from the same prompt ids: tokens that agree before the
    first difference. bf16 kernels differ, so an early difference after many tokens is normal."""
    import torch
    rows = []
    by_id = {c["item_id"]: c for c in (ref or {}).get("chat", [])}
    for c in spec["chat"]:
        _text, ids = model_io.render_messages(state.tok, c["messages"])
        ids = ids.to(state.model.device)
        with torch.no_grad(), model_io.attn_ctx():
            g = state.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=96, do_sample=False,
                                     temperature=None, top_p=None, top_k=None)
        mine = g[0, ids.shape[1]:].tolist()
        row = {"item_id": c["item_id"], "n_prompt_tokens": int(ids.shape[1]),
               "reply_text": state.tok.decode(mine, skip_special_tokens=True)[:300]}
        r = by_id.get(c["item_id"])
        if r:
            theirs = r["reply_ids"]
            same = 0
            for x, y in zip(mine, theirs):
                if x != y:
                    break
                same += 1
            row.update({"same_prompt_ids": r.get("prompt_ids", r["prompt_ids_head"]) ==
                        (ids[0].tolist() if "prompt_ids" in r else ids[0, :8].tolist()),
                        "same_prompt_len": r["n_prompt_tokens"] == int(ids.shape[1]),
                        "agree_tokens": same, "of": min(len(mine), len(theirs)),
                        "vllm_reply_text": r["reply_text"][:300]})
        rows.append(row)
    return rows


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--model-name", default="gemma-4-31b-it")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--vllm-ref", action="store_true", help="get vLLM's reference numbers first")
    a = ap.parse_args(argv)
    state = State()
    state.model_name = a.model_name
    state.info = {"usable": False, "unusable_why": "not loaded yet", "code_sha256": code_hashes()}
    spec = json.load(open(os.path.join(HERE, "diag_prompts.json")))
    try:
        if a.manifest:
            log("MANIFEST_RECHECK ok " + " ".join(p for p, _ in model_io.recheck_manifest(a.model_dir, a.manifest)))
        ref = None
        if a.vllm_ref:
            log("VLLM_REF starting (vLLM in its own process, before transformers loads)")
            ref = vllm_reference(a.model_dir)
            log("VLLM_REF " + json.dumps({k: v for k, v in ref.items() if k != "chat"})[:600])
        t0 = time.time()
        state.model, state.tok, info = model_io.load_text_only(a.model_dir)
        info["load_secs"] = round(time.time() - t0, 1)
        state.info.update(info)
        state.info["vllm_ref"] = ref
        log("LOADED " + json.dumps(info))
        table, choice, why = diagnose(state, spec["plain_text"], ref)
        state.info["diagnosis"] = table
        state.info["attention_choice"] = choice
        state.info["self_check_nll"] = (table.get(choice) or {}).get("plain") if choice else None
        log("DIAGNOSIS " + json.dumps(table) + f" choice={choice} why={why}")
        state.info["greedy_vs_vllm"] = greedy_match(state, spec, ref)
        log("GREEDY " + json.dumps(state.info["greedy_vs_vllm"])[:800])
        if choice is None and os.environ.get("SI_INTERP_REHEARSAL") == "1":
            choice = "default"
            state.info["attention_choice"] = "default (REHEARSAL: random weights, check skipped)"
            why = None
        if choice is None:
            state.info["unusable_why"] = why
        else:
            import game_engine as ge
            settings = json.load(open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "prereg", "settings.json")))
            lp, how = tasks._grammar_processor(state.tok, ge.response_schema(settings["max_message_chars"]["value"]),
                                               state.info["vocab_size"])
            state.info["grammar"] = how if lp is not None else "UNAVAILABLE: " + how
            log(f"GRAMMAR {state.info['grammar']}")
            state.info["usable"], state.info["unusable_why"] = True, None
    except Exception as e:                    # noqa: BLE001  keep serving so the laptop can read why
        state.info["unusable_why"] = f"{type(e).__name__}: {e}"[:800]
        state.info["traceback"] = traceback.format_exc()[-3000:]
        log(f"INTERP_LOAD_FAILED {type(e).__name__}: {e}")
        traceback.print_exc()
    if state.info["usable"]:
        log(f"INTERP_READY transformers={state.info['transformers_version']} layers={state.info['num_hidden_layers']} "
            f"hidden={state.info['hidden_size']} attention={state.info['attention_choice']}")
    else:
        log("INTERP_UNUSABLE " + str(state.info["unusable_why"]))
    httpd = HTTPServer(("127.0.0.1", a.port), make_handler(state))
    httpd.stop = False
    log(f"CONTROL_API 127.0.0.1:{a.port}")
    while not httpd.stop:
        httpd.handle_request()
    log("CONTROL_API stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
