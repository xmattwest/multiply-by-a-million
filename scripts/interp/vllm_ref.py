#!/usr/bin/env python3
"""vllm_ref.py (POD-SIDE): ground truth from vLLM, the engine every behavioural run used.

Run by run_interp.py in a separate process BEFORE transformers loads the model (so the GPU is
free afterwards). For the fixed diagnostic inputs (diag_prompts.json) it reports:
  - the mean next-token loss of the plain English sentence (same token ids as transformers sees),
  - the greedy reply (token ids and text) to each chat prompt, from the same prompt token ids.
The token ids are made with the model's own tokenizer and chat template (thinking off), exactly
as model_io.render_messages makes them, so transformers can be compared token for token.

    python3 vllm_ref.py <model_dir> <diag_prompts.json> <out.json>
"""
import json
import sys
import time


def main(argv):
    model_dir, prompts_path, out_path = argv
    t0 = time.time()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import vllm

    tok = AutoTokenizer.from_pretrained(model_dir)
    spec = json.load(open(prompts_path))
    # Gemma needs <bos> first. The tokenizer does not add it to plain text (2026-10-05 session:
    # without it both engines scored the sentence ~10 nats); the chat template does add it.
    bos = tok.bos_token_id if tok.bos_token_id is not None else 2
    plain_ids = [bos] + tok(spec["plain_text"], add_special_tokens=False)["input_ids"]
    chats = []
    for c in spec["chat"]:
        text = tok.apply_chat_template(c["messages"], add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
        chats.append((c["item_id"], tok(text, add_special_tokens=False)["input_ids"]))

    kw = dict(model=model_dir, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85,
              enforce_eager=True, seed=0)
    try:
        llm, how = LLM(**kw, language_model_only=True), "language_model_only"
    except TypeError:                         # argument unknown in this vLLM: load as is
        llm, how = LLM(**kw), "default"

    out = {"vllm_version": vllm.__version__, "load": how, "plain_n_tokens": len(plain_ids)}
    res = llm.generate([{"prompt_token_ids": plain_ids}],
                       SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0))[0]
    lps = [d[t].logprob for t, d in zip(plain_ids[1:], res.prompt_logprobs[1:])]
    out["plain_nll"] = -sum(lps) / len(lps)
    gens = llm.generate([{"prompt_token_ids": ids} for _, ids in chats],
                        SamplingParams(max_tokens=96, temperature=0.0, logprobs=0))
    out["chat"] = []
    for (iid, ids), g in zip(chats, gens):
        o = g.outputs[0]
        toks = list(o.token_ids)
        lps = [d[t].logprob for t, d in zip(toks, o.logprobs or [])]
        out["chat"].append({"item_id": iid, "n_prompt_tokens": len(ids), "prompt_ids": ids,
                            "prompt_ids_head": ids[:8], "reply_ids": toks, "reply_text": o.text,
                            "reply_nll": (-sum(lps) / len(lps)) if lps else None})
    out["secs"] = round(time.time() - t0, 1)
    json.dump(out, open(out_path, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
