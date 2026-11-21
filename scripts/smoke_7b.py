#!/usr/bin/env python3
"""smoke_7b.py: Phase 7b checks against a live vLLM, through the SSH tunnel.

Run while scripts/launch_pod.sh holds a serve session (SESSION_READY):
    python3 scripts/smoke_7b.py <model name>

Uses only scripts/vllm_client.py (127.0.0.1 only) and the first move of the
first PILOT game (config/pilot_fixtures.json, phrase
"scale-invariance/fixtures/pilot"; data plan E4). No real fixture is played.

Checks, each printed as one screen-safe line:
  1. /tokenize + /detokenize: the rendered prompt ends with the
     thinking-off marker for this family (api_map 2.3).
  2. One real chat request, exactly as game_engine.build_request makes it,
     plus return_prompt_text: vLLM accepts our JSON schema (anyOf +
     minimum), the reply passes game_engine.validate_reply, reasoning is
     null, and usage has real token counts (C15).
Raw requests and responses go to logs/smoke_7b_<UTC>/ for the record.
Exit 0 only if every check passes.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import game_engine as ge          # noqa: E402
import render_prompt as rp        # noqa: E402
import vllm_client as v           # noqa: E402

PORT = 18000
TAILS = {
    "qwen": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "gemma": "<|turn>model\n<|channel>thought\n<channel|>",
}


def main(argv):
    if len(argv) != 1:
        print(__doc__.split("\n\n")[1])
        return 2
    model = argv[0]
    family = "qwen" if model.startswith("qwen") else "gemma"
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    fx = json.load(open(os.path.join(ROOT, "config", "pilot_fixtures.json")))
    assert fx["seed_phrase"] == "scale-invariance/fixtures/pilot", "not the pilot fixtures"
    out = os.path.join(ROOT, "logs", "smoke_7b_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    os.makedirs(out)
    ok = True

    def save(name, data):
        with open(os.path.join(out, name), "wb") as f:
            f.write(data if isinstance(data, bytes) else json.dumps(data, indent=1).encode())

    def post(path, body, name):
        raw = json.dumps(body).encode()
        save(name + "_request.json", raw)
        status, data, err = v.request(PORT, "POST", path, raw, total_timeout=120)
        save(name + "_response.json", data)
        return status, data, err

    game = sorted(fx["games"], key=lambda g: g["run_order"])[0]
    role = game["opener"]
    limit = game["F"] if role == "seller" else game["C"]
    max_msgs = settings["max_messages"]["value"]
    max_chars = settings["max_message_chars"]["value"]
    messages = rp.render(role, game["V"], limit, [], 1, tuple(fx["f_pct_range"]),
                         tuple(fx["c_pct_range"]), max_msgs, max_chars)
    print(f"Test move: pilot game {game['game_id']}, message 1, {role} (scale {game['scale']})")

    # 1. The prompt as the server renders it
    st, data, err = post("/tokenize", {"model": model, "messages": messages, "add_generation_prompt": True,
                                       "chat_template_kwargs": {"enable_thinking": False},
                                       "return_token_strs": True}, "tokenize")
    if st != 200:
        print(f"  ✗ tokenize: {err or 'HTTP %s: %s' % (st, data[:300].decode('utf-8', 'replace'))}")
        return 1
    tok = json.loads(data)
    st, data, err = post("/detokenize", {"model": model, "tokens": tok["tokens"]}, "detokenize")
    rendered = json.loads(data).get("prompt", "") if st == 200 else ""
    save("rendered_prompt.txt", rendered.encode())
    if rendered.endswith(TAILS[family]):
        print(f"  ✓ thinking off: the {tok['count']}-token prompt ends with the empty-thinking marker")
    else:
        ok = False
        print(f"  ✗ thinking off: prompt ends with {rendered[-60:]!r}, expected {TAILS[family]!r}")

    # 2. One real move
    seed = ge.call_seed(game["fixture"], game["opener"], 1, 1)
    body = ge.build_request(model, messages, settings, seed)
    body["return_prompt_text"] = True
    v.check_chat_body(json.dumps(body).encode())
    t0 = time.monotonic()
    st, data, err = post("/v1/chat/completions", body, "chat")
    secs = time.monotonic() - t0
    if st != 200:
        print(f"  ✗ schema/request: {err or 'HTTP %s: %s' % (st, data[:400].decode('utf-8', 'replace'))}")
        return 1
    print(f"  ✓ JSON schema accepted (anyOf integer>=1 | null, enum, maxLength {max_chars}); reply in {secs:.1f} s")
    resp = json.loads(data)
    ch = resp["choices"][0]
    msg = ch["message"]
    reasoning = msg.get("reasoning", msg.get("reasoning_content"))
    move, errors = ge.validate_reply(msg.get("content"), ch.get("finish_reason"), False, max_chars)
    if errors:
        ok = False
        print(f"  ✗ valid move: {errors}")
    else:
        print(f"  ✓ valid move: {move['action']} at {move['price']}, message {len(move['message'])} chars")
    if reasoning in (None, ""):
        print("  ✓ reasoning: null")
    else:
        ok = False
        print(f"  ✗ reasoning present ({len(reasoning)} chars)")
    u = resp.get("usage") or {}
    pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
    if isinstance(pt, int) and isinstance(ct, int) and pt > 0 and ct > 0:
        limit_note = "under" if ct < settings["max_tokens"]["value"] else "AT"
        print(f"  ✓ token counts: prompt {pt} (tokenize said {tok['count']}), reply {ct} ({limit_note} the "
              f"{settings['max_tokens']['value']} cap), finish_reason {ch.get('finish_reason')}")
        if pt != tok["count"]:
            print("    ! prompt token counts differ between /tokenize and the chat call")
    else:
        ok = False
        print(f"  ✗ token counts missing: {u}")
    ptext = resp.get("prompt_text") or ch.get("prompt_text")
    if ptext is not None:
        same = ptext == rendered
        print(f"  {'✓' if same else '!'} return_prompt_text: {'identical to' if same else 'DIFFERS from'} the /tokenize rendering")
    else:
        print(f"  ! return_prompt_text: no prompt text in the reply (keys: {sorted(resp)})")
    print(f"  request id {resp.get('id')} (look it up in vLLM's request log); raw files in {os.path.relpath(out, ROOT)}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
