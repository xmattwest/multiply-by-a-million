#!/usr/bin/env python3
"""run_addendum_d_open.py: Run Addendum D1 (Thinking-On on Gemma/Qwen vLLM pods) and
D2 (3-Model Cross-Play Matrix {Gemma, Qwen, Gemini} on G01-G40).

Strictly read-only with respect to all 37 frozen Study 3 files.
Talks to:
  - Gemma 4 31B vLLM on http://127.0.0.1:<gemma_port>/v1/chat/completions
  - Qwen3.8 27B vLLM on http://127.0.0.1:<qwen_port>/v1/chat/completions
  - gemini-3.5-flash-lite via scripts/run_addendum_d.call_gemini_native (GEMINI_API_KEY)
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import game_engine as ge            # noqa: E402
import render_prompt_s2 as rp2      # noqa: E402
import run_addendum_d as rad        # noqa: E402
import study3 as s3                 # noqa: E402

SCALES = {"x1e2": 100, "x1e8": 100_000_000}

# Exact Study 3 model-native sampling settings (from data/runs/s3-*/run_manifest.json)
VLLM_SAMPLING = {
    "gemma-4-31b-it": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "qwen3.8-27b-fp8": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
}

RESPONSE_SCHEMA_OPENAI = {
    "type": "json_schema",
    "json_schema": {
        "name": "negotiation_move",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["OFFER", "ACCEPT", "WALKAWAY"]},
                "price": {"type": ["integer", "null"]},
                "message": {"type": "string"},
            },
            "required": ["action", "price", "message"],
            "additionalProperties": False,
        },
    },
}

THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)


def seed_for(block, tag, fixture_id, opener, sample, msg_no):
    s = f"scale-invariance/addendum-d/{block}/{tag}/v1|{fixture_id}|{opener}|{sample}|{msg_no}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def wait_vllm(port, timeout_s=900):
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                if data.get("data"):
                    model_id = data["data"][0]["id"]
                    print(f"[vllm:{port}] ready: {model_id}", flush=True)
                    return model_id
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM on port {port} did not become ready within {timeout_s}s")


def call_vllm(port, model_id, arm, messages, seed, enable_thinking, max_tokens):
    samp = VLLM_SAMPLING[arm]
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": samp["temperature"],
        "top_p": samp["top_p"],
        "top_k": samp["top_k"],
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "n": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    # When thinking is off, enforce structured JSON schema as in Study 3.
    # When thinking is on, let the model emit <think>/<thought> freely before JSON so guided decoding doesn't block thoughts.
    if not enable_thinking:
        body["response_format"] = RESPONSE_SCHEMA_OPENAI

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    data = json.dumps(body).encode("utf-8")
    last_err = None
    for attempt in range(1, 4):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            choice = raw["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            # Also extract inline <think>...</think> or <thought>...</thought> if present in content
            if not reasoning and ("<think>" in content.lower() or "<thought>" in content.lower()):
                m = THINK_RE.search(content)
                if m:
                    reasoning = m.group(0)
                    content = THINK_RE.sub("", content).strip()
            # Extract first JSON object if wrapped in markdown fences or trailing text
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            usage = raw.get("usage", {})
            fr = choice.get("finish_reason", "stop") or "stop"
            # Approximate thought token count if server doesn't break it out
            comp_toks = int(usage.get("completion_tokens", 0) or 0)
            r_toks = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
            if r_toks == 0 and reasoning:
                r_toks = max(1, len(reasoning) // 4)
            return {
                "content": cleaned,
                "reasoning_preview": reasoning[:300],
                "thoughts_tokens": r_toks,
                "completion_tokens": comp_toks,
                "finish_reason": fr,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 * attempt)
    raise RuntimeError(f"vLLM call failed on port {port}: {last_err}")


def extract_first_json(text):
    """Match Study 3's first_json_only rule: ignore trailing text after first complete JSON object."""
    s = text.strip()
    start = s.find("{")
    if start < 0:
        return s, False
    dec = json.JSONDecoder()
    try:
        _, end = dec.raw_decode(s[start:])
        obj_str = s[start : start + end]
        trailing = bool(s[start + end :].strip())
        return obj_str, trailing
    except ValueError:
        return s, False


def play_game(job, block, endpoints, enable_thinking, max_tokens):
    fx = job["fixture"]
    V = SCALES[job["scale"]]
    floor = fx["f_pct"] * V // 100
    ceiling = fx["c_pct"] * V // 100
    opener = job["opener"]
    sample = job["sample"]
    seller_arm = job["seller_arm"]
    buyer_arm = job["buyer_arm"]
    tag = f"{seller_arm}_vs_{buyer_arm}"

    history = []
    pending_offer = None
    terminal = None
    deal_price = None
    forfeit_reason = None
    first_offer = None
    thought_toks = []
    trailing_calls = 0

    for msg_no in range(1, s3.MAX_MSGS + 1):
        role = opener if (msg_no % 2 == 1) else s3.OTHER[opener]
        arm = seller_arm if role == "seller" else buyer_arm
        limit = floor if role == "seller" else ceiling
        messages = rp2.render(role, V, limit, history, msg_no, s3.F_RANGE, s3.C_RANGE, s3.MAX_MSGS, s3.MAX_CHARS)
        sd = seed_for(block, tag, fx["id"], opener, sample, msg_no)

        if arm == "gemini-3.5-flash-lite":
            res = rad.call_gemini_native(
                model="gemini-3.5-flash-lite",
                messages=messages,
                max_tokens=1024,
                seed=sd,
                thinking_budget=0,
                include_thoughts=False,
            )
            raw_reply = res["content"]
            fr_map = {"STOP": "stop", "MAX_TOKENS": "length"}
            norm_fr = fr_map.get(res["finish_reason"], res["finish_reason"].lower())
            thought_toks.append(res["thoughts_tokens"])
        else:
            port, model_id = endpoints[arm]
            res = call_vllm(port, model_id, arm, messages, sd, enable_thinking=enable_thinking, max_tokens=max_tokens)
            raw_reply = res["content"]
            norm_fr = res["finish_reason"]
            thought_toks.append(res["thoughts_tokens"])

        obj_str, has_trailing = extract_first_json(raw_reply)
        if has_trailing:
            trailing_calls += 1
        parsed, errs = ge.validate_reply(obj_str, norm_fr, (pending_offer is not None), s3.MAX_CHARS)
        if errs:
            terminal = "PROTOCOL_FORFEIT"
            forfeit_reason = f"{role}({arm})@msg{msg_no}:{','.join(errs)}"
            break
        action = parsed["action"]
        if action == "OFFER":
            price = parsed["price"]
            if msg_no == 1:
                first_offer = price
            pending_offer = price
            history.append((role, price, parsed["message"]))
        elif action == "ACCEPT":
            deal_price = pending_offer
            history.append((role, None, parsed["message"]))
            terminal = "AGREEMENT"
            break
        elif action == "WALKAWAY":
            history.append((role, None, parsed["message"]))
            terminal = "WALKAWAY"
            break
    if terminal is None:
        terminal = "TIMEOUT"

    seller_share = (deal_price - floor) / (ceiling - floor) if terminal == "AGREEMENT" else None
    outside_zopa = (terminal == "AGREEMENT" and (deal_price < floor or deal_price > ceiling))
    opener_gap = None
    if first_offer is not None:
        opener_gap = (first_offer / floor - 1.0) if opener == "seller" else (1.0 - first_offer / ceiling)

    return {
        "block": block,
        "seller_arm": seller_arm,
        "buyer_arm": buyer_arm,
        "condition": "A0",
        "scale": job["scale"],
        "V": V,
        "fixture": fx["id"],
        "floor": floor,
        "ceiling": ceiling,
        "opener": opener,
        "sample": sample,
        "enable_thinking": enable_thinking,
        "terminal": terminal,
        "forfeit_reason": forfeit_reason,
        "deal_price": deal_price,
        "seller_share": seller_share,
        "outside_zopa": outside_zopa,
        "opener_gap": opener_gap,
        "n_messages": len(history),
        "trailing_text_calls": trailing_calls,
        "mean_thoughts_tokens": sum(thought_toks) / max(len(thought_toks), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=["d1_open", "d2_cross"])
    ap.add_argument("--gemma-port", type=int, default=18001)
    ap.add_argument("--qwen-port", type=int, default=18002)
    ap.add_argument("--model", default=None, help="For d1_open: gemma-4-31b-it or qwen3.8-27b-fp8")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--n-fixtures", type=int, default=40)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    endpoints = {}
    if args.block == "d1_open":
        assert args.model in ("gemma-4-31b-it", "qwen3.8-27b-fp8")
        port = args.gemma_port if args.model == "gemma-4-31b-it" else args.qwen_port
        endpoints[args.model] = (port, wait_vllm(port))
        pairs = [(args.model, args.model)]
        enable_thinking = True
        max_tokens = 4096
    else:
        endpoints["gemma-4-31b-it"] = (args.gemma_port, wait_vllm(args.gemma_port))
        endpoints["qwen3.8-27b-fp8"] = (args.qwen_port, wait_vllm(args.qwen_port))
        arms = ["gemma-4-31b-it", "qwen3.8-27b-fp8", "gemini-3.5-flash-lite"]
        pairs = [(s, b) for s in arms for b in arms if s != b]
        enable_thinking = False
        max_tokens = 1024

    fxs = s3.fixtures()[: args.n_fixtures]
    jobs = []
    for s_arm, b_arm in pairs:
        for sc in ("x1e2", "x1e8"):
            for fx in fxs:
                for opener in s3.OPENERS:
                    for sample in range(1, args.samples + 1):
                        jobs.append({
                            "seller_arm": s_arm, "buyer_arm": b_arm,
                            "scale": sc, "fixture": fx, "opener": opener, "sample": sample,
                        })
    random.Random(20261122).shuffle(jobs)

    os.makedirs(args.out_dir, exist_ok=True)
    games_path = os.path.join(args.out_dir, "games.jsonl")
    print(f"[{args.block}] games={len(jobs)} pairs={len(pairs)} thinking={enable_thinking} out={args.out_dir}", flush=True)

    lock = threading.Lock()
    done = errors = 0
    t0 = time.time()
    with open(games_path, "w", encoding="utf-8") as fh:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(play_game, j, args.block, endpoints, enable_thinking, max_tokens) for j in jobs]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    rec = fut.result()
                    with lock:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        done += 1
                        if done % 40 == 0 or done == len(jobs):
                            print(f"  [{args.block}] {done}/{len(jobs)} ({time.time() - t0:.1f}s)", flush=True)
                except Exception as e:
                    with lock:
                        errors += 1
                        print(f"  ERROR: {e}", file=sys.stderr, flush=True)
    print(f"[{args.block}] finished {done}/{len(jobs)} errors={errors} in {time.time() - t0:.1f}s", flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
