#!/usr/bin/env python3
"""run_addendum_d.py: Execute Addendum D blocks (D1 Thinking-On, D3 $1,000 Anchor Check, D2 Cross-Play).

Reuses frozen Study 3 modules (scripts/study3.py, scripts/game_engine.py, scripts/render_prompt_s2.py,
scripts/render_prompt_s3.py) strictly read-only. Never modifies any file in prereg/study3/inputs.sha256.

Supports:
  - Direct Gemini API via GEMINI_API_KEY (gemini-3.5-flash-lite with native thinkingConfig)
  - OpenRouter via OPENROUTER_API_KEY (for cross-model / Haiku / Luna)

Usage:
  python3 scripts/run_addendum_d.py --block d1 --model gemini-3.5-flash-lite --conditions A0,A1 --samples 2
  python3 scripts/run_addendum_d.py --block d3 --model gemini-3.5-flash-lite --samples 2
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import random
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
import render_prompt_s3 as rp3      # noqa: E402
import study3 as s3                 # noqa: E402

SCALES_D = {
    "x1e2": 100,
    "x1e3": 1_000,
    "x1e5": 100_000,
    "x1e8": 100_000_000,
}

RESPONSE_SCHEMA_GEMINI = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": ["OFFER", "ACCEPT", "WALKAWAY"]},
        "price": {"type": "INTEGER", "nullable": True},
        "message": {"type": "STRING"},
    },
    "required": ["action", "price", "message"],
}


def seed_for(block, fixture_id, opener, sample, msg_no, attempt=1):
    s = f"scale-invariance/addendum-d/{block}/v1|{fixture_id}|{opener}|{sample}|{msg_no}|{attempt}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def call_gemini_native(model, messages, max_tokens, seed, thinking_budget=None, include_thoughts=False):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    sys_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
    gen_cfg = {
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json",
        "responseSchema": RESPONSE_SCHEMA_GEMINI,
        "seed": seed,
    }
    if thinking_budget is not None:
        if int(thinking_budget) <= 0:
            gen_cfg["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}
        else:
            gen_cfg["thinkingConfig"] = {
                "thinkingBudget": int(thinking_budget),
                "includeThoughts": bool(include_thoughts),
            }
    body = {
        "systemInstruction": {"parts": [{"text": sys_text}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": gen_cfg,
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    data = json.dumps(body).encode("utf-8")

    last_err = None
    for attempt in range(1, 5):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            cand = raw.get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [])
            text_parts, thought_parts = [], []
            for p in parts:
                if p.get("thought"):
                    thought_parts.append(p.get("text", ""))
                elif "text" in p:
                    text_parts.append(p.get("text", ""))
            usage = raw.get("usageMetadata", {})
            return {
                "content": "".join(text_parts),
                "thought": "".join(thought_parts),
                "thoughts_tokens": int(usage.get("thoughtsTokenCount", 0) or 0),
                "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
                "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
                "finish_reason": cand.get("finishReason", ""),
                "model_version": raw.get("modelVersion", model),
            }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:400]
            last_err = f"HTTP {e.code}: {err_body}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 16) + random.uniform(0, 1))
                continue
            raise RuntimeError(last_err)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def render_turn(condition, role, floor, ceiling, V, history, msg_no):
    limit = floor if role == "seller" else ceiling
    if condition == "A0":
        return rp2.render(role, V, limit, history, msg_no, s3.F_RANGE, s3.C_RANGE, s3.MAX_MSGS, s3.MAX_CHARS)
    if condition == "A1":
        return rp3.render(role, V, limit, history, msg_no, s3.F_RANGE, s3.C_RANGE, s3.MAX_MSGS, s3.MAX_CHARS)
    raise ValueError(f"unknown condition {condition}")


def play_one_game(job, block, model, thinking_budget, max_tokens):
    fx = job["fixture"]
    V = SCALES_D[job["scale"]]
    floor = fx["f_pct"] * V // 100
    ceiling = fx["c_pct"] * V // 100
    opener = job["opener"]
    cond = job["condition"]
    sample = job["sample"]

    call_logs = []

    def call_fn(role, history, msg_no):
        messages = render_turn(cond, role, floor, ceiling, V, history, msg_no)
        sd = seed_for(block, fx["id"], opener, sample, msg_no)
        res = call_gemini_native(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            seed=sd,
            thinking_budget=thinking_budget,
            include_thoughts=(thinking_budget is not None and thinking_budget > 0),
        )
        call_logs.append({
            "msg_no": msg_no,
            "role": role,
            "seed": sd,
            "thoughts_tokens": res["thoughts_tokens"],
            "prompt_tokens": res["prompt_tokens"],
            "output_tokens": res["output_tokens"],
            "finish_reason": res["finish_reason"],
            "model_version": res["model_version"],
            "thought_preview": res["thought"][:400],
            "raw_content": res["content"],
        })
        fr_map = {"STOP": "stop", "MAX_TOKENS": "length"}
        return res["content"], fr_map.get(res["finish_reason"], res["finish_reason"].lower())

    history = []
    pending_offer = None
    terminal = None
    deal_price = None
    forfeit_reason = None
    trailing_text_calls = 0
    first_offer = None

    for msg_no in range(1, s3.MAX_MSGS + 1):
        role = opener if (msg_no % 2 == 1) else s3.OTHER[opener]
        raw_reply, norm_fr = call_fn(role, history, msg_no)
        parsed, errs = ge.validate_reply(raw_reply, norm_fr, (pending_offer is not None), s3.MAX_CHARS)
        if errs:
            terminal = "PROTOCOL_FORFEIT"
            forfeit_reason = f"{role}@msg{msg_no}:{','.join(errs)}"
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
        "model": model,
        "condition": cond,
        "scale": job["scale"],
        "V": V,
        "fixture": fx["id"],
        "floor": floor,
        "ceiling": ceiling,
        "opener": opener,
        "sample": sample,
        "thinking_budget": thinking_budget,
        "terminal": terminal,
        "forfeit_reason": forfeit_reason,
        "deal_price": deal_price,
        "seller_share": seller_share,
        "outside_zopa": outside_zopa,
        "opener_gap": opener_gap,
        "n_messages": len(history),
        "trailing_text_calls": trailing_text_calls,
        "total_thoughts_tokens": sum(c["thoughts_tokens"] for c in call_logs),
        "mean_thoughts_tokens": sum(c["thoughts_tokens"] for c in call_logs) / max(len(call_logs), 1),
        "model_versions": sorted(set(c["model_version"] for c in call_logs)),
        "calls": call_logs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=["d1", "d3"])
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--conditions", default="A0,A1")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--thinking-budget", type=int, default=None, help="Default: 1024 for d1, 0 for d3")
    ap.add_argument("--max-tokens", type=int, default=None, help="Default: 4096 for d1, 1024 for d3")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--n-fixtures", type=int, default=40)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    thinking_budget = args.thinking_budget if args.thinking_budget is not None else (1024 if args.block == "d1" else 0)
    max_tokens = args.max_tokens if args.max_tokens is not None else (4096 if args.block == "d1" else 1024)
    fxs = s3.fixtures()[: args.n_fixtures]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    scales = ("x1e2", "x1e8") if args.block == "d1" else ("x1e3", "x1e8")

    jobs = []
    for cond in conds:
        for sc in scales:
            for fx in fxs:
                for opener in s3.OPENERS:
                    for sample in range(1, args.samples + 1):
                        jobs.append({"condition": cond, "scale": sc, "fixture": fx, "opener": opener, "sample": sample})

    rng = random.Random(20261122)
    rng.shuffle(jobs)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or os.path.join(ROOT, "data", "runs", f"addD-{args.block}-{args.model}-{ts}")
    os.makedirs(out_dir, exist_ok=True)
    games_path = os.path.join(out_dir, "games.jsonl")
    manifest_path = os.path.join(out_dir, "run_manifest.json")

    manifest = {
        "block": args.block,
        "model": args.model,
        "conditions": conds,
        "scales": list(scales),
        "n_fixtures": args.n_fixtures,
        "samples": args.samples,
        "thinking_budget": thinking_budget,
        "max_tokens": max_tokens,
        "concurrency": args.concurrency,
        "total_games": len(jobs),
        "started_utc": ts,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[addendum_d] block={args.block} model={args.model} games={len(jobs)} thinking_budget={thinking_budget} out={out_dir}", flush=True)
    lock = threading.Lock()
    done = 0
    errors = 0
    t0 = time.time()

    with open(games_path, "w", encoding="utf-8") as out_fh:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(play_one_game, j, args.block, args.model, thinking_budget, max_tokens) for j in jobs]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    rec = fut.result()
                    with lock:
                        out_fh.write(json.dumps(rec) + "\n")
                        out_fh.flush()
                        done += 1
                        if done % 40 == 0 or done == len(jobs):
                            elapsed = time.time() - t0
                            print(f"  progress: {done}/{len(jobs)} ({elapsed:.1f}s)", flush=True)
                except Exception as e:
                    with lock:
                        errors += 1
                        print(f"  ERROR: {e}", file=sys.stderr, flush=True)

    print(f"[addendum_d] finished {done}/{len(jobs)} games, errors={errors} in {time.time() - t0:.1f}s", flush=True)
    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
