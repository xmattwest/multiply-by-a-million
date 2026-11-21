#!/usr/bin/env python3
"""run_addendum_c.py: Study 2's full games on the three Addendum B API models
(prereg/addendum_C/ADDENDUM_C.md, DRAFT until frozen).

THIS SPENDS MONEY unless --preview. Prompts (only prompts) go to OpenRouter,
which forwards them to OpenAI, Anthropic and Google.

    python3 scripts/run_addendum_c.py --pilot --preview --confirm-cost X     free rehearsal (fake server)
    python3 scripts/run_addendum_c.py --pilot                                dry run: prints the worst case, sends nothing
    python3 scripts/run_addendum_c.py --pilot --confirm-cost X.XX            PAID format pilot (X01-X05)
    python3 scripts/run_addendum_c.py --confirm-cost key-limit               PAID registered run (F01-F30)

WHAT ONE GAME IS. Exactly Study 2's game: scripts/game_engine.play_game with
the Study 2 renderer (scripts/render_prompt_s2.py, no appraisal), at most 16
messages, 500-character messages, strict JSON moves checked by
game_engine.validate_reply (no repair: an invalid reply forfeits the game).
Both sides are the same model. Fixture f, opener (seller or buyer), scale
V = $100, $100K or $100M, sample s = 1..k.

HOW THE ENGINE IS REUSED UNCHANGED. play_game builds a vLLM request body for
every move. The `call` passed to it here takes only body["messages"] from
that body and builds the API request itself (Addendum B's arm settings:
pinned vendor endpoint, thinking off, vendor-default sampling, strict JSON
schema without minimum/maxLength). The engine's own seed is ignored; the
seed sent is this file's rule (fixture, opener, sample, message, attempt;
not the scale), masked to 31 bits, and only to endpoints that accept one.

ARMS: imported from scripts/run_addendum_b.py (frozen), so the three models,
their pinned providers and their thinking switches are exactly Addendum B's.
Only max_tokens differs in the worst-case arithmetic: later messages carry
the history, so a call is reserved at 6,000 input tokens, not 2,000.

RECORDS (scripts/run_records.py hash chains): calls.jsonl has, for every
try, the exact request body and the raw response text with its sha256.
games.jsonl has one record per game attempt (game_engine's game/1 record)
plus "api": the per-call model IDs, provider, dated version, fingerprint,
token counts, reasoning tokens, refusals and cost.

RETRIES. A call is retried (identical request, up to 3 tries) only when no
completed reply came back. A completed reply is never re-asked. If a call
still has no completed reply, the game attempt is INFRASTRUCTURE_VOID and
is replayed from the start as attempt 2 (then 3), with the attempt in the
seed, as in Studies 1-2 (data plan E8); after 3 void attempts the game is
missing. A 4xx rejection other than 408/429 stops the run.

MONEY. The pilot keeps its $1 in-process cap and numeric --confirm-cost.
The registered run uses the OpenRouter key's cap alone: --confirm-cost
key-limit acknowledges that policy. A capped key with positive remaining
credit is required; its limit, usage and remaining credit are recorded at
the start. Costs are still logged, but there is no main-run client cap or
requirement for credit to cover the theoretical worst case. A key-limit
rejection (HTTP 402) stops the run under the existing 4xx rule.
The registered run refuses to start
unless prereg/addendum_C/inputs.sha256 is committed and every file it lists
matches.

Standard library only.
"""
import argparse
import hashlib
import http.server
import json
import math
import os
import queue
import random
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import game_engine as ge            # noqa: E402
import make_fixtures as mf          # noqa: E402
import openrouter_client as oc      # noqa: E402
import render_prompt_s2 as rp2      # noqa: E402
import run_addendum_b as rb         # noqa: E402
import run_records as rr            # noqa: E402

ARMS = rb.ARMS                      # Addendum B's arms, unchanged
MAX_CHARS = rb.MAX_CHARS            # 500 (prereg/settings.json)
MAX_MSGS = rb.MAX_MSGS              # 16 (prereg/settings.json)
MAX_TOKENS = rb.MAX_TOKENS          # 1,024 (Addendum B)
IN_TOKENS_WORST = 6000              # per-call input ceiling: the longest Study 2 prompt is ~11k characters
MAX_TRIES = rb.MAX_TRIES            # 3 tries per call
MAX_ATTEMPTS = 3                    # game attempts (void games are replayed)
SCALES = (("x1e2", 100), ("x1e5", 100_000), ("x1e8", 100_000_000))
PILOT_SCALES = (("x1e2", 100), ("x1e8", 100_000_000))
OPENERS = ("seller", "buyer")
MAIN_SAMPLES = 4
PILOT_FIXTURES, PILOT_SAMPLES = 5, 1
PILOT_PHRASE = rb.PILOT_PHRASE      # X01..X05 (format pilot only, excluded from every test)
SEED_LABEL = "scale-invariance/addendum-c/v1"
ORDER_LABEL = "scale-invariance/addendum-c/order/v1"
FROZEN_LIST = os.path.join(ROOT, "prereg", "addendum_C", "inputs.sha256")
PILOT_CAP = 1.0                    # dollars; the main run relies on the OpenRouter key's cap
ENGINE_MODEL = "api"                # the key play_game looks up in settings["sampling"]; never sent


def worst_per_call(arm):
    return IN_TOKENS_WORST * arm["price_in"] + MAX_TOKENS * arm["price_out"]


def call_seed(fixture, opener, sample, msg_no, attempt):
    """Common random numbers: the scale is deliberately not an input. 31 bits (Google, amendment B1)."""
    label = f"{SEED_LABEL}|{fixture}|{opener}|{sample}|{msg_no}|{attempt}"
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def engine_settings():
    """prereg/settings.json's values the engine reads, plus a placeholder sampling entry that only
    satisfies game_engine.build_request (its body is never sent; only its messages are used)."""
    with open(os.path.join(ROOT, "prereg", "settings.json"), encoding="utf-8") as fh:
        s = json.load(fh)
    assert s["max_messages"]["value"] == MAX_MSGS and s["max_message_chars"]["value"] == MAX_CHARS
    s = json.loads(json.dumps(s))
    s["sampling"][ENGINE_MODEL] = {"temperature": None, "top_p": None, "top_k": None, "min_p": None}
    return s


def fixtures_for(pilot):
    return rb.fixtures_for(pilot)


def build_games(pilot, arms, samples):
    fixtures, phrase = fixtures_for(pilot)
    scales = PILOT_SCALES if pilot else SCALES
    games = []
    for name in arms:
        for f in fixtures:
            for tag, V in scales:
                for o in OPENERS:
                    for s in range(1, samples + 1):
                        games.append({"game_id": f"{name}|{f['id']}-{tag}-{o[0].upper()}-s{s}", "arm": name,
                                      "fixture": f["id"], "scale": tag, "V": V, "opener": o, "sample": s,
                                      "F": f["f_pct"] * V // 100, "C": f["c_pct"] * V // 100})
    random.Random(rr.sha256_hex(f"{ORDER_LABEL}|{'pilot' if pilot else 'main'}".encode())).shuffle(games)
    for i, g in enumerate(games, 1):
        g["run_order"] = i
    return games, {"fixtures": fixtures, "seed_phrase": phrase, "scales": dict(scales)}


def render(role, V, limit, history, msg_no, f_range, c_range, max_msgs, max_chars):
    return rp2.render(role, V, limit, history, msg_no, f_range, c_range, max_msgs, max_chars)


def build_body(arm_name, messages, seed, arms=ARMS):
    arm = arms[arm_name]
    body = {"model": arm["model"], "messages": messages, "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "negotiation_move", "schema": rb.response_schema(),
                                                "strict": True}},
            "provider": oc.provider_block(arm), "reasoning": arm["reasoning"], "stream": False}
    if arm["send_seed"]:
        body["seed"] = seed
    return body


class VoidGame(Exception):
    pass


def recording_call(writer, transport, run_id, arms, cap, sleep=time.sleep):
    """One API call per move, every try recorded before sending and before parsing.
    Same retry, billing and stop rules as run_addendum_b.recording_call, with this file's worst case.
    The main run uses an unbounded counter; only the pilot has a client-side cap."""

    def call(body, call_key, arm_name):
        arm = arms[arm_name]
        req_bytes = rr.canonical(body).encode("utf-8")
        req_sha = rr.sha256_hex(req_bytes)
        problem, call_id = None, None
        for t in range(1, MAX_TRIES + 1):
            call_id = f"{call_key}/t{t}"
            wpc = worst_per_call(arm)
            if not cap.reserve(wpc):
                writer.event("spend_cap_reached", call_id=call_id, spent=round(cap.spent, 6), cap=cap.cap)
                raise rb.CapReached(f"spend cap ${cap.cap:.2f} reached (spent ${cap.spent:.4f})")
            writer.calls.append({"schema": "api-call-request/1", "run_id": run_id, "call_id": call_id,
                                 "arm": arm_name, "try": t, "t_sent": rr.utc_now(),
                                 "request_sha256": req_sha, "request": body})
            t0 = time.monotonic()
            status, raw, terr = transport(req_bytes)
            latency = round((time.monotonic() - t0) * 1000)
            raw = raw or b""
            fields, problem = (None, f"transport: {terr}") if terr else rb.parse_reply(status, raw)
            if fields is not None:
                actual = fields["cost"] if fields["cost"] is not None else wpc
                if fields["cost"] is None:
                    cap.unpriced += 1
            else:
                actual = rb._cost_of_failed(raw)
            cap.settle(wpc, actual)
            writer.calls.append({"schema": "api-call-result/1", "run_id": run_id, "call_id": call_id,
                                 "t_received": rr.utc_now(), "latency_ms": latency, "http_status": status,
                                 "transport_error": terr, "response_sha256": rr.sha256_hex(raw),
                                 "response_text": raw[:oc.MAX_RESPONSE + 1].decode("utf-8", errors="replace"),
                                 "cost_counted": actual, "spent_after": round(cap.spent, 8)})
            if fields is not None:
                return {"kind": "model_output", "call_id": call_id, "tries": t, **fields}
            if status is not None and 400 <= status < 500 and status not in (408, 429):
                writer.event("request_rejected", call_id=call_id, http_status=status,
                             detail=raw[:2000].decode("utf-8", errors="replace"))
                raise rb.RequestRejected(f"HTTP {status} for {call_id}: the request was rejected; see calls.jsonl")
            writer.event("infrastructure_retry" if t < MAX_TRIES else "infrastructure_give_up",
                         call_id=call_id, problem=problem)
            if t < MAX_TRIES:
                sleep((rb.BACKOFF_429 if status == 429 else rb.BACKOFF)[min(t - 1, 1)])
        return {"kind": "infrastructure", "call_id": call_id, "tries": MAX_TRIES, "problem": problem}

    return call


API_FIELDS = ("model_returned", "model_version", "provider_returned", "system_fingerprint", "service_tier",
              "finish_reason", "native_finish_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens",
              "cost", "tries")


def play(game, attempt, api_call, settings, arms=ARMS):
    """One attempt of one game through the unchanged engine. Returns the game/1 record + "api"."""
    calls = []

    def engine_call(vllm_body, info):
        # Only the messages are taken from the engine's body; everything else is the arm's.
        seed = call_seed(game["fixture"], game["opener"], game["sample"], info["msg_no"], info["attempt"])
        body = build_body(game["arm"], vllm_body["messages"], seed, arms)
        reply = api_call(body, f"{game['game_id']}/a{info['attempt']}/m{info['msg_no']}", game["arm"])
        meta = {k: reply.get(k) for k in API_FIELDS}
        meta.update({"msg_no": info["msg_no"], "role": info["role"], "kind": reply.get("kind"),
                     "seed_sent": body.get("seed"), "refusal": reply.get("refusal") not in (None, ""),
                     "reasoning_text_present": reply.get("reasoning") not in (None, "")})
        calls.append(meta)
        return reply

    eg = {"game_id": game["game_id"], "fixture": game["fixture"], "scale": game["scale"], "V": game["V"],
          "F": game["F"], "C": game["C"], "opener": game["opener"], "block": None, "run_order": game["run_order"]}
    rec = ge.play_game(eg, ENGINE_MODEL, settings, (75, 95), (105, 125), engine_call, attempt=attempt, render=render)
    rec["schema"] = "game-api/1"
    rec["model"] = game["arm"]
    rec["arm"], rec["sample"] = game["arm"], game["sample"]
    rec["model_requested"] = arms[game["arm"]]["model"]
    rec["provider_requested"] = arms[game["arm"]]["provider"]
    rec["api"] = calls
    rec["cost"] = sum(c.get("cost") or 0.0 for c in calls)
    rec["refusals"] = sum(c["refusal"] for c in calls)
    rec["reasoning_token_calls"] = sum(1 for c in calls if (c.get("reasoning_tokens") or 0) > 0)
    return rec


def worst_case(games, arms=ARMS):
    """Every game runs 16 calls, every call bills its worst case; then with every retry and void replay."""
    one = sum(MAX_MSGS * worst_per_call(arms[g["arm"]]) for g in games)
    return one, one * MAX_TRIES * MAX_ATTEMPTS


def money(x):
    return rb.money(x)


def check_frozen():
    """Registered run only: inputs.sha256 committed and clean, every listed file matching."""
    if not os.path.exists(FROZEN_LIST):
        return "prereg/addendum_C/inputs.sha256 does not exist (the addendum isn't frozen yet)"
    rel = os.path.relpath(FROZEN_LIST, ROOT)
    r = subprocess.run(["git", "-C", ROOT, "ls-files", "--error-unmatch", rel], capture_output=True, text=True)
    if r.returncode != 0:
        return f"{rel} is not committed"
    with open(FROZEN_LIST, encoding="utf-8") as fh:
        lines = [l.split() for l in fh if l.strip() and not l.startswith("#")]
    files = [rel] + [p for _, p in lines]
    r = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--"] + files, capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.strip():
        return "frozen files have uncommitted changes:\n" + r.stdout
    for h, p in lines:
        with open(os.path.join(ROOT, p), "rb") as fh:
            if rr.sha256_hex(fh.read()) != h:
                return f"{p} does not match prereg/addendum_C/inputs.sha256"
    return None


# ---------------------------------------------------------------- fake server (preview only)

FAKE_MODESTY = {"gpt-6-luna": (0.08, 0.04), "claude-haiku-5.5": (0.15, 0.10), "gemini-3.5-flash-lite": (0.30, 0.10)}
HISTORY_RE = re.compile(r"Message (\d+), (you|the seller|the buyer): OFFER at (\d+) dollars")


def fake_reply(body, seen, lock):
    """A synthetic multi-turn negotiator for preview. Reads only the prompt: role, own limit, history.
    Opens at a markup/discount that shrinks at x10^6 (more for sellers), then concedes halfway and
    accepts once the gap is small. Never a result."""
    arm_name = next(n for n, a in ARMS.items() if a["model"] == body["model"])
    arm = ARMS[arm_name]
    user = body["messages"][1]["content"]
    role = "seller" if user.startswith("You are the seller") else "buyer"
    limit = int(re.search(r"Your private (?:floor|ceiling) is (\d+) dollars", user).group(1))
    hist = [(int(n), role if who == "you" else who[4:], int(p)) for n, who, p in HISTORY_RE.findall(user)]
    big = limit >= 10 ** 6
    h = int(rr.sha256_hex(rr.canonical(body).encode()), 16)
    with lock:
        first = h not in seen
        seen.add(h)
    if h % 97 == 0 and first:
        return 502, {"error": {"code": 502, "message": "fake upstream error (preview)"}}
    rng = random.Random(h)
    mine = [p for _, who, p in hist if who == role]
    theirs = [p for _, who, p in hist if who != role]
    if not mine:
        base = 0.40 + rng.uniform(-0.05, 0.05)
        cut = FAKE_MODESTY[arm_name][0 if role == "seller" else 1] if big else 0.0
        g = max(0.02, base - cut)
        price = round(limit * (1 + g)) if role == "seller" else max(1, round(limit * (1 - g)))
        move = {"action": "OFFER", "price": price, "message": "Opening offer (preview)."}
    else:
        last, other = mine[-1], theirs[-1]
        if abs(last - other) <= 0.03 * max(last, other) or len(hist) >= 12:
            move = {"action": "ACCEPT", "price": None, "message": "Deal (preview)."}
        else:
            price = round((last + other) / 2)
            price = max(price, limit) if role == "seller" else min(price, limit)
            move = {"action": "OFFER", "price": price, "message": "Meet me halfway (preview)."}
    content, refusal = json.dumps(move), None
    if h % 211 == 5:
        content = content + " Thanks!"
    pt, ct = 600 + 60 * len(hist), 45
    cost = pt * arm["price_in"] + ct * arm["price_out"]
    return 200, {"id": f"gen-preview-{h % 10**12}", "provider": arm["vendor"], "model": arm["model"],
                 "object": "chat.completion", "created": int(time.time()), "system_fingerprint": "fp_preview",
                 "choices": [{"index": 0, "finish_reason": "stop", "native_finish_reason": "stop",
                              "message": {"role": "assistant", "content": content, "refusal": refusal,
                                          "reasoning": None}}],
                 "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "cost": cost,
                           "completion_tokens_details": {"reasoning_tokens": 0}},
                 "openrouter_metadata": {"endpoints": {"available": [{"provider": arm["vendor"],
                                                                      "model": arm["canonical"], "selected": True}]}}}


def fake_server(key_limit=10.0):
    seen, lock = set(), threading.Lock()
    auth_seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            auth_seen.append(self.headers.get("Authorization"))
            if self.path != oc.KEY_PATH:
                return self._send(404, {"error": "not found"})
            self._send(200, {"data": {"label": "preview", "limit": key_limit, "usage": 0.0,
                                      "limit_remaining": key_limit}})

        def do_POST(self):
            auth_seen.append(self.headers.get("Authorization"))
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path != oc.CHAT_PATH:
                return self._send(404, {"error": "not found"})
            try:
                oc.check_chat_body(rr.canonical(body).encode(), ARMS)
            except ValueError as e:
                return self._send(400, {"error": {"code": 400, "message": str(e)}})
            self._send(*fake_reply(body, seen, lock))

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.auth_seen = auth_seen
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------- pilot gate

def pilot_gate(records, arms):
    """Format pilot verdict per arm (ADDENDUM_C.md §9): every call answered, >= 95% of replies valid,
    the vendor's own provider, no reasoning tokens (gpt-6-luna, claude-haiku-5.5). Format counts only."""
    out = {}
    for name in arms:
        rs = [r for r in records if r["arm"] == name]
        calls = [c for r in rs for c in r["api"]]
        done = [c for c in calls if c["kind"] == "model_output"]
        invalid = {}
        for r in rs:
            for d in r["diagnostics"]:
                if d.startswith("INVALID:"):
                    for e in d.split(":", 2)[2].split(","):
                        invalid[e] = invalid.get(e, 0) + 1
        n_invalid = sum(1 for r in rs if r["terminal"] == "PROTOCOL_FORFEIT")
        frac = 1 - n_invalid / len(done) if done else 0.0
        rtok = sum(1 for c in done if (c.get("reasoning_tokens") or 0) > 0)
        prov = sorted({str(c.get("provider_returned")) for c in done})
        models = sorted({f"{c.get('model_returned')} = {c.get('model_version')}" for c in done})
        voids = sum(1 for r in rs if r["terminal"] == "INFRASTRUCTURE_VOID")
        checks = {"replies": len(done), "calls": len(calls), "games": len(rs), "void_attempts": voids,
                  "valid_share": round(frac, 4), "valid_ok": frac >= 0.95 and len(done) == len(calls),
                  "invalid_by_error": invalid, "replies_with_reasoning_tokens": rtok,
                  "reasoning_ok": name == "gemini-3.5-flash-lite" or rtok == 0,
                  "providers_returned": prov, "provider_ok": prov == [arms[name]["vendor"]],
                  "models_returned": models,
                  "mean_calls_per_game_pooled": round(statistics.fmean(len(r["api"]) for r in rs), 2) if rs else None,
                  "mean_cost_per_game": round(statistics.fmean(r["cost"] for r in rs), 6) if rs else None}
        checks["pass"] = checks["valid_ok"] and checks["reasoning_ok"] and checks["provider_ok"]
        out[name] = checks
    return out


# ---------------------------------------------------------------- run

def main(argv, stdout=sys.stdout, client_factory=None, sleep=time.sleep):
    def say(*a):
        print(*a, file=stdout, flush=True)

    ap = argparse.ArgumentParser(description="Addendum C full games (see the top of this file).")
    ap.add_argument("--pilot", action="store_true", help="format pilot on X01-X05 (excluded from every test)")
    ap.add_argument("--preview", action="store_true", help="fake server on 127.0.0.1; nothing is sent anywhere")
    ap.add_argument("--confirm-cost", default=None,
                    help="pilot: repeat the printed cap/worst case; main run: key-limit")
    ap.add_argument("--spend-cap", type=float, default=None,
                    help=f"pilot only: hard cap in dollars (default {PILOT_CAP:g})")
    ap.add_argument("--arms", default=",".join(ARMS), help="subset of arms (pilot or preview only)")
    ap.add_argument("--samples", type=int, default=None, help=argparse.SUPPRESS)   # preview/tests only
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.pilot and a.spend_cap is None:
        a.spend_cap = PILOT_CAP

    names = a.arms.split(",")
    if any(n not in ARMS for n in names) or len(set(names)) != len(names):
        say(f"bad --arms (choose from {', '.join(ARMS)})")
        return 2
    if not a.pilot and not a.preview and (names != list(ARMS) or a.samples is not None):
        say("the registered run uses all three arms and k = 4")
        return 2
    if not a.pilot and a.spend_cap is not None:
        say("--spend-cap is pilot-only; the main run uses the OpenRouter key's cap")
        return 2
    if (a.pilot and not 0 < a.spend_cap <= 25) or not 1 <= a.workers <= 16:
        say("bad pilot --spend-cap (0-25) or --workers (1-16)")
        return 2
    arms = {n: ARMS[n] for n in names}
    samples = a.samples or (PILOT_SAMPLES if a.pilot else MAIN_SAMPLES)
    games, fmeta = build_games(a.pilot, arms, samples)
    one, worst = worst_case(games, arms)
    if a.pilot:
        worst = min(worst, a.spend_cap)
    worst_s = money(worst)
    confirm_s = worst_s if a.pilot else "key-limit"
    kind = ("pilot" if a.pilot else "registered run") if not a.preview else ("pilot rehearsal" if a.pilot else "run rehearsal")
    say(f"Addendum C {kind}{' (PREVIEW: fake server, nothing sent)' if a.preview else ''}")
    say(f"  {len(games)} games: {len(arms)} models x {len(fmeta['fixtures'])} fixtures "
        f"({fmeta['fixtures'][0]['id']}..{fmeta['fixtures'][-1]['id']}) x {len(fmeta['scales'])} scales "
        f"x 2 openers x {samples} samples; up to {MAX_MSGS} calls per game")
    for n, arm in arms.items():
        k = sum(g["arm"] == n for g in games)
        say(f"  {n:22s} {arm['model']} via provider '{arm['provider']}', {arm['thinking_off']}; "
            f"{k} games, worst ${k * MAX_MSGS * worst_per_call(arm):.2f}")
    say(f"  Worst case if every game runs {MAX_MSGS} calls of {IN_TOKENS_WORST} input + {MAX_TOKENS} output tokens:"
        f" ${one:.2f}; with every retry and void replay billed: ${one * MAX_TRIES * MAX_ATTEMPTS:.2f}.")
    if a.pilot:
        say(f"  Hard spend cap ${a.spend_cap:.2f} (never crossed; the run stops instead). WORST CASE: ${worst_s}")
    else:
        say("  Spending limit: OpenRouter key cap only; no main-run client cap.")
    if a.confirm_cost is None:
        say(f"Dry run: nothing sent. To run, add:  --confirm-cost {confirm_s}")
        return 0
    if a.confirm_cost.lstrip("$") != confirm_s:
        say(f"--confirm-cost {a.confirm_cost} does not match the required confirmation {confirm_s}. Nothing sent.")
        return 2
    if not a.pilot and not a.preview:
        problem = check_frozen()
        if problem:
            say("Not running: " + problem)
            return 3

    srv = fake_server(key_limit=25.0) if a.preview else None
    try:
        client = client_factory(arms, srv) if client_factory else \
            oc.Client(ARMS, preview_port=srv.server_address[1] if srv else None)
    except KeyError as e:
        say(f"Not running: {e}")
        return 3
    try:
        limit, usage, remaining = client.key_status()
    except RuntimeError as e:
        say(f"Not running: {e}")
        return 3
    say(f"  OpenRouter key: limit {limit}, used {usage}, remaining {remaining}")
    if a.pilot and remaining is not None and remaining < worst:
        say(f"Not running: the key's remaining credit ({remaining}) is below the cap/worst case ({worst_s}).")
        return 3
    if not a.pilot and not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                              and math.isfinite(x) and x > 0 for x in (limit, remaining)):
        say("Not running: the main run needs a capped OpenRouter key with positive remaining credit.")
        return 3

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if a.out:
        data_root = a.out
    elif a.preview:
        data_root = tempfile.mkdtemp(prefix="si-addc-preview-")
    else:
        data_root = os.path.join(ROOT, "data", "pilot" if a.pilot else "runs")
    run_id = f"addC-{'pilot-' if a.pilot else ''}{'preview-' if a.preview else ''}{stamp}"
    run_dir = os.path.join(data_root, run_id)
    settings = engine_settings()
    manifest = {"schema": "run_manifest/1", "run_id": run_id, "kind": "addendum_C_" + ("pilot" if a.pilot else "main"),
                "preview": a.preview, "registered": not a.pilot and not a.preview,
                "prereg": "prereg/addendum_C/ADDENDUM_C.md", "arms": arms,
                "provider_block_rule": "only [provider], allow_fallbacks false, require_parameters true, "
                                       "data_collection deny",
                "max_tokens": MAX_TOKENS, "max_messages": MAX_MSGS, "max_message_chars": MAX_CHARS,
                "samples": samples, "fixture_meta": fmeta, "response_schema": rb.response_schema(),
                "renderer": "scripts/render_prompt_s2.py (Study 2, no appraisal)",
                "seed_rule": f"sha256('{SEED_LABEL}|fixture|opener|sample|msg_no|attempt')[:8] & 0x7fffffff; "
                             "same at every scale; sent only where send_seed",
                "sampling": "vendor defaults (no temperature/top_p/top_k sent)",
                "spend_cap": a.spend_cap, "worst_case": worst, "git": rb.git_state(),
                "spend_policy": "client_and_key_caps" if a.pilot else "openrouter_key_cap_only",
                "key_credit_at_start": {"limit": limit, "usage": usage, "remaining": remaining},
                "games_planned": [g["game_id"] for g in games], "started_utc": rr.utc_now()}
    w = rb.locked_writer(rr.RunWriter(run_dir, manifest))
    cap = rb.SpendCap(a.spend_cap if a.pilot else math.inf)
    api_call = recording_call(w, client.chat_transport(), run_id, arms, cap, sleep=sleep)
    stop, stop_why, lock = threading.Event(), [], threading.Lock()
    counts = {"games_done": 0, "games_missing": 0, "void_attempts": 0, "forfeits": 0}
    records = []

    def on_int(signum, frame):
        stop_why.append("stopped by you")
        stop.set()
    old = {s: signal.signal(s, on_int) for s in (signal.SIGINT, signal.SIGTERM)} \
        if threading.current_thread() is threading.main_thread() else {}

    q = queue.Queue()
    for g in games:
        q.put(g)

    def worker():
        while not stop.is_set():
            try:
                g = q.get_nowait()
            except queue.Empty:
                return
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if stop.is_set():
                    return
                try:
                    rec = play(g, attempt, api_call, settings, arms)
                except (rb.CapReached, rb.RequestRejected) as e:
                    stop_why.append(str(e))
                    stop.set()
                    w.event("game_interrupted", game_id=g["game_id"], attempt=attempt, why=str(e))
                    return
                w.game(rec)
                with lock:
                    records.append(rec)
                    if rec["terminal"] == "INFRASTRUCTURE_VOID":
                        counts["void_attempts"] += 1
                        if attempt == MAX_ATTEMPTS:
                            counts["games_missing"] += 1
                        continue
                    counts["games_done"] += 1
                    counts["forfeits"] += rec["terminal"] == "PROTOCOL_FORFEIT"
                    n = counts["games_done"] + counts["games_missing"]
                    if n % 50 == 0 or n == len(games):
                        say(f"  {n}/{len(games)} games done; spent ${cap.spent:.4f}")
                break

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(0.5)
    for s, h in old.items():
        signal.signal(s, h)
    finished = counts["games_done"] + counts["games_missing"]
    status = "complete" if finished == len(games) else "stopped"
    gate = pilot_gate(records, arms) if a.pilot else None
    if gate:
        w.event("pilot_format_gate", result=gate)
    w.close(status=status, why=stop_why[0] if stop_why else None, games=len(games), spent=round(cap.spent, 6),
            unpriced_replies=cap.unpriced, **counts)
    if srv:
        assert all(h is None for h in srv.auth_seen), "preview sent an Authorization header"
        srv.shutdown()
        srv.server_close()
    say(f"Addendum C {kind} {status}" + (f": {stop_why[0]}" if stop_why else "") + ".")
    say(f"  {counts['games_done']} of {len(games)} games scored ({counts['forfeits']} forfeits), "
        f"{counts['games_missing']} missing, {counts['void_attempts']} void attempts replayed. "
        f"Spent ${cap.spent:.4f} (from usage.cost"
        f"{'; ' + str(cap.unpriced) + ' replies had no cost and were counted at their worst case' if cap.unpriced else ''}).")
    if gate:
        for n, g in gate.items():
            say(f"  pilot gate {n:22s} {'PASS' if g['pass'] else 'FAIL'}: valid {g['valid_share']:.0%} "
                f"({g['replies']}/{g['calls']} calls answered), reasoning-token replies "
                f"{g['replies_with_reasoning_tokens']}, providers {g['providers_returned']}, "
                f"models {g['models_returned']}; {g['mean_calls_per_game_pooled']} calls and "
                f"${g['mean_cost_per_game']} per game (pooled, for the cost estimate)"
                + (f", errors {g['invalid_by_error']}" if g['invalid_by_error'] else ""))
    say(f"  Data: {run_dir}")
    if a.pilot and not a.preview:
        say("  The pilot is format only (ADDENDUM_C.md §9): no split, opening or midpoint number is computed from it.")
    else:
        say(f"  Next: python3 scripts/analyze_addendum_c.py {run_dir}")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
