#!/usr/bin/env python3
"""run_addendum_b.py: the API-model panel for Addendum B (prereg/addendum_B/ADDENDUM_B.md).

THIS SPENDS MONEY unless --preview. Prompts (only prompts) go to OpenRouter,
which forwards them to OpenAI, Anthropic and Google.

    python3 scripts/run_addendum_b.py --pilot --preview --confirm-cost X     free rehearsal (fake server)
    python3 scripts/run_addendum_b.py --pilot                                dry run: prints the worst case, sends nothing
    python3 scripts/run_addendum_b.py --pilot --confirm-cost X.XX            PAID format pilot (X-fixtures)
    python3 scripts/run_addendum_b.py --confirm-cost X.XX                    PAID registered run (F01-F30)

WHAT ONE CALL IS. The Study 2 prompt (scripts/render_prompt_s2.py, no
appraisal), message 1, no history: the opener's first move. Fixture f,
role (seller or buyer), scale V = $100 (x1e2) or $100M (x1e8), sample s.
The same seed rule as the digits probe (common random numbers): the seed
depends on (fixture, role, sample), not on the scale. It is sent only to
endpoints that accept one (OpenAI, Google); Anthropic's doesn't.

ARMS (all thinking off, vendor-default sampling, nothing else set):
  gpt-6-luna            openai/gpt-6-luna,            provider openai,           reasoning effort none
  claude-haiku-5.5      anthropic/claude-haiku-5.5,   provider anthropic,        reasoning enabled false
  gemini-3.5-flash-lite google/gemini-3.5-flash-lite, provider google-ai-studio, reasoning effort minimal
Every request pins the vendor's own endpoint (provider.only, no fallbacks,
require_parameters, data_collection deny).

RECORDS (scripts/run_records.py hash chains): calls.jsonl has, for every
try, the exact request body (headers are never logged; the key is only in a
header) and the raw response text with its sha256. games.jsonl has one
record per item: the first completed reply, scored with
game_engine.validate_reply, plus the returned model ID, provider, system
fingerprint, token counts (incl. reasoning tokens) and cost.

RETRIES. Only when no completed reply came back (transport error, non-200,
an error object, finish_reason "error"): up to 3 tries with the identical
request, each recorded with its own call_id. A completed reply is never
re-asked, whatever it says (invalid, refusal, walkaway all count as-is).

MONEY. The worst case is printed before anything is sent. A paid run needs
--confirm-cost equal to it (as scripts/launch_pod.sh). A hard spend cap
(--spend-cap, default $10) is enforced in this process from each response's
usage.cost; a call that could take the total past the cap is never sent.
The OpenRouter key's own credit limit is checked first (GET /api/v1/key, no
prompt). The registered run also refuses to start unless
prereg/addendum_B/inputs.sha256 is committed and every file it lists matches.

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
import run_records as rr            # noqa: E402

MAX_CHARS = 500                     # prereg/settings.json max_message_chars
MAX_MSGS = 16                       # prereg/settings.json max_messages
MAX_TOKENS = 1024                   # Addendum B: room for a little thinking on Gemini minimal
IN_TOKENS_WORST = 2000              # per-call input-token ceiling for the worst case (prompt ~2,700 chars)
MAX_TRIES = 3
BACKOFF = (2.0, 10.0)
BACKOFF_429 = (10.0, 30.0)
SCALES = (("x1e2", 100), ("x1e8", 100_000_000))
ROLES = ("seller", "buyer")
MAIN_SAMPLES = 4
PILOT_FIXTURES, PILOT_SAMPLES = 5, 2
PILOT_PHRASE = "scale-invariance/fixtures/explore-v1"          # X01.. (format pilot only, excluded from tests)
SEED_LABEL = "scale-invariance/addendum-b/v1"
ORDER_LABEL = "scale-invariance/addendum-b/order/v1"
FROZEN_LIST = os.path.join(ROOT, "prereg", "addendum_B", "inputs.sha256")

# Prices are list prices per token (OpenRouter /models/<id>/endpoints, read 2026-10-10;
# logs/api_panel_facts_20261010.txt). Used only for the worst case and when a response has no cost.
ARMS = {
    "gpt-6-luna": {"model": "openai/gpt-6-luna", "provider": "openai", "vendor": "OpenAI",
                   "canonical": "openai/gpt-6-luna-20260922", "reasoning": {"effort": "none"},
                   "send_seed": True, "price_in": 0.10e-6, "price_out": 0.50e-6,
                   "thinking_off": "reasoning.effort none (vendor: reasoning.effort none)"},
    "claude-haiku-5.5": {"model": "anthropic/claude-haiku-5.5", "provider": "anthropic", "vendor": "Anthropic",
                         "canonical": "anthropic/claude-haiku-5.5-20261007", "reasoning": {"enabled": False},
                         "send_seed": False, "price_in": 0.10e-6, "price_out": 0.50e-6,
                         "thinking_off": "reasoning.enabled false (vendor: thinking {type: disabled})"},
    "gemini-3.5-flash-lite": {"model": "google/gemini-3.5-flash-lite", "provider": "google-ai-studio",
                              "vendor": "Google AI Studio", "canonical": "google/gemini-3.5-flash-lite-20260721",
                              "reasoning": {"effort": "minimal"}, "send_seed": True,
                              "price_in": 0.30e-6, "price_out": 2.50e-6,
                              "thinking_off": "reasoning.effort minimal (vendor: thinking_level minimal; "
                                              "lowest level, not guaranteed zero)"},
}
for _a in ARMS.values():
    _a["max_tokens_cap"] = MAX_TOKENS


def worst_per_call(arm):
    return IN_TOKENS_WORST * arm["price_in"] + MAX_TOKENS * arm["price_out"]


def item_seed(fixture, role, sample):
    """Common random numbers: the scale is deliberately not an input. Masked to 31 bits:
    Google rejects seeds >= 2^31 (TYPE_INT32; format pilot 2026-10-10, amendment B1)."""
    label = f"{SEED_LABEL}|{fixture}|{role}|{sample}"
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def fixtures_for(pilot):
    if pilot:
        doc = mf.build(PILOT_PHRASE)
        return [{**f, "id": f"X{f['id'][1:]}"} for f in doc["fixtures"][:PILOT_FIXTURES]], PILOT_PHRASE
    with open(os.path.join(ROOT, "config", "fixtures.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc["fixtures"], doc["seed_phrase"]


def build_items(pilot, arms, samples):
    fixtures, phrase = fixtures_for(pilot)
    items = []
    for name in arms:
        for f in fixtures:
            for role in ROLES:
                pct = f["f_pct"] if role == "seller" else f["c_pct"]
                for tag, V in SCALES:
                    for s in range(1, samples + 1):
                        items.append({"item_id": f"{name}|{f['id']}-{role[0].upper()}-{tag}-s{s}", "arm": name,
                                      "fixture": f["id"], "role": role, "scale": tag, "V": V, "sample": s,
                                      "limit": pct * V // 100, "seed": item_seed(f["id"], role, s)})
    random.Random(rr.sha256_hex(f"{ORDER_LABEL}|{'pilot' if pilot else 'main'}".encode())).shuffle(items)
    for i, it in enumerate(items, 1):
        it["run_order"] = i
    return items, {"fixtures": fixtures, "seed_phrase": phrase}


def render_item(item):
    return rp2.render(item["role"], item["V"], item["limit"], [], 1, (75, 95), (105, 125), MAX_MSGS, MAX_CHARS)


def response_schema():
    """game_engine.response_schema without minimum/maxLength (not enforced by every vendor's
    structured-output mode); validate_reply still checks both."""
    s = ge.response_schema(MAX_CHARS)
    s["properties"]["price"] = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    s["properties"]["message"] = {"type": "string"}
    return s


def build_body(item, arms=ARMS):
    arm = arms[item["arm"]]
    body = {"model": arm["model"], "messages": render_item(item), "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "negotiation_move", "schema": response_schema(),
                                                "strict": True}},
            "provider": oc.provider_block(arm), "reasoning": arm["reasoning"], "stream": False}
    if arm["send_seed"]:
        body["seed"] = item["seed"]
    return body


def parse_reply(status, raw):
    """(fields, None) for a completed reply, or (None, problem) = infrastructure (retried)."""
    if status != 200:
        return None, f"HTTP {status}"
    if len(raw) > oc.MAX_RESPONSE:
        return None, "response larger than 1 MB"
    try:
        env = json.loads(raw.decode("utf-8"))
        if not isinstance(env, dict):
            return None, "not a JSON object"
        if env.get("error"):
            return None, "error object in response"
        choice = env["choices"][0]
        if choice.get("finish_reason") == "error" or choice.get("error"):
            return None, "finish_reason error"
        msg = choice["message"]
        content = msg.get("content")
        if content is not None and not isinstance(content, str):
            return None, "content is not text"
        usage = env.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        meta = env.get("openrouter_metadata") or {}
        selected = [e for e in ((meta.get("endpoints") or {}).get("available") or [])
                    if isinstance(e, dict) and e.get("selected")]
        cost = usage.get("cost")
        return {"content": content, "finish_reason": choice.get("finish_reason"),
                "native_finish_reason": choice.get("native_finish_reason"),
                "refusal": msg.get("refusal"), "reasoning": msg.get("reasoning"),
                "response_id": env.get("id"), "model_returned": env.get("model"),
                "provider_returned": env.get("provider") or (selected[0].get("provider") if selected else None),
                "model_version": selected[0].get("model") if selected else None,
                "system_fingerprint": env.get("system_fingerprint"), "service_tier": env.get("service_tier"),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": details.get("reasoning_tokens"),
                "cost": cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None}, None
    except (UnicodeDecodeError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None, "not a chat-completion response"


class SpendCap:
    """Hard cap. A try is sent only if spent + reserved + its own worst case <= cap."""

    def __init__(self, cap):
        self.cap, self.spent, self.reserved, self.unpriced = cap, 0.0, 0.0, 0
        self.lock = threading.Lock()

    def reserve(self, amount):
        with self.lock:
            if self.spent + self.reserved + amount > self.cap + 1e-12:
                return False
            self.reserved += amount
            return True

    def settle(self, reserved, actual):
        with self.lock:
            self.reserved -= reserved
            self.spent += actual


class CapReached(Exception):
    pass


class RequestRejected(Exception):
    """A 4xx other than 408/429: our request is wrong, so retrying can't help and the run stops."""


def recording_call(writer, transport, run_id, arms, cap, sleep=time.sleep):
    """Record every try (request before sending, raw response before parsing); never re-ask a
    completed reply. Returns {"kind": "model_output"|"infrastructure", ...}."""

    def call(body, item):
        arm = arms[item["arm"]]
        req_bytes = rr.canonical(body).encode("utf-8")
        req_sha = rr.sha256_hex(req_bytes)
        problem, call_id = None, None
        for t in range(1, MAX_TRIES + 1):
            call_id = f"{item['item_id']}/t{t}"
            wpc = worst_per_call(arm)
            if not cap.reserve(wpc):
                writer.event("spend_cap_reached", call_id=call_id, spent=round(cap.spent, 6), cap=cap.cap)
                raise CapReached(f"spend cap ${cap.cap:.2f} reached (spent ${cap.spent:.4f})")
            writer.calls.append({"schema": "api-call-request/1", "run_id": run_id, "call_id": call_id,
                                 "item_id": item["item_id"], "arm": item["arm"], "try": t, "t_sent": rr.utc_now(),
                                 "request_sha256": req_sha, "request": body})
            t0 = time.monotonic()
            status, raw, terr = transport(req_bytes)
            latency = round((time.monotonic() - t0) * 1000)
            raw = raw or b""
            fields, problem = (None, f"transport: {terr}") if terr else parse_reply(status, raw)
            # Billing: a completed reply bills usage.cost (or its worst case if absent); a failed try
            # is assumed unbilled only if it has no usage at all, else its usage.cost is counted.
            if fields is not None:
                actual = fields["cost"] if fields["cost"] is not None else wpc
                if fields["cost"] is None:
                    cap.unpriced += 1
            else:
                actual = _cost_of_failed(raw)
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
                raise RequestRejected(f"HTTP {status} for {call_id}: the request was rejected; see calls.jsonl")
            writer.event("infrastructure_retry" if t < MAX_TRIES else "infrastructure_give_up",
                         call_id=call_id, problem=problem)
            if t < MAX_TRIES:
                sleep((BACKOFF_429 if status == 429 else BACKOFF)[min(t - 1, 1)])
        return {"kind": "infrastructure", "call_id": call_id, "tries": MAX_TRIES, "problem": problem}

    return call


def _cost_of_failed(raw):
    try:
        c = json.loads(raw.decode("utf-8")).get("usage", {}).get("cost")
        return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else 0.0
    except (ValueError, AttributeError, UnicodeDecodeError, TypeError):
        return 0.0


def gap(role, price, limit):
    return price / limit - 1 if role == "seller" else 1 - price / limit


def score(item, reply):
    """One games.jsonl record per item (schema firstmove-api/1)."""
    rec = {"schema": "firstmove-api/1", **{k: item[k] for k in (
        "item_id", "arm", "fixture", "role", "scale", "V", "sample", "limit", "seed", "run_order")},
        "model_requested": ARMS[item["arm"]]["model"], "provider_requested": ARMS[item["arm"]]["provider"],
        "call_id": reply.get("call_id"), "tries": reply.get("tries"), "action": None, "price": None, "gap": None,
        "errors": [], "message_chars": None}
    for k in ("model_returned", "model_version", "provider_returned", "system_fingerprint", "service_tier", "response_id",
              "finish_reason", "native_finish_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens",
              "cost"):
        rec[k] = reply.get(k)
    rec["reasoning_text_present"] = reply.get("reasoning") not in (None, "")
    if reply.get("kind") != "model_output":
        rec["errors"] = ["INFRASTRUCTURE"]
        rec["problem"] = reply.get("problem")
        return rec
    if reply.get("refusal") or reply.get("finish_reason") == "content_filter":
        rec["errors"] = ["REFUSAL"]
        return rec
    move, errors = ge.validate_reply(reply.get("content"), reply.get("finish_reason"), False, MAX_CHARS)
    if errors:
        rec["errors"] = errors
        return rec
    rec["action"], rec["message_chars"] = move["action"], len(move["message"])
    if move["action"] == "OFFER":
        rec["price"] = move["price"]
        rec["gap"] = gap(item["role"], move["price"], item["limit"])
    return rec


def worst_case(items, arms=ARMS):
    one = sum(worst_per_call(arms[it["arm"]]) for it in items)
    return one, one * MAX_TRIES


def money(x):
    """Dollars rounded UP to the cent, as text (the gate compares this string)."""
    return f"{math.ceil(round(x * 100, 6)) / 100:.2f}"


def check_frozen():
    """Registered run only: inputs.sha256 committed and clean, every listed file matching."""
    if not os.path.exists(FROZEN_LIST):
        return "prereg/addendum_B/inputs.sha256 does not exist (the addendum isn't frozen yet)"
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
                return f"{p} does not match prereg/addendum_B/inputs.sha256"
    return None


def git_state():
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout
        return {"head": head, "uncommitted_paths": [l[3:] for l in dirty.splitlines()][:200]}
    except OSError:
        return {"head": None, "uncommitted_paths": None}


# ---------------------------------------------------------------- fake server (preview only)

FAKE_EFFECT = {"gpt-6-luna": -0.10, "claude-haiku-5.5": -0.04, "gemini-3.5-flash-lite": 0.0}


def fake_reply(body, seen, lock):
    """A synthetic OpenRouter response for preview. Reads role and limit from the prompt only."""
    arm_name = next(n for n, a in ARMS.items() if a["model"] == body["model"])
    arm = ARMS[arm_name]
    user = body["messages"][1]["content"]
    role = "seller" if user.startswith("You are the seller") else "buyer"
    limit = int(re.search(r"Your private (?:floor|ceiling) is (\d+) dollars", user).group(1))
    big = limit >= 10 ** 6
    h = int(rr.sha256_hex(rr.canonical(body).encode()), 16)
    with lock:
        first = h not in seen
        seen.add(h)
    if h % 41 == 0 and first:
        return 502, {"error": {"code": 502, "message": "fake upstream error (preview)"}}
    rng = random.Random(h)
    base = (0.30 if role == "seller" else 0.35) + (h % 7) / 100
    g = max(0.01, base + (FAKE_EFFECT[arm_name] if big else 0.0) + rng.gauss(0, 0.03))
    price = round(limit * (1 + g)) if role == "seller" else max(1, round(limit * (1 - g)))
    content = json.dumps({"action": "OFFER", "price": price, "message": "Opening offer (preview)."})
    refusal, finish = None, "stop"
    if h % 53 == 1:
        content = content + " Thanks!"
    elif h % 89 == 2:
        content, refusal = None, "I can't help with that. (preview)"
    elif h % 71 == 3:
        content = json.dumps({"action": "WALKAWAY", "price": None, "message": "No."})
    rtok = (h % 3) * 4 if arm_name == "gemini-3.5-flash-lite" else 0
    pt, ct = 680, 45 + rtok
    cost = pt * arm["price_in"] + ct * arm["price_out"]
    return 200, {"id": f"gen-preview-{h % 10**12}", "provider": arm["vendor"], "model": arm["model"],
                 "object": "chat.completion", "created": int(time.time()), "system_fingerprint": "fp_preview",
                 "choices": [{"index": 0, "finish_reason": finish, "native_finish_reason": finish,
                              "message": {"role": "assistant", "content": content, "refusal": refusal,
                                          "reasoning": None}}],
                 "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "cost": cost,
                           "completion_tokens_details": {"reasoning_tokens": rtok}},
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


# ---------------------------------------------------------------- run

def locked_writer(w):
    lock = threading.Lock()
    for ch in (w.calls, w.games, w.events):
        orig = ch.append

        def append(obj, _orig=orig):
            with lock:
                return _orig(obj)
        ch.append = append
    return w


def pilot_gate(records, arms):
    """Format pilot verdict per arm (the addendum's format criteria). Returns {arm: {...}}."""
    out = {}
    for name in arms:
        rs = [r for r in records if r["arm"] == name]
        done = [r for r in rs if r["errors"] != ["INFRASTRUCTURE"]]
        valid = [r for r in done if not r["errors"]]
        rtok = [r for r in done if (r.get("reasoning_tokens") or 0) > 0]
        prov = sorted({str(r.get("provider_returned")) for r in done})
        models = sorted({f"{r.get('model_returned')} = {r.get('model_version')}" for r in done})
        frac = len(valid) / len(done) if done else 0.0
        checks = {"valid_share": round(frac, 4), "valid_ok": frac >= 0.95 and len(done) == len(rs),
                  "replies": len(done), "items": len(rs), "invalid_by_error": {},
                  "replies_with_reasoning_tokens": len(rtok),
                  "reasoning_ok": name == "gemini-3.5-flash-lite" or not rtok,
                  "providers_returned": prov, "provider_ok": prov == [arms[name]["vendor"]],
                  "models_returned": models}
        for r in done:
            for e in r["errors"]:
                checks["invalid_by_error"][e] = checks["invalid_by_error"].get(e, 0) + 1
        checks["pass"] = checks["valid_ok"] and checks["reasoning_ok"] and checks["provider_ok"]
        out[name] = checks
    return out


def main(argv, stdout=sys.stdout, client_factory=None, sleep=time.sleep):
    def say(*a):
        print(*a, file=stdout, flush=True)

    ap = argparse.ArgumentParser(description="Addendum B API panel (see the top of this file).")
    ap.add_argument("--pilot", action="store_true", help="format pilot on X01-X05 (excluded from the test)")
    ap.add_argument("--preview", action="store_true", help="fake server on 127.0.0.1; nothing is sent anywhere")
    ap.add_argument("--confirm-cost", default=None, help="repeat the printed worst case to run")
    ap.add_argument("--spend-cap", type=float, default=10.0, help="hard cap in dollars (default 10)")
    ap.add_argument("--arms", default=",".join(ARMS), help="subset of arms (pilot or preview only)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    names = a.arms.split(",")
    if any(n not in ARMS for n in names) or len(set(names)) != len(names):
        say(f"bad --arms (choose from {', '.join(ARMS)})")
        return 2
    if not a.pilot and not a.preview and names != list(ARMS):
        say("the registered run uses all three arms")
        return 2
    if not 0 < a.spend_cap <= 25 or not 1 <= a.workers <= 16:
        say("bad --spend-cap (0-25) or --workers (1-16)")
        return 2
    arms = {n: ARMS[n] for n in names}
    samples = PILOT_SAMPLES if a.pilot else MAIN_SAMPLES
    items, fmeta = build_items(a.pilot, arms, samples)
    one, worst = worst_case(items, arms)
    worst = min(worst, a.spend_cap)
    worst_s = money(worst)
    kind = "pilot" if a.pilot else "registered run"
    say(f"Addendum B {kind}{' (PREVIEW: fake server, nothing sent)' if a.preview else ''}")
    say(f"  {len(items)} first moves: {len(arms)} models x {len(fmeta['fixtures'])} fixtures "
        f"({fmeta['fixtures'][0]['id']}..{fmeta['fixtures'][-1]['id']}) x 2 roles x 2 scales x {samples} samples")
    for n, arm in arms.items():
        k = sum(it["arm"] == n for it in items)
        say(f"  {n:22s} {arm['model']} via provider '{arm['provider']}', {arm['thinking_off']}; "
            f"{k} calls, worst ${k * worst_per_call(arm):.3f}")
    say(f"  Worst case if every reply bills {IN_TOKENS_WORST} input + {MAX_TOKENS} output tokens: ${one:.2f};"
        f" if every retry were billed too: ${one * MAX_TRIES:.2f}.")
    say(f"  Hard spend cap ${a.spend_cap:.2f}. WORST CASE: ${worst_s}")
    if a.confirm_cost is None:
        say(f"Dry run: nothing sent. To run, add:  --confirm-cost {worst_s}")
        return 0
    if a.confirm_cost.lstrip("$") != worst_s:
        say(f"--confirm-cost {a.confirm_cost} does not match the worst case {worst_s}. Nothing sent.")
        return 2
    if not a.pilot and not a.preview:
        problem = check_frozen()
        if problem:
            say("Not running: " + problem)
            return 3

    srv = fake_server() if a.preview else None
    try:
        if client_factory:
            client = client_factory(arms, srv)
        else:
            client = oc.Client(ARMS, preview_port=srv.server_address[1] if srv else None)
    except KeyError as e:
        say(f"Not running: {e}")
        return 3
    try:
        limit, usage, remaining = client.key_status()
    except RuntimeError as e:
        say(f"Not running: {e}")
        return 3
    say(f"  OpenRouter key: limit {limit}, used {usage}, remaining {remaining}")
    if remaining is not None and remaining < worst:
        say(f"Not running: the key's remaining credit ({remaining}) is below the worst case ({worst_s}).")
        return 3

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if a.out:
        data_root = a.out
    elif a.preview:
        data_root = tempfile.mkdtemp(prefix="si-addb-preview-")
    else:
        data_root = os.path.join(ROOT, "data", "pilot" if a.pilot else "runs")
    run_id = f"addB-{'pilot-' if a.pilot else ''}{'preview-' if a.preview else ''}{stamp}"
    run_dir = os.path.join(data_root, run_id)
    manifest = {"schema": "run_manifest/1", "run_id": run_id, "kind": "addendum_B_" + ("pilot" if a.pilot else "main"),
                "preview": a.preview, "registered": not a.pilot and not a.preview,
                "prereg": "prereg/addendum_B/ADDENDUM_B.md", "arms": arms, "provider_block_rule": "only [provider], "
                "allow_fallbacks false, require_parameters true, data_collection deny",
                "max_tokens": MAX_TOKENS, "samples": samples, "scales": dict(SCALES), "fixture_meta": fmeta,
                "response_schema": response_schema(),
                "seed_rule": f"sha256('{SEED_LABEL}|fixture|role|sample')[:8]; same at both scales; "
                             "sent only where send_seed",
                "sampling": "vendor defaults (no temperature/top_p/top_k sent)",
                "example_body_seller_x1e8": build_body({**items[0], "role": "seller", "V": 100_000_000,
                                                        "limit": 81_000_000}, arms),
                "spend_cap": a.spend_cap, "worst_case": worst, "git": git_state(),
                "items_planned": [it["item_id"] for it in items], "started_utc": rr.utc_now()}
    w = locked_writer(rr.RunWriter(run_dir, manifest))
    cap = SpendCap(a.spend_cap)
    call = recording_call(w, client.chat_transport(), run_id, arms, cap, sleep=sleep)
    stop, stop_why, lock = threading.Event(), [], threading.Lock()
    counts = {"done": 0, "valid": 0, "invalid": 0, "infra": 0}
    records = []

    def on_int(signum, frame):
        stop_why.append("stopped by you")
        stop.set()
    old = {s: signal.signal(s, on_int) for s in (signal.SIGINT, signal.SIGTERM)} \
        if threading.current_thread() is threading.main_thread() else {}

    q = queue.Queue()
    for it in items:
        q.put(it)

    def worker():
        while not stop.is_set():
            try:
                it = q.get_nowait()
            except queue.Empty:
                return
            try:
                reply = call(build_body(it, arms), it)
            except (CapReached, RequestRejected) as e:
                stop_why.append(str(e))
                stop.set()
                return
            rec = score(it, reply)
            w.game(rec)
            with lock:
                records.append(rec)
                counts["done"] += 1
                if rec["errors"] == ["INFRASTRUCTURE"]:
                    counts["infra"] += 1
                elif rec["errors"]:
                    counts["invalid"] += 1
                else:
                    counts["valid"] += 1
                if counts["done"] % 50 == 0 or counts["done"] == len(items):
                    say(f"  {counts['done']}/{len(items)} done; spent ${cap.spent:.4f}")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(0.5)
    for s, h in old.items():
        signal.signal(s, h)
    status = "complete" if counts["done"] == len(items) else "stopped"
    gate = pilot_gate(records, arms) if a.pilot else None
    if gate:
        w.event("pilot_format_gate", result=gate)
    w.close(status=status, why=stop_why[0] if stop_why else None, items=len(items), spent=round(cap.spent, 6),
            unpriced_replies=cap.unpriced, **counts)
    if srv:
        assert all(h is None for h in srv.auth_seen), "preview sent an Authorization header"
        srv.shutdown()
        srv.server_close()
    say(f"Addendum B {kind} {status}" + (f": {stop_why[0]}" if stop_why else "") + ".")
    say(f"  {counts['done']} of {len(items)} items; {counts['valid']} valid, {counts['invalid']} invalid or "
        f"refused, {counts['infra']} infrastructure failures. Spent ${cap.spent:.4f} (from usage.cost"
        f"{'; ' + str(cap.unpriced) + ' replies had no cost and were counted at their worst case' if cap.unpriced else ''}).")
    if gate:
        for n, g in gate.items():
            say(f"  pilot gate {n:22s} {'PASS' if g['pass'] else 'FAIL'}: valid {g['valid_share']:.0%} "
                f"({g['replies']}/{g['items']}), reasoning-token replies {g['replies_with_reasoning_tokens']}, "
                f"providers {g['providers_returned']}, models {g['models_returned']}"
                + (f", errors {g['invalid_by_error']}" if g['invalid_by_error'] else ""))
    say(f"  Data: {run_dir}")
    if a.pilot and not a.preview:
        say("  The pilot is format only (ADDENDUM_B.md §9): no HA number is computed from it.")
    else:
        say(f"  Next: python3 scripts/analyze_addendum_b.py {run_dir}")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
