#!/usr/bin/env python3
"""study3.py: the parts of Study 3 shared by its runner, its replay audit and its analysis
(prereg/study3/PREREGISTRATION.md, DRAFT until frozen).

WHAT ONE GAME IS. Study 2's game, played by scripts/game_engine.play_game unchanged: at most 16
messages, 500-character messages, strict JSON moves checked by game_engine.validate_reply (no repair;
an invalid reply forfeits). One exception, decided after the format pilot (2026-10-14): text after the
first JSON object is ignored (first_json_only) and counted; the object itself is still checked in full.
Both sides are the same model. A game is one arm (model), one fixture,
one cell, one opener and one sample s = 1..4.

CELLS
  Core grid (§3.1): A0 (Study 2 prompt) and A1 (Study 2 prompt + the appraisal line,
  scripts/render_prompt_s3.py), each at $100, $100K and $100M: "A0-x1e2" ... "A1-x1e8".
  Opening clamp (§3.2), A0 prompt: game scale G and opening source O, each $100 or $100M:
  "K-x1e2-o1e2", "K-x1e2-o1e8", "K-x1e8-o1e2", "K-x1e8-o1e8". Messages 1 and 2 are scripted: the
  opener's first offer and the responder's counter-offer, at the frozen first-offer positions in zone
  units for (arm, opener, O) in prereg/study3/clamp_values.json, priced on this game's own limits
  (F + x (C - F), rounded to a multiple of V/100, so the $100M prices are the $100 prices with six zeros
  appended), with clamp.txt's text.

HOW THE CLAMP REUSES THE ENGINE UNCHANGED. The `call` given to play_game answers messages 1 and 2
itself, with exactly the JSON a model would have written ({"action": "OFFER", ...}); no request is
sent. The engine validates them, appends them to the public history and renders them like any other
move ("Message 1, you: OFFER at ..."). From message 3 every move is the model's. The game record
lists the scripted message numbers, and the replay audit re-derives them from the frozen values.

REQUESTS. Only the engine's messages are used; the body is built here.
  API arms: exactly Addendum C's body (scripts/run_addendum_c.build_body: pinned vendor endpoint,
    thinking off, vendor-default sampling, max_tokens 1,024, the schema without minimum/maxLength).
  vLLM arms: game_engine.build_request (family sampling, penalties, thinking off), with max_tokens
    1,024 and the same schema as the API arms (decided 2026-10-13: one schema for every arm).
  Seed: sha256('scale-invariance/study3/v1|fixture|opener|sample|msg_no|attempt')[:8] & 0x7fffffff.
    Scale, appraisal and clamp cell are not inputs (common random numbers). Not sent to Anthropic.

Standard library only. No network in this file.
"""
import hashlib
import json
import os
import random
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import game_engine as ge            # noqa: E402
import make_fixtures as mf          # noqa: E402
import render_prompt as rp          # noqa: E402
import render_prompt_s2 as rp2      # noqa: E402
import render_prompt_s3 as rp3      # noqa: E402
import run_addendum_b as rb         # noqa: E402
import run_addendum_c as rac        # noqa: E402
import run_records as rr            # noqa: E402

PREREG = os.path.join(ROOT, "prereg", "study3")
FROZEN_LIST = os.path.join(PREREG, "inputs.sha256")
CLAMP_VALUES = os.path.join(PREREG, "clamp_values.json")
CLAMP_TEXT = os.path.join(PREREG, "prompts", "clamp.txt")

SEED_LABEL = "scale-invariance/study3/v1"
ORDER_LABEL = "scale-invariance/study3/order/v1"
FIXTURE_PHRASE, N_FIXTURES, FIXTURE_PREFIX = "scale-invariance/fixtures/study3-v1", 40, "G"
PILOT_PHRASE, N_PILOT, PILOT_PREFIX = "scale-invariance/fixtures/study3-pilot", 5, "Q"
F_RANGE, C_RANGE = mf.F_RANGE, mf.C_RANGE          # (75, 95), (105, 125): Studies 1-2's geometry
SCALES = {"x1e2": 100, "x1e5": 100_000, "x1e8": 100_000_000}
LOW, MID, HIGH = "x1e2", "x1e5", "x1e8"
OPENERS = ("seller", "buyer")
OTHER = {"seller": "buyer", "buyer": "seller"}
SAMPLES = 4
SESSIONS = {1: (1, 2), 2: (3, 4)}                  # samples per session (§3.3)
MAX_MSGS, MAX_CHARS, MAX_TOKENS = 16, 500, 1024
MAX_ATTEMPTS = 3

API_ARMS = rb.ARMS                                  # Addendum B/C's three arms, unchanged
VLLM_ARMS = ("qwen3.8-27b-fp8", "gemma-4-31b-it")
ARMS = tuple(VLLM_ARMS) + tuple(API_ARMS)

CORE_CELLS = tuple(f"{a}-{s}" for a in ("A0", "A1") for s in (LOW, MID, HIGH))
CLAMP_CELLS = tuple(f"K-{g}-o{o[1:]}" for g in (LOW, HIGH) for o in (LOW, HIGH))
CELLS = CORE_CELLS + CLAMP_CELLS
PILOT_CELLS = ("A0-x1e2", "A0-x1e8", "A1-x1e2", "A1-x1e8", "K-x1e8-o1e2")


def backend(arm):
    if arm in VLLM_ARMS:
        return "vllm"
    if arm in API_ARMS:
        return "api"
    raise ValueError(f"unknown arm {arm!r}")


def parse_cell(cell):
    """'A1-x1e5' -> ('A1', 'x1e5', None); 'K-x1e8-o1e2' -> ('K', 'x1e8', 'x1e2')."""
    parts = cell.split("-")
    if parts[0] in ("A0", "A1") and len(parts) == 2 and parts[1] in SCALES:
        return parts[0], parts[1], None
    if parts[0] == "K" and len(parts) == 3 and parts[1] in (LOW, HIGH) and parts[2] in ("o1e2", "o1e8"):
        return "K", parts[1], "x" + parts[2][1:]
    raise ValueError(f"bad cell {cell!r}")


# ---------------------------------------------------------------- fixtures

def fixtures(phrase=FIXTURE_PHRASE, n=N_FIXTURES, prefix=FIXTURE_PREFIX):
    """make_fixtures' draws (same labels "fixture/<i>/f" and "fixture/<i>/c", same ranges), n of them.
    The first 30 equal make_fixtures.build(phrase)'s 30; make_fixtures.py itself is not changed."""
    seed = mf.seed_from_phrase(phrase)
    out = []
    for i in range(1, n + 1):
        f = F_RANGE[0] + mf.derive(seed, f"fixture/{i}/f", F_RANGE[1] - F_RANGE[0] + 1)
        c = C_RANGE[0] + mf.derive(seed, f"fixture/{i}/c", C_RANGE[1] - C_RANGE[0] + 1)
        out.append({"id": f"{prefix}{i:02d}", "f_pct": f, "c_pct": c})
    return out


def appraisal_position(fx):
    """Where V sits in the zone, floor 0 and ceiling 1 (the S9 moderator)."""
    return Fraction(100 - fx["f_pct"], fx["c_pct"] - fx["f_pct"])


# ---------------------------------------------------------------- games

def game_id(arm, fixture, cell, opener, sample):
    return f"{arm}|{fixture}-{cell}-{opener[0].upper()}-s{sample}"


def make_game(arm, fx, cell, opener, sample):
    cond, scale, source = parse_cell(cell)
    V = SCALES[scale]
    return {"game_id": game_id(arm, fx["id"], cell, opener, sample), "arm": arm, "backend": backend(arm),
            "fixture": fx["id"], "f_pct": fx["f_pct"], "c_pct": fx["c_pct"], "cell": cell, "condition": cond,
            "scale": scale, "source": source, "V": V, "F": fx["f_pct"] * V // 100, "C": fx["c_pct"] * V // 100,
            "opener": opener, "sample": sample}


def build_games(arms, pilot=False, session=None):
    """Every game of one run (one backend's arms; one session, or the pilot), in one fixed shuffle."""
    if pilot:
        fxs, cells, samples, tag = fixtures(PILOT_PHRASE, N_PILOT, PILOT_PREFIX), PILOT_CELLS, (1,), "pilot"
    else:
        if session not in SESSIONS:
            raise ValueError("session must be 1 or 2")
        fxs, cells, samples, tag = fixtures(), CELLS, SESSIONS[session], f"session{session}"
    games = [make_game(a, f, c, o, s) for a in arms for f in fxs for c in cells for o in OPENERS for s in samples]
    random.Random(rr.sha256_hex(f"{ORDER_LABEL}|{'+'.join(sorted(arms))}|{tag}".encode())).shuffle(games)
    for i, g in enumerate(games, 1):
        g["run_order"] = i
    return games, {"fixtures": fxs, "seed_phrase": PILOT_PHRASE if pilot else FIXTURE_PHRASE, "cells": list(cells),
                   "samples": list(samples), "order_label": f"{ORDER_LABEL}|{'+'.join(sorted(arms))}|{tag}"}


def call_seed(fixture, opener, sample, msg_no, attempt):
    label = f"{SEED_LABEL}|{fixture}|{opener}|{sample}|{msg_no}|{attempt}"
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def renderer(game):
    return rp3.render if game["condition"] == "A1" else rp2.render


# ---------------------------------------------------------------- the clamp

def load_clamp_values(path=CLAMP_VALUES):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("schema") != "study3-clamp-values/1":
        raise ValueError(f"{path}: not a study3 clamp-values file")
    return doc


def round_half_up(x):
    return int((Fraction(x) + Fraction(1, 2)).__floor__())


def clamp_prices(game, values):
    """(opener's price, responder's price) for a clamp game: the frozen first-offer positions in zone units
    (decimal strings x, floor 0 and ceiling 1), priced on this game's own limits, F + x (C - F), rounded half up
    to a multiple of V/100 (whole dollars at $100, $1M at $100M). The scripted prices of a fixture's $100M
    game are therefore its $100 game's prices with six zeros appended, as every other amount is."""
    v = values["values"][game["arm"]][game["opener"]][game["source"]]
    xs, xb = Fraction(v["x_seller"]), Fraction(v["x_buyer"])
    w, unit = game["C"] - game["F"], game["V"] // 100
    ps = round_half_up((game["F"] + xs * w) / unit) * unit
    pb = round_half_up((game["F"] + xb * w) / unit) * unit
    if pb < 1 or ps < 1 or not pb < ps:
        raise ValueError(f"{game['game_id']}: scripted prices not usable (seller {ps}, buyer {pb})")
    return (ps, pb) if game["opener"] == "seller" else (pb, ps)


_CLAMP_TEXT = {}


def clamp_text(role, price):
    if not _CLAMP_TEXT:
        _CLAMP_TEXT.update(rp.load_fragments(CLAMP_TEXT))
    return rp.fill(_CLAMP_TEXT[f"clamp_{role}"], {"PRICE": rp.money(price)})


def clamp_prefix(game, values):
    """[(role, price, text), (role, price, text)] for messages 1 and 2, or [] for a core game."""
    if game["condition"] != "K":
        return []
    p1, p2 = clamp_prices(game, values)
    o, r = game["opener"], OTHER[game["opener"]]
    return [(o, p1, clamp_text(o, p1)), (r, p2, clamp_text(r, p2))]


# ---------------------------------------------------------------- request bodies

def engine_settings():
    """prereg/settings.json plus a placeholder sampling entry; the engine's own body is never sent."""
    return rac.engine_settings()


def vllm_settings():
    with open(os.path.join(ROOT, "prereg", "settings.json"), encoding="utf-8") as fh:
        return json.load(fh)


def response_schema():
    return rb.response_schema()            # the same schema for every arm (§4)


def vllm_body(model, messages, seed, settings):
    body = ge.build_request(model, messages, settings, seed)
    body["max_tokens"] = MAX_TOKENS
    body["response_format"] = {"type": "json_schema", "json_schema": {
        "name": "negotiation_move", "schema": response_schema(), "strict": True}}
    return body


def build_body(game, messages, seed, vsettings=None):
    if game["backend"] == "api":
        return rac.build_body(game["arm"], messages, seed, API_ARMS)
    return vllm_body(game["arm"], messages, seed, vsettings or vllm_settings())


# ---------------------------------------------------------------- one game

API_FIELDS = rac.API_FIELDS
VLLM_FIELDS = ("finish_reason", "prompt_tokens", "completion_tokens", "response_id")


class Stop(Exception):
    """Stop playing (Ctrl-C, spend cap, rejected request, machine gone, time nearly up)."""


def first_json_only(content):
    """Decided 2026-10-14 (format pilot): text after the first JSON object is ignored. Returns
    (content for the engine, True if non-whitespace text followed the first object). Only cuts; it never
    edits the object, so game_engine.validate_reply still checks the move itself in full (fields, types,
    price, 500 characters, duplicate keys, NaN). A reply that doesn't start with a parseable object, or
    that was truncated (validate_reply checks finish_reason first), is passed on unchanged and forfeits."""
    if not isinstance(content, str):
        return content, False
    text = content.strip(ge.JSON_WS)
    if not text.startswith("{"):
        return content, False
    try:
        _, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return content, False
    if not text[end:].strip(ge.JSON_WS):
        return content, False
    return text[:end], True


def play(game, attempt, send, clamp_values, vsettings=None):
    """One attempt of one game. send(body, call_key, game, info) -> the reply dict for the engine.
    Returns the games.jsonl record (schema study3-game/1)."""
    prefix = clamp_prefix(game, clamp_values) if game["condition"] == "K" else []
    calls = []

    def engine_call(engine_body, info):
        n = info["msg_no"]
        if n <= len(prefix):
            role, price, text = prefix[n - 1]
            if role != info["role"]:
                raise AssertionError("scripted move out of turn")
            return {"kind": "model_output", "call_id": None, "finish_reason": "stop", "reasoning": None,
                    "content": json.dumps({"action": "OFFER", "price": price, "message": text})}
        seed = call_seed(game["fixture"], game["opener"], game["sample"], n, info["attempt"])
        body = build_body(game, engine_body["messages"], seed, vsettings)
        reply = send(body, f"{game['game_id']}/a{info['attempt']}/m{n}", game, info)
        fields = API_FIELDS if game["backend"] == "api" else VLLM_FIELDS
        meta = {k: reply.get(k) for k in fields}
        meta.update({"msg_no": n, "role": info["role"], "kind": reply.get("kind"), "seed_sent": body.get("seed"),
                     "refusal": reply.get("refusal") not in (None, ""),
                     "reasoning_text_present": reply.get("reasoning") not in (None, "")})
        trailing = False
        if reply.get("kind") == "model_output":
            content, trailing = first_json_only(reply.get("content"))
            if trailing:
                reply = {**reply, "content": content}
        meta["trailing_text_ignored"] = trailing
        calls.append(meta)
        return reply

    eg = {"game_id": game["game_id"], "fixture": game["fixture"], "scale": game["scale"], "V": game["V"],
          "F": game["F"], "C": game["C"], "opener": game["opener"], "block": None, "run_order": game["run_order"]}
    rec = ge.play_game(eg, rac.ENGINE_MODEL, engine_settings(), F_RANGE, C_RANGE, engine_call,
                       attempt=attempt, render=renderer(game))
    rec["schema"] = "study3-game/1"
    rec["model"] = game["arm"]
    for k in ("arm", "backend", "cell", "condition", "source", "sample", "f_pct", "c_pct"):
        rec[k] = game[k]
    rec["scripted_msgs"] = list(range(1, len(prefix) + 1))
    if game["backend"] == "api":
        rec["model_requested"] = API_ARMS[game["arm"]]["model"]
        rec["provider_requested"] = API_ARMS[game["arm"]]["provider"]
    rec["calls"] = calls
    rec["cost"] = sum(c.get("cost") or 0.0 for c in calls)
    rec["refusals"] = sum(c["refusal"] for c in calls)
    rec["reasoning_token_calls"] = sum(1 for c in calls if (c.get("reasoning_tokens") or 0) > 0)
    rec["reasoning_text_calls"] = sum(1 for c in calls if c["reasoning_text_present"])
    rec["trailing_text_calls"] = sum(1 for c in calls if c["trailing_text_ignored"])
    return rec


# ---------------------------------------------------------------- frozen inputs

def check_frozen(list_path=FROZEN_LIST, root=ROOT):
    """Registered sessions only: the hash list committed and clean, every listed file matching."""
    import subprocess
    if not os.path.exists(list_path):
        return f"{os.path.relpath(list_path, root)} does not exist (Study 3 isn't frozen yet)"
    rel = os.path.relpath(list_path, root)
    r = subprocess.run(["git", "-C", root, "ls-files", "--error-unmatch", rel], capture_output=True, text=True)
    if r.returncode != 0:
        return f"{rel} is not committed"
    with open(list_path, encoding="utf-8") as fh:
        lines = [l.split() for l in fh if l.strip() and not l.startswith("#")]
    files = [rel] + [p for _, p in lines]
    r = subprocess.run(["git", "-C", root, "status", "--porcelain", "--"] + files, capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.strip():
        return "frozen files have uncommitted changes:\n" + r.stdout
    for h, p in lines:
        with open(os.path.join(root, p), "rb") as fh:
            if rr.sha256_hex(fh.read()) != h:
                return f"{p} does not match {rel}"
    return None


# ---------------------------------------------------------------- replay audit

def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def split_call_id(call_id):
    """'<game_id>/a2/m5/t1' -> (game_id, 2, 5, 1)."""
    gid, a, m, t = call_id.rsplit("/", 3)
    return gid, int(a[1:]), int(m[1:]), int(t[1:])


def audit(run_dir):
    """Re-derive every game of a Study 3 run from its raw call records. Returns (problems, n_records).
    Checks the hash chains and run_summary, rebuilds every request byte for byte, re-derives the scripted
    moves from the clamp values in the manifest, and requires each replayed record to equal the stored one."""
    problems = []
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    for name in rr.RunWriter.FILES:
        ok, _, problem, _ = rr.verify_chain(os.path.join(run_dir, name))
        if not ok:
            problems.append(f"{name}: {problem}")
    spath = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(spath):
        with open(spath, encoding="utf-8") as fh:
            summary = json.load(fh)
        for name, want in summary["sha256"].items():
            with open(os.path.join(run_dir, name), "rb") as fh:
                if rr.sha256_hex(fh.read()) != want:
                    problems.append(f"{name}: sha256 differs from run_summary.json")
    if problems:
        return problems, 0
    calls = _load_jsonl(os.path.join(run_dir, "calls.jsonl"))
    reqs = {c["call_id"]: c for c in calls if c["schema"].endswith("call-request/1")}
    ress = {c["call_id"]: c for c in calls if c["schema"].endswith("call-result/1")}
    by_move = {}
    for cid in reqs:
        gid, a, m, t = split_call_id(cid)
        by_move.setdefault((gid, a, m), []).append((t, cid))
    events = _load_jsonl(os.path.join(run_dir, "events.jsonl"))
    interrupted = {(e["game_id"], e["attempt"]) for e in events if e.get("kind") == "game_interrupted"}
    cut = lambda cid: split_call_id(cid)[:2] in interrupted    # noqa: E731
    for cid in reqs:
        if cid not in ress and not cut(cid):
            problems.append(f"{cid}: request without a result line")
    games = {g["game_id"]: g for g in man["games"]}
    values = man["clamp_values"]
    vsettings = man.get("vllm_settings")
    used = set()
    records = _load_jsonl(os.path.join(run_dir, "games.jsonl"))
    for rec in records:
        gid, attempt = rec["game_id"], rec["attempt"]
        g = games.get(gid)
        if g is None:
            problems.append(f"{gid}: not a planned game of this run")
            continue
        if (gid, attempt) in interrupted:
            problems.append(f"{gid} attempt {attempt}: has a game record but is also marked interrupted")

        def send(body, key, game, info, _gid=gid):
            tries = sorted(by_move.get((_gid, info["attempt"], info["msg_no"]), []))
            if not tries:
                raise AssertionError(f"{key}: no recorded call")
            want = rr.canonical(body).encode("utf-8")
            for _, cid in tries:
                used.add(cid)
                if rr.canonical(reqs[cid]["request"]).encode("utf-8") != want:
                    raise AssertionError(f"{cid}: recorded request differs from what Study 3 builds for this move")
                if rr.sha256_hex(want) != reqs[cid]["request_sha256"]:
                    raise AssertionError(f"{cid}: request_sha256 does not match")
            last_id = tries[-1][1]
            last = ress[last_id]
            raw = last["response_text"].encode("utf-8")
            if rr.sha256_hex(raw) != last["response_sha256"]:
                raise AssertionError(f"{last_id}: response_text does not match its sha256")
            api = game["backend"] == "api"
            if last["transport_error"]:
                fields, problem = None, f"transport: {last['transport_error']}"
            else:
                fields, problem = (rb.parse_reply if api else rr.read_envelope)(last["http_status"], raw)
            if fields is None:
                out = {"kind": "infrastructure", "call_id": last_id, "problem": problem}
                if api:
                    out["tries"] = len(tries)
                return out
            out = {"kind": "model_output", "call_id": last_id, **fields}
            if api:
                out = {"kind": "model_output", "call_id": last_id, "tries": len(tries), **fields}
            return out

        try:
            again = play(g, attempt, send, values, vsettings)
        except (AssertionError, ValueError) as e:
            problems.append(str(e))
            continue
        again = json.loads(rr.canonical(again))
        stored = {k: v for k, v in rec.items() if k != "prev_sha256"}
        if again != stored:
            diff = sorted(k for k in set(again) | set(stored) if again.get(k) != stored.get(k))
            problems.append(f"{gid} attempt {attempt}: game record differs on {', '.join(diff)}")
    for cid in reqs:
        if cid not in used and not cut(cid):
            problems.append(f"{cid}: call not part of any recorded game")
    return problems, len(records)


def main(argv):
    if len(argv) != 2 or argv[0] != "audit" or not os.path.isdir(argv[1]):
        print("usage: python3 scripts/study3.py audit data/runs/<s3 run>", file=sys.stderr)
        return 2
    problems, n = audit(argv[1])
    if problems:
        print(f"STUDY 3 REPLAY AUDIT FAILED: {len(problems)} problem(s)")
        for p in problems[:50]:
            print("  ✗ " + p)
        return 1
    print(f"replay audit passed: {n} game records reproduce exactly from the raw calls")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
