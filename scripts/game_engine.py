#!/usr/bin/env python3
"""game_engine.py: the rules of one negotiation, with no network in it.

WHAT IT DOES
  - validate_reply(): the strict reply check from docs/data_plan.md section 7.
    No repair: anything that isn't exactly one valid JSON move is a forfeit.
  - build_request(): the exact JSON body sent to vLLM for one move, with
    every sampling field set explicitly (prereg/settings.json).
  - call_seed(): the per-move sampling seed (data plan E5): the same at all
    three scales, so paired games share their random numbers.
  - play_game(): plays one game attempt. It asks a `call` function for each
    move; that function is the only thing that talks to a server, so the
    whole game can be tested offline with scripted replies.
  - score(): seller share and utilities as exact fractions (ZOPA spec 2.4).

  The engine never sees a server's raw bytes; the caller records those
  (docs/data_plan.md section 6.2) before handing the parsed reply here.

THE `call` CONTRACT
  call(request_body: dict, info: dict) -> dict with
    kind           "model_output" (a completed assistant reply) or
                   "infrastructure" (no usable completion after the client's
                   own retries; data plan E8)
    content        the reply text (model_output only)
    finish_reason  e.g. "stop", "length"
    reasoning      message.reasoning from the response (should be None)
    call_id        opaque pointer into calls.jsonl
  info carries game_id, attempt, msg_no, role.

Standard library only.
"""

import hashlib
import json
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_prompt  # noqa: E402

ACTIONS = ("OFFER", "ACCEPT", "WALKAWAY")
KEYS = ("action", "price", "message")
JSON_WS = " \t\n\r"
TERMINALS = ("AGREEMENT", "WALKAWAY", "TIMEOUT", "PROTOCOL_FORFEIT", "INFRASTRUCTURE_VOID")


class _Reject(Exception):
    pass


def _no_dupes(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _Reject("DUPLICATE_KEY")
    return dict(pairs)


def _no_constants(name):
    raise _Reject("NONFINITE_NUMBER")


def validate_reply(content, finish_reason, has_pending_offer, max_chars):
    """Return (move or None, errors). move = {"action", "price", "message"}."""
    if finish_reason == "length":
        return None, ["TRUNCATED"]
    if finish_reason != "stop":
        return None, [f"FINISH_REASON:{finish_reason}"]
    if not isinstance(content, str):
        return None, ["NO_CONTENT"]
    text = content.strip(JSON_WS)
    if not text.startswith("{"):
        return None, ["NOT_JSON_OBJECT"]
    dec = json.JSONDecoder(object_pairs_hook=_no_dupes, parse_constant=_no_constants)
    try:
        obj, end = dec.raw_decode(text)
    except _Reject as e:
        return None, [str(e)]
    except ValueError:
        return None, ["NOT_JSON"]
    if end != len(text):
        return None, ["TEXT_AFTER_JSON"]
    if not isinstance(obj, dict):
        return None, ["NOT_JSON_OBJECT"]

    errors = []
    errors += [f"MISSING_FIELD:{k}" for k in KEYS if k not in obj]
    errors += [f"EXTRA_FIELD:{k}" for k in sorted(obj) if k not in KEYS]
    if errors:
        return None, errors

    action, price, message = obj["action"], obj["price"], obj["message"]
    if action not in ACTIONS:
        errors.append("BAD_ACTION")
    if action == "OFFER":
        if price is None:
            errors.append("PRICE_REQUIRED")
        elif isinstance(price, bool) or not isinstance(price, int):
            errors.append("BAD_PRICE_TYPE")   # true, 94000.0, "94000" all land here
        elif price < 1:
            errors.append("NONPOSITIVE_PRICE")
    elif action in ("ACCEPT", "WALKAWAY") and price is not None:
        errors.append("PRICE_MUST_BE_NULL")
    if not isinstance(message, str):
        errors.append("BAD_MESSAGE_TYPE")
    elif len(message) > max_chars:
        errors.append("MESSAGE_TOO_LONG")
    if action == "ACCEPT" and not has_pending_offer:
        errors.append("ACCEPT_WITHOUT_OFFER")
    if errors:
        return None, errors
    return {"action": action, "price": price, "message": message}, []


def call_seed(fixture, opener, msg_no, attempt):
    """E5. fixture and opener exactly as in config/fixtures.json ("F12", "seller").
    The scale is deliberately not an input."""
    label = f"scale-invariance/call-seed/v1|{fixture}|{opener}|{msg_no}|{attempt}"
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def response_schema(max_chars):
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "price": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
            "message": {"type": "string", "maxLength": max_chars},
        },
        "required": list(KEYS),
        "additionalProperties": False,
    }


# Default price field used by the plain (Study-1/2) schema, so the strict
# variant below can reuse exactly the OFFER price constraint.
PRICE_OFFER_DEFAULT = {"type": "integer", "minimum": 1}


def strict_action_schema(max_chars, price_offer=None):
    """Opt-in reply schema that ties `price` to `action` IN THE GRAMMAR, not
    only in validate_reply: ACCEPT/WALKAWAY force price=null, OFFER forces a
    positive price. Built as a oneOf over two closed object variants, which
    xgrammar accepts more reliably than if/then/allOf (newgame SCHEMA_PRICE
    note: vLLM 0.30.0 is fussy about JSON-Schema features).

    `price_offer` is the schema for an OFFER's price (default: a positive
    integer; newgame passes the notation-specific string/number form). The
    null branch is handled by the ACCEPT/WALKAWAY variant, so `price_offer`
    should describe only a *present* price.

    This changes nothing on its own: callers opt in. validate_reply is the
    unchanged ground truth, and this grammar is a strict subset of what it
    accepts, so every tests/reply_cases.jsonl outcome is identical.
    """
    offer = price_offer or dict(PRICE_OFFER_DEFAULT)
    message = {"type": "string", "maxLength": max_chars}
    return {
        "oneOf": [
            {   # an OFFER must carry a price
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["OFFER"]},
                    "price": offer,
                    "message": message,
                },
                "required": list(KEYS),
                "additionalProperties": False,
            },
            {   # ACCEPT or WALKAWAY must carry price=null
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["ACCEPT", "WALKAWAY"]},
                    "price": {"type": "null"},
                    "message": message,
                },
                "required": list(KEYS),
                "additionalProperties": False,
            },
        ]
    }


def build_request(model, messages, settings, seed):
    """The complete request body. Every sampling field is sent, so nothing
    falls back to a server or model default (research notes C3, V5)."""
    samp = settings["sampling"][model]
    max_chars = settings["max_message_chars"]["value"]
    return {
        "model": model,
        "messages": messages,
        "temperature": samp["temperature"],
        "top_p": samp["top_p"],
        "top_k": samp["top_k"],
        "min_p": samp["min_p"],
        "presence_penalty": settings["presence_penalty"]["value"],
        "frequency_penalty": settings["frequency_penalty"]["value"],
        "repetition_penalty": settings["repetition_penalty"]["value"],
        "max_tokens": settings["max_tokens"]["value"],
        "seed": seed,
        "n": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "negotiation_move", "schema": response_schema(max_chars), "strict": True},
        },
    }


def score(game, terminal, price=None, ended_by=None):
    """Exact outcome numbers. Fractions are kept as (numerator, denominator)."""
    F, C = game["F"], game["C"]
    width = C - F
    out = {"seller_share": None, "u_seller": None, "u_buyer": None,
           "violation": False, "outside_zopa": False}
    if terminal == "AGREEMENT":
        s = Fraction(price - F, width)
        out["seller_share"] = s
        out["u_seller"], out["u_buyer"] = s, 1 - s
        out["outside_zopa"] = not (F <= price <= C)
        # ZOPA spec 2.4: accepting outside your own limit is a violation.
        out["violation"] = (ended_by == "seller" and price < F) or (ended_by == "buyer" and price > C)
    elif terminal in ("WALKAWAY", "TIMEOUT"):
        out["u_seller"] = out["u_buyer"] = Fraction(0)
    elif terminal == "PROTOCOL_FORFEIT":
        out["u_seller"] = Fraction(-1) if ended_by == "seller" else Fraction(0)
        out["u_buyer"] = Fraction(-1) if ended_by == "buyer" else Fraction(0)
    return out


def frac(x):
    return None if x is None else {"num": x.numerator, "den": x.denominator, "float": float(x)}


def play_game(game, model, settings, f_range, c_range, call, attempt=1, render=None):
    """Play one attempt of one game. Returns the games.jsonl record (data plan 6.3)."""
    max_msgs = settings["max_messages"]["value"]
    max_chars = settings["max_message_chars"]["value"]
    other = {"seller": "buyer", "buyer": "seller"}
    role = game["opener"]
    history = []              # earlier moves, all OFFERs: (role, price, text)
    offers, call_ids, diagnostics = [], [], []
    terminal, price, ended_by, n_messages = None, None, None, 0

    for msg_no in range(1, max_msgs + 1):
        limit = game["F"] if role == "seller" else game["C"]
        messages = (render or render_prompt.render)(role, game["V"], limit, history, msg_no,
                                        f_range, c_range, max_msgs, max_chars)
        seed = call_seed(game["fixture"], game["opener"], msg_no, attempt)
        body = build_request(model, messages, settings, seed)
        info = {"game_id": game["game_id"], "attempt": attempt, "msg_no": msg_no, "role": role}
        reply = call(body, info)
        call_ids.append(reply.get("call_id"))
        n_messages = msg_no

        if reply.get("kind") != "model_output":
            terminal = "INFRASTRUCTURE_VOID"
            break
        if reply.get("reasoning") not in (None, ""):
            diagnostics.append(f"REASONING_PRESENT:{msg_no}")
        move, errors = validate_reply(reply.get("content"), reply.get("finish_reason"),
                                      bool(history), max_chars)
        if errors:
            terminal, ended_by = "PROTOCOL_FORFEIT", role
            diagnostics.append(f"INVALID:{msg_no}:" + ",".join(errors))
            break
        if move["action"] == "OFFER":
            history.append((role, move["price"], move["message"]))
            offers.append([msg_no, role, move["price"]])
            role = other[role]
            continue
        if move["action"] == "ACCEPT":
            terminal, price, ended_by = "AGREEMENT", history[-1][1], role
        else:
            terminal, ended_by = "WALKAWAY", role
        break
    else:
        terminal = "TIMEOUT"

    rec = {
        "schema": "game/1",
        "game_id": game["game_id"], "attempt": attempt, "model": model,
        "block": game.get("block"), "run_order": game.get("run_order"),
        "fixture": game["fixture"], "scale": game["scale"], "V": game["V"],
        "F": game["F"], "C": game["C"], "opener": game["opener"],
        "terminal": terminal, "n_messages": n_messages, "price": price, "ended_by": ended_by,
        "offers": offers, "diagnostics": diagnostics, "call_ids": call_ids,
    }
    if terminal != "INFRASTRUCTURE_VOID":
        sc = score(game, terminal, price, ended_by)
        rec.update({"seller_share": frac(sc["seller_share"]), "u_seller": frac(sc["u_seller"]),
                    "u_buyer": frac(sc["u_buyer"]), "violation": sc["violation"],
                    "outside_zopa": sc["outside_zopa"]})
    return rec
