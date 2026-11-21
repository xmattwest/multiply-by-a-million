#!/usr/bin/env python3
"""newgame.py: the redesigned negotiation game for the new-game probe (STORY §12, Q12). No network.

WHAT CHANGES FROM STUDY 2 (and why; STORY §7, 2026-10-04)
  1. Value ranges. Study 2 put every floor at 75-95% and every ceiling at 105-125% of V = 100 x 10^k,
     so the deal zone always contained exactly $100 (or $100M): a focal point. Here each fixture has a
     hidden, non-round centre M (log-uniform over $30-$900); floor = M x f, ceiling = M x c with f and c
     log-uniform over [0.50, 0.95] and [1/0.95, 2.00] (log-symmetric), so the zone's midpoint moves
     around and no round number is special. Limits have 4 significant figures and are never a multiple
     of $5 at the small scale.
  2. Notation. Every price the model sees (its limit, the history, the pending offer, the appraisal) is
     written in business format: "$243,700,000", "$243.70", "$1,234". The model replies with a plain
     JSON number of dollars.
  3. Price grid. Cents are allowed at every scale (at most two decimal places), and the deal zone is at
     least 25% of M wide, so seller share is measured finely at both scales.
  4. A scale-proof bot opponent (cell BOT): a scripted side whose moves depend only on ratios, so it
     behaves identically at x1 and x10^6. The model plays the other role. Any scale effect in BOT
     games comes from the model's side alone.
  5. A public reference price (cell SPR): an appraisal both sides see, placed off-centre in the zone
     (25-40% or 60-75% of the way from floor to ceiling) and never round.
  The x10^6 game is the x1 game with six zeros appended to every amount (same digits, as in Studies
  1-2). Amounts are held as integer cents throughout.

Standard library only.
"""
import hashlib
import json
import math
import os
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
import sys                                   # noqa: E402
sys.path.insert(0, HERE)
import game_engine as ge                     # noqa: E402
import make_fixtures as mf                   # noqa: E402
import render_prompt as rp                   # noqa: E402

PROMPTS = os.path.join(HERE, "..", "config", "explore", "newgame-v2")      # v1 (Q12 run, commit 3495cf2): newgame/
PHRASE = "scale-invariance/fixtures/explore-v3"
SCALES = {"x1": 1, "x1e6": 10 ** 6}
M_RANGE = (30.0, 900.0)                      # hidden centre, dollars, log-uniform
F_RATIO = (0.50, 0.95)                       # floor / M, log-uniform
C_RATIO = (1 / 0.95, 2.00)                   # ceiling / M, log-uniform (log-symmetric to F_RATIO)
MIN_WIDTH = 0.25                             # (C - F) / M at least this
REF_POS = ((0.25, 0.40), (0.60, 0.75))       # appraisal position in the zone, floor = 0, ceiling = 1
SIG = 4                                      # significant figures of every fixture amount
BOT = {"open": Fraction(13, 10), "step": Fraction(7, 20), "close": Fraction(1, 100)}   # bot: open 30% beyond its limit; concede 35% of the gap;
                                                    # accept when within 1% of its own limit's size
BOT_TEXT = {"open": "My price is {P}.", "counter": "I can do {P}.", "hold": "{P} is as far as I can go."}
BIZ_RE = r"^\$[1-9][0-9]{0,2}(,[0-9]{3})*(\.[0-9]{2})?$"    # "$68,640,000", "$95.49", "$1,234"
SCHEMA_PRICE = {
    # biz: a quoted string, so the grammar can't end the number at a comma (the Q12 slip, STORY 2026-10-04).
    # Checked offline: xgrammar 0.2.8 compiles this and rejects "68,640,000", "$68640000", "$95.5", 68.
    # No minLength/maxLength next to the pattern (vLLM 0.30.0 refuses that mix for xgrammar).
    "biz": {"anyOf": [{"type": "string", "pattern": BIZ_RE}, {"type": "null"}]},
    # raw: plain digits everywhere in the conversation, so a JSON number is safe (no commas to copy).
    "raw": {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]},
}
# The OFFER-only (non-null) price per notation, for the opt-in strict schema
# (ge.strict_action_schema). Same constraints as SCHEMA_PRICE minus the null
# branch, since the ACCEPT/WALKAWAY variant owns the null.
PRICE_OFFER = {
    "biz": {"type": "string", "pattern": BIZ_RE},
    "raw": {"type": "number", "exclusiveMinimum": 0},
}
NOTATIONS = ("biz", "raw")
STYLES_FILE = os.path.join(HERE, "..", "config", "explore", "styles", "styles.txt")
INFO_FILE = os.path.join(HERE, "..", "config", "explore", "info", "info.txt")
INFO_CODES = ("N", "A", "P")           # nothing, a private appraisal, the other side's range (Q17)
_INFO = {}
STYLE_CODES = ("N", "A", "C", "D")     # neutral (no text), aggressive, cooperative, scale-debias (Q14)
_STYLES = {}


def style_text(code):
    """The extra paragraph for a style code; '' for N (the prompt is then unchanged)."""
    if code in (None, "N"):
        return ""
    if not _STYLES:
        _STYLES.update(rp.load_fragments(STYLES_FILE))
    return _STYLES["style_" + code.lower()]


def info_text(code, role, game):
    """The private-information line for one side ('' for N or no code). game["R_private"]: the
    appraisal in cents; game["ranges"]: {"F": (lo, hi), "C": (lo, hi)} in cents, at this scale."""
    if code in (None, "N"):
        return ""
    if not _INFO:
        _INFO.update(rp.load_fragments(INFO_FILE))
    nt = game.get("notation", "biz")
    other = "buyer" if role == "seller" else "seller"
    if code == "A":
        return rp.fill(_INFO["info_a"], {"R": fmt(game["R_private"], nt), "OTHER": other})
    if code == "P":
        lo, hi = game["ranges"]["C" if role == "seller" else "F"]
        return rp.fill(_INFO[f"info_p_{role}"], {"LO": fmt(lo, nt), "HI": fmt(hi, nt)})
    raise ValueError(f"unknown info code {code!r}")


# ---------------------------------------------------------------- amounts and notation
def usd(cents):
    """Business format: $243,700,000 / $243.70 / $1,234 (cents shown only when not zero)."""
    if isinstance(cents, bool) or not isinstance(cents, int) or cents < 1:
        raise ValueError("amount must be a positive whole number of cents")
    d, c = divmod(cents, 100)
    return f"${d:,}" + (f".{c:02d}" if c else "")


def raw(cents):
    """Study 2 style, plain digits: 243700000 dollars / 243.70 dollars (cents shown only when not zero)."""
    if isinstance(cents, bool) or not isinstance(cents, int) or cents < 1:
        raise ValueError("amount must be a positive whole number of cents")
    d, c = divmod(cents, 100)
    return f"{d}" + (f".{c:02d}" if c else "") + " dollars"


def fmt(cents, notation):
    return usd(cents) if notation == "biz" else raw(cents)


def parse_usd(text):
    """'$68,640,000' -> 6864000000 cents; None if it isn't exactly business format."""
    if not isinstance(text, str) or not re.fullmatch(BIZ_RE[1:-1], text):
        return None
    d, _, c = text[1:].replace(",", "").partition(".")
    return int(d) * 100 + (int(c) if c else 0)


def sig_cents(dollars, sig=SIG):
    """Round a positive dollar amount to `sig` significant figures; return integer cents."""
    q = Decimal(f"{dollars:.{sig}g}")
    cents = q * 100
    if cents != cents.to_integral_value():
        raise ValueError("amount below a cent at this precision")
    return int(cents)


def scale_free(cents, mult):
    """An amount on the x1 scale (for comparing across scales)."""
    return Decimal(cents) / mult


# ---------------------------------------------------------------- fixtures
def _u(seed_hex, label):
    return mf.derive(seed_hex, label, 10 ** 9) / 10 ** 9


def _logu(seed_hex, label, lo, hi):
    return math.exp(math.log(lo) + _u(seed_hex, label) * (math.log(hi) - math.log(lo)))


GEOMETRIES = {
    # wide: the Q12 redesign. Hidden non-round centre, log-symmetric wide zone.
    "wide": {"f": F_RATIO, "c": C_RATIO, "min_width": MIN_WIDTH, "log": True},
    # narrow: Study 2's widths (floor 75-95%, ceiling 105-125%) around a hidden non-round centre.
    "narrow": {"f": (0.75, 0.95), "c": (1.05, 1.25), "min_width": 0.10, "log": False},
    # round: Study 2 exactly: centre $100, whole-dollar floor 75-95 and ceiling 105-125.
    "round": None,
}


def build_fixtures(n, phrase=PHRASE, geom="wide", prefix="Z"):
    """n fixtures <prefix>01..; amounts in cents at the x1 scale. Deterministic from the phrase."""
    seed = mf.seed_from_phrase(phrase)
    if geom == "round":
        doc = mf.build(phrase)
        out = []
        for i, f in enumerate(doc["fixtures"][:n], 1):
            F, C = f["f_pct"] * 100, f["c_pct"] * 100
            out.append({"id": f"{prefix}{i:02d}", "geom": geom, "M": 100.0, "F": F, "C": C,
                        "R": sig_cents((F + 0.33 * (C - F)) / 100), "redraws": 0, "ref_pos": 0.33})
        return out
    G = GEOMETRIES[geom]
    draw = (lambda lab, lo, hi: _logu(seed, lab, lo, hi)) if G["log"] else \
        (lambda lab, lo, hi: lo + _u(seed, lab) * (hi - lo))
    out = []
    for i in range(1, n + 1):
        fid = f"{prefix}{i:02d}"
        for k in range(1000):                      # redraw until every rule holds
            lab = f"{fid}|{k}" if geom == "wide" else f"{geom}|{fid}|{k}"
            M = _logu(seed, lab + "|M", *M_RANGE)
            F = sig_cents(M * draw(lab + "|f", *G["f"]))
            C = sig_cents(M * draw(lab + "|c", *G["c"]))
            side = mf.derive(seed, lab + "|refside", 2)
            lo, hi = REF_POS[side]
            pos = lo + _u(seed, lab + "|refpos") * (hi - lo)
            R = sig_cents((F + pos * (C - F)) / 100)
            if (C - F) >= G["min_width"] * M * 100 and all(x % 500 for x in (F, C, R)) and F < R < C:
                break
        else:
            raise RuntimeError(f"no valid draw for {fid}")
        out.append({"id": fid, "geom": geom, "M": round(M, 4), "F": F, "C": C, "R": R, "redraws": k,
                    "ref_pos": round((R - F) / (C - F), 4)})
    return out


# ---------------------------------------------------------------- prompts
def load_templates():
    return rp.load_templates(PROMPTS)


def render(role, game, history, msg_no, max_msgs, max_chars, templates=None):
    """Messages for one model turn. history: earlier moves (role, cents, text), all OFFERs.
    game["notation"] ("biz" default, or "raw") sets how every price is written and how the reply's price is asked for."""
    system_t, user_t, fr = templates or load_templates()
    nt = game.get("notation", "biz")
    usd = lambda c: fmt(c, nt)                                                    # noqa: E731
    other = "buyer" if role == "seller" else "seller"
    if len(history) != msg_no - 1:
        raise ValueError("msg_no must be len(history) + 1")
    ref = (rp.fill(fr["reference"], {"R": usd(game["R"])}) + "\n") if game.get("R") else ""
    system = rp.fill(system_t, {"REFERENCE": ref, "MAX_MSGS": str(max_msgs), "MAX_CHARS": str(max_chars),
                                "PRICE_FIELD": fr[f"price_field_{nt}"], "PRICE_RULE": fr[f"price_rule_{nt}"]})
    if history:
        lines = [fr["history_head"]]
        for i, (who, cents, text) in enumerate(history, start=1):
            lines.append(rp.fill(fr["history_line"], {"N": str(i), "WHO": "you" if who == role else f"the {who}",
                                                      "PRICE": usd(cents),
                                                      "TEXT": json.dumps(text, ensure_ascii=False)}))
        hist = "\n".join(lines)
        pending = rp.fill(fr["pending_offer"], {"OTHER": other, "PRICE": usd(history[-1][1])})
    else:
        hist, pending = fr["history_empty"], rp.fill(fr["pending_none"], {"OTHER": other})
    goal = fr["goal_seller"] if role == "seller" else fr["goal_buyer"]
    extra = "\n".join(t for t in (info_text((game.get("info") or {}).get(role), role, game),
                                   style_text((game.get("styles") or {}).get(role))) if t)
    user = rp.fill(user_t, {"ROLE": role, "LIMIT_NAME": "floor" if role == "seller" else "ceiling",
                            "LIMIT": usd(game["F"] if role == "seller" else game["C"]), "OTHER": other,
                            "GOAL": goal + ("\n" + extra if extra else ""),
                            "HISTORY": hist, "MSG_NO": str(msg_no), "MAX_MSGS": str(max_msgs),
                            "PENDING": pending})
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_request(model, messages, settings, seed, notation="biz", strict_action=False):
    """Study 2's request body, with the price field for this notation (biz: a patterned string; raw: a number).

    strict_action (opt-in, default off): tie price to action in the grammar
    itself (OFFER needs a price, ACCEPT/WALKAWAY force null) via
    ge.strict_action_schema. Used for the Gemma 4 12B rescue (WP2a), which
    kept emitting ACCEPT-with-a-price. Off by default, so every existing run
    is byte-identical. Must pass the offline xgrammar-compile preflight on a
    pod before any paid use.
    """
    body = ge.build_request(model, messages, settings, seed)
    if strict_action:
        max_chars = settings["max_message_chars"]["value"]
        body["response_format"]["json_schema"]["schema"] = ge.strict_action_schema(
            max_chars, price_offer=PRICE_OFFER[notation])
    else:
        body["response_format"]["json_schema"]["schema"]["properties"]["price"] = SCHEMA_PRICE[notation]
    return body


def call_seed(fixture, cell, opener, model_role, msg_no, attempt, ns="newgame/v1"):
    """Common random numbers across scales (and, in the geometry probe, notations): neither is an input."""
    label = f"scale-invariance/{ns}|{fixture}|{cell}|{opener}|{model_role}|{msg_no}|{attempt}"
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------- reply check
def validate_reply(content, finish_reason, has_pending_offer, max_chars, notation="biz"):
    """Study 2's strict check (no repair). The price is, for biz, a string in exact business format
    ("$68,640,000"); for raw, a JSON number of dollars with at most two decimal places.
    Returns (move or None, errors); move["price"] is integer cents."""
    if finish_reason == "length":
        return None, ["TRUNCATED"]
    if finish_reason != "stop":
        return None, [f"FINISH_REASON:{finish_reason}"]
    if not isinstance(content, str):
        return None, ["NO_CONTENT"]
    text = content.strip(ge.JSON_WS)
    if not text.startswith("{"):
        return None, ["NOT_JSON_OBJECT"]
    dec = json.JSONDecoder(object_pairs_hook=ge._no_dupes, parse_constant=ge._no_constants,
                           parse_float=Decimal, parse_int=Decimal)
    try:
        obj, end = dec.raw_decode(text)
    except ge._Reject as e:
        return None, [str(e)]
    except (ValueError, InvalidOperation):
        return None, ["NOT_JSON"]
    if end != len(text):
        return None, ["TEXT_AFTER_JSON"]
    if not isinstance(obj, dict):
        return None, ["NOT_JSON_OBJECT"]
    errors = [f"MISSING_FIELD:{k}" for k in ge.KEYS if k not in obj]
    errors += [f"EXTRA_FIELD:{k}" for k in sorted(obj) if k not in ge.KEYS]
    if errors:
        return None, errors
    action, price, message = obj["action"], obj["price"], obj["message"]
    cents = None
    if action not in ge.ACTIONS:
        errors.append("BAD_ACTION")
    if action == "OFFER":
        if price is None:
            errors.append("PRICE_REQUIRED")
        elif notation == "biz":
            cents = parse_usd(price)
            if cents is None:
                errors.append("BAD_PRICE_FORMAT")
        elif not isinstance(price, Decimal) or not price.is_finite():
            errors.append("BAD_PRICE_TYPE")
        elif price <= 0:
            errors.append("NONPOSITIVE_PRICE")
        elif price * 100 != (price * 100).to_integral_value():
            errors.append("PRICE_PRECISION")          # more than two decimal places
        else:
            cents = int(price * 100)
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
    return {"action": action, "price": cents, "message": message}, []


# ---------------------------------------------------------------- the scale-proof bot
def _sig_round_cents(cents, sig=SIG):
    """Round an exact amount of cents (int or Fraction) to `sig` significant figures, half up.
    Exact arithmetic, so it is scale-proof: x10^6 shifts the digits and nothing else."""
    x = Fraction(cents)
    if x <= 0:
        raise ValueError("amount must be positive")
    e = len(str(math.floor(x))) - sig
    q = Fraction(10) ** max(e, 0)
    return max(1, int(math.floor(x / q + Fraction(1, 2)) * q))


def bot_move(role, limit, history, notation="biz"):
    """One bot move from its own limit and the history only. Returns (action, cents or None, text).
    Every quantity is a ratio, and rounding keeps 4 significant figures, so the bot's moves at x10^6 are
    its moves at x1 with six zeros appended."""
    sell = role == "seller"
    mine = [p for who, p, _ in history if who == role]
    theirs = [p for who, p, _ in history if who != role]
    if not mine:
        target = _sig_round_cents(limit * BOT["open"] if sell else limit / BOT["open"])
        if theirs and ((sell and theirs[-1] >= target) or (not sell and theirs[-1] <= target)):
            return "ACCEPT", None, "Agreed."
        return "OFFER", target, BOT_TEXT["open"].replace("{P}", fmt(target, notation))
    own, other = mine[-1], theirs[-1]
    ok_for_me = other >= limit if sell else other <= limit
    planned = own - BOT["step"] * (own - other)
    planned = max(planned, limit) if sell else min(planned, limit)
    planned = _sig_round_cents(planned)
    if ok_for_me and ((sell and other >= planned) or (not sell and other <= planned)
                      or abs(own - other) <= BOT["close"] * limit):
        return "ACCEPT", None, "Agreed."
    text = BOT_TEXT["hold"] if planned == own else BOT_TEXT["counter"]
    return "OFFER", planned, text.replace("{P}", fmt(planned, notation))


# ---------------------------------------------------------------- one game
def play_game(game, model, settings, call, attempt=1, templates=None, strict_action=False):
    """Play one attempt. game: fixture, scale, mult, F, C (cents), R (cents or None), cell, opener (role),
    bot_role (None for self-play). Returns the games.jsonl record (schema newgame/1)."""
    max_msgs, max_chars = settings["max_messages"]["value"], settings["max_message_chars"]["value"]
    templates = templates or load_templates()
    other = {"seller": "buyer", "buyer": "seller"}
    role, history, offers, call_ids, diagnostics = game["opener"], [], [], [], []
    terminal, price, ended_by, n_messages, model_calls = None, None, None, 0, 0
    model_role = other[game["bot_role"]] if game.get("bot_role") else None
    nt = game.get("notation", "biz")
    for msg_no in range(1, max_msgs + 1):
        n_messages = msg_no
        limit = game["F"] if role == "seller" else game["C"]
        if role == game.get("bot_role"):
            action, cents, text = bot_move(role, limit, history, nt)
            by = "bot"
        else:
            msgs = render(role, game, history, msg_no, max_msgs, max_chars, templates)
            mdl = (game.get("models") or {}).get(role, model)          # cross-play: each role its own model
            seed = call_seed(game["fixture"], game["cell"], game["opener"], model_role or "both", msg_no, attempt,
                             game.get("seed_ns", "newgame/v1"))
            info = {"game_id": game["game_id"], "attempt": attempt, "msg_no": msg_no, "role": role}
            reply = call(build_request(mdl, msgs, settings, seed, nt, strict_action=strict_action), info)
            call_ids.append(reply.get("call_id"))
            model_calls += 1
            if reply.get("kind") != "model_output":
                terminal = "INFRASTRUCTURE_VOID"
                break
            if reply.get("reasoning") not in (None, ""):
                diagnostics.append(f"REASONING_PRESENT:{msg_no}")
            move, errors = validate_reply(reply.get("content"), reply.get("finish_reason"), bool(history),
                                          max_chars, nt)
            if errors:
                terminal, ended_by = "PROTOCOL_FORFEIT", role
                diagnostics.append(f"INVALID:{msg_no}:" + ",".join(errors))
                break
            action, cents, text, by = move["action"], move["price"], move["message"], "model"
        if action == "OFFER":
            history.append((role, cents, text))
            offers.append([msg_no, role, cents, by])
            role = other[role]
            continue
        if action == "ACCEPT":
            terminal, price, ended_by = "AGREEMENT", history[-1][1], role
        else:
            terminal, ended_by = "WALKAWAY", role
        break
    else:
        terminal = "TIMEOUT"
    rec = {"schema": "newgame/1", "game_id": game["game_id"], "attempt": attempt, "model": model,
           "run_order": game.get("run_order"), "kind": "new", **{k: game.get(k) for k in (
               "fixture", "scale", "mult", "cell", "opener", "bot_role", "F", "C", "R", "geom", "label",
               "styles", "models", "info", "R_private", "ranges")},
           "notation": nt,
           "model_role": model_role, "units": "cents", "terminal": terminal, "n_messages": n_messages,
           "model_calls": model_calls, "price": price, "ended_by": ended_by, "offers": offers,
           "diagnostics": diagnostics, "call_ids": call_ids}
    if terminal != "INFRASTRUCTURE_VOID":
        sc = ge.score(game, terminal, price, ended_by)
        rec.update({"seller_share": ge.frac(sc["seller_share"]), "u_seller": ge.frac(sc["u_seller"]),
                    "u_buyer": ge.frac(sc["u_buyer"]), "violation": sc["violation"],
                    "outside_zopa": sc["outside_zopa"]})
    return rec
