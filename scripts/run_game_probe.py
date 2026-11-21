#!/usr/bin/env python3
"""run_game_probe.py: the new-game probe (STORY §12, Q12). Full negotiations. NOT pre-registered.

THIS SPENDS MONEY (except with --preview). One pod, full games played concurrently.

    python3 scripts/run_game_probe.py --model gemma-4-31b-it --preview      (free: fake server)
    python3 scripts/run_game_probe.py --model gemma-4-31b-it                (paid)

WHY. Before a confirmatory Study 3 we test a redesigned game (scripts/newgame.py): non-round,
off-centre value ranges (no $100 focal point); business notation for every price the model sees;
cents allowed with a wider deal zone; a scale-proof bot opponent; a public appraisal. Does Gemma 31B's
Study 2 effect (seller squeezed at $100M) survive the redesign, and which side moves?

CELLS (per fixture, at both scales x1 and x1e6 = the same digits with six zeros appended)
  SPN  self-play, no reference                  openers seller, buyer                  2 games
  SPR  self-play, public appraisal (off-centre)  openers seller, buyer                  2 games
  BOT  the model vs the scale-proof bot, no ref  model is seller or buyer x opener      4 games
  LEG  the exact Study 2 game (bridge cell): Study 2 prompt, ranges and whole dollars, at
       $100 and $100M, on its own fresh fixtures L01.. (phrase .../explore-v3-legacy)    2 games
  Default 30 fixtures: 30 x (8 x 2 + 2 x 2) = 600 games.

Seeds depend on fixture, cell, opener, model role, message and attempt, never the scale (common
random numbers between the paired x1 and x1e6 games). Games are shuffled so cells and scales interleave.
Reused from scripts/run_job.py: launcher, teardown, panel, server checks, the first-reply
logged-settings check, hash-chained records. Data: data/explore/X-newgame-<model>-<stamp>/.
Analyse with scripts/analyze_game_probe.py. Standard library only.

DESIGN geom (--design geom; STORY §12 Q13): geometry x notation, with the fixed reply format.
    python3 scripts/run_game_probe.py --model gemma-4-31b-it --design geom
  Geometry (fresh fixtures, phrase .../explore-v4, 30 each):
    round   R01..  Study 2 exactly: centre $100, floor 75-95, ceiling 105-125, whole dollars
    narrow  N01..  Study 2's widths around a hidden non-round centre ($30-900), 4 significant figures
    wide    W01..  the Q12 redesign (floor 0.50-0.95, ceiling 1.05-2.00 of the centre)
  Notation: raw ("95490000 dollars", JSON number) vs biz ("$95,490,000", JSON string with a pattern).
  Cells: SPN self-play (2 openers) and BOT (model seller/buyer x who opens), at x1 and x1e6.
  30 x 3 x 2 x 2 x 6 = 2,160 games. Seeds are shared across scale and notation.
  A preflight call checks that vLLM accepts the patterned-string schema before any game.

DESIGN prompt (--design prompt; STORY §12 Q14): does an instruction change the big-number timidity?
    python3 scripts/run_game_probe.py --model gemma-4-31b-it --design prompt
  Study 2's zone (round, R-style fixtures P01-P30, phrase .../explore-v5), raw notation, x1 and x1e6.
  One extra paragraph after a side's goal (config/explore/styles/styles.txt):
    N none (= Q13's round raw prompt)  A "negotiate hard"  C "a deal matters more than the price"
    D "the size of the numbers doesn't change good negotiating"
  Self-play pairs (seller style, buyer style): NN AA CC DD AN NA CN NC AC CA, each with both openers.
  Bot cells: the model in style N, A, C or D, as seller or buyer x who opens.
  30 x 2 x (10 x 2 + 4 x 4) = 2,160 games. Seeds are shared across style and scale.

DESIGN cross (--design cross; STORY §12 Q15): Gemma 31B vs Qwen 27B, one pod each (two slots).
    python3 scripts/run_game_probe.py --design cross --slot 4        (uses slots 4 and 5)
  Pairings (seller/buyer): G/G, Q/Q, G/Q, Q/G; both openers; x1 and x1e6.
  Zones: round + raw notation (CR01-CR30) and wide + business notation (CW01-CW30), phrase .../explore-v6.
  30 x 2 x 2 x 4 x 2 = 960 games. Seeds shared across pairing and scale.
  Each model runs on the same GPU type and server settings as its solo runs.

DESIGN info (--design info; STORY §12 Q17): asymmetric information.
    python3 scripts/run_game_probe.py --model gemma-4-31b-it --design info
  Study 2's zone (round, fixtures I01-I30, phrase .../explore-v7), raw notation, x1 and x1e6.
  Private information for one side, added after its goal (config/explore/info/info.txt); the other
  side's prompt and the shared system message are unchanged:
    N nothing   A a private appraisal, off-centre (25-40% or 60-75% of the zone), "the other side has
    not seen it"   P the range the other side's limit was drawn from (Study 1's public ranges, private)
  Self-play (seller info / buyer info): NN AN NA AA PN NP PP, both openers.
  Bot cells: the model with N, A or P, as seller or buyer x who opens (the bot never sees it).
  30 x 2 x (7 x 2 + 3 x 4) = 1,560 games. Seeds are shared across information and scale.

--gpu-id "<RunPod GPU id>" picks another GPU type if the default is out of stock.
"""
import argparse
import json
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import game_engine as ge          # noqa: E402
import make_fixtures as mf        # noqa: E402
import newgame as N               # noqa: E402
import render_prompt_s2 as rp2    # noqa: E402
import run_job as J               # noqa: E402
import run_records as rr          # noqa: E402
import vllm_client as v           # noqa: E402

P = J.P
PROBE = "newgame"
LEGACY_PHRASE = "scale-invariance/fixtures/explore-v3-legacy"
LEGACY_SCALES = ("x1e2", "x1e8")
CELLS = {
    "SPN": "self-play, no reference",
    "SPR": "self-play, public appraisal (off-centre, non-round)",
    "BOT": "the model vs a scale-proof scripted opponent, no reference",
    "LEG": "bridge: the exact Study 2 game (prompt, ranges, whole dollars) on fresh fixtures",
}


def build_games(n_fix, cells):
    """All planned games, shuffled. New-game amounts are integer cents."""
    games = []
    fx = N.build_fixtures(n_fix)
    for f in fx:
        for scale, mult in N.SCALES.items():
            base = {"kind": "new", "fixture": f["id"], "scale": scale, "mult": mult,
                    "F": f["F"] * mult, "C": f["C"] * mult}
            for opener in ("seller", "buyer"):
                o = opener[0].upper()
                if "SPN" in cells:
                    games.append({**base, "cell": "SPN", "opener": opener, "bot_role": None, "R": None,
                                  "game_id": f"{f['id']}-{scale}-SPN-{o}"})
                if "SPR" in cells:
                    games.append({**base, "cell": "SPR", "opener": opener, "bot_role": None, "R": f["R"] * mult,
                                  "game_id": f"{f['id']}-{scale}-SPR-{o}"})
                if "BOT" in cells:
                    for model_role in ("seller", "buyer"):
                        bot_role = "buyer" if model_role == "seller" else "seller"
                        who = "M" if opener == model_role else "B"         # who opens: model or bot
                        games.append({**base, "cell": "BOT", "opener": opener, "bot_role": bot_role, "R": None,
                                      "game_id": f"{f['id']}-{scale}-BOT-m{model_role[0].upper()}-{who}"})
    legacy = []
    if "LEG" in cells:
        doc = mf.build(LEGACY_PHRASE)
        ids = {g["id"]: f"L{g['id'][1:]}" for g in doc["fixtures"][:n_fix]}
        for g in doc["games"]:
            if g["fixture"] in ids and g["scale"] in LEGACY_SCALES:
                lid = ids[g["fixture"]]
                legacy.append({**g, "kind": "legacy", "cell": "LEG", "fixture": lid,
                               "game_id": f"{lid}-{g['scale']}-LEG-{g['opener'][0].upper()}"})
        meta_leg = {"seed_phrase": LEGACY_PHRASE, "f_pct_range": doc["f_pct_range"],
                    "c_pct_range": doc["c_pct_range"],
                    "fixtures": [{**x, "id": ids[x["id"]]} for x in doc["fixtures"][:n_fix]]}
    else:
        meta_leg = None
    games += legacy
    random.Random(rr.sha256_hex((N.PHRASE + "|newgame-order").encode())).shuffle(games)
    for i, g in enumerate(games, 1):
        g["run_order"], g["block"] = i, None
    meta = {"seed_phrase": N.PHRASE, "fixtures": fx, "legacy": meta_leg,
            "design": {"M_range": N.M_RANGE, "F_ratio": N.F_RATIO, "C_ratio": N.C_RATIO, "min_width": N.MIN_WIDTH,
                       "ref_pos": N.REF_POS, "sig_figs": N.SIG, "bot": {k: str(x) for k, x in N.BOT.items()},
                       "bot_text": N.BOT_TEXT,
                       "scales": N.SCALES}}
    return meta, games


GEOM_PHRASE = "scale-invariance/fixtures/explore-v4"
GEOM_SETS = (("round", "R"), ("narrow", "N"), ("wide", "W"))


def build_games_geom(n_fix):
    """Q13: geometry x notation x scale x (self-play, bot). Amounts in integer cents."""
    games, fx_all = [], {}
    for geom, prefix in GEOM_SETS:
        fx = N.build_fixtures(n_fix, GEOM_PHRASE, geom, prefix)
        fx_all[geom] = fx
        for f in fx:
            for scale, mult in N.SCALES.items():
                for nt in N.NOTATIONS:
                    base = {"kind": "new", "geom": geom, "fixture": f["id"], "scale": scale, "mult": mult,
                            "notation": nt, "F": f["F"] * mult, "C": f["C"] * mult, "R": None,
                            "seed_ns": "geomprobe/v1"}
                    for opener in ("seller", "buyer"):
                        o = opener[0].upper()
                        games.append({**base, "cell": "SPN", "opener": opener, "bot_role": None,
                                      "game_id": f"{f['id']}-{scale}-{nt}-SPN-{o}"})
                        for model_role in ("seller", "buyer"):
                            bot_role = "buyer" if model_role == "seller" else "seller"
                            who = "M" if opener == model_role else "B"
                            games.append({**base, "cell": "BOT", "opener": opener, "bot_role": bot_role,
                                          "game_id": f"{f['id']}-{scale}-{nt}-BOT-m{model_role[0].upper()}-{who}"})
    random.Random(rr.sha256_hex((GEOM_PHRASE + "|geom-order").encode())).shuffle(games)
    for i, g in enumerate(games, 1):
        g["run_order"], g["block"] = i, None
    meta = {"seed_phrase": GEOM_PHRASE, "fixtures": fx_all, "legacy": None,
            "design": {"geometries": {k: (v if v is None else {a: (list(b) if isinstance(b, tuple) else b)
                                                              for a, b in v.items()})
                                      for k, v in N.GEOMETRIES.items()},
                       "M_range": N.M_RANGE, "sig_figs": N.SIG, "notations": N.NOTATIONS,
                       "schema_price": N.SCHEMA_PRICE, "bot": {k: str(x) for k, x in N.BOT.items()},
                       "bot_text": N.BOT_TEXT, "scales": N.SCALES}}
    return meta, games


PROMPT_PHRASE = "scale-invariance/fixtures/explore-v5"
SP_PAIRS = ("NN", "AA", "CC", "DD", "AN", "NA", "CN", "NC", "AC", "CA")   # seller style, buyer style
BOT_STYLES = N.STYLE_CODES


def build_games_prompt(n_fix):
    """Q14: prompt style x scale, Study 2's zone, raw notation."""
    games = []
    fx = N.build_fixtures(n_fix, PROMPT_PHRASE, "round", "P")
    for f in fx:
        for scale, mult in N.SCALES.items():
            base = {"kind": "new", "geom": "round", "fixture": f["id"], "scale": scale, "mult": mult,
                    "notation": "raw", "F": f["F"] * mult, "C": f["C"] * mult, "R": None,
                    "seed_ns": "promptprobe/v1"}
            for opener in ("seller", "buyer"):
                o = opener[0].upper()
                for pair in SP_PAIRS:
                    games.append({**base, "cell": "SPN", "opener": opener, "bot_role": None,
                                  "styles": {"seller": pair[0], "buyer": pair[1]},
                                  "label": f"SP {pair[0]}/{pair[1]}",
                                  "game_id": f"{f['id']}-{scale}-SP{pair}-{o}"})
                for sty in BOT_STYLES:
                    for model_role in ("seller", "buyer"):
                        bot_role = "buyer" if model_role == "seller" else "seller"
                        who = "M" if opener == model_role else "B"
                        games.append({**base, "cell": "BOT", "opener": opener, "bot_role": bot_role,
                                      "styles": {model_role: sty},
                                      "label": f"BOT {sty} model={model_role}",
                                      "game_id": f"{f['id']}-{scale}-BOT{sty}-m{model_role[0].upper()}-{who}"})
    random.Random(rr.sha256_hex((PROMPT_PHRASE + "|prompt-order").encode())).shuffle(games)
    for i, g in enumerate(games, 1):
        g["run_order"], g["block"] = i, None
    meta = {"seed_phrase": PROMPT_PHRASE, "fixtures": {"round": fx}, "legacy": None,
            "design": {"geometry": "round (Study 2)", "notation": "raw", "sp_pairs": SP_PAIRS,
                       "bot_styles": BOT_STYLES, "styles": {c: N.style_text(c) for c in N.STYLE_CODES},
                       "schema_price": N.SCHEMA_PRICE, "bot": {k: str(x) for k, x in N.BOT.items()},
                       "bot_text": N.BOT_TEXT, "scales": N.SCALES}}
    return meta, games


INFO_PHRASE = "scale-invariance/fixtures/explore-v7"
INFO_PAIRS = ("NN", "AN", "NA", "AA", "PN", "NP", "PP")                 # seller info, buyer info
INFO_RANGES = {"F": (7500, 9500), "C": (10500, 12500)}                    # Study 1's ranges, x1, cents


def info_appraisal(f, phrase=INFO_PHRASE):
    """A private appraisal for a round fixture: off-centre, 25-40% of the zone for odd fixtures and 60-75%
    for even ones (balanced: half favour each side), 4 significant figures, never a multiple of $5.
    Returns (cents, position in the zone)."""
    seed = mf.seed_from_phrase(phrase)
    side = 1 - int(f["id"][1:]) % 2
    for k in range(1000):
        lab = f"info|{f['id']}|{k}"
        lo, hi = N.REF_POS[side]
        pos = lo + N._u(seed, lab + "|pos") * (hi - lo)
        R = N.sig_cents((f["F"] + pos * (f["C"] - f["F"])) / 100)
        if f["F"] < R < f["C"] and R % 500:
            return R, round((R - f["F"]) / (f["C"] - f["F"]), 4)
    raise RuntimeError(f"no appraisal for {f['id']}")


def build_games_info(n_fix):
    """Q17: private information x scale, Study 2's zone, raw notation."""
    games = []
    fx = N.build_fixtures(n_fix, INFO_PHRASE, "round", "I")
    for f in fx:
        f["R"], f["ref_pos"] = info_appraisal(f)             # used only as the PRIVATE appraisal
        for scale, mult in N.SCALES.items():
            base = {"kind": "new", "geom": "round", "fixture": f["id"], "scale": scale, "mult": mult,
                    "notation": "raw", "F": f["F"] * mult, "C": f["C"] * mult, "R": None,
                    "R_private": f["R"] * mult,
                    "ranges": {k: (lo * mult, hi * mult) for k, (lo, hi) in INFO_RANGES.items()},
                    "seed_ns": "infoprobe/v1"}
            for opener in ("seller", "buyer"):
                o = opener[0].upper()
                for pair in INFO_PAIRS:
                    games.append({**base, "cell": "SPN", "opener": opener, "bot_role": None,
                                  "info": {"seller": pair[0], "buyer": pair[1]},
                                  "label": f"SP {pair[0]}/{pair[1]}",
                                  "game_id": f"{f['id']}-{scale}-SP{pair}-{o}"})
                for code in N.INFO_CODES:
                    for model_role in ("seller", "buyer"):
                        bot_role = "buyer" if model_role == "seller" else "seller"
                        who = "M" if opener == model_role else "B"
                        games.append({**base, "cell": "BOT", "opener": opener, "bot_role": bot_role,
                                      "info": {model_role: code},
                                      "label": f"BOT {code} model={model_role}",
                                      "game_id": f"{f['id']}-{scale}-BOT{code}-m{model_role[0].upper()}-{who}"})
    random.Random(rr.sha256_hex((INFO_PHRASE + "|info-order").encode())).shuffle(games)
    for i, g in enumerate(games, 1):
        g["run_order"], g["block"] = i, None
    meta = {"seed_phrase": INFO_PHRASE, "fixtures": {"round": fx}, "legacy": None,
            "design": {"geometry": "round (Study 2)", "notation": "raw", "sp_pairs": INFO_PAIRS,
                       "bot_info": N.INFO_CODES, "ranges_x1_cents": INFO_RANGES,
                       "info_text": N.rp.load_fragments(N.INFO_FILE), "schema_price": N.SCHEMA_PRICE,
                       "bot": {k: str(x) for k, x in N.BOT.items()}, "bot_text": N.BOT_TEXT,
                       "scales": N.SCALES}}
    return meta, games


CROSS_PHRASE = "scale-invariance/fixtures/explore-v6"
CROSS_MODELS = {"G": "gemma-4-31b-it", "Q": "qwen3.8-27b-fp8"}
CROSS_SETS = (("round", "CR", "raw"), ("wide", "CW", "biz"))
CROSS_PAIRS = ("GG", "QQ", "GQ", "QG")                                   # seller model, buyer model


def build_games_cross(n_fix):
    """Q15: model pairing x scale in two zone/notation settings. Self-play cells only (no bot)."""
    games, fx_all = [], {}
    for geom, prefix, nt in CROSS_SETS:
        fx = N.build_fixtures(n_fix, CROSS_PHRASE, geom, prefix)
        fx_all[geom] = fx
        for f in fx:
            for scale, mult in N.SCALES.items():
                base = {"kind": "new", "geom": geom, "fixture": f["id"], "scale": scale, "mult": mult,
                        "notation": nt, "F": f["F"] * mult, "C": f["C"] * mult, "R": None,
                        "seed_ns": "crossprobe/v1", "cell": "SPN", "bot_role": None}
                for opener in ("seller", "buyer"):
                    for pair in CROSS_PAIRS:
                        games.append({**base, "opener": opener,
                                      "models": {"seller": CROSS_MODELS[pair[0]], "buyer": CROSS_MODELS[pair[1]]},
                                      "label": f"{pair[0]}/{pair[1]}",
                                      "game_id": f"{f['id']}-{scale}-{nt}-{pair}-{opener[0].upper()}"})
    random.Random(rr.sha256_hex((CROSS_PHRASE + "|cross-order").encode())).shuffle(games)
    for i, g in enumerate(games, 1):
        g["run_order"], g["block"] = i, None
    meta = {"seed_phrase": CROSS_PHRASE, "fixtures": fx_all, "legacy": None,
            "design": {"sets": CROSS_SETS, "models": CROSS_MODELS, "pairs": CROSS_PAIRS,
                       "schema_price": N.SCHEMA_PRICE, "scales": N.SCALES}}
    return meta, games


class SubPanel:
    """One machine's view of the shared panel (cross-play): its launcher lines are tagged and its
    machine/model-files/vLLM stages fold into one "pod <tag>" stage."""

    def __init__(self, panel, tag):
        self._p, self.tag = panel, tag
        self.t_created, self.cost_final = None, None

    def __getattr__(self, k):
        return getattr(self._p, k)

    def set_stage(self, name, state):
        if name in ("machine", "model files", "vLLM"):
            self._p.set_stage(f"pod {self.tag}", "ok" if (name == "vLLM" and state == "ok") else
                              ("fail" if state == "fail" else "run"))
        else:
            self._p.set_stage(name, state)

    def set_now(self, text):
        self._p.set_now(f"{self.tag}: {text}")

    def note(self, text):
        self._p.note(f"{self.tag}: {text}")


def preflight(model, settings, templates, port=None, strict_action=False, preview=False):
    """One real request with each notation's schema before any game: vLLM must accept it (HTTP 200) and
    return a reply our validator passes. Returns a list of problems (empty = fine).

    With strict_action, the request carries the opt-in action-dependent grammar
    (ge.strict_action_schema via N.build_request). HTTP 200 means xgrammar
    compiled the oneOf, so guided decoding will structurally enforce
    price-with-OFFER / null-with-ACCEPT-or-WALKAWAY. The reply (an OFFER from
    the seller's first move) must still validate under the stricter grammar.

    preview=True (fake server) can't honour an arbitrary schema, so only the
    plumbing/HTTP path is exercised and reply validation is skipped; the real
    xgrammar-compile check needs a pod."""
    problems = []
    mm, mc = settings["max_messages"]["value"], settings["max_message_chars"]["value"]
    for nt in N.NOTATIONS:
        g = {"F": 9549 * 10 ** 6, "C": 19320 * 10 ** 6, "R": None, "notation": nt}
        body = N.build_request(model, N.render("seller", g, [], 1, mm, mc, templates), settings, 12345, nt,
                               strict_action=strict_action)
        for _try in range(3):          # retry a transient server hiccup (no reply, 429, 5xx); not a refusal
            st, raw_, err = v.request(port or J.PORT, "POST", "/v1/chat/completions",
                                      body=json.dumps(body).encode(), total_timeout=120)
            if st not in (None, 0, 429) and not 500 <= (st or 0) <= 599:
                break
            time.sleep(3)
        tag = (nt + " strict") if strict_action else nt
        if st != 200:
            problems.append(f"{tag} schema: HTTP {st} {(raw_ or b'')[:200]!r}")
            continue
        if preview:                     # fake server: don't judge the canned reply against the schema
            continue
        msg = json.loads(raw_)["choices"][0]
        mv, errs = N.validate_reply(msg["message"].get("content"), msg.get("finish_reason"), False, mc, nt)
        if errs:
            problems.append(f"{tag} schema: reply failed validation {errs}")
    return problems


def example_prompts(fx0, settings):
    """What the model sees, for the manifest: Z-fixture seller, first move and a later move, both scales."""
    mm, mc = settings["max_messages"]["value"], settings["max_message_chars"]["value"]
    out = {}
    for scale, mult in N.SCALES.items():
        g = {"F": fx0["F"] * mult, "C": fx0["C"] * mult, "R": fx0["R"] * mult}
        out[scale] = {"biz_seller_first": N.render("seller", {**g, "R": None, "notation": "biz"}, [], 1, mm, mc),
                      "raw_seller_first": N.render("seller", {**g, "R": None, "notation": "raw"}, [], 1, mm, mc),
                      "SPR_seller_first": N.render("seller", g, [], 1, mm, mc),
                      "SPN_seller_msg3": N.render("seller", {**g, "R": None},
                                                  [("seller", g["C"], "Opening."),
                                                   ("buyer", g["F"], "Too high.")], 3, mm, mc)}
    return out


def main(argv):
    ap = argparse.ArgumentParser(description="New-game probe (see the top of this file).")
    ap.add_argument("--model", help="required except for --design cross")
    ap.add_argument("--design", choices=("newgame", "geom", "prompt", "cross", "info"), default="newgame")
    ap.add_argument("--cells", default="SPN,SPR,BOT,LEG", help="newgame design only")
    ap.add_argument("--fixtures", type=int, default=30)
    ap.add_argument("--workers", type=int, default=None, help="default 48 (newgame) or 64 (others)")
    ap.add_argument("--lifetime-min", type=int, default=None, help="default 60 (newgame, cross) or 90 (others)")
    ap.add_argument("--slot", type=int, default=int(os.environ.get("SI_SLOT", "0") or 0),
                    help="0-9: run alongside other jobs, one slot each (cross uses this slot and the next)")
    ap.add_argument("--gpu-id", default=None, help="RunPod GPU id (default: the launcher's)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview-games", type=int, default=40, help=argparse.SUPPRESS)
    ap.add_argument("--strict-action", action="store_true",
                    help="use the opt-in action-dependent reply grammar (OFFER needs a price, "
                         "ACCEPT/WALKAWAY force null); for the Gemma 4 12B rescue (WP2a)")
    ap.add_argument("--preflight", action="store_true",
                    help="only check that vLLM accepts the schema(s) and that the replies validate, "
                         "then stop before any full game (schema-compile + smoke; still needs a pod)")
    a = ap.parse_args(argv)
    cross = a.design == "cross"
    if cross and not 0 <= a.slot <= 8:
        print("--design cross needs --slot 0-8 (it uses that slot and the next)")
        return 2
    J.use_slot(a.slot)

    pins = json.load(open(P("config", "pins.json")))
    mp = {m["name"]: m for m in pins["models"]}
    cells = a.cells.split(",")
    models = list(CROSS_MODELS.values()) if cross else [a.model]
    if any(m not in mp for m in models) or any(c not in CELLS for c in cells) or not 1 <= a.fixtures <= 30:
        print("bad --model, --cells or --fixtures (1-30)")
        return 2
    model = " vs ".join(models)
    slots = [a.slot, a.slot + 1] if cross else [None]
    J.PREVIEW_MODEL, J.MANIFEST_TSV = models[0], P("config", "manifests", models[0] + ".tsv")
    settings = json.load(open(P("prereg", "settings.json")))
    templates = N.load_templates()
    a.workers = a.workers or (48 if a.design == "newgame" else 64)
    a.lifetime_min = a.lifetime_min or (60 if a.design in ("newgame", "cross") else 90)
    global PROBE
    PROBE = a.design
    if a.design == "geom":
        cells = ["SPN", "BOT"]
        meta, games = build_games_geom(a.fixtures)
    elif a.design == "prompt":
        cells = ["SPN", "BOT"]
        meta, games = build_games_prompt(a.fixtures)
    elif a.design == "info":
        cells = ["SPN", "BOT"]
        meta, games = build_games_info(a.fixtures)
    elif cross:
        cells = ["SPN"]
        meta, games = build_games_cross(a.fixtures)
    else:
        meta, games = build_games(a.fixtures, cells)
    if a.preview:
        games = games[:a.preview_games]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    data_root = tempfile.mkdtemp(prefix="si-newgame-preview-") if a.preview else P("data", "explore")
    run_id = f"X-{PROBE}-{'-vs-'.join(models)}-{stamp}"
    run_dir = os.path.join(data_root, run_id)
    n_by = {c: sum(g["cell"] == c for g in games) for c in cells}
    plan = (f"Probe {PROBE}: {len(games)} games (" + ", ".join(f"{c} {n}" for c, n in n_by.items())
            + f"), {a.fixtures} fixtures, {a.workers} at a time")
    gpu = ["--gpu-id", a.gpu_id] if a.gpu_id else []

    worsts, prices = [], []
    if a.preview:
        worsts, prices = ["0.00"] * len(models), [0.0] * len(models)
        print(f"PREVIEW: fake machine and server on this laptop. Nothing is rented.\n  {plan}.\n")
    else:
        if J.uncommitted():
            print("Note: uncommitted changes are recorded in the manifest (probe runs don't require a commit).")
        for m, sl in zip(models, slots):
            rc, out = J.dry_run(m, a.lifetime_min, gpu, sl)
            if cross:
                print(f"--- {m} (slot {sl})")
            print("\n".join(l for l in out.rstrip().splitlines() if not l.startswith("Dry run:")).rstrip())
            mm = re.search(r"--confirm-cost ([0-9.]+)", out)
            if rc != 0 or not mm:
                print("\nNot launching (the launcher's checks above say why).")
                return 3
            worsts.append(mm.group(1))
            prices.append(float(re.search(r"\$([0-9.]+)/hr", out).group(1)))
    worst = f"{sum(float(x) for x in worsts):.2f}"
    if not a.preview:
        print(f"\nThis probe: {plan}.\nThe machine{'s delete themselves' if cross else ' deletes itself'} after"
              f" {a.lifetime_min} min at the latest." + (f" Two pods: worst case {' + '.join(worsts)}." if cross else ""))
        try:
            typed = input(f"Type the worst-case cost ({worst}) to launch, or anything else to cancel: ").strip()
        except (EOFError, KeyboardInterrupt):
            typed = ""
        if typed.lstrip("$") != worst:
            print("Cancelled. Nothing was created.")
            return 0
        print()
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    tags = ["gemma", "qwen"] if cross else [None]
    stages = (("pod gemma", "pod qwen") if cross else ("machine", "model files", "vLLM")) + \
        ("checks", "probe", "teardown")
    panel = J.Panel(f"PROBE {PROBE} · {model}" + (f" · slot {a.slot}" + ("+1" if cross else "") if a.slot or cross
                                                  else ""), stages, sys.stdout.isatty(), a.preview)
    panel.lifetime_s, panel.price, panel.worst = a.lifetime_min * 60, sum(prices), worst
    os.makedirs(P("logs"), exist_ok=True)
    Mach = J.FakeMachine if a.preview else J.Machine
    machines, subs = [], []
    for m, sl, tag in zip(models, slots, tags):
        sp = SubPanel(panel, tag) if cross else panel
        subs.append(sp)
        log = P("logs", f"probe_{PROBE}_{stamp}" + (f".{tag}" if tag else "") + ".launcher.out")
        machines.append(Mach(sp, log, slot=sl, manifest_tsv=P("config", "manifests", m + ".tsv")))
    by_model = dict(zip(models, machines))
    ports = {m: (18000 + sl if sl is not None else J.PORT) for m, sl in zip(models, slots)}
    srvs = [J.fake_server(ports[m], m) for m in models] if a.preview else []
    if a.preview:
        panel.t_created = time.time()
    interrupts = [0]
    # server_checks() renders one Study 2 first move to confirm the thinking-off marker.
    check_game = {"game_id": "check", "opener": "seller", "V": 100, "F": 81, "C": 124, "run_order": 1}

    def ctx_for(m):
        return {"panel": panel, "machine": by_model[m], "args": a, "model": m, "settings": settings, "pins": pins,
                "mp": mp, "pilot_fx": {"games": [check_game], "f_pct_range": (75, 95), "c_pct_range": (105, 125)},
                "family": J.family_of(m), "interrupts": interrupts, "study": 2,
                "render": rp2.render}
    ctx = ctx_for(models[0])

    def on_int(signum, frame):
        interrupts[0] += 1
        panel.note("stop requested: finishing current moves, then deleting the machine"
                   if interrupts[0] == 1 else "stop requested again: skipping the log copy")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_int)
    threading.Thread(target=panel.ticker, daemon=True).start()
    panel.draw()

    result, w = {"status": "not started"}, None
    counts = {"agreements": 0, "walkaways": 0, "timeouts": 0, "forfeits": 0, "model_calls": 0}
    try:
        for m, mach, wst in zip(models, machines, worsts):
            mach.start(["--mode", "serve", "--model", m, "--lifetime-min", str(a.lifetime_min),
                        "--confirm-cost", wst] + gpu)
            if cross:
                time.sleep(3)           # let the first launcher claim its slot before the second checks
        while not all(mach.ready.is_set() for mach in machines):
            time.sleep(1.0)
            if interrupts[0]:
                raise J.JobStop("stopped by you before the games began")
            for mach in machines:
                if mach.ready.is_set() and (not mach.alive() or mach.failed):
                    raise J.JobStop(mach.failed or "a launcher ended before vLLM was ready")
        for mach in machines:
            if not mach.alive() or mach.failed:
                raise J.JobStop(mach.failed or "the launcher ended before vLLM was ready")
        if cross and not a.preview:
            panel.t_created = min(sp.t_created for sp in subs if sp.t_created) if any(
                sp.t_created for sp in subs) else time.time()
        panel.set_stage("checks", "run"); panel.set_now("checking the server before any game")
        infos = {}
        for m, sl in zip(models, slots):
            if sl is not None:
                J.use_slot(sl)
            J.MANIFEST_TSV = P("config", "manifests", m + ".tsv")
            problems, infos[m] = J.server_checks(ctx_for(m))
            if problems:
                raise J.JobStop(f"check failed ({m}): " + "; ".join(problems))
        J.use_slot(a.slot)
        J.MANIFEST_TSV = P("config", "manifests", models[0] + ".tsv")
        panel.set_stage("checks", "ok")
        panel.note("checks passed: weights, version 0.30.0, one model, thinking off"
                   + ("" if a.preview else ", no watermark") + (" (both pods)" if cross else ""))

        panel.set_now("checking the price schemas with one real request each")
        for m in models:
            pf = preflight(m, settings, templates, ports[m], strict_action=a.strict_action, preview=a.preview)
            if pf:
                raise J.JobStop(f"preflight failed ({m}): " + "; ".join(pf))
        panel.note("preflight: vLLM accepts "
                   + ("the strict action grammar" if a.strict_action else "both price schemas")
                   + " and the replies validate" + (" (both models)" if cross else ""))
        if a.preflight:
            raise J.JobStop("--preflight: schema check passed; stopping before any full game")

        sessions = {m: J.session_record(ctx_for(m), infos[m]) for m in models}
        fx0 = meta["fixtures"]
        fx0 = (fx0.get("wide") or fx0.get("round")) if isinstance(fx0, dict) else fx0
        manifest = {
            "schema": "run_manifest/1", "run_id": run_id, "kind": "probe", "preview": a.preview,
            "study": "probe", "probe": PROBE, "not_preregistered": True, "model": model,
            "repo": " vs ".join(mp[m]["repo"] for m in models),
            "revision": " vs ".join(mp[m]["revision"] for m in models),
            "image": "vllm/vllm-openai@" + pins["vllm_image"]["index_digest"], **sessions[models[0]],
            **({"sessions": sessions, "models": models} if cross else {}),
            "cells": {c: CELLS[c] for c in cells},
            "prompts": "config/explore/newgame-v2" + (" + config/explore/styles" if a.design == "prompt" else "")
                       + (" + config/explore/info" if a.design == "info" else "")
                       + (" (LEG: prereg/study2/prompts)" if a.design == "newgame" else ""),
            "design": a.design, "gpu_id": a.gpu_id, "strict_action": a.strict_action,
            "example_prompts_first_fixture": example_prompts(fx0[0], settings),
            "seed_rule": "sha256('scale-invariance/<seed_ns>|fixture|cell|opener|model_role|msg|attempt')[:8];"
                         " the scale (and style, notation, pairing) is never an input",
            "workers": a.workers, "fixture_meta": meta,
            "games_planned": [g["game_id"] for g in games], "started_utc": rr.utc_now(),
        }
        w = J_locked(rr.RunWriter(run_dir, manifest))
        with open(os.path.join(run_dir, "games_planned.json"), "w") as fh:
            json.dump(games, fh, indent=0)
        panel.set_stage("probe", "run"); panel.reset("Probe", len(games))
        panel.note(f"probe started: {len(games)} games, {a.workers} at a time")

        transports = {m: v.make_chat_transport(ports[m], total_timeout=300) for m in models}

        def transport(body_bytes):          # route each request to the pod serving its model
            return transports[json.loads(body_bytes)["model"]](body_bytes)
        base_call = rr.recording_call(w, transport, run_id,
                                      sleep=(lambda s: time.sleep(0.1)) if a.preview else time.sleep)
        first, first_lock, stop = {}, threading.Lock(), threading.Event()
        by_id = {g["game_id"]: g for g in games}

        def call(body, info_):
            if interrupts[0] or stop.is_set():
                raise J.JobStop("stopped")
            if not all(mach.alive() for mach in machines):
                raise J.JobStop("a machine session ended")
            g = by_id[info_["game_id"]]
            # Fake server only: plausible whole-dollar prices near this side's limit.
            lim = g["F"] if info_["role"] == "seller" else g["C"]
            J.FAKE["msg_no"], J.FAKE["V"] = info_["msg_no"], (lim / 100 if g["kind"] == "new" else lim)
            t0 = time.monotonic()
            reply = base_call(body, info_)
            with panel.lock:
                panel.reply_secs.append(time.monotonic() - t0)
                panel.retries += int(reply["call_id"].rsplit("/t", 1)[1]) - 1
            m = body["model"]
            if reply.get("kind") == "model_output" and not a.preview:
                with first_lock:
                    if m not in first:
                        first[m] = reply.get("response_id")
                        ok, why = J.logged_settings_match(by_model[m], first[m], body)
                        w.event("gate_effective_settings", ok=ok, detail=why, response_id=first[m], model=m)
                        if not ok:
                            stop.set()
                            raise J.JobStop(f"check failed ({m}): logged settings " + why)
                        panel.note("settings vLLM logged for the first reply equal settings.json"
                                   + (f" ({m})" if cross else ""))
            return reply

        q = queue.Queue()
        for g in games:
            q.put((g, 1))
        stop_why = []
        leg_f, leg_c = (meta["legacy"]["f_pct_range"], meta["legacy"]["c_pct_range"]) if meta["legacy"] \
            else ((75, 95), (105, 125))
        play_model = model if cross else models[0]

        def worker():
            while not stop.is_set() and not interrupts[0]:
                if not a.preview and panel.t_created:
                    recent = panel.game_secs[-15:]
                    avg = sum(recent) / len(recent) if recent else 120
                    if panel.t_created + a.lifetime_min * 60 - time.time() < max(300, 2 * avg):
                        stop_why.append("machine time nearly up; no new game started")
                        stop.set()
                        return
                try:
                    g, attempt = q.get_nowait()
                except queue.Empty:
                    return
                t0 = time.monotonic()
                try:
                    if g["kind"] == "new":
                        rec = N.play_game(g, play_model, settings, call, attempt=attempt, templates=templates,
                                          strict_action=a.strict_action)
                    else:
                        rec = ge.play_game(g, play_model, settings, leg_f, leg_c, call, attempt=attempt,
                                           render=rp2.render)
                        rec.update({"kind": "legacy", "cell": "LEG", "units": "dollars",
                                    "model_calls": len(rec["call_ids"]), "bot_role": None, "model_role": None})
                except J.JobStop as e:
                    w.event("game_interrupted", game_id=g["game_id"], attempt=attempt, why=str(e))
                    stop_why.append(str(e))
                    stop.set()
                    return
                w.game(rec)
                with panel.lock:
                    d = rec["diagnostics"]
                    panel.reasoning += sum(1 for x in d if x.startswith("REASONING_PRESENT"))
                    panel.truncated += sum(1 for x in d if x.startswith("INVALID:") and "TRUNCATED" in x)
                    if rec["terminal"] == "INFRASTRUCTURE_VOID":
                        panel.voided += 1
                        panel.replies += rec["model_calls"] - 1
                        if attempt < J.MAX_ATTEMPTS:
                            q.put((g, attempt + 1))
                        else:
                            panel.games_done += 1
                    else:
                        panel.replies += rec["model_calls"]
                        counts["model_calls"] += rec["model_calls"]
                        panel.invalid += int(rec["terminal"] == "PROTOCOL_FORFEIT")
                        key = {"AGREEMENT": "agreements", "WALKAWAY": "walkaways", "TIMEOUT": "timeouts",
                               "PROTOCOL_FORFEIT": "forfeits"}[rec["terminal"]]
                        counts[key] += 1
                        panel.games_done += 1
                        panel.game_secs.append((time.monotonic() - t0) / max(1, a.workers))
                panel.set_now(f"probe: {panel.games_done} of {len(games)} games done")
                panel.draw()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(1.0)
        if interrupts[0]:
            raise J.JobStop("stopped by you")
        if stop_why:
            raise J.JobStop(stop_why[0])
        panel.set_stage("probe", "ok")
        result = {"status": "complete"}
    except J.JobStop as e:
        preflight_ok = str(e).startswith("--preflight:")
        result = {"status": "preflight-ok" if preflight_ok else "stopped", "why": str(e)}
        panel.note(("preflight ok: " if preflight_ok else "stopped: ") + str(e))
    except Exception as e:
        result = {"status": "error", "why": f"{type(e).__name__}: {e}"}
        panel.note(f"error: {type(e).__name__}: {e}")
    finally:
        if panel.stage.get("probe") == "run":
            panel.set_stage("probe", "fail")
        if w is not None:
            try:
                for m in models:
                    J.copy_server_logs(ctx_for(m), os.path.join(run_dir, m) if cross else run_dir)
                w.close(status=result["status"], games_done=panel.games_done, replies=panel.replies,
                        voided=panel.voided, reasoning=panel.reasoning, **counts)
            except Exception as e:          # never let record-keeping block the teardown below
                panel.note(f"could not close the run records: {type(e).__name__}: {e}")
        panel.set_now("deleting the machine" + ("s" if cross else ""))
        fins = [threading.Thread(target=mach.finish) for mach in machines]   # tear down in parallel
        for t in fins:
            t.start()
        for t in fins:
            t.join()
        if cross:
            costs = [sp.cost_final for sp in subs]
            panel.cost_final = sum(costs) if all(c is not None for c in costs) else None
        for srv in srvs:
            srv.shutdown()
        panel.set_now("done" if result["status"] in ("complete", "preflight-ok") else "stopped")
        panel.stop.set()
        panel.draw()

    print()
    print(f"Probe {result['status']}" + (f": {result['why']}" if result.get("why") else "") + ".")
    print(f"  {panel.games_done} of {len(games)} games; {panel.replies - panel.invalid} of {panel.replies} model"
          f" replies valid; voided {panel.voided}; reasoning {panel.reasoning}")
    print("  Data: " + (os.path.relpath(run_dir, J.ROOT) if not a.preview else run_dir + " (preview)"))
    script = {"geom": "analyze_geom_probe.py", "prompt": "analyze_cell_probe.py", "info": "analyze_info_probe.py",
              "cross": "analyze_cell_probe.py"}.get(a.design, "analyze_game_probe.py")
    print(f"  Next: python3 scripts/{script} " + (os.path.relpath(run_dir, J.ROOT) if not a.preview else run_dir))
    ok = all(mach.deleted is True for mach in machines)
    if ok:
        cost = f"${panel.cost_final:.2f}" if panel.cost_final is not None else "see logs/budget.tsv"
        print(f"  Machine{'s' if cross else ''} deleted; 0 of our pods left. Estimated cost {cost}.")
    else:
        print("  !!! COULD NOT CONFIRM " + ("BOTH MACHINES ARE" if cross else "THE MACHINE IS")
              + " GONE. Check the RunPod console now. !!!")
    return 9 if not ok else (0 if result["status"] in ("complete", "preflight-ok") else 1)


def J_locked(w):
    """Thread-safe appends to the three hash chains (same as run_probe.locked_writer)."""
    lock = threading.Lock()
    for ch in (w.calls, w.games, w.events):
        orig = ch.append

        def append(obj, _orig=orig):
            with lock:
                return _orig(obj)
        ch.append = append
    return w


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
