#!/usr/bin/env python3
"""Offline checks for scripts/newgame.py and scripts/run_game_probe.py (no network, no model).

    python3 tests/test_game_probe.py
"""
import json
import math
import os
import sys
import unittest
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import newgame as N                 # noqa: E402
import run_game_probe as G          # noqa: E402

SETTINGS = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
MM, MC = SETTINGS["max_messages"]["value"], SETTINGS["max_message_chars"]["value"]
FX = N.build_fixtures(30)


def reply(move):
    return {"kind": "model_output", "content": json.dumps(move), "finish_reason": "stop", "reasoning": None,
            "call_id": "x/a1/m1/t1"}


class Geometries(unittest.TestCase):
    def test_three_geometries(self):
        rnd = N.build_fixtures(30, "t-round", "round", "R")
        nar = N.build_fixtures(30, "t-narrow", "narrow", "N")
        wid = N.build_fixtures(30, "t-wide", "wide", "W")
        self.assertTrue(all(f["F"] % 100 == 0 and 7500 <= f["F"] <= 9500 and 10500 <= f["C"] <= 12500 for f in rnd))
        for f in nar:
            self.assertTrue(0.75 <= f["F"] / (f["M"] * 100) <= 0.951 and 1.049 <= f["C"] / (f["M"] * 100) <= 1.25)
        cf = lambda fx: sum(f["C"] / f["F"] for f in fx) / len(fx)               # noqa: E731
        self.assertLess(cf(nar), 1.5)
        self.assertGreater(cf(wid), 1.8)
        self.assertEqual({f["geom"] for f in rnd + nar + wid}, {"round", "narrow", "wide"})


class Fixtures(unittest.TestCase):
    def test_rules_hold(self):
        for f in FX:
            M = f["M"] * 100
            self.assertTrue(N.M_RANGE[0] <= f["M"] <= N.M_RANGE[1])
            self.assertTrue(f["F"] < f["R"] < f["C"])
            self.assertGreaterEqual(f["C"] - f["F"], N.MIN_WIDTH * M - 1)
            for x in (f["F"], f["C"], f["R"]):
                self.assertNotEqual(x % 500, 0, "no multiples of $5")
                self.assertLessEqual(len(str(x).rstrip("0")), N.SIG, "4 significant figures")
            self.assertTrue(0.25 <= f["ref_pos"] <= 0.40 or 0.60 <= f["ref_pos"] <= 0.75)

    def test_no_focal_point(self):
        mids = [(f["F"] + f["C"]) / 2 / (f["M"] * 100) for f in FX]
        self.assertGreater(max(mids) - min(mids), 0.3)          # the zone's midpoint moves around
        self.assertGreater(len({round(math.log10(f["M"])) for f in FX}), 1)   # centres span magnitudes

    def test_deterministic(self):
        self.assertEqual(FX, N.build_fixtures(30))


class Notation(unittest.TestCase):
    def test_usd(self):
        self.assertEqual(N.usd(24370), "$243.70")
        self.assertEqual(N.usd(24370 * 10 ** 6), "$243,700,000")
        self.assertEqual(N.usd(123400), "$1,234")
        self.assertEqual(N.usd(5), "$0.05")
        with self.assertRaises(ValueError):
            N.usd(0)

    def test_every_price_in_business_format(self):
        f = FX[0]
        for mult in N.SCALES.values():
            g = {"F": f["F"] * mult, "C": f["C"] * mult, "R": f["R"] * mult}
            hist = [("seller", g["C"], "a"), ("buyer", g["F"], "b")]
            sysm, user = (m["content"] for m in N.render("seller", g, hist, 3, MM, MC))
            self.assertIn(f"Your private floor is {N.usd(g['F'])}.", user)
            self.assertIn(f"OFFER at {N.usd(g['C'])}, with", user)
            self.assertIn(f"The buyer's current offer is {N.usd(g['F'])}.", user)
            self.assertIn(f"It values the lot at {N.usd(g['R'])}.", sysm)
            self.assertNotIn("dollars", user)                    # no "N dollars" anywhere
            self.assertIn("cents only if needed", sysm)
        no_ref = N.render("buyer", {**g, "R": None}, [], 1, MM, MC)[0]["content"]
        self.assertNotIn("appraisal", no_ref)
        self.assertNotIn("{{", no_ref)

    def test_scales_differ_only_by_zeros(self):
        f = FX[3]
        small = N.render("seller", {"F": f["F"], "C": f["C"], "R": f["R"]}, [], 1, MM, MC)
        big = N.render("seller", {"F": f["F"] * 10 ** 6, "C": f["C"] * 10 ** 6, "R": f["R"] * 10 ** 6}, [], 1, MM, MC)
        strip = lambda s: s.replace(",", "").replace(".", "").replace("0", "")   # noqa: E731
        self.assertEqual(strip(small[1]["content"]), strip(big[1]["content"]))


class Replies(unittest.TestCase):
    def test_raw_cents_accepted(self):
        for price, cents in ((243.7, 24370), (243, 24300), (243700000, 24370000000), (12.05, 1205)):
            mv, err = N.validate_reply(json.dumps({"action": "OFFER", "price": price, "message": "x"}), "stop",
                                       False, MC, "raw")
            self.assertEqual(err, [])
            self.assertEqual(mv["price"], cents)

    def test_biz_strings_accepted(self):
        for price, cents in (("$68,640,000", 6864000000), ("$95.49", 9549), ("$1,234", 123400),
                             ("$68,640,000.00", 6864000000), ("$7", 700)):
            mv, err = N.validate_reply(json.dumps({"action": "OFFER", "price": price, "message": "x"}), "stop",
                                       False, MC, "biz")
            self.assertEqual(err, [], price)
            self.assertEqual(mv["price"], cents)

    def test_biz_rejects_the_slip_and_near_misses(self):
        for price in (68, 68640000, "68,640,000", "$68640000", "$6,86,40", "$95.5", "$068", " $68", "$68,640,000 "):
            mv, err = N.validate_reply(json.dumps({"action": "OFFER", "price": price, "message": "x"}), "stop",
                                       False, MC, "biz")
            self.assertIsNone(mv, price)
            self.assertIn("BAD_PRICE_FORMAT", err)
        self.assertIsNone(N.parse_usd("$1,2345"))

    def test_raw_strict(self):
        bad = {'{"action": "OFFER", "price": 10.123, "message": "x"}': "PRICE_PRECISION",
               '{"action": "OFFER", "price": "10", "message": "x"}': "BAD_PRICE_TYPE",
               '{"action": "OFFER", "price": 0, "message": "x"}': "NONPOSITIVE_PRICE",
               '{"action": "ACCEPT", "price": 10, "message": "x"}': "PRICE_MUST_BE_NULL",
               '{"action": "OFFER", "price": 10, "message": "x"} extra': "TEXT_AFTER_JSON",
               '{"action": "OFFER", "price": NaN, "message": "x"}': "NONFINITE_NUMBER"}
        for text, why in bad.items():
            mv, err = N.validate_reply(text, "stop", True, MC, "raw")
            self.assertIsNone(mv)
            self.assertIn(why, err, text)
        self.assertEqual(N.validate_reply('{"action": "ACCEPT", "price": null, "message": ""}', "stop", False, MC)[1],
                         ["ACCEPT_WITHOUT_OFFER"])

    def test_schema_per_notation(self):
        msgs = N.render("seller", {"F": 100, "C": 200, "R": None}, [], 1, MM, MC)
        for nt in N.NOTATIONS:
            body = N.build_request("gemma-4-31b-it", msgs, SETTINGS, 1, nt)
            sch = body["response_format"]["json_schema"]["schema"]["properties"]["price"]
            self.assertEqual(sch, N.SCHEMA_PRICE[nt])
        biz = N.SCHEMA_PRICE["biz"]["anyOf"][0]
        self.assertNotIn("maxLength", biz)            # vLLM refuses pattern + length bounds for xgrammar
        self.assertNotIn("minLength", biz)

    def test_price_instructions_per_notation(self):
        g = {"F": 9549, "C": 19320, "R": None}
        biz = N.render("seller", {**g, "notation": "biz"}, [], 1, MM, MC)
        rawm = N.render("seller", {**g, "notation": "raw"}, [], 1, MM, MC)
        self.assertIn("as text in quotes", biz[0]["content"])
        self.assertIn("Your private floor is $95.49.", biz[1]["content"])
        self.assertIn("plain number of dollars", rawm[0]["content"])
        self.assertIn("Your private floor is 95.49 dollars.", rawm[1]["content"])
        g6 = {"F": 9549 * 10 ** 6, "C": 19320 * 10 ** 6, "R": None, "notation": "raw"}
        self.assertIn("Your private floor is 95490000 dollars.", N.render("seller", g6, [], 1, MM, MC)[1]["content"])

    def test_seed_ignores_scale(self):
        self.assertEqual(N.call_seed("Z01", "SPN", "seller", "both", 3, 1), N.call_seed("Z01", "SPN", "seller", "both", 3, 1))
        self.assertNotEqual(N.call_seed("Z01", "SPN", "seller", "both", 3, 1), N.call_seed("Z01", "SPR", "seller", "both", 3, 1))


class Bot(unittest.TestCase):
    def play(self, f, mult, model_role, opener, notation="biz"):
        """A scripted, scale-proof 'model': opens 20% beyond its limit, concedes half the gap each turn."""
        bot_role = "buyer" if model_role == "seller" else "seller"
        g = {"game_id": "t", "fixture": f["id"], "scale": "s", "mult": mult, "cell": "BOT", "opener": opener,
             "bot_role": bot_role, "F": f["F"] * mult, "C": f["C"] * mult, "R": None, "notation": notation}

        def call(body, info):
            user = body["messages"][1]["content"]
            lim = g["F"] if model_role == "seller" else g["C"]
            mine = [l for l in user.splitlines() if l.startswith("Message") and ", you:" in l]
            if "current offer is" in user and len(mine) >= 4:
                return reply({"action": "ACCEPT", "price": None, "message": "ok"})
            k = len(mine)
            mult_ = 1 + Fraction(1, 5) / 2 ** k if model_role == "seller" else 1 - Fraction(1, 5) / 2 ** k
            cents = N._sig_round_cents(lim * mult_)
            price = N.usd(cents) if notation == "biz" else cents / 100
            return reply({"action": "OFFER", "price": price, "message": "x"})
        return N.play_game(g, "gemma-4-31b-it", SETTINGS, call)

    def test_bot_is_scale_proof(self):
        for f in FX[:10]:
            for model_role in ("seller", "buyer"):
                for opener, nt in (("seller", "biz"), ("buyer", "biz"), ("seller", "raw"), ("buyer", "raw")):
                    a, b = self.play(f, 1, model_role, opener, nt), self.play(f, 10 ** 6, model_role, opener, nt)
                    self.assertNotIn("PROTOCOL_FORFEIT", (a["terminal"], b["terminal"]))
                    self.assertEqual(a["terminal"], b["terminal"])
                    self.assertEqual([o[2] * 10 ** 6 for o in a["offers"]], [o[2] for o in b["offers"]])
                    self.assertEqual(a["seller_share"], b["seller_share"])

    def test_bot_respects_its_limit(self):
        f = FX[5]
        h = [("seller", f["F"] * 3, "")]
        for _ in range(12):
            act, p, _t = N.bot_move("buyer", f["C"], h)
            if act != "OFFER":
                break
            self.assertLessEqual(p, f["C"])
            h += [("buyer", p, ""), ("seller", h[-1][1], "")]  # the seller never moves


class Analysis(unittest.TestCase):
    def test_slip_detector(self):
        import analyze_game_probe as AG
        r = {"F": 3108000000, "C": 7914000000, "offers": [[1, "buyer", 6088000000, "bot"], [2, "seller", 6800, "model"]]}
        self.assertTrue(AG.slipped(r))
        r["offers"][1][2] = 6864000000
        self.assertFalse(AG.slipped(r))
        r["offers"][0][2] = 6088                     # a bot price never counts
        self.assertFalse(AG.slipped(r))


class Design(unittest.TestCase):
    def test_game_counts_and_pairs(self):
        meta, games = G.build_games(30, ["SPN", "SPR", "BOT", "LEG"])
        cnt = {c: sum(g["cell"] == c for g in games) for c in ("SPN", "SPR", "BOT", "LEG")}
        self.assertEqual(cnt, {"SPN": 120, "SPR": 120, "BOT": 240, "LEG": 120})
        self.assertEqual(len({g["game_id"] for g in games}), len(games))
        new = [g for g in games if g["kind"] == "new"]
        key = lambda g: (g["fixture"], g["cell"], g["opener"], g.get("bot_role"))   # noqa: E731
        by = {}
        for g in new:
            by.setdefault(key(g), {})[g["scale"]] = g
        for pair in by.values():
            self.assertEqual(set(pair), {"x1", "x1e6"})
            self.assertEqual(pair["x1"]["F"] * 10 ** 6, pair["x1e6"]["F"])
        leg = [g for g in games if g["cell"] == "LEG"]
        self.assertEqual({g["scale"] for g in leg}, {"x1e2", "x1e8"})
        self.assertTrue(all(g["fixture"].startswith("L") for g in leg))


class PromptAndCross(unittest.TestCase):
    def test_prompt_design(self):
        meta, games = G.build_games_prompt(30)
        self.assertEqual(len(games), 2160)
        self.assertEqual(len({g["game_id"] for g in games}), 2160)
        labels = {g["label"] for g in games}
        self.assertEqual(len(labels), 10 + 4 * 2)
        for g in games:
            self.assertEqual((g["geom"], g["notation"]), ("round", "raw"))
            if g["cell"] == "BOT":       # only the model's side has a style
                self.assertEqual(list(g["styles"]), ["seller" if g["bot_role"] == "buyer" else "buyer"])
        # each label: 30 fixtures x 2 scales x (2 openers, or 2 openers for BOT per role)
        c = {}
        for g in games:
            c[g["label"]] = c.get(g["label"], 0) + 1
        self.assertEqual(set(c.values()), {120})

    def test_style_text_only_where_asked(self):
        g = {"F": 8100, "C": 12400, "R": None, "notation": "raw"}
        plain = N.render("seller", g, [], 1, 10, 300)
        same = N.render("seller", {**g, "styles": {"seller": "N", "buyer": "A"}}, [], 1, 10, 300)
        self.assertEqual(plain, same)                     # N adds nothing; the other side's style is invisible
        hard = N.render("seller", {**g, "styles": {"seller": "A"}}, [], 1, 10, 300)
        self.assertIn("Negotiate hard.", hard[1]["content"])
        self.assertEqual(plain[0], hard[0])               # the system message never changes
        for code in "ACD":
            self.assertTrue(N.style_text(code))

    def test_cross_design_and_routing(self):
        meta, games = G.build_games_cross(30)
        self.assertEqual(len(games), 960)
        g = next(x for x in games if x["label"] == "G/Q" and x["opener"] == "seller")
        settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
        seen = []

        def call(body, info):
            seen.append((info["role"], body["model"]))
            return {"kind": "model_output", "call_id": "c/t1", "finish_reason": "stop", "reasoning": None,
                    "content": json.dumps({"action": "ACCEPT" if info["msg_no"] > 1 else "OFFER",
                                           "price": None if info["msg_no"] > 1 else
                                           (N.usd(g["C"]) if g["notation"] == "biz" else g["C"] / 100),
                                           "message": "ok"})}
        rec = N.play_game(g, "cross", settings, call)
        self.assertEqual(seen, [("seller", "gemma-4-31b-it"), ("buyer", "qwen3.8-27b-fp8")])
        self.assertEqual(rec["terminal"], "AGREEMENT")
        self.assertEqual(rec["models"], {"seller": "gemma-4-31b-it", "buyer": "qwen3.8-27b-fp8"})

    def test_machine_slot_paths(self):
        import run_job as J
        m = J.Machine(None, os.devnull, slot=5)
        self.assertTrue(m.stopfile().endswith("pod_session.5.stop"))
        self.assertEqual(m.env()["SI_SLOT"], "5")


class Info(unittest.TestCase):
    def test_info_design(self):
        meta, games = G.build_games_info(30)
        self.assertEqual(len(games), 1560)
        self.assertEqual(len({g["game_id"] for g in games}), 1560)
        fx = meta["fixtures"]["round"]
        self.assertEqual(sum(f["ref_pos"] > 0.5 for f in fx), 15)          # balanced appraisal sides
        for f in fx:
            self.assertTrue(f["F"] < f["R"] < f["C"] and f["R"] % 500)
        for g in games:
            self.assertIsNone(g["R"])                                      # never the shared appraisal
            if g["cell"] == "BOT":
                self.assertEqual(list(g["info"]), ["seller" if g["bot_role"] == "buyer" else "buyer"])

    def test_only_the_informed_side_sees_it(self):
        g = {"F": 8100, "C": 12400, "R": None, "notation": "raw", "R_private": 9137,
             "ranges": {"F": (7500, 9500), "C": (10500, 12500)}}
        plain_s, plain_b = (N.render(r, g, [], 1, 10, 300) for r in ("seller", "buyer"))
        a = {**g, "info": {"seller": "A", "buyer": "N"}}
        self.assertIn("91.37 dollars", N.render("seller", a, [], 1, 10, 300)[1]["content"])
        self.assertEqual(N.render("buyer", a, [], 1, 10, 300), plain_b)   # uninformed side unchanged
        self.assertEqual(N.render("seller", a, [], 1, 10, 300)[0], plain_s[0])   # system message unchanged
        p = N.render("buyer", {**g, "info": {"buyer": "P"}}, [], 1, 10, 300)[1]["content"]
        self.assertIn("seller's floor was drawn at random from the range 75 dollars to 95 dollars", p)
        big = {**g, "F": 8100 * 10 ** 6, "C": 12400 * 10 ** 6, "R_private": 9137 * 10 ** 6,
               "ranges": {"F": (7500 * 10 ** 6, 9500 * 10 ** 6), "C": (10500 * 10 ** 6, 12500 * 10 ** 6)},
               "info": {"seller": "P"}}
        self.assertIn("from the range 105000000 dollars to 125000000 dollars",
                      N.render("seller", big, [], 1, 10, 300)[1]["content"])


class StrictAction(unittest.TestCase):
    """WP2a: the opt-in action-dependent reply grammar and the runner flags."""

    def test_build_request_default_vs_strict(self):
        off = N.build_request("gemma-4-12b-it", [], SETTINGS, 1, "raw")["response_format"]["json_schema"]["schema"]
        self.assertIn("anyOf", off["properties"]["price"])
        on = N.build_request("gemma-4-12b-it", [], SETTINGS, 1, "raw",
                             strict_action=True)["response_format"]["json_schema"]["schema"]
        self.assertEqual(len(on["oneOf"]), 2)
        self.assertEqual(on["oneOf"][0]["properties"]["price"], {"type": "number", "exclusiveMinimum": 0})
        self.assertEqual(on["oneOf"][1]["properties"]["price"], {"type": "null"})

    def test_play_game_runs_with_strict_action(self):
        """A scripted game under strict_action=True still plays and scores; the
        request bodies carry the oneOf grammar."""
        f = FX[0]
        g = {"game_id": "T1-x1-SPN-S", "fixture": "T1", "cell": "SPN", "opener": "seller",
             "scale": "x1", "mult": 1, "F": f["F"], "C": f["C"], "R": None, "notation": "raw",
             "bot_role": None, "models": None, "kind": "new", "geom": "round"}
        mid = (f["F"] + f["C"]) / 2
        seen = []

        def call(body, info_):
            seen.append(body)
            # seller opens high, buyer accepts on the second message
            if info_["role"] == "seller" and info_["msg_no"] == 1:
                move = {"action": "OFFER", "price": f["C"] / 100, "message": "open"}
            else:
                move = {"action": "ACCEPT", "price": None, "message": "ok"}
            return {"kind": "model_output", "content": json.dumps(move), "finish_reason": "stop",
                    "reasoning": None, "call_id": f"x/a1/m{info_['msg_no']}/t1"}

        rec = N.play_game(g, "gemma-4-12b-it", SETTINGS, call, templates=N.load_templates(), strict_action=True)
        self.assertEqual(rec["terminal"], "AGREEMENT")
        # every request carried the strict oneOf grammar
        for body in seen:
            self.assertIn("oneOf", body["response_format"]["json_schema"]["schema"])

    def test_runner_declares_the_new_flags(self):
        src = open(os.path.join(ROOT, "scripts", "run_game_probe.py")).read()
        self.assertIn('"--strict-action"', src)
        self.assertIn('"--preflight"', src)
        self.assertIn("strict_action=a.strict_action", src)
        self.assertIn('result["status"] in ("complete", "preflight-ok")', src)


class ModelFamilies(unittest.TestCase):
    """Amendment 2026-10-04: every pinned model has a chat-template tail and a sampling profile."""

    def test_every_pinned_model_has_family_tail_and_sampling(self):
        import run_job as J
        pins = json.load(open(os.path.join(ROOT, "config", "pins.json")))
        for m in pins["models"]:
            fam = J.family_of(m["name"])
            self.assertIn(fam, J.TAILS)
            self.assertIn(m["name"], SETTINGS["sampling"], m["name"])
            self.assertTrue(os.path.exists(os.path.join(ROOT, "config", "manifests", m["name"] + ".tsv")))

    def test_new_families(self):
        import run_job as J
        self.assertEqual(J.family_of("mistral-small-24b-it"), "mistral")
        self.assertEqual(J.family_of("phi-4-14b"), "phi")
        self.assertEqual(J.family_of("gemma-4-12b-it"), "gemma")
        self.assertEqual(J.family_of("qwen3.8-27b-fp8"), "qwen")
        self.assertEqual(J.TAILS["mistral"], "[/INST]")
        self.assertEqual(J.TAILS["phi"], "<|im_start|>assistant<|im_sep|>")

    def test_unknown_model_refused(self):
        import run_job as J
        with self.assertRaises(ValueError):
            J.family_of("llama-3-8b")

    def test_phi_not_greedy(self):
        self.assertEqual(SETTINGS["sampling"]["phi-4-14b"]["temperature"], 0.3)
        self.assertEqual(SETTINGS["sampling"]["mistral-small-24b-it"]["temperature"], 0.15)


if __name__ == "__main__":
    unittest.main()
