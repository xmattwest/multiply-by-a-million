"""test_game_engine.py: offline tests for scripts/game_engine.py.

Plays whole games against scripted "models" (no server), checks the strict
reply rules against tests/reply_cases.jsonl, and checks requests, seeds and
scoring against the pre-registration and data plan.

Usage: python3 tests/test_game_engine.py      (GE_SCRIPT overrides the engine)
Standard library only, no network.
"""

import hashlib
import importlib.util
import json
import os
import re
import sys
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SCRIPT = os.environ.get("GE_SCRIPT", os.path.join(ROOT, "scripts", "game_engine.py"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
spec = importlib.util.spec_from_file_location("game_engine", SCRIPT)
ge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ge)


def slurp(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


FX = json.loads(slurp(os.path.join(ROOT, "config", "fixtures.json")))
ST = json.loads(slurp(os.path.join(ROOT, "prereg", "settings.json")))
PINS = json.loads(slurp(os.path.join(ROOT, "config", "pins.json")))
MODELS = [m["name"] for m in PINS["models"]]
MAXC = ST["max_message_chars"]["value"]
GAMES = {g["game_id"]: g for g in FX["games"]}
G = GAMES["F12-x1e5-S"]          # seller opens; F 88000, C 107000


def move(action, price=None, message="ok"):
    return {"kind": "model_output", "finish_reason": "stop", "reasoning": None,
            "content": json.dumps({"action": action, "price": price, "message": message})}


class Script:
    """A fake model: hands out scripted replies and records every request."""

    def __init__(self, replies):
        self.replies, self.bodies, self.infos = list(replies), [], []

    def __call__(self, body, info):
        self.bodies.append(body)
        self.infos.append(info)
        r = dict(self.replies.pop(0))
        r["call_id"] = f"c{len(self.bodies)}"
        return r


def play(replies, game=G, model=MODELS[0], attempt=1):
    s = Script(replies)
    rec = ge.play_game(game, model, ST, FX["f_pct_range"], FX["c_pct_range"], s, attempt=attempt)
    return rec, s


class Replies(unittest.TestCase):
    def test_shared_cases(self):
        cases = [json.loads(l) for l in slurp(os.path.join(HERE, "reply_cases.jsonl")).splitlines() if l.strip()]
        self.assertGreaterEqual(len(cases), 30)
        for c in cases:
            content = c["content"]
            if isinstance(content, str):
                content = re.sub(r"@@(\d+)@@", lambda m: "x" * int(m.group(1)), content)
            got, errors = ge.validate_reply(content, c.get("finish_reason", "stop"), c["pending"], MAXC)
            if c.get("ok"):
                self.assertEqual(errors, [], c["why"])
                self.assertIsNotNone(got, c["why"])
            else:
                self.assertIsNone(got, c["why"])
                self.assertEqual(errors, c["errors"], c["why"])

    def test_limit_in_cases_matches_settings(self):
        self.assertEqual(MAXC, 500)


class Requests(unittest.TestCase):
    def test_every_sampling_field_sent_with_the_registered_value(self):
        for model in MODELS:
            body = ge.build_request(model, [{"role": "user", "content": "x"}], ST, 7)
            samp = ST["sampling"][model]
            for k in ("temperature", "top_p", "top_k", "min_p"):
                self.assertEqual(body[k], samp[k], (model, k))
            self.assertEqual(body["presence_penalty"], 0)
            self.assertEqual(body["frequency_penalty"], 0)
            self.assertEqual(body["repetition_penalty"], 1.0)
            self.assertEqual(body["max_tokens"], ST["max_tokens"]["value"])
            self.assertEqual(body["seed"], 7)
            self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False)
            self.assertIs(body["stream"], False)

    def test_schema(self):
        sch = ge.build_request(MODELS[0], [], ST, 1)["response_format"]["json_schema"]
        self.assertTrue(sch["strict"])
        s = sch["schema"]
        self.assertEqual(s["required"], ["action", "price", "message"])
        self.assertIs(s["additionalProperties"], False)
        self.assertEqual(s["properties"]["message"]["maxLength"], MAXC)
        self.assertEqual(s["properties"]["action"]["enum"], ["OFFER", "ACCEPT", "WALKAWAY"])
        # xgrammar silently drops length limits on strings that also carry a
        # pattern or format (api_map 2.4), so the message must have neither.
        self.assertNotIn("pattern", s["properties"]["message"])
        self.assertNotIn("format", s["properties"]["message"])


def _conforms(obj, schema):
    """A tiny stdlib checker, just enough for the closed oneOf schemas here.
    Supports: object (properties, required, additionalProperties:false),
    string (pattern, maxLength), integer/number (minimum, exclusiveMinimum),
    null, enum, const, oneOf, anyOf. Returns True/False."""
    import re as _re
    if "oneOf" in schema:
        return sum(1 for s in schema["oneOf"] if _conforms(obj, s)) == 1
    if "anyOf" in schema:
        return any(_conforms(obj, s) for s in schema["anyOf"])
    t = schema.get("type")
    if "const" in schema and obj != schema["const"]:
        return False
    if "enum" in schema and obj not in schema["enum"]:
        return False
    if t == "null":
        return obj is None
    if t == "integer":
        if isinstance(obj, bool) or not isinstance(obj, int):
            return False
        return obj >= schema.get("minimum", obj)
    if t == "number":
        if isinstance(obj, bool) or not isinstance(obj, (int, float)):
            return False
        if "exclusiveMinimum" in schema and not obj > schema["exclusiveMinimum"]:
            return False
        return True
    if t == "string":
        if not isinstance(obj, str):
            return False
        if "pattern" in schema and not _re.search(schema["pattern"], obj):
            return False
        if "maxLength" in schema and len(obj) > schema["maxLength"]:
            return False
        return True
    if t == "object":
        if not isinstance(obj, dict):
            return False
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            if any(k not in props for k in obj):
                return False
        if any(k not in obj for k in schema.get("required", [])):
            return False
        return all(_conforms(obj[k], props[k]) for k in props if k in obj)
    return True


class StrictActionSchema(unittest.TestCase):
    """The opt-in action-dependent grammar (WP2a, Gemma 12B rescue)."""

    def test_default_build_request_unchanged(self):
        # The plain schema still has an anyOf integer/null price (no oneOf).
        s = ge.build_request(MODELS[0], [], ST, 1)["response_format"]["json_schema"]["schema"]
        self.assertIn("properties", s)
        self.assertIn("anyOf", s["properties"]["price"])
        self.assertNotIn("oneOf", s)

    def test_oneof_shape(self):
        s = ge.strict_action_schema(MAXC)
        self.assertEqual(len(s["oneOf"]), 2)
        offer, term = s["oneOf"]
        self.assertEqual(offer["properties"]["action"]["enum"], ["OFFER"])
        self.assertEqual(offer["properties"]["price"], {"type": "integer", "minimum": 1})
        self.assertEqual(term["properties"]["action"]["enum"], ["ACCEPT", "WALKAWAY"])
        self.assertEqual(term["properties"]["price"], {"type": "null"})
        for v in s["oneOf"]:
            self.assertIs(v["additionalProperties"], False)
            self.assertEqual(v["required"], ["action", "price", "message"])
            self.assertNotIn("pattern", v["properties"]["message"])  # xgrammar length rule

    def test_parity_with_validator_on_shared_cases(self):
        """Every shared reply case that the validator accepts as a well-formed
        move must conform to the strict grammar; and the two cases the
        validator rejects purely for a price/action mismatch
        (ACCEPT/WALKAWAY naming a price, OFFER with null price) must NOT
        conform. This is why the strict grammar is a safe subset."""
        s = ge.strict_action_schema(MAXC)
        cases = [json.loads(l) for l in slurp(os.path.join(HERE, "reply_cases.jsonl")).splitlines() if l.strip()]
        for c in cases:
            content = c["content"]
            if not isinstance(content, str):
                continue
            txt = content.strip(" \t\n\r")
            try:
                obj = json.loads(txt)
            except ValueError:
                continue
            if not isinstance(obj, dict) or set(obj) != set(ge.KEYS):
                continue
            move, errors = ge.validate_reply(content, c.get("finish_reason", "stop"), c["pending"], MAXC)
            conforms = _conforms(obj, s)
            if move is not None:
                self.assertTrue(conforms, ("valid move must conform", c["why"], obj))
            if errors in (["PRICE_MUST_BE_NULL"], ["PRICE_REQUIRED"]):
                self.assertFalse(conforms, ("price/action mismatch must be blocked", c["why"], obj))

    def test_newgame_strict_composes_notation_price(self):
        import importlib.util as _u
        spec2 = _u.spec_from_file_location("newgame", os.path.join(ROOT, "scripts", "newgame.py"))
        ng = _u.module_from_spec(spec2)
        spec2.loader.exec_module(ng)
        # default (off): notation price on the plain schema
        off = ng.build_request(MODELS[0], [], ST, 1, notation="biz")["response_format"]["json_schema"]["schema"]
        self.assertIn("anyOf", off["properties"]["price"])
        # strict on: oneOf with the biz pattern on the OFFER branch, null on the other
        on = ng.build_request(MODELS[0], [], ST, 1, notation="biz",
                              strict_action=True)["response_format"]["json_schema"]["schema"]
        self.assertEqual(len(on["oneOf"]), 2)
        self.assertEqual(on["oneOf"][0]["properties"]["price"]["pattern"], ng.BIZ_RE)
        self.assertEqual(on["oneOf"][1]["properties"]["price"], {"type": "null"})
        # a biz OFFER string conforms; an ACCEPT with a price does not
        self.assertTrue(_conforms({"action": "OFFER", "price": "$1,234", "message": "x"}, on))
        self.assertFalse(_conforms({"action": "ACCEPT", "price": "$1,234", "message": "x"}, on))
        self.assertTrue(_conforms({"action": "ACCEPT", "price": None, "message": "x"}, on))


class Seeds(unittest.TestCase):
    def test_seed_rule(self):
        want = int(hashlib.sha256(b"scale-invariance/call-seed/v1|F12|seller|3|1").hexdigest()[:8], 16)
        self.assertEqual(ge.call_seed("F12", "seller", 3, 1), want)

    def test_same_seeds_at_every_scale(self):
        seeds = {}
        for gid in ("F12-x1e2-S", "F12-x1e5-S", "F12-x1e8-S"):
            _, s = play([move("OFFER", GAMES[gid]["V"]), move("WALKAWAY")], game=GAMES[gid])
            seeds[gid] = [b["seed"] for b in s.bodies]
        self.assertEqual(len({tuple(v) for v in seeds.values()}), 1)
        self.assertEqual(len(set(seeds["F12-x1e2-S"])), 2)

    def test_seed_changes_with_attempt(self):
        _, a = play([move("WALKAWAY")], attempt=1)
        _, b = play([move("WALKAWAY")], attempt=2)
        self.assertNotEqual(a.bodies[0]["seed"], b.bodies[0]["seed"])


class Games(unittest.TestCase):
    def test_agreement(self):
        rec, s = play([move("OFFER", 110000), move("OFFER", 95000), move("ACCEPT")])
        self.assertEqual(rec["terminal"], "AGREEMENT")
        self.assertEqual(rec["price"], 95000)           # the buyer's offer, accepted by the seller
        self.assertEqual(rec["ended_by"], "seller")
        self.assertEqual(rec["n_messages"], 3)
        self.assertEqual(rec["seller_share"], {"num": 7, "den": 19, "float": 7 / 19})   # (95-88)/(107-88)
        self.assertEqual(Fraction(rec["u_seller"]["num"], rec["u_seller"]["den"]) +
                         Fraction(rec["u_buyer"]["num"], rec["u_buyer"]["den"]), 1)
        self.assertEqual([i["role"] for i in s.infos], ["seller", "buyer", "seller"])
        self.assertEqual(rec["offers"], [[1, "seller", 110000], [2, "buyer", 95000]])
        self.assertEqual(rec["call_ids"], ["c1", "c2", "c3"])
        self.assertFalse(rec["violation"])

    def test_each_side_sees_only_its_own_limit(self):
        _, s = play([move("OFFER", 110000), move("OFFER", 95000), move("ACCEPT")])
        for body, info in zip(s.bodies, s.infos):
            text = json.dumps(body["messages"])
            hidden = G["C"] if info["role"] == "seller" else G["F"]
            own = G["F"] if info["role"] == "seller" else G["C"]
            self.assertNotRegex(text, rf"(?<![0-9]){hidden}(?![0-9])")
            self.assertIn(f"{own} dollars", text)

    def test_history_reaches_the_next_move(self):
        _, s = play([move("OFFER", 110000, "start"), move("WALKAWAY")])
        self.assertIn("110000 dollars", s.bodies[1]["messages"][1]["content"])
        self.assertIn('\\"start\\"', json.dumps(s.bodies[1]["messages"][1]["content"]))

    def test_walkaway(self):
        rec, _ = play([move("OFFER", 120000), move("WALKAWAY")])
        self.assertEqual((rec["terminal"], rec["ended_by"], rec["seller_share"]), ("WALKAWAY", "buyer", None))
        self.assertEqual(rec["u_seller"]["float"], 0.0)

    def test_timeout_after_sixteen(self):
        offers = [move("OFFER", 100000 + (i % 2) * 1000) for i in range(16)]
        rec, s = play(offers)
        self.assertEqual(rec["terminal"], "TIMEOUT")
        self.assertEqual(rec["n_messages"], 16)
        self.assertEqual(len(s.bodies), 16)
        self.assertEqual(rec["u_buyer"]["float"], 0.0)

    def test_accept_on_the_last_message(self):
        rec, _ = play([move("OFFER", 100000)] * 15 + [move("ACCEPT")])
        self.assertEqual((rec["terminal"], rec["n_messages"]), ("AGREEMENT", 16))

    def test_opening_accept_is_a_forfeit(self):
        rec, _ = play([move("ACCEPT")])
        self.assertEqual((rec["terminal"], rec["ended_by"]), ("PROTOCOL_FORFEIT", "seller"))
        self.assertEqual(rec["u_seller"]["float"], -1.0)
        self.assertEqual(rec["u_buyer"]["float"], 0.0)
        self.assertIn("ACCEPT_WITHOUT_OFFER", rec["diagnostics"][0])

    def test_invalid_reply_is_a_forfeit_for_that_side(self):
        bad = {"kind": "model_output", "finish_reason": "stop", "reasoning": None, "content": "OFFER 5"}
        rec, _ = play([move("OFFER", 110000), bad])
        self.assertEqual((rec["terminal"], rec["ended_by"]), ("PROTOCOL_FORFEIT", "buyer"))
        self.assertEqual(rec["u_buyer"]["float"], -1.0)

    def test_infrastructure_failure_is_void_and_unscored(self):
        rec, _ = play([move("OFFER", 110000), {"kind": "infrastructure"}])
        self.assertEqual(rec["terminal"], "INFRASTRUCTURE_VOID")
        self.assertNotIn("seller_share", rec)
        self.assertNotIn("u_seller", rec)

    def test_reasoning_is_flagged(self):
        r = move("WALKAWAY")
        r["reasoning"] = "thinking..."
        rec, _ = play([r])
        self.assertIn("REASONING_PRESENT:1", rec["diagnostics"])

    def test_accepting_below_your_floor_is_a_violation(self):
        rec, _ = play([move("OFFER", 110000), move("OFFER", 80000), move("ACCEPT")])
        self.assertTrue(rec["violation"])
        self.assertTrue(rec["outside_zopa"])
        self.assertLess(rec["seller_share"]["float"], 0)   # not clamped (ZOPA 2.4)

    def test_buyer_opens(self):
        g = GAMES["F12-x1e5-B"]
        rec, s = play([move("OFFER", 90000), move("ACCEPT")], game=g)
        self.assertEqual([i["role"] for i in s.infos], ["buyer", "seller"])
        self.assertEqual((rec["price"], rec["ended_by"]), (90000, "seller"))
        self.assertEqual(rec["seller_share"]["num"], 2)

    def test_record_copies_fixture_fields(self):
        rec, _ = play([move("WALKAWAY")])
        for k in ("fixture", "scale", "V", "F", "C", "opener", "block", "run_order"):
            self.assertEqual(rec[k], G[k], k)


if __name__ == "__main__":
    unittest.main(verbosity=1)
