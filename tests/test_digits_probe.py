#!/usr/bin/env python3
"""Offline checks for scripts/run_digits_probe.py (no network, no model).

    python3 tests/test_digits_probe.py
"""
import json
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import make_fixtures as mf          # noqa: E402
import render_prompt as rp          # noqa: E402
import render_prompt_s2 as rp2      # noqa: E402
import run_digits_probe as D        # noqa: E402

SETTINGS = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
MM, MC = SETTINGS["max_messages"]["value"], SETTINGS["max_message_chars"]["value"]


def s2(role, limit):
    return rp2.render(role, 100, limit, [], 1, (75, 95), (105, 125), MM, MC)


def item(cond, role="seller", b=81):
    return {"condition": cond, "role": role, "base_limit": b}


class Prompts(unittest.TestCase):
    def test_d_conditions_are_study2_word_for_word(self):
        for role, b in (("seller", 81), ("buyer", 124)):
            self.assertEqual(D.render_item(item("D2", role, b), SETTINGS), s2(role, b))
            self.assertEqual(D.render_item(item("D5", role, b), SETTINGS), s2(role, b * 10**3))
            self.assertEqual(D.render_item(item("D8", role, b), SETTINGS), s2(role, b * 10**6))

    def test_only_the_intended_spans_change(self):
        for role, b in (("seller", 81), ("buyer", 124)):
            base = s2(role, b)
            for c in D.CONDITIONS:
                m = D.render_item(item(c, role, b), SETTINGS)
                text = D.CONDITIONS[c][0](b)
                self.assertEqual(m[1]["content"], base[1]["content"].replace(rp.money(b) + ".", text + ".", 1))
                self.assertIn(f"is {text}.", m[1]["content"])
                unit = D.CONDITIONS[c][2]
                if unit is None:
                    self.assertEqual(m[0]["content"], base[0]["content"])
                else:
                    self.assertIn(f"a whole number of {unit} for an OFFER", m[0]["content"])
                    self.assertIn(f"plain whole number of {unit}:", m[0]["content"])
                    # Removing the unit phrases gives back the Study 2 system message.
                    back = m[0]["content"].replace(f"of {unit} for", "of dollars for") \
                                          .replace(f"whole number of {unit}:", "whole number:")
                    self.assertEqual(back, base[0]["content"])

    def test_same_digits_in_stakes_only_conditions(self):
        digits = lambda msgs: re.findall(r"\d+", msgs[1]["content"].split("Your private")[1].split(".")[0])  # noqa
        self.assertEqual(digits(D.render_item(item("U8"), SETTINGS)), ["81"])
        self.assertEqual(digits(D.render_item(item("U5"), SETTINGS)), ["81"])
        self.assertEqual(digits(D.render_item(item("D2"), SETTINGS)), ["81"])
        self.assertEqual(D.render_item(item("C8"), SETTINGS)[1]["content"].count("81,000,000 dollars"), 1)
        user = lambda c: D.render_item(item(c), SETTINGS)[1]["content"]  # noqa: E731
        self.assertIn("Your private floor is $81. The buyer", user("S2"))
        self.assertIn("Your private floor is $81,000,000. The buyer", user("S8"))
        self.assertIn("Your private floor is $81000000. The buyer", user("SN"))
        self.assertIn("Your private floor is $81 million. The buyer", user("SM"))
        self.assertIn("Your private floor is 81.00 dollars. The buyer", user("X2"))
        for c in ("S2", "SN", "S8", "SM", "X2"):     # replies stay in plain whole dollars
            self.assertEqual(D.render_item(item(c), SETTINGS)[0]["content"], s2("seller", 81)[0]["content"])

    def test_per100_and_round_tens(self):
        import digits_figures as F
        r = lambda price, units, base=81: {"price": price, "limit_units": units, "base_limit": base}  # noqa: E731
        self.assertEqual(F.per100(r(120_000_000, 81_000_000)), 120)
        self.assertEqual(F.per100(r(12000, 8100)), 120)              # cents
        self.assertEqual(F.per100(r(120, 81)), 120)                  # millions or dollars
        self.assertTrue(F.round_ten(r(120_000_000, 81_000_000)))
        self.assertFalse(F.round_ten(r(95_000_000, 81_000_000)))
        self.assertFalse(F.round_ten(r(115, 81)))

    def test_condition_sets(self):
        for name in (D.FIRST_SET, D.DOLLAR_SET):
            self.assertTrue(all(c in D.CONDITIONS for c in name.split(",")))
        self.assertEqual(D.FIRST_SET.split(","), ["D2", "D5", "D8", "C8", "W8", "U5", "U8", "K2"])


class Design(unittest.TestCase):
    def test_counts_ids_and_common_seeds(self):
        meta, items = D.build_items(30, 4, list(D.CONDITIONS))
        self.assertEqual(len(items), 30 * 2 * len(D.CONDITIONS) * 4)
        self.assertEqual(len({i["item_id"] for i in items}), len(items))
        by = {}
        for i in items:
            by.setdefault((i["fixture"], i["role"], i["sample"]), set()).add(i["seed"])
        self.assertTrue(all(len(v) == 1 for v in by.values()), "seed must not depend on the condition")
        self.assertEqual(len({i["seed"] for i in items}), 30 * 2 * 4)
        self.assertEqual(sorted(i["run_order"] for i in items), list(range(1, len(items) + 1)))

    def test_fixtures_are_fresh(self):
        new = [(f["f_pct"], f["c_pct"]) for f in mf.build(D.PHRASE)["fixtures"]]
        real = [(f["f_pct"], f["c_pct"]) for f in json.load(open(os.path.join(ROOT, "config", "fixtures.json")))["fixtures"]]
        old = [(f["f_pct"], f["c_pct"]) for f in mf.build("scale-invariance/fixtures/explore-v1")["fixtures"]]
        self.assertNotEqual(new, real)
        self.assertNotEqual(new, old)

    def test_limit_units_and_gap(self):
        _, items = D.build_items(1, 1, list(D.CONDITIONS))
        for i in items:
            per = {"D2": 1, "D5": 10**3, "D8": 10**6, "C8": 10**6, "W8": 10**6, "U5": 1, "U8": 1, "K2": 100,
                   "S2": 1, "SN": 10**6, "S8": 10**6, "SM": 10**6, "X2": 1}
            self.assertEqual(i["limit_units"], i["base_limit"] * per[i["condition"]])
        self.assertAlmostEqual(D.gap("seller", 95, 81), 95 / 81 - 1)
        self.assertAlmostEqual(D.gap("buyer", 90, 120), 0.25)
        self.assertAlmostEqual(D.gap("seller", 9500, 8100), 95 / 81 - 1)    # K2 in cents

    def test_score(self):
        it = {"item_id": "Y01-S-U8-s1", "fixture": "Y01", "role": "seller", "condition": "U8", "sample": 1,
              "base_limit": 81, "limit_text": "81 million dollars", "limit_units": 81,
              "reply_unit": "millions of dollars", "stakes_dollars": 81 * 10**6, "seed": 1, "run_order": 1}
        ok = {"kind": "model_output", "call_id": "x", "finish_reason": "stop", "reasoning": None,
              "content": json.dumps({"action": "OFFER", "price": 95, "message": "hi"})}
        r = D.score(it, ok, MC)
        self.assertEqual((r["action"], r["price"], r["errors"]), ("OFFER", 95, []))
        self.assertAlmostEqual(r["gap"], 95 / 81 - 1)
        bad = {**ok, "content": json.dumps({"action": "ACCEPT", "price": None, "message": "x"})}
        self.assertEqual(D.score(it, bad, MC)["errors"], ["ACCEPT_WITHOUT_OFFER"])
        self.assertEqual(D.score(it, {"kind": "infrastructure", "call_id": "x"}, MC)["errors"], ["INFRASTRUCTURE"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
