#!/usr/bin/env python3
"""Offline checks for Study 3: scripts/study3.py, run_study3.py, analyze_study3.py, render_prompt_s3.py,
study3_clamp_values.py. No network beyond 127.0.0.1, no Keychain, no money. About two minutes.

    python3 tests/test_study3.py
"""
import glob
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from fractions import Fraction
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_study3 as AS          # noqa: E402
import make_fixtures as mf           # noqa: E402
import render_prompt_s2 as rp2       # noqa: E402
import render_prompt_s3 as rp3       # noqa: E402
import run_addendum_c as rac         # noqa: E402
import run_records as rr             # noqa: E402
import run_study3 as R               # noqa: E402
import study3 as S                   # noqa: E402
import study3_clamp_values as CV     # noqa: E402

API = list(S.API_ARMS)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def run(argv):
    buf = io.StringIO()
    rc = R.main(argv, stdout=buf, sleep=lambda s: None)
    return rc, buf.getvalue()


class Prompts(unittest.TestCase):
    def test_a1_adds_exactly_the_appraisal_line(self):
        for role, limit, V in (("seller", 83, 100), ("buyer", 118_000_000, 100_000_000)):
            hist = [("seller" if role == "buyer" else "buyer", 5 * V // 4, "x")] if role == "buyer" else []
            n = len(hist) + 1
            a0 = rp2.render(role, V, limit, hist, n, (75, 95), (105, 125), 16, 500)
            a1 = rp3.render(role, V, limit, hist, n, (75, 95), (105, 125), 16, 500)
            self.assertEqual(a0[1], a1[1])
            l0, l1 = a0[0]["content"].split("\n"), a1[0]["content"].split("\n")
            extra = [l for l in l1 if l not in l0]
            self.assertEqual(extra, [f"- An independent appraisal values the lot at {V} dollars. Both sides know this."])
            self.assertEqual([l for l in l1 if l not in extra], l0)
            self.assertNotIn("drawn at random", a1[0]["content"])        # no ranges in A1


class Design(unittest.TestCase):
    def test_fixtures(self):
        g = S.fixtures()
        self.assertEqual(len(g), 40)
        self.assertEqual(g[0]["id"], "G01")
        first30 = mf.build(S.FIXTURE_PHRASE)["fixtures"]
        self.assertEqual([(f["f_pct"], f["c_pct"]) for f in g[:30]], [(f["f_pct"], f["c_pct"]) for f in first30])
        for f in g:
            self.assertTrue(75 <= f["f_pct"] <= 95 and 105 <= f["c_pct"] <= 125)
        self.assertEqual(len(S.fixtures(S.PILOT_PHRASE, S.N_PILOT, S.PILOT_PREFIX)), 5)

    def test_games(self):
        s1, _ = S.build_games(API, session=1)
        s2, _ = S.build_games(API, session=2)
        p, _ = S.build_games(API, pilot=True)
        self.assertEqual((len(s1), len(s2), len(p)), (4800, 4800, 150))
        ids = [g["game_id"] for g in s1 + s2]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual({g["sample"] for g in s1}, {1, 2})
        self.assertEqual({g["sample"] for g in s2}, {3, 4})
        self.assertEqual([g["game_id"] for g in S.build_games(API, session=1)[0]], [g["game_id"] for g in s1])
        v, _ = S.build_games(["gemma-4-31b-it"], session=1)
        self.assertEqual(len(v), 1600)
        self.assertEqual({g["backend"] for g in v}, {"vllm"})
        self.assertEqual(sorted({g["cell"] for g in s1}), sorted(S.CELLS))

    def test_seed_ignores_scale_and_cell(self):
        a = S.call_seed("G01", "seller", 2, 5, 1)
        self.assertEqual(a, S.call_seed("G01", "seller", 2, 5, 1))
        self.assertNotEqual(a, S.call_seed("G01", "seller", 3, 5, 1))
        self.assertLess(a, 2 ** 31)

    def test_parse_cell(self):
        self.assertEqual(S.parse_cell("A1-x1e5"), ("A1", "x1e5", None))
        self.assertEqual(S.parse_cell("K-x1e8-o1e2"), ("K", "x1e8", "x1e2"))
        for bad in ("A2-x1e2", "K-x1e5-o1e2", "K-x1e2-o1e5", "A0"):
            with self.assertRaises(ValueError):
                S.parse_cell(bad)


class Clamp(unittest.TestCase):
    def setUp(self):
        self.v = S.load_clamp_values()

    def test_values_file_is_current_and_usable(self):
        self.assertEqual(CV.main(["--check"]), 0)
        problems, _ = CV.feasibility(self.v)
        self.assertEqual(problems, [])

    def test_prices_scale_exactly(self):
        for arm in S.ARMS:
            for fx in S.fixtures():
                for o in S.OPENERS:
                    for src in ("o1e2", "o1e8"):
                        lo = S.clamp_prices(S.make_game(arm, fx, f"K-x1e2-{src}", o, 1), self.v)
                        hi = S.clamp_prices(S.make_game(arm, fx, f"K-x1e8-{src}", o, 1), self.v)
                        self.assertEqual(hi, (lo[0] * 10 ** 6, lo[1] * 10 ** 6))

    def test_prefix_and_play(self):
        fx = S.fixtures()[0]
        g = {**S.make_game("gpt-6-luna", fx, "K-x1e8-o1e2", "buyer", 1), "run_order": 1}
        pre = S.clamp_prefix(g, self.v)
        self.assertEqual([p[0] for p in pre], ["buyer", "seller"])
        self.assertEqual(pre[0][2], f"I can pay {pre[0][1]} dollars for the lot.")
        sent = []

        def send(body, key, game, info):
            sent.append((key, body))
            text = body["messages"][1]["content"]
            self.assertIn(f"Message 1, you: OFFER at {pre[0][1]} dollars", text)
            self.assertIn(f"Message 2, the seller: OFFER at {pre[1][1]} dollars", text)
            return {"kind": "model_output", "call_id": key + "/t1", "content": json.dumps(
                {"action": "ACCEPT", "price": None, "message": "ok"}), "finish_reason": "stop", "reasoning": None}

        rec = S.play(g, 1, send, self.v)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0][0].endswith("/a1/m3"))
        self.assertEqual(rec["terminal"], "AGREEMENT")
        self.assertEqual(rec["price"], pre[1][1])
        self.assertEqual(rec["scripted_msgs"], [1, 2])
        self.assertEqual(rec["offers"][:2], [[1, "buyer", pre[0][1]], [2, "seller", pre[1][1]]])
        self.assertEqual(sent[0][1]["seed"], S.call_seed(fx["id"], "buyer", 1, 3, 1))
        self.assertEqual(sent[0][1]["max_tokens"], 1024)

    def test_vllm_body_uses_the_shared_schema(self):
        b = S.vllm_body("gemma-4-31b-it", [{"role": "user", "content": "x"}], 7, S.vllm_settings())
        self.assertEqual(b["max_tokens"], 1024)
        self.assertEqual(b["response_format"]["json_schema"]["schema"], S.response_schema())
        self.assertEqual(b["temperature"], 1.0)
        api = rac.build_body("gpt-6-luna", [{"role": "user", "content": "x"}], 7)
        self.assertEqual(api["response_format"]["json_schema"]["schema"], b["response_format"]["json_schema"]["schema"])


class TrailingText(unittest.TestCase):
    """Decided 2026-10-14: text after the first JSON object is ignored and counted; the object is still
    checked in full by the unchanged validate_reply."""
    LUNA = ('{"action":"OFFER","price":98000000,"message":"I can make a substantial move to 98000000 dollars."} '
            '\n</|assistant|>\u0e0e\u0e32\u0e04\u0e21\n? Must exact json no trailing.\n\n'
            '{"action":"OFFER","price":98000000,"message":"I can make a substantial move to 98000000 dollars."}')

    def check(self, content, finish="stop", pending=True):
        c, flag = S.first_json_only(content)
        return S.ge.validate_reply(c, finish, pending, 500), flag

    def test_cuts_only_after_a_complete_object(self):
        (mv, err), flag = self.check(self.LUNA)
        self.assertEqual((err, flag), ([], True))
        self.assertEqual((mv["action"], mv["price"]), ("OFFER", 98000000))
        clean = '{"action": "ACCEPT", "price": null, "message": "ok"}'
        for text in (clean, "\n " + clean + " \n"):
            self.assertEqual(S.first_json_only(text), (text, False))     # nothing to ignore: untouched
        self.assertEqual(self.check(clean + " Thanks!")[1], True)

    def test_the_move_itself_is_still_checked(self):
        bad = {'{"action": "OFFER", "price": 10} junk': "MISSING_FIELD:message",
               '{"action": "ACCEPT", "price": 10, "message": "x"} junk': "PRICE_MUST_BE_NULL",
               '{"action": "OFFER", "price": 1, "price": 2, "message": "x"} junk': "DUPLICATE_KEY",
               '{"action": "OFFER", "price": NaN, "message": "x"} junk': "NONFINITE_NUMBER",
               '{"action": "OFFER", "price": 10, "message": "' + "x" * 501 + '"} junk': "MESSAGE_TOO_LONG",
               'Sure! {"action": "OFFER", "price": 10, "message": "x"}': "NOT_JSON_OBJECT",
               '{"action": "OFFER", "price": 10, "mess': "NOT_JSON"}
        for text, why in bad.items():
            (mv, err), _ = self.check(text)
            self.assertIsNone(mv, text)
            self.assertIn(why, err, text)
        (mv, err), _ = self.check(self.LUNA, finish="length")
        self.assertEqual((mv, err), (None, ["TRUNCATED"]))

    def test_game_continues_and_counts(self):
        fx = S.fixtures()[0]
        g = {**S.make_game("gpt-6-luna", fx, "A0-x1e8", "seller", 1), "run_order": 1}
        moves = iter([self.LUNA, '{"action": "ACCEPT", "price": null, "message": "ok"}'])

        def send(body, key, game, info):
            return {"kind": "model_output", "call_id": key + "/t1", "content": next(moves),
                    "finish_reason": "stop", "reasoning": None}
        rec = S.play(g, 1, send, S.load_clamp_values())
        self.assertEqual((rec["terminal"], rec["price"]), ("AGREEMENT", 98000000))
        self.assertEqual(rec["trailing_text_calls"], 1)
        self.assertEqual([c["trailing_text_ignored"] for c in rec["calls"]], [True, False])


class Runs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="si-s3-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_registered_session_needs_the_frozen_list(self):
        with patch.object(S, "check_frozen", return_value="not frozen (test)"):
            rc, out = run(["--backend", "api", "--session", "1", "--confirm-cost", "key-limit"])
        self.assertEqual(rc, 3)
        self.assertIn("not frozen (test)", out)

    def test_api_dry_run_sends_nothing(self):
        rc, out = run(["--backend", "api", "--pilot"])
        self.assertEqual(rc, 0)
        self.assertIn("Dry run: nothing sent", out)

    def test_api_pilot_and_tamper(self):
        rc, out = run(["--backend", "api", "--pilot", "--preview", "--out", self.tmp])
        self.assertEqual(rc, 0, out)
        self.assertIn("Replay audit: every game reproduces", out)
        run_dir = glob.glob(os.path.join(self.tmp, "s3-api-pilot-*"))[0]
        lines = read(os.path.join(run_dir, "games.jsonl")).splitlines()
        recs = [json.loads(l) for l in lines]
        self.assertGreater(sum(r["trailing_text_calls"] for r in recs), 0)   # the fake's " Thanks!" replies
        self.assertFalse([d for r in recs for d in r["diagnostics"] if "TEXT_AFTER_JSON" in d])
        rec = json.loads(lines[3])
        rec["price"] = (rec["price"] or 0) + 1
        lines[3] = rr.canonical(rec)
        write(os.path.join(run_dir, "games.jsonl"), "\n".join(lines) + "\n")
        problems, _ = S.audit(run_dir)
        self.assertTrue(problems)

    def test_vllm_pilot_preview(self):
        rc, out = run(["--backend", "vllm", "--model", "gemma-4-31b-it", "--pilot", "--preview", "--out", self.tmp,
                       "--slot", "7"])
        self.assertEqual(rc, 0, out)
        self.assertIn("pilot gate gemma-4-31b-it", out)
        self.assertIn("Replay audit: every game reproduces", out)

    def test_interrupt_then_resume(self):
        real, n = S.play, [0]

        def flaky(g, attempt, send, values, vs=None):
            n[0] += 1
            if n[0] == 40:
                raise S.Stop("simulated stop")
            return real(g, attempt, send, values, vs)
        with patch.object(R.S, "play", flaky):
            rc, out = run(["--backend", "api", "--session", "1", "--preview", "--out", self.tmp, "--games", "120",
                           "--workers", "2"])
        self.assertEqual(rc, 1, out)
        self.assertIn("Resume with", out)
        run_dir = glob.glob(os.path.join(self.tmp, "s3-api-s1-*"))[0]
        self.assertFalse(os.path.exists(os.path.join(run_dir, "run_summary.json")))
        self.assertEqual(S.audit(run_dir)[0], [])
        rc, out = run(["--backend", "api", "--resume", run_dir, "--preview"])
        self.assertEqual(rc, 0, out)
        self.assertTrue(os.path.exists(os.path.join(run_dir, "run_summary.json")))
        man = json.loads(read(os.path.join(run_dir, "run_manifest.json")))
        recs = [json.loads(l) for l in read(os.path.join(run_dir, "games.jsonl")).splitlines()]
        scored = {r["game_id"] for r in recs if r["terminal"] != "INFRASTRUCTURE_VOID"}
        self.assertEqual(scored, set(man["games_planned"]))


class Analysis(unittest.TestCase):
    """A fake API whose openers are not modest under the appraisal: P1 must find it; the fake concedes
    halfway, so the clamp's concession channel is ~0 and omega tracks the scripted midpoint."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="si-s3-analysis-")
        orig, lock = rac.fake_reply, threading.Lock()
        saved = dict(rac.FAKE_MODESTY)

        def planted(body, seen, lk):
            with lock:
                if "An independent appraisal" in body["messages"][0]["content"]:
                    rac.FAKE_MODESTY.update({k: (0.0, 0.0) for k in saved})
                try:
                    return orig(body, seen, lk)
                finally:
                    rac.FAKE_MODESTY.update(saved)
        with patch.object(rac, "fake_reply", planted):
            for s in ("1", "2"):
                rc, out = run(["--backend", "api", "--session", s, "--preview", "--out", cls.tmp])
                assert rc == 0, out
        cls.res = AS.analyze(sorted(glob.glob(os.path.join(cls.tmp, "s3-api-s*"))), flips=2000)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_label_and_family(self):
        self.assertTrue(self.res["label"].startswith("PREVIEW"))
        self.assertEqual(sorted(self.res["holm_family"]), sorted(API))
        self.assertEqual(self.res["per_model"]["gemma-4-31b-it"], {"status": "not run"})

    def test_planted_effects(self):
        for arm in API:
            v = self.res["per_model"][arm]
            self.assertLess(v["P1"]["h_A0_mean"], 0)
            self.assertGreater(v["P1"]["mean"], 0)
            self.assertTrue(v["P1"]["supported"], arm)
            self.assertLess(abs(v["P2"]["A1"]["mean"]), 0.05)
            self.assertLess(abs(v["P3"]["P3a_kappa"]["mean"]), 0.03)
            self.assertLess(abs(v["P3"]["P3b_omega_minus_delta_m"]["mean"]), 0.05)
            self.assertEqual(v["games_scored"] + len(v["games_missing"]), 3200)
            s7 = v["secondary"]["S7_session_consistency"]
            self.assertIn("P2_d_A0", s7)
        gem = self.res["per_model"]["gemini-3.5-flash-lite"]
        self.assertLess(gem["P2"]["A0"]["mean"], self.res["per_model"]["gpt-6-luna"]["P2"]["A0"]["mean"])

    def test_scripted_midpoint_shift_matches_the_values_file(self):
        v = S.load_clamp_values()
        fx = S.fixtures()[0]
        g_lo = S.make_game("gemma-4-31b-it", fx, "K-x1e8-o1e2", "seller", 1)
        g_hi = S.make_game("gemma-4-31b-it", fx, "K-x1e8-o1e8", "seller", 1)
        dm = AS.scripted_midpoint(g_hi, v) - AS.scripted_midpoint(g_lo, v)
        vals = v["values"]["gemma-4-31b-it"]["seller"]
        want = (Fraction(vals["x1e8"]["x_seller"]) + Fraction(vals["x1e8"]["x_buyer"])
                - Fraction(vals["x1e2"]["x_seller"]) - Fraction(vals["x1e2"]["x_buyer"])) / 2
        self.assertLess(abs(float(dm - want)), 0.03)        # rounding to the V/100 grid only


if __name__ == "__main__":
    unittest.main(verbosity=1)
