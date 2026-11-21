#!/usr/bin/env python3
"""Offline checks for Addendum C: scripts/run_addendum_c.py and analyze_addendum_c.py.
No network beyond 127.0.0.1, no Keychain, no money.

    python3 tests/test_addendum_c.py
"""
import glob
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze as A                  # noqa: E402
import analyze_addendum_c as AC      # noqa: E402
import game_engine as ge             # noqa: E402
import openrouter_client as oc       # noqa: E402
import render_prompt_s2 as rp2       # noqa: E402
import run_addendum_b as RB          # noqa: E402
import run_addendum_c as R           # noqa: E402
import run_records as rr             # noqa: E402

FAKE_KEY = "sk-or-v1-TESTKEY-0123456789abcdef"


_UNSET = object()


def envelope(move=None, content=_UNSET, finish="stop", cost=0.0001, rtok=0, refusal=None, arm="gpt-6-luna"):
    a = R.ARMS[arm]
    if content is _UNSET:
        content = json.dumps(move or {"action": "OFFER", "price": 100, "message": "hi"})
    return json.dumps({
        "id": "gen-1", "provider": a["vendor"], "model": a["model"], "object": "chat.completion", "created": 1,
        "system_fingerprint": "fp_x",
        "choices": [{"index": 0, "finish_reason": finish, "native_finish_reason": finish,
                     "message": {"role": "assistant", "content": content, "refusal": refusal}}],
        "usage": {"prompt_tokens": 700, "completion_tokens": 40, "total_tokens": 740, "cost": cost,
                  "completion_tokens_details": {"reasoning_tokens": rtok}},
        "openrouter_metadata": {"endpoints": {"available": [{"provider": a["vendor"], "model": a["canonical"],
                                                             "selected": True}]}}}).encode()


def game(arm="gpt-6-luna", fixture="F01", scale="x1e8", V=100_000_000, opener="seller", sample=1, f=88, c=118):
    return {"game_id": f"{arm}|{fixture}-{scale}-{opener[0].upper()}-s{sample}", "arm": arm, "fixture": fixture,
            "scale": scale, "V": V, "opener": opener, "sample": sample, "F": f * V // 100, "C": c * V // 100,
            "run_order": 1}


class KeyedClient(oc.Client):
    def __init__(self, port, **kw):
        self._port = port
        super().__init__(R.ARMS, preview_port=None, key_reader=lambda: FAKE_KEY, **kw)

    def _conn(self, timeout):
        return http.client.HTTPConnection("127.0.0.1", self._port, timeout=timeout)


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="si-addc-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def writer(self, name="run"):
        w = RB.locked_writer(rr.RunWriter(os.path.join(self.tmp, name), {"schema": "test"}))
        self.addCleanup(lambda: [c.fh.close() for c in (w.calls, w.games, w.events)])
        return w

    def scripted(self, replies):
        """A transport that returns the scripted (status, bytes, err) in order and records what was sent."""
        sent, it = [], iter(replies)

        def t(b):
            sent.append(b)
            return next(it)
        return t, sent


# ------------------------------------------------------------------ design

class Design(unittest.TestCase):
    def test_arms_are_addendum_b_arms(self):
        self.assertIs(R.ARMS, RB.ARMS)
        self.assertEqual(sorted(R.ARMS), ["claude-haiku-5.5", "gemini-3.5-flash-lite", "gpt-6-luna"])

    def test_main_games(self):
        gs, meta = R.build_games(False, R.ARMS, R.MAIN_SAMPLES)
        self.assertEqual(len(gs), 3 * 30 * 3 * 2 * 4)
        self.assertEqual(len({g["game_id"] for g in gs}), len(gs))
        self.assertEqual(meta["fixtures"][0]["id"], "F01")
        self.assertEqual(sorted({g["scale"] for g in gs}), ["x1e2", "x1e5", "x1e8"])
        self.assertEqual(sorted(g["run_order"] for g in gs), list(range(1, len(gs) + 1)))
        self.assertGreater(len({g["arm"] for g in gs[:12]}), 1)            # models interleave in time

    def test_limits_match_study2_fixtures(self):
        fx = {f["id"]: f for f in json.load(open(os.path.join(ROOT, "config", "fixtures.json")))["fixtures"]}
        for g in R.build_games(False, R.ARMS, 1)[0]:
            f = fx[g["fixture"]]
            self.assertEqual((g["F"], g["C"]), (f["f_pct"] * g["V"] // 100, f["c_pct"] * g["V"] // 100))

    def test_pilot_uses_x_fixtures_two_scales(self):
        gs, meta = R.build_games(True, R.ARMS, R.PILOT_SAMPLES)
        self.assertEqual(len(gs), 3 * 5 * 2 * 2)
        self.assertTrue(all(g["fixture"].startswith("X") for g in gs))
        self.assertEqual(sorted({g["scale"] for g in gs}), ["x1e2", "x1e8"])

    def test_seeds(self):
        s = R.call_seed("F01", "seller", 1, 3, 1)
        allseeds = [R.call_seed(f"F{i:02d}", o, k, m, 1) for i in range(1, 31) for o in ("seller", "buyer")
                    for k in range(1, 5) for m in range(1, 17)]
        self.assertTrue(all(0 <= x < 2 ** 31 for x in allseeds))        # Google: TYPE_INT32 (amendment B1)
        label = "scale-invariance/addendum-c/v1|F01|seller|1|3|1"
        self.assertEqual(s, int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF)
        self.assertNotEqual(s, R.call_seed("F01", "seller", 2, 3, 1))     # samples differ
        self.assertNotEqual(s, R.call_seed("F01", "seller", 1, 4, 1))     # moves differ
        self.assertNotEqual(s, R.call_seed("F01", "seller", 1, 3, 2))     # replayed attempts differ
        self.assertNotEqual(s, ge.call_seed("F01", "seller", 3, 1) & 0x7FFFFFFF)   # not the engine's vLLM rule
        for _ in range(3):
            self.assertEqual(s, R.call_seed("F01", "seller", 1, 3, 1))   # deterministic, no scale input

    def test_worst_case_and_caps(self):
        gs, _ = R.build_games(False, R.ARMS, R.MAIN_SAMPLES)
        one, allr = R.worst_case(gs)
        self.assertAlmostEqual(allr, one * 9)
        self.assertEqual(R.PILOT_CAP, 1.0)
        self.assertGreater(R.worst_per_call(R.ARMS["gpt-6-luna"]), RB.worst_per_call(RB.ARMS["gpt-6-luna"]))


# ------------------------------------------------------------------ one game through the engine

class OneGame(TempDir):
    def play(self, replies, g=None, attempt=1):
        self.n = getattr(self, "n", 0) + 1
        w = self.writer(f"run{self.n}")
        t, sent = self.scripted(replies)
        cap = RB.SpendCap(10.0)
        call = R.recording_call(w, t, "run", R.ARMS, cap, sleep=lambda s: None)
        rec = R.play(g or game(), attempt, call, R.engine_settings())
        return rec, [json.loads(b) for b in sent], w, cap

    def test_agreement_and_body_rules(self):
        g = game(arm="claude-haiku-5.5")
        rec, bodies, _, _ = self.play([
            (200, envelope({"action": "OFFER", "price": 100_000_000, "message": "100M"}, arm="claude-haiku-5.5"), None),
            (200, envelope({"action": "OFFER", "price": 95_000_000, "message": "95M"}, arm="claude-haiku-5.5"), None),
            (200, envelope({"action": "ACCEPT", "price": None, "message": "ok"}, arm="claude-haiku-5.5"), None)], g)
        self.assertEqual((rec["terminal"], rec["price"], rec["ended_by"], rec["n_messages"]),
                         ("AGREEMENT", 95_000_000, "seller", 3))
        s = Fraction(95_000_000 - g["F"], g["C"] - g["F"])
        self.assertEqual(Fraction(rec["seller_share"]["num"], rec["seller_share"]["den"]), s)
        for i, b in enumerate(bodies, 1):
            oc.check_chat_body(rr.canonical(b).encode(), R.ARMS)          # passes the client allowlist
            self.assertNotIn("seed", b)                                   # Anthropic: no seed
            for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "chat_template_kwargs"):
                self.assertNotIn(k, b)                                    # the engine's vLLM fields never leak
            self.assertEqual(b["messages"], rp2.render(
                "seller" if i % 2 else "buyer", g["V"], g["F"] if i % 2 else g["C"],
                [("seller", 100_000_000, "100M"), ("buyer", 95_000_000, "95M")][:i - 1], i,
                (75, 95), (105, 125), 16, 500))                           # exactly the Study 2 prompt
        self.assertEqual(rec["schema"], "game-api/1")
        self.assertEqual([c["msg_no"] for c in rec["api"]], [1, 2, 3])
        self.assertAlmostEqual(rec["cost"], 0.0003)

    def test_seed_sent_where_supported_and_common_across_scales(self):
        seeds = {}
        for scale, V in (("x1e2", 100), ("x1e8", 100_000_000)):
            g = game(scale=scale, V=V)
            _, bodies, _, _ = self.play([(200, envelope({"action": "WALKAWAY", "price": None, "message": "no"}), None)], g)
            seeds[scale] = bodies[0]["seed"]
        self.assertEqual(seeds["x1e2"], seeds["x1e8"])
        self.assertEqual(seeds["x1e2"], R.call_seed("F01", "seller", 1, 1, 1))

    def test_invalid_reply_forfeits_and_is_not_reasked(self):
        rec, bodies, _, _ = self.play([(200, envelope(content='{"action": "OFFER", "price": 5, "message": "' +
                                                               "x" * 501 + '"}'), None)])
        self.assertEqual((rec["terminal"], rec["ended_by"], len(bodies)), ("PROTOCOL_FORFEIT", "seller", 1))
        self.assertIn("INVALID:1:MESSAGE_TOO_LONG", rec["diagnostics"])

    def test_refusal_forfeits_and_is_counted(self):
        rec, _, _, _ = self.play([(200, envelope(content=None, refusal="no"), None)])
        self.assertEqual((rec["terminal"], rec["refusals"]), ("PROTOCOL_FORFEIT", 1))

    def test_infrastructure_voids_the_attempt(self):
        rec, bodies, w, _ = self.play([(200, envelope({"action": "OFFER", "price": 100_000_000, "message": "a"}), None),
                                       (500, b"", None), (502, b"{}", None), (None, b"", "timeout")])
        self.assertEqual(rec["terminal"], "INFRASTRUCTURE_VOID")
        self.assertEqual(len({rr.canonical(b) for b in bodies[1:]}), 1)  # identical retries of message 2

    def test_rejected_request_stops(self):
        for status in (400, 402):
            with self.subTest(status=status), self.assertRaises(RB.RequestRejected):
                self.play([(status, json.dumps({"error": {"code": status}}).encode(), None)])

    def test_spend_cap_never_exceeded(self):
        w = self.writer()
        wpc = R.worst_per_call(R.ARMS["gpt-6-luna"])
        cap = RB.SpendCap(wpc * 2.5)
        t, sent = self.scripted([(200, envelope({"action": "OFFER", "price": 100_000_000 + i, "message": "a"},
                                                cost=wpc), None) for i in range(10)])
        call = R.recording_call(w, t, "run", R.ARMS, cap, sleep=lambda s: None)
        with self.assertRaises(RB.CapReached):
            R.play(game(), 1, call, R.engine_settings())
        self.assertEqual(len(sent), 2)
        self.assertLessEqual(cap.spent, cap.cap)


# ------------------------------------------------------------------ main()

class Main(TempDir):
    def run_main(self, args, factory=None):
        buf = io.StringIO()
        if factory is None and "--preview" not in args:
            def factory(arms, srv):
                raise AssertionError("unexpected real client: tests must stay offline")
        rc = R.main(args + ["--out", self.tmp], stdout=buf, client_factory=factory, sleep=lambda s: None)
        return rc, buf.getvalue()

    def test_dry_run_sends_nothing(self):
        for args, confirm in ((["--pilot"], "1.00"), ([], "key-limit")):
            rc, out = self.run_main(args)
            self.assertEqual(rc, 0)
            self.assertIn("Dry run: nothing sent", out)
            self.assertIn("--confirm-cost " + confirm, out)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_wrong_confirm_cost(self):
        for args in (["--pilot", "--confirm-cost", "0.01"], ["--confirm-cost", "15.00"]):
            rc, out = self.run_main(args)
            self.assertEqual((rc, "does not match" in out), (2, True))

    def test_registered_run_needs_frozen_inputs(self):
        if R.check_frozen() is None:
            self.skipTest("addendum already frozen in this checkout")

        def no_client(arms, srv):                     # never reach the Keychain or the network
            raise AssertionError("a client was built before the freeze check")
        rc, out = self.run_main(["--confirm-cost", "key-limit"], factory=no_client)
        self.assertEqual(rc, 3)
        self.assertIn("Not running:", out)

    def test_registered_run_uses_all_arms_and_k4(self):
        self.assertEqual(self.run_main(["--arms", "gpt-6-luna"])[0], 2)
        self.assertEqual(self.run_main(["--samples", "1"])[0], 2)

    def test_main_rejects_client_cap_override(self):
        rc, out = self.run_main(["--spend-cap", "15.00", "--confirm-cost", "key-limit"])
        self.assertEqual(rc, 2)
        self.assertIn("pilot-only", out)

    def test_main_requires_capped_key_with_credit(self):
        class Client:
            def key_status(self):
                return status

            def chat_transport(self):
                raise AssertionError("prompt transport built without a capped key and credit")

        for status in ((None, 0, None), (0, 0, 0), (10, 10, 0), (10, 0, None),
                       (float("inf"), 0, 10), (10, 0, float("nan"))):
            with self.subTest(status=status), patch.object(R, "check_frozen", return_value=None):
                rc, out = self.run_main(["--confirm-cost", "key-limit"], factory=lambda arms, srv: Client())
            self.assertEqual(rc, 3)
            self.assertIn("capped OpenRouter key with positive remaining credit", out)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_main_uses_key_cap_and_stops_on_402(self):
        sent = []

        class Client:
            def key_status(self):
                return 10.0, 0.6, 9.4

            def chat_transport(self):
                def transport(body):
                    b = json.loads(body)
                    sent.append(b)
                    if len(sent) > 1:
                        return 402, b'{"error": {"code": 402, "message": "key limit reached"}}', None
                    arm = next(n for n, a in R.ARMS.items() if a["model"] == b["model"])
                    return 200, envelope({"action": "WALKAWAY", "price": None, "message": "test"},
                                         arm=arm, cost=0.1), None
                return transport

        gs, meta = R.build_games(False, R.ARMS, R.MAIN_SAMPLES)
        # Even a reservation above both the old $15 cap and the key's available credit must reach
        # the provider; the provider's 402 is the stopping condition, not a local spending cap.
        with patch.object(R, "check_frozen", return_value=None), \
                patch.object(R, "build_games", return_value=(gs[:2], meta)), \
                patch.object(R, "worst_per_call", return_value=20.0):
            rc, out = self.run_main(["--confirm-cost", "key-limit", "--workers", "1"],
                                    factory=lambda arms, srv: Client())
        self.assertEqual(rc, 1, out)
        self.assertEqual(len(sent), 2)                    # 402 is not retried
        self.assertIn("HTTP 402", out)
        run_dir = glob.glob(os.path.join(self.tmp, "addC-*"))[0]
        manifest = json.load(open(os.path.join(run_dir, "run_manifest.json")))
        self.assertIsNone(manifest["spend_cap"])
        self.assertEqual(manifest["spend_policy"], "openrouter_key_cap_only")
        self.assertEqual(manifest["key_credit_at_start"], {"limit": 10.0, "usage": 0.6, "remaining": 9.4})
        summary = json.load(open(os.path.join(run_dir, "run_summary.json")))["counts"]
        self.assertEqual((summary["status"], summary["games_done"]), ("stopped", 1))
        self.assertAlmostEqual(summary["spent"], 0.1)

    def test_low_key_credit_stops_before_any_prompt(self):
        srv = R.fake_server(key_limit=0.50)
        try:
            rc, out = self.run_main(["--pilot", "--confirm-cost", "1.00"],
                                    factory=lambda arms, s: KeyedClient(srv.server_address[1]))
        finally:
            srv.shutdown(); srv.server_close()
        self.assertEqual(rc, 3)
        self.assertIn("below the cap", out)
        self.assertEqual(len(srv.auth_seen), 1)                           # only GET /api/v1/key

    def test_keyed_pilot_end_to_end_key_never_written(self):
        srv = R.fake_server()
        try:
            rc, out = self.run_main(["--pilot", "--confirm-cost", "1.00"],
                                    factory=lambda arms, s: KeyedClient(srv.server_address[1]))
        finally:
            srv.shutdown(); srv.server_close()
        self.assertEqual(rc, 0, out)
        self.assertTrue(all(h == "Bearer " + FAKE_KEY for h in srv.auth_seen))
        self.assertNotIn(FAKE_KEY, out)
        run_dir = glob.glob(os.path.join(self.tmp, "addC-pilot-*"))[0]
        for p in glob.glob(os.path.join(run_dir, "*")):
            self.assertNotIn(FAKE_KEY, open(p, encoding="utf-8").read(), p)
        for name in ("calls.jsonl", "games.jsonl", "events.jsonl"):
            self.assertTrue(rr.verify_chain(os.path.join(run_dir, name))[0])
        games = [json.loads(l) for l in open(os.path.join(run_dir, "games.jsonl"))]
        self.assertEqual(len({g["game_id"] for g in games}), 60)
        self.assertIn("pilot gate", out)
        self.assertNotIn("analyze_addendum_c", out)
        with self.assertRaises(ValueError):                               # a real pilot is never analysed
            AC.analyze(run_dir, flips=100)

    def test_void_games_are_replayed_with_a_new_attempt(self):
        g = game()
        calls = []

        def flaky(body, key, arm):
            calls.append((key, body["seed"]))
            if "/a1/" in key and key.endswith("/m2"):
                return {"kind": "infrastructure", "call_id": key + "/t3", "tries": 3, "problem": "x"}
            move = {"action": "ACCEPT", "price": None, "message": "ok"} if key.endswith("/m2") else \
                {"action": "OFFER", "price": 100_000_000, "message": "a"}
            return {"kind": "model_output", "call_id": key + "/t1", "tries": 1, "content": json.dumps(move),
                    "finish_reason": "stop", "reasoning": None, "refusal": None, "cost": 0.0}
        recs = []
        for attempt in range(1, R.MAX_ATTEMPTS + 1):
            r = R.play(g, attempt, flaky, R.engine_settings())
            recs.append(r)
            if r["terminal"] != "INFRASTRUCTURE_VOID":
                break
        self.assertEqual([r["terminal"] for r in recs], ["INFRASTRUCTURE_VOID", "AGREEMENT"])
        self.assertEqual(recs[1]["attempt"], 2)
        seeds = dict(calls)
        self.assertNotEqual(seeds[f"{g['game_id']}/a1/m1"], seeds[f"{g['game_id']}/a2/m1"])   # attempt in the seed
        self.assertEqual(seeds[f"{g['game_id']}/a2/m1"], R.call_seed("F01", "seller", 1, 1, 2))


# ------------------------------------------------------------------ analysis

def grec(arm, f, opener, scale, offers, terminal="AGREEMENT", F=80, C=120, sample=1, rtok=0, diag=()):
    """A minimal scored game record. offers = [(role, price), ...]; an agreement is at the last offer."""
    V = {"x1e2": 1, "x1e5": 1000, "x1e8": 1_000_000}[scale]
    F, C = F * V, C * V
    offs = [[i + 1, who, p * V] for i, (who, p) in enumerate(offers)]
    price = offs[-1][2] if terminal == "AGREEMENT" else None
    r = {"game_id": f"{arm}|{f}-{scale}-{opener[0].upper()}-s{sample}", "arm": arm, "fixture": f, "opener": opener,
         "scale": scale, "F": F, "C": C, "terminal": terminal, "price": price, "offers": offs,
         "n_messages": len(offs) + 1, "diagnostics": list(diag), "api": [], "refusals": 0,
         "reasoning_token_calls": rtok, "cost": 0.0}
    sc = ge.score({"F": F, "C": C}, terminal, price, "seller")
    r.update({"seller_share": ge.frac(sc["seller_share"]), "u_seller": ge.frac(sc["u_seller"]),
              "violation": sc["violation"], "outside_zopa": sc["outside_zopa"]})
    return r


class Analysis(TempDir):
    def test_d_and_e_by_hand(self):
        # F01, seller opens: x1: S 112, B 96 -> deal 104 (share 0.6, midpoint 0.6)
        #                    x1e6: S 100, B 96 -> deal 98 (share 0.45, midpoint 0.45)
        gs = [grec("m", "F01", "seller", "x1e2", [("seller", 112), ("buyer", 96), ("seller", 104)]),
              grec("m", "F01", "seller", "x1e8", [("seller", 100), ("buyer", 96), ("seller", 98)])]
        cells = AC.cells_of(gs)
        d = AC.paired(cells, ["F01"], AC.share)
        self.assertEqual(d["F01"], Fraction(98 - 80, 40) - Fraction(104 - 80, 40))
        parts = AC.hm_parts(cells, ["F01"])
        self.assertEqual(parts["F01"], (Fraction(-6, 40), Fraction(-6, 40)))
        h = AC.paired(cells, ["F01"], AC.opening_gap)
        self.assertEqual(h["F01"], Fraction(100, 80) - Fraction(112, 80))

    def test_opening_gap_both_roles(self):
        s = grec("m", "F01", "seller", "x1e2", [("seller", 100), ("buyer", 90), ("seller", 95)])
        b = grec("m", "F01", "buyer", "x1e2", [("buyer", 90), ("seller", 110), ("buyer", 100)])
        self.assertEqual(AC.opening_gap(s), Fraction(100, 80) - 1)
        self.assertEqual(AC.opening_gap(b), 1 - Fraction(90, 120))

    def test_accept_of_the_first_offer_has_no_midpoint(self):
        r = grec("m", "F01", "seller", "x1e2", [("seller", 110)])
        self.assertEqual(r["terminal"], "AGREEMENT")
        self.assertIsNone(AC.midpoint(r))
        self.assertIsNotNone(AC.share(r))

    def test_cell_means_over_samples(self):
        gs = [grec("m", "F01", "buyer", "x1e2", [("buyer", 90), ("seller", 110), ("buyer", 100)], sample=1),
              grec("m", "F01", "buyer", "x1e2", [("buyer", 90), ("seller", 110), ("buyer", 108)], sample=2),
              grec("m", "F01", "buyer", "x1e8", [("buyer", 90), ("seller", 110), ("buyer", 100)], sample=1)]
        d = AC.paired(AC.cells_of(gs), ["F01"], AC.share)
        self.assertEqual(d["F01"], Fraction(20, 40) - (Fraction(20, 40) + Fraction(28, 40)) / 2)

    def test_spearman(self):
        self.assertAlmostEqual(AC.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(AC.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        import random
        rho, p = AC.spearman_perm_greater(list(range(20)), list(range(20)), random.Random(1), perms=2000)
        self.assertLess(p, 0.001)

    def test_reading_table(self):
        self.assertEqual(AC.reading(True, False), "scale shifts this model's split")
        self.assertEqual(AC.reading(False, True), "invariant to within +-0.10")
        self.assertEqual(AC.reading(False, False), "inconclusive")

    def test_wilcoxon_matches_analyze_py(self):
        import random
        vals = [Fraction(i - 12, 50) for i in range(30)]
        p1 = A.wilcoxon_signflip(vals, random.Random(A.seed("addendumC-wilcoxon-H1", "m")), 2000)
        self.assertEqual(p1, A.wilcoxon_signflip(vals, random.Random(A.seed("addendumC-wilcoxon-H1", "m")), 2000))

    def test_preview_run_recovers_planted_effects(self):
        buf = io.StringIO()
        rc = R.main(["--preview", "--confirm-cost", "key-limit", "--workers", "16", "--out", self.tmp],
                    stdout=buf, sleep=lambda s: None)
        self.assertEqual(rc, 0, buf.getvalue())
        run_dir = glob.glob(os.path.join(self.tmp, "addC-preview-*"))[0]
        res = AC.analyze(run_dir, flips=5000)
        self.assertTrue(res["label"].startswith("PREVIEW"))
        self.assertEqual(sorted(res["holm_family"]), sorted(R.ARMS))
        pm = res["per_model"]
        # The fake negotiators open more modestly at x1e6 (gemini most), then concede halfway: the split
        # follows the first-offer midpoint, and the shift is ordered by the planted seller-buyer asymmetry.
        self.assertLess(pm["gemini-3.5-flash-lite"]["H1"]["mean"], pm["gpt-6-luna"]["H1"]["mean"])
        for arm in R.ARMS:
            self.assertLess(pm[arm]["HA"]["mean"], 0)
            self.assertLess(abs(pm[arm]["HM"]["HM1"]["mean"]), 0.05)
            self.assertGreater(pm[arm]["HM"]["HM2"]["spearman"], 0.5)
            self.assertIn("B_expectation_not_a_test", pm[arm])
            self.assertEqual(pm[arm]["games_scored"] + len(pm[arm]["games_missing"]), 720)


if __name__ == "__main__":
    unittest.main(verbosity=1)
