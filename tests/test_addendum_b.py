#!/usr/bin/env python3
"""Offline checks for Addendum B: scripts/openrouter_client.py, run_addendum_b.py,
analyze_addendum_b.py. No network beyond 127.0.0.1, no Keychain, no money.

    python3 tests/test_addendum_b.py
"""
import glob
import http.client
import http.server
import io
import json
import os
import random
import re
import shutil
import ssl
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze as A                  # noqa: E402
import analyze_addendum_b as AB      # noqa: E402
import openrouter_client as oc       # noqa: E402
import render_prompt_s2 as rp2       # noqa: E402
import run_addendum_b as R           # noqa: E402
import run_digits_probe as D         # noqa: E402
import run_records as rr             # noqa: E402

SETTINGS = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
FAKE_KEY = "sk-or-v1-TESTKEY-0123456789abcdef"


def item(arm="gpt-6-luna", role="seller", scale="x1e8", V=100_000_000, pct=81, sample=1, fixture="F01"):
    return {"item_id": f"{arm}|{fixture}-{role[0].upper()}-{scale}-s{sample}", "arm": arm, "fixture": fixture,
            "role": role, "scale": scale, "V": V, "sample": sample, "limit": pct * V // 100,
            "seed": R.item_seed(fixture, role, sample), "run_order": 1}


def body_bytes(b):
    return rr.canonical(b).encode()


class KeyedClient(oc.Client):
    """The real (keyed) client, but connected to a local fake server instead of openrouter.ai."""

    def __init__(self, port, **kw):
        self._port = port
        super().__init__(R.ARMS, preview_port=None, key_reader=lambda: FAKE_KEY, **kw)

    def _conn(self, timeout):
        return http.client.HTTPConnection("127.0.0.1", self._port, timeout=timeout)


def envelope(content='{"action": "OFFER", "price": 100000000, "message": "hi"}', finish="stop", cost=0.0001,
             rtok=0, refusal=None, provider="OpenAI", model="openai/gpt-6-luna", **extra):
    env = {"id": "gen-1", "provider": provider, "model": model, "object": "chat.completion", "created": 1,
           "system_fingerprint": "fp_x",
           "choices": [{"index": 0, "finish_reason": finish, "native_finish_reason": finish,
                        "message": {"role": "assistant", "content": content, "refusal": refusal}}],
           "usage": {"prompt_tokens": 700, "completion_tokens": 40, "total_tokens": 740, "cost": cost,
                     "completion_tokens_details": {"reasoning_tokens": rtok}},
           "openrouter_metadata": {"endpoints": {"total": 7, "available": [
               {"provider": provider, "model": "openai/gpt-6-luna-20260922", "selected": True}]}}}
    env.update(extra)
    return json.dumps(env).encode()


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="si-addb-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def writer(self, name="run"):
        w = R.locked_writer(rr.RunWriter(os.path.join(self.tmp, name), {"schema": "test"}))
        self.addCleanup(lambda: [c.fh.close() for c in (w.calls, w.games, w.events)])
        return w


# ------------------------------------------------------------------ client

class Body(unittest.TestCase):
    def test_every_arm_body_passes(self):
        for arm in R.ARMS:
            for role in R.ROLES:
                oc.check_chat_body(body_bytes(R.build_body(item(arm, role))), R.ARMS)

    def mutate(self, arm, fn):
        b = R.build_body(item(arm))
        fn(b)
        with self.assertRaises(ValueError):
            oc.check_chat_body(body_bytes(b), R.ARMS)

    def test_rejections(self):
        self.mutate("gpt-6-luna", lambda b: b.update(temperature=0.7))
        self.mutate("gpt-6-luna", lambda b: b.pop("provider"))
        self.mutate("gpt-6-luna", lambda b: b["provider"].update(allow_fallbacks=True))
        self.mutate("gpt-6-luna", lambda b: b["provider"].update(only=["azure"]))
        self.mutate("gpt-6-luna", lambda b: b["provider"].update(require_parameters=False))
        self.mutate("gpt-6-luna", lambda b: b.update(reasoning={"effort": "low"}))
        self.mutate("gpt-6-luna", lambda b: b.pop("seed"))
        self.mutate("gpt-6-luna", lambda b: b.update(seed=True))
        self.mutate("claude-haiku-5.5", lambda b: b.update(seed=5))
        self.mutate("gpt-6-luna", lambda b: b.update(model="openai/gpt-6-astra"))
        self.mutate("gpt-6-luna", lambda b: b.update(stream=True))
        self.mutate("gpt-6-luna", lambda b: b.update(max_tokens=4096))
        self.mutate("gpt-6-luna", lambda b: b["messages"].append({"role": "user", "content": "x"}))
        self.mutate("gpt-6-luna", lambda b: b["response_format"]["json_schema"].update(strict=False))

    def test_duplicate_key_rejected(self):
        raw = body_bytes(R.build_body(item()))
        with self.assertRaises(ValueError):
            oc.check_chat_body(raw[:-1] + b',"stream":false}', R.ARMS)

    def test_seed_rule_per_arm(self):
        self.assertIn("seed", R.build_body(item("gpt-6-luna")))
        self.assertIn("seed", R.build_body(item("gemini-3.5-flash-lite")))
        self.assertNotIn("seed", R.build_body(item("claude-haiku-5.5")))

    def test_no_sampling_fields_sent(self):
        for arm in R.ARMS:
            b = R.build_body(item(arm))
            for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty"):
                self.assertNotIn(k, b)


class Keychain(unittest.TestCase):
    def runner(self, out, rc=0):
        calls = []

        class Res:
            returncode, stdout = rc, out

        def run(cmd, **kw):
            calls.append(cmd)
            return Res()
        return run, calls

    def test_reads_the_named_item(self):
        run, calls = self.runner(FAKE_KEY + "\n")
        self.assertEqual(oc.read_keychain_key(account="me", runner=run), FAKE_KEY)
        self.assertEqual(calls[0], ["security", "find-generic-password", "-s", "scale-invariance-openrouter",
                                    "-a", "me", "-w"])

    def test_missing_or_odd_key_is_an_error_without_the_value(self):
        for out, rc in (("", 0), (FAKE_KEY, 44), ("two words", 0)):
            run, _ = self.runner(out, rc)
            with self.assertRaises(KeyError) as cm:
                oc.read_keychain_key(account="me", runner=run)
            self.assertNotIn("words", str(cm.exception))
            self.assertNotIn(FAKE_KEY, str(cm.exception))

    def test_no_key_from_environment(self):
        src = open(os.path.join(ROOT, "scripts", "openrouter_client.py")).read()
        self.assertNotIn("OPENROUTER_API_KEY", src)
        self.assertNotRegex(src, r"environ\.get\(\s*['\"][A-Z_]*KEY")


class Transport(TempDir):
    def setUp(self):
        super().setUp()
        self.srv = R.fake_server()
        self.port = self.srv.server_address[1]

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close()
        super().tearDown()

    def test_preview_never_reads_keychain_or_sends_auth(self):
        def boom():
            raise AssertionError("keychain read in preview")
        c = oc.Client(R.ARMS, preview_port=self.port, key_reader=boom)
        st, raw, err = c.chat_transport()(body_bytes(R.build_body(item())))
        self.assertIsNone(err)
        self.assertIn(st, (200, 502))
        self.assertEqual(self.srv.auth_seen, [None])

    def test_real_client_targets_openrouter_over_verified_tls(self):
        c = oc.Client(R.ARMS, key_reader=lambda: FAKE_KEY)
        conn = c._conn(5)
        self.assertIsInstance(conn, http.client.HTTPSConnection)
        self.assertEqual((conn.host, conn.port), ("openrouter.ai", 443))
        self.assertTrue(conn._context.check_hostname)
        self.assertEqual(conn._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertNotIn(FAKE_KEY, repr(c))

    def test_routes_are_allowlisted(self):
        c = oc.Client(R.ARMS, preview_port=self.port)
        for m, p in (("GET", "/api/v1/models"), ("POST", "/api/v1/key"), ("GET", "/api/v1/credits")):
            with self.assertRaises(ValueError):
                c.request(m, p, b"{}" if m == "POST" else None)

    def test_key_only_in_authorization_header(self):
        c = KeyedClient(self.port)
        c.chat_transport()(body_bytes(R.build_body(item())))
        self.assertEqual(self.srv.auth_seen, ["Bearer " + FAKE_KEY])

    def test_response_echoing_the_key_is_discarded(self):
        class Echo(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                data = json.dumps({"echo": self.headers["Authorization"]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Echo)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        try:
            st, raw, err = KeyedClient(s.server_address[1]).chat_transport()(body_bytes(R.build_body(item())))
        finally:
            s.shutdown(); s.server_close()
        self.assertEqual(raw, b"")
        self.assertIn("contained the API key", err)
        self.assertNotIn(FAKE_KEY, err)

    def test_deadline(self):
        class Slow(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):          # headers at once, then one byte every 0.1 s
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                for _ in range(20):
                    try:
                        self.wfile.write(b" ")
                        self.wfile.flush()
                    except OSError:
                        return
                    time.sleep(0.1)
        s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        try:
            c = oc.Client(R.ARMS, preview_port=s.server_address[1], total_timeout=0.3)
            t0 = time.monotonic()
            st, raw, err = c.chat_transport()(body_bytes(R.build_body(item())))
        finally:
            s.shutdown(); s.server_close()
        self.assertIsNone(st)
        self.assertIn("timeout", err)
        self.assertLess(time.monotonic() - t0, 1.5)


# ------------------------------------------------------------------ design

class Design(unittest.TestCase):
    def test_main_items(self):
        items, meta = R.build_items(False, R.ARMS, 4)
        self.assertEqual(len(items), 3 * 30 * 2 * 2 * 4)
        self.assertEqual([f["id"] for f in meta["fixtures"]], [f"F{i:02d}" for i in range(1, 31)])
        self.assertEqual(len({i["item_id"] for i in items}), len(items))
        for arm in R.ARMS:
            for sc in ("x1e2", "x1e8"):
                self.assertEqual(sum(i["arm"] == arm and i["scale"] == sc for i in items), 240)

    def test_pilot_uses_x_fixtures_only(self):
        items, meta = R.build_items(True, R.ARMS, R.PILOT_SAMPLES)
        self.assertTrue(all(i["fixture"].startswith("X") for i in items))
        self.assertEqual(len(items), 3 * 5 * 2 * 2 * 2)

    def test_limits_match_study2(self):
        fx = {f["id"]: f for f in json.load(open(os.path.join(ROOT, "config", "fixtures.json")))["fixtures"]}
        for it in R.build_items(False, R.ARMS, 1)[0]:
            pct = fx[it["fixture"]]["f_pct" if it["role"] == "seller" else "c_pct"]
            self.assertEqual(it["limit"], pct * it["V"] // 100)

    def test_seeds_fit_int32(self):            # Google: TYPE_INT32 (format pilot 2026-10-10)
        for pilot, k in ((False, 4), (True, R.PILOT_SAMPLES)):
            items, _ = R.build_items(pilot, R.ARMS, k)
            self.assertTrue(all(0 <= i["seed"] <= 2**31 - 1 for i in items))

    def test_seeds_common_across_scales_distinct_across_samples(self):
        items, _ = R.build_items(False, R.ARMS, 4)
        by = {}
        for i in items:
            by.setdefault((i["fixture"], i["role"], i["sample"]), set()).add(i["seed"])
        self.assertTrue(all(len(s) == 1 for s in by.values()))
        self.assertEqual(len({next(iter(s)) for s in by.values()}), len(by))

    def test_prompt_is_study2_and_matches_digits_probe_D2_D8(self):
        for role, b in (("seller", 81), ("buyer", 124)):
            lo = R.render_item(item(role=role, scale="x1e2", V=100, pct=b))
            hi = R.render_item(item(role=role, scale="x1e8", V=100_000_000, pct=b))
            self.assertEqual(lo, rp2.render(role, 100, b, [], 1, (75, 95), (105, 125), 16, 500))
            self.assertEqual(lo, D.render_item({"condition": "D2", "role": role, "base_limit": b}, SETTINGS))
            self.assertEqual(hi, D.render_item({"condition": "D8", "role": role, "base_limit": b}, SETTINGS))
            strip = lambda m: [re.sub(r"\d+", "#", x["content"]) for x in m]
            self.assertEqual(strip(lo), strip(hi))

    def test_schema_is_the_study_schema_minus_unenforced_bounds(self):
        s = R.response_schema()
        self.assertEqual(s["required"], ["action", "price", "message"])
        self.assertFalse(s["additionalProperties"])
        self.assertNotIn("maxLength", json.dumps(s))
        self.assertNotIn("minimum", json.dumps(s))

    def test_worst_case_and_money(self):
        items, _ = R.build_items(False, R.ARMS, 4)
        one, every = R.worst_case(items)
        self.assertAlmostEqual(one, 480 * sum(R.worst_per_call(a) for a in R.ARMS.values()))
        self.assertAlmostEqual(every, 3 * one)
        self.assertEqual(R.money(0.5500001), "0.56")
        self.assertEqual(R.money(2.0), "2.00")


class Scoring(unittest.TestCase):
    def reply(self, raw):
        f, p = R.parse_reply(200, raw)
        self.assertIsNone(p)
        return {"kind": "model_output", "call_id": "c/t1", "tries": 1, **f}

    def test_offer_gap(self):
        r = R.score(item(role="seller", pct=80), self.reply(envelope('{"action":"OFFER","price":100000000,"message":"x"}')))
        self.assertEqual(r["errors"], [])
        self.assertAlmostEqual(r["gap"], 0.25)
        r = R.score(item(role="buyer", pct=120), self.reply(envelope('{"action":"OFFER","price":90000000,"message":"x"}')))
        self.assertAlmostEqual(r["gap"], 0.25)
        self.assertEqual((r["provider_returned"], r["model_returned"], r["model_version"], r["cost"],
                          r["reasoning_tokens"]), ("OpenAI", "openai/gpt-6-luna", "openai/gpt-6-luna-20260922",
                                                   0.0001, 0))

    def test_invalid_refusal_walkaway(self):
        self.assertEqual(R.score(item(), self.reply(envelope('{"action":"OFFER","price":5,"message":"x"} ok')))["errors"],
                         ["TEXT_AFTER_JSON"])
        self.assertEqual(R.score(item(), self.reply(envelope(None, refusal="no")))["errors"], ["REFUSAL"])
        self.assertEqual(R.score(item(), self.reply(envelope("x", finish="content_filter")))["errors"], ["REFUSAL"])
        self.assertEqual(R.score(item(), self.reply(envelope("{}", finish="length")))["errors"], ["TRUNCATED"])
        self.assertEqual(R.score(item(), self.reply(envelope('{"action":"OFFER","price":1,"message":"' + "a" * 501 + '"}')))["errors"],
                         ["MESSAGE_TOO_LONG"])
        self.assertEqual(R.score(item(), self.reply(envelope('{"action":"OFFER","price":0,"message":"x"}')))["errors"],
                         ["NONPOSITIVE_PRICE"])
        w = R.score(item(), self.reply(envelope('{"action":"WALKAWAY","price":null,"message":"x"}')))
        self.assertEqual((w["errors"], w["action"], w["gap"]), ([], "WALKAWAY", None))

    def test_parse_reply_infrastructure(self):
        self.assertIsNotNone(R.parse_reply(502, b"{}")[1])
        self.assertIsNotNone(R.parse_reply(200, b'{"error": {"code": 500}}')[1])
        self.assertIsNotNone(R.parse_reply(200, envelope(finish="error"))[1])
        self.assertIsNotNone(R.parse_reply(200, b"not json")[1])


class Recording(TempDir):
    def transport(self, script):
        sent = []

        def t(b):
            sent.append(b)
            return script.pop(0)
        return t, sent

    def test_every_try_recorded_and_completed_reply_never_reasked(self):
        w = self.writer()
        t, sent = self.transport([(502, b"{}", None), (None, b"", "timeout"), (200, envelope('{"x":1}'), None)])
        cap = R.SpendCap(10.0)
        rep = R.recording_call(w, t, "run", R.ARMS, cap, sleep=lambda s: None)(R.build_body(item()), item())
        self.assertEqual((rep["kind"], rep["tries"]), ("model_output", 3))
        self.assertEqual(len(set(sent)), 1)                        # identical request every try
        self.assertIn("MISSING_FIELD:action", R.score(item(), rep)["errors"])   # scored as-is, not re-asked
        w.close()
        calls = [json.loads(l) for l in open(os.path.join(self.tmp, "run", "calls.jsonl"))]
        self.assertEqual([c["schema"] for c in calls], ["api-call-request/1", "api-call-result/1"] * 3)
        self.assertEqual([c["call_id"] for c in calls[::2]], [f"{item()['item_id']}/t{i}" for i in (1, 2, 3)])
        self.assertTrue(rr.verify_chain(os.path.join(self.tmp, "run", "calls.jsonl"))[0])

    def test_invalid_reply_is_not_retried(self):
        w = self.writer()
        t, sent = self.transport([(200, envelope("garbage"), None)])
        rep = R.recording_call(w, t, "run", R.ARMS, R.SpendCap(10.0), sleep=lambda s: None)(R.build_body(item()), item())
        self.assertEqual(len(sent), 1)
        self.assertEqual(R.score(item(), rep)["errors"], ["NOT_JSON_OBJECT"])

    def test_rejected_request_stops_without_retry(self):
        w = self.writer()
        t, sent = self.transport([(400, b'{"error": {"code": 400, "message": "bad seed"}}', None)])
        with self.assertRaises(R.RequestRejected):
            R.recording_call(w, t, "run", R.ARMS, R.SpendCap(10.0), sleep=lambda s: None)(R.build_body(item()), item())
        self.assertEqual(len(sent), 1)
        for st in (408, 429):
            t, sent = self.transport([(st, b"{}", None), (200, envelope(), None)])
            rep = R.recording_call(w, t, "run", R.ARMS, R.SpendCap(10.0), sleep=lambda s: None)(R.build_body(item()), item())
            self.assertEqual((rep["kind"], len(sent)), ("model_output", 2))

    def test_give_up_after_three(self):
        w = self.writer()
        t, sent = self.transport([(500, b"", None)] * 3)
        rep = R.recording_call(w, t, "run", R.ARMS, R.SpendCap(10.0), sleep=lambda s: None)(R.build_body(item()), item())
        self.assertEqual((rep["kind"], len(sent)), ("infrastructure", 3))
        self.assertEqual(R.score(item(), rep)["errors"], ["INFRASTRUCTURE"])

    def test_spend_cap_is_never_exceeded(self):
        w = self.writer()
        wpc = R.worst_per_call(R.ARMS["gpt-6-luna"])
        cap = R.SpendCap(wpc * 3.5)
        t, sent = self.transport([(200, envelope(cost=wpc), None)] * 10)
        call = R.recording_call(w, t, "run", R.ARMS, cap, sleep=lambda s: None)
        n = 0
        with self.assertRaises(R.CapReached):
            for _ in range(10):
                call(R.build_body(item()), item())
                n += 1
        self.assertEqual((n, len(sent)), (3, 3))
        self.assertLessEqual(cap.spent, cap.cap)

    def test_missing_cost_counts_at_worst(self):
        w = self.writer()
        t, _ = self.transport([(200, envelope(cost=None), None)])
        cap = R.SpendCap(10.0)
        R.recording_call(w, t, "run", R.ARMS, cap, sleep=lambda s: None)(R.build_body(item()), item())
        self.assertAlmostEqual(cap.spent, R.worst_per_call(R.ARMS["gpt-6-luna"]))
        self.assertEqual(cap.unpriced, 1)


class Gates(TempDir):
    def no_client(self, *a):
        raise AssertionError("a client was created")

    def run_main(self, argv, factory=None):
        out = io.StringIO()
        rc = R.main(argv + ["--out", self.tmp], stdout=out, client_factory=factory or self.no_client,
                    sleep=lambda s: None)
        return rc, out.getvalue()

    def test_dry_run_sends_nothing(self):
        rc, out = self.run_main(["--pilot"])
        self.assertEqual(rc, 0)
        self.assertIn("Dry run: nothing sent", out)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_wrong_confirm_cost(self):
        rc, out = self.run_main(["--pilot", "--confirm-cost", "0.01"])
        self.assertEqual(rc, 2)
        self.assertIn("does not match", out)

    def test_registered_run_needs_frozen_inputs(self):
        if R.check_frozen() is None:
            self.skipTest("addendum already frozen in this checkout")
        _, out = self.run_main([])
        cc = re.search(r"--confirm-cost (\S+)", out).group(1)
        rc, out = self.run_main(["--confirm-cost", cc])
        self.assertEqual(rc, 3)
        self.assertIn("Not running:", out)

    def test_check_frozen_in_a_scratch_repo(self):
        import subprocess
        repo = os.path.join(self.tmp, "repo")
        os.makedirs(os.path.join(repo, "prereg", "addendum_B"))
        open(os.path.join(repo, "a.py"), "w").write("x = 1\n")
        lst = os.path.join(repo, "prereg", "addendum_B", "inputs.sha256")
        open(lst, "w").write("# c\n" + rr.sha256_hex(b"x = 1\n") + "  a.py\n")
        old_root, old_list = R.ROOT, R.FROZEN_LIST
        R.ROOT, R.FROZEN_LIST = repo, lst
        try:
            git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, check=True)
            git("init", "-q")
            self.assertIn("not committed", R.check_frozen())
            git("add", ".")
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "freeze")
            self.assertIsNone(R.check_frozen())
            open(os.path.join(repo, "a.py"), "w").write("x = 2\n")
            self.assertIn("uncommitted", R.check_frozen())
        finally:
            R.ROOT, R.FROZEN_LIST = old_root, old_list

    def test_registered_run_uses_all_arms(self):
        rc, _ = self.run_main(["--arms", "gpt-6-luna"])
        self.assertEqual(rc, 2)

    def test_low_key_credit_stops_before_any_prompt(self):
        srv = R.fake_server(key_limit=0.05)
        try:
            _, out = self.run_main(["--pilot"])
            cc = re.search(r"--confirm-cost (\S+)", out).group(1)
            rc, out = self.run_main(["--pilot", "--confirm-cost", cc],
                                    factory=lambda arms, s: KeyedClient(srv.server_address[1]))
        finally:
            srv.shutdown(); srv.server_close()
        self.assertEqual(rc, 3)
        self.assertIn("below the worst case", out)
        self.assertEqual(len(srv.auth_seen), 1)               # only GET /api/v1/key

    def test_keyed_pilot_end_to_end_key_never_written(self):
        srv = R.fake_server()
        try:
            _, out = self.run_main(["--pilot"])
            cc = re.search(r"--confirm-cost (\S+)", out).group(1)
            rc, out = self.run_main(["--pilot", "--confirm-cost", cc],
                                    factory=lambda arms, s: KeyedClient(srv.server_address[1]))
        finally:
            srv.shutdown(); srv.server_close()
        self.assertEqual(rc, 0, out)
        self.assertTrue(all(h == "Bearer " + FAKE_KEY for h in srv.auth_seen))
        self.assertNotIn(FAKE_KEY, out)
        run_dir = glob.glob(os.path.join(self.tmp, "addB-pilot-*"))[0]
        for p in glob.glob(os.path.join(run_dir, "*")):
            self.assertNotIn(FAKE_KEY, open(p, encoding="utf-8").read(), p)
        for name in ("calls.jsonl", "games.jsonl", "events.jsonl"):
            self.assertTrue(rr.verify_chain(os.path.join(run_dir, name))[0])
        games = [json.loads(l) for l in open(os.path.join(run_dir, "games.jsonl"))]
        self.assertEqual(len(games), 120)
        self.assertIn("pilot gate", out)
        self.assertNotIn("analyze_addendum_b", out)
        with self.assertRaises(ValueError):                    # never analysed for HA
            AB.analyze(run_dir, flips=100, boots=10)

    def test_preview_run_and_analysis(self):
        _, out = self.run_main(["--preview", "--arms", "gpt-6-luna,gemini-3.5-flash-lite"])
        cc = re.search(r"--confirm-cost (\S+)", out).group(1)
        rc = R.main(["--preview", "--arms", "gpt-6-luna,gemini-3.5-flash-lite", "--confirm-cost", cc,
                     "--out", self.tmp], stdout=io.StringIO(), sleep=lambda s: None)
        self.assertEqual(rc, 0)
        run_dir = glob.glob(os.path.join(self.tmp, "addB-preview-*"))[0]
        res = AB.analyze(run_dir, flips=20_000, boots=1_000)
        self.assertTrue(res["label"].startswith("PREVIEW"))
        self.assertEqual(res["holm_family"], ["gemini-3.5-flash-lite", "gpt-6-luna"])
        self.assertLess(res["per_model"]["gpt-6-luna"]["HA"]["mean_h"], -0.05)   # fake effect -0.10
        self.assertNotIn("claude-haiku-5.5", res["per_model"])


# ------------------------------------------------------------------ analysis

def rec(arm, f, role, sc, gap, errors=(), action="OFFER", rtok=0):
    return {"item_id": f"{arm}|{f}-{role}-{sc}-{random.random()}", "arm": arm, "fixture": f, "role": role,
            "scale": sc, "errors": list(errors), "action": None if errors else action,
            "gap": None if errors or action != "OFFER" else gap, "reasoning_tokens": rtok}


class Analysis(unittest.TestCase):
    def test_h_by_hand(self):
        rs = [rec("m", "F01", "seller", "x1e2", 0.40), rec("m", "F01", "seller", "x1e2", 0.30),
              rec("m", "F01", "seller", "x1e8", 0.20),
              rec("m", "F01", "buyer", "x1e2", 0.50), rec("m", "F01", "buyer", "x1e8", 0.45),
              rec("m", "F02", "seller", "x1e2", 0.30), rec("m", "F02", "seller", "x1e8", None, ["TEXT_AFTER_JSON"]),
              rec("m", "F02", "buyer", "x1e2", 0.30), rec("m", "F02", "buyer", "x1e8", 0.40),
              rec("m", "F03", "seller", "x1e2", 0.30), rec("m", "F03", "seller", "x1e8", None, action="WALKAWAY")]
        h, per_role = AB.h_values(rs)
        self.assertAlmostEqual(h["F01"], ((0.20 - 0.35) + (0.45 - 0.50)) / 2)
        self.assertAlmostEqual(h["F02"], 0.10)          # seller side undefined: buyer only
        self.assertNotIn("F03", h)                      # no role defined at both scales
        h2, _ = AB.h_values(rs + [rec("m", "F01", "seller", "x1e8", 0.9, rtok=5)],
                            keep=lambda r: not r["reasoning_tokens"])
        self.assertAlmostEqual(h2["F01"], h["F01"])

    def test_test_matches_analyze_py_and_holm(self):
        vals = {f"F{i:02d}": -0.025 + 0.01 * (i % 5) for i in range(1, 31)}
        res = AB.ha_test(vals, "gpt-6-luna", flips=50_000, boots=1_000)
        p, wp, n = A.wilcoxon_signflip_less([vals[f] for f in sorted(vals)],
                                            random.Random(A.seed("addendumB-wilcoxon-HA", "gpt-6-luna")), 50_000)
        self.assertEqual((res["p_one_sided_unadjusted"], res["W_plus"], res["n_nonzero"]), (p, wp, n))
        self.assertTrue(0 < p < 1)                     # a p that depends on the seed
        ci = A.bootstrap_ci([vals[f] for f in sorted(vals)],
                            random.Random(A.seed("addendumB-bootstrap-HA", "gpt-6-luna")), levels=(0.95,), boots=1_000)
        self.assertEqual(res["ci95_mean_h"], ci["95"])
        self.assertEqual(res["fixtures_h_below_0"], sum(v < 0 for v in vals.values()))
        adj = A.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)

    def test_flags(self):
        rs = []
        for f in range(1, 11):
            rs += [rec("m", f"F{f:02d}", "seller", "x1e2", 0.3), rec("m", f"F{f:02d}", "seller", "x1e8", 0.2)]
        rs += [rec("m", f"F{f:02d}", "buyer", "x1e8", None, ["REFUSAL"]) for f in range(1, 11)]
        q = AB.quality(rs)
        self.assertEqual(q["x1e8"]["refusals"], 10)
        self.assertAlmostEqual(q["x1e8"]["invalid_or_refused_share"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=1)
