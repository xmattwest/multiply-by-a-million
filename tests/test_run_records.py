"""test_run_records.py: offline tests for scripts/run_records.py and
scripts/replay_audit.py.

Plays real games through the recorder against a fake vLLM (a scripted
negotiator behind a fake transport), then checks the replay audit
reproduces them, and catches each kind of tampering or failure.

Usage: python3 tests/test_run_records.py      Standard library only, no network.
"""

import json
import os
import re
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import game_engine as ge  # noqa: E402
import replay_audit as ra  # noqa: E402
import run_records as rr  # noqa: E402


def slurp(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


FX = json.loads(slurp(os.path.join(ROOT, "config", "fixtures.json")))
ST = json.loads(slurp(os.path.join(ROOT, "prereg", "settings.json")))
GAMES = {g["game_id"]: g for g in FX["games"]}
MODEL = "qwen3.8-27b-fp8"


def envelope(content, finish="stop", reasoning=None):
    return json.dumps({"id": "chatcmpl-x", "object": "chat.completion",
                       "choices": [{"index": 0, "finish_reason": finish,
                                    "message": {"role": "assistant", "content": content,
                                                "reasoning": reasoning}}],
                       "usage": {"prompt_tokens": 900, "completion_tokens": 30}}).encode()


def negotiator(body_bytes):
    """A fake model: reads its own prompt and plays a simple, deterministic
    concession strategy. Enough to produce agreements, offers and history."""
    body = json.loads(body_bytes)
    u = body["messages"][1]["content"]
    role = re.search(r"You are the (seller|buyer)\.", u).group(1)
    limit = int(re.search(r"Your private (?:floor|ceiling) is (\d+) dollars", u).group(1))
    n = int(re.search(r"This is message (\d+) of", u).group(1))
    pend = re.search(r"current offer is (\d+) dollars", u)
    if pend:
        p = int(pend.group(1))
        if (role == "seller" and p >= limit * 1.08) or (role == "buyer" and p <= limit * 0.92):
            return 200, envelope(json.dumps({"action": "ACCEPT", "price": None, "message": "Deal."})), None
    step = 0.04 * n
    price = round(limit * (1.30 - step)) if role == "seller" else round(limit * (0.70 + step))
    return 200, envelope(json.dumps({"action": "OFFER", "price": price, "message": f"I can do {price}."})), None


def manifest():
    return {"schema": "run_manifest/1", "run_id": "test-run", "model": MODEL}


def run_games(run_dir, game_ids, transport=negotiator, close=True):
    w = rr.RunWriter(run_dir, manifest())
    call = rr.recording_call(w, transport, "test-run", sleep=lambda s: None)
    recs = []
    for gid in game_ids:
        rec = ge.play_game(GAMES[gid], MODEL, ST, FX["f_pct_range"], FX["c_pct_range"], call)
        w.game(rec)
        recs.append(rec)
    if close:
        w.close(games=len(recs))
    else:
        for c in (w.calls, w.games, w.events):
            c.close()
    return recs


def rewrite_chain(path, edit):
    """Edit records, then recompute every prev_sha256, the way a careful
    tamperer would. Returns nothing; rewrites the file."""
    recs = [json.loads(l) for l in slurp(path).splitlines()]
    for r in recs:
        r.pop("prev_sha256")
    recs = edit(recs)
    os.remove(path)
    c = rr.Chain(path)
    for r in recs:
        c.append(r)
    c.close()


F12 = ["F12-x1e2-S", "F12-x1e5-S", "F12-x1e8-S", "F12-x1e2-B", "F12-x1e5-B", "F12-x1e8-B"]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.run = os.path.join(self.tmp, "run")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def path(self, name):
        return os.path.join(self.run, name)


class Chains(Base):
    def test_chain_detects_edits_deletions_reordering_and_cut_writes(self):
        p = os.path.join(self.tmp, "x.jsonl")
        c = rr.Chain(p)
        for i in range(4):
            c.append({"i": i})
        c.close()
        self.assertEqual(rr.verify_chain(p)[:2], (True, 4))
        lines = slurp(p).splitlines(keepends=True)
        variants = {
            "edit": lines[:1] + [lines[1].replace('"i":1', '"i":9')] + lines[2:],
            "delete": lines[:1] + lines[2:],
            "reorder": [lines[0], lines[2], lines[1], lines[3]],
            "cut": lines[:3] + [lines[3][:-5]],
        }
        for name, ls in variants.items():
            q = os.path.join(self.tmp, f"{name}.jsonl")
            with open(q, "w", encoding="utf-8") as fh:
                fh.write("".join(ls))
            self.assertFalse(rr.verify_chain(q)[0], name)

    def test_chain_resumes_and_refuses_broken_files(self):
        p = os.path.join(self.tmp, "x.jsonl")
        c = rr.Chain(p); c.append({"a": 1}); c.close()
        c = rr.Chain(p); c.append({"a": 2}); c.close()
        self.assertEqual(rr.verify_chain(p)[:2], (True, 2))
        with open(p, "a", encoding="utf-8") as fh:
            fh.write('{"a":3,"prev_sha256":"' + "0" * 64 + '"}\n')
        with self.assertRaises(ValueError):
            rr.Chain(p)

    def test_records_cannot_set_their_own_hash(self):
        c = rr.Chain(os.path.join(self.tmp, "x.jsonl"))
        with self.assertRaises(ValueError):
            c.append({"prev_sha256": "f" * 64})
        c.close()


class Writer(Base):
    def test_run_folder_and_manifest_are_never_reused(self):
        rr.RunWriter(self.run, manifest()).close()
        with self.assertRaises(FileExistsError):
            rr.RunWriter(self.run, manifest())

    def test_closed_run_cannot_be_reopened(self):
        rr.RunWriter(self.run, manifest()).close()
        with self.assertRaises(ValueError):
            rr.RunWriter(self.run)

    def test_summary_hashes_every_file(self):
        run_games(self.run, F12[:1])
        s = json.loads(slurp(self.path("run_summary.json")))
        self.assertEqual(sorted(s["sha256"]), ["calls.jsonl", "events.jsonl", "games.jsonl", "run_manifest.json"])


class Recording(Base):
    def test_request_is_on_disk_before_sending_and_raw_reply_kept(self):
        seen = {}

        def transport(b):
            seen["lines_at_send"] = slurp(self.path("calls.jsonl")).count("\n")
            seen["sent"] = b
            return 200, envelope('{"action":"WALKAWAY","price":null,"message":"no"}'), None

        w = rr.RunWriter(self.run, manifest())
        reply = rr.recording_call(w, transport, "r")({"x": 1}, {"game_id": "G", "attempt": 1, "msg_no": 1, "role": "seller"})
        w.close()
        lines = [json.loads(l) for l in slurp(self.path("calls.jsonl")).splitlines()]
        self.assertEqual(seen["lines_at_send"], 1)                     # request line already written
        self.assertEqual([l["schema"] for l in lines], ["call-request/1", "call-result/1"])
        self.assertEqual(lines[1]["response_text"].encode(), envelope('{"action":"WALKAWAY","price":null,"message":"no"}'))
        self.assertEqual(seen["sent"], rr.canonical({"x": 1}).encode())
        self.assertEqual(reply["kind"], "model_output")

    def test_retries_identical_bytes_then_gives_up(self):
        sent, sleeps = [], []

        def failing(b):
            sent.append(b)
            return 503, b'{"error":"busy"}', None

        w = rr.RunWriter(self.run, manifest())
        call = rr.recording_call(w, failing, "r", sleep=sleeps.append)
        reply = call({"x": 1}, {"game_id": "G", "attempt": 1, "msg_no": 1, "role": "seller"})
        w.close()
        self.assertEqual(reply["kind"], "infrastructure")
        self.assertEqual(len(sent), 3)                                 # 1 try + 2 retries
        self.assertEqual(len(set(sent)), 1)
        self.assertEqual(sleeps, [2.0, 5.0])
        self.assertIn("infrastructure_give_up", slurp(self.path("events.jsonl")))

    def test_recovers_after_one_failure(self):
        answers = [(None, b"", "connection reset"), (200, envelope('{"action":"OFFER","price":5,"message":""}'), None)]
        w = rr.RunWriter(self.run, manifest())
        reply = rr.recording_call(w, lambda b: answers.pop(0), "r", sleep=lambda s: None)(
            {"x": 1}, {"game_id": "G", "attempt": 1, "msg_no": 1, "role": "seller"})
        w.close()
        self.assertEqual((reply["kind"], reply["call_id"]), ("model_output", "G/a1/m1/t2"))

    def test_what_counts_as_infrastructure(self):
        # An error status is infrastructure even when the body looks like a
        # completion (ZOPA spec 4.4: never parse an error body as a move).
        looks_real = envelope('{"action":"ACCEPT","price":null,"message":""}')
        cases = [(500, looks_real), (429, looks_real), (500, b"{}"), (200, b"<html>gateway</html>"), (200, b'{"choices":[]}'),
                 (200, b"x" * (rr.MAX_RESPONSE + 1)), (200, b'{"choices":[{"message":{"content":5}}]}')]
        for status, raw in cases:
            fields, problem = rr.read_envelope(status, raw)
            self.assertIsNone(fields, raw[:30])
            self.assertTrue(problem)

    def test_a_cut_off_reply_is_model_output_not_infrastructure(self):
        fields, problem = rr.read_envelope(200, envelope('{"action":"OFF', finish="length"))
        self.assertIsNone(problem)
        self.assertEqual(fields["finish_reason"], "length")


class Audit(Base):
    def test_a_clean_run_reproduces(self):
        recs = run_games(self.run, F12)
        self.assertIn("AGREEMENT", {r["terminal"] for r in recs})
        problems, n = ra.audit(self.run)
        self.assertEqual(problems, [])
        self.assertEqual(n, 6)

    def test_run_with_retries_and_a_void_reproduces(self):
        state = {"n": 0}

        def flaky(b):
            state["n"] += 1
            if state["n"] in (2, 3):
                return 502, b"bad gateway", None
            if 8 <= state["n"] <= 10:
                return None, b"", "timed out"
            return negotiator(b)

        recs = run_games(self.run, F12[:3], transport=flaky)
        self.assertIn("INFRASTRUCTURE_VOID", {r["terminal"] for r in recs})
        self.assertEqual(ra.audit(self.run)[0], [])

    def test_naive_tampering_breaks_the_chain(self):
        run_games(self.run, F12[:2])
        p = self.path("games.jsonl")
        with open(p, encoding="utf-8") as fh:
            text = fh.read()
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text.replace('"terminal":"AGREEMENT"', '"terminal":"WALKAWAY"', 1))
        problems, _ = ra.audit(self.run)
        self.assertTrue(any("chain broken" in p or "sha256 differs" in p for p in problems), problems)

    def test_careful_tampering_is_caught_by_the_summary(self):
        run_games(self.run, F12[:2])
        rewrite_chain(self.path("games.jsonl"), lambda rs: rs[:1])     # drop a game, fix the chain
        problems, _ = ra.audit(self.run)
        self.assertIn("games.jsonl: sha256 differs from run_summary.json", problems)

    def test_edited_game_record_is_caught_by_replay(self):
        run_games(self.run, F12[:2], close=False)

        def edit(rs):
            for r in rs:
                if r.get("price"):
                    r["price"] += 1
            return rs
        rewrite_chain(self.path("games.jsonl"), edit)
        problems, _ = ra.audit(self.run)
        self.assertTrue(any("game record differs on price" in p for p in problems), problems)

    def test_changed_settings_are_caught_by_replay(self):
        run_games(self.run, F12[:1], close=False)

        def edit(rs):
            for r in rs:
                if r["schema"] == "call-request/1":
                    r["request"]["temperature"] = 0.0
            return rs
        rewrite_chain(self.path("calls.jsonl"), edit)
        problems, _ = ra.audit(self.run)
        self.assertTrue(any("recorded request differs" in p for p in problems), problems)

    def test_edited_reply_is_caught(self):
        run_games(self.run, F12[:1], close=False)

        def edit(rs):
            for r in rs:
                if r["schema"] == "call-result/1":
                    r["response_text"] = r["response_text"].replace("Deal.", "Fine.")
            return rs
        rewrite_chain(self.path("calls.jsonl"), edit)
        problems, _ = ra.audit(self.run)
        self.assertTrue(any("does not match its sha256" in p for p in problems), problems)

    def test_stray_call_is_reported(self):
        run_games(self.run, F12[:1], close=False)

        def edit(rs):
            extra = dict(rs[0]); extra["call_id"] = "ghost"; extra["game_id"] = "F99"
            return rs + [extra]
        rewrite_chain(self.path("calls.jsonl"), edit)
        problems, _ = ra.audit(self.run)
        self.assertIn("ghost: request without a result line", problems)
        self.assertIn("ghost: call not part of any recorded game", problems)

    def test_interrupted_attempt_is_accounted_for(self):
        # E9: calls of an attempt cut off mid-game are fine once an event marks it interrupted.
        run_games(self.run, F12[:1], close=False)
        gid = []

        def edit(rs):
            extra = dict(rs[0]); extra["attempt"] = 9
            extra["call_id"] = extra["call_id"].replace("/a1/", "/a9/")
            gid.append(extra["game_id"])
            return rs + [extra]
        rewrite_chain(self.path("calls.jsonl"), edit)
        problems, _ = ra.audit(self.run)
        self.assertTrue(any("call not part of any recorded game" in p for p in problems), problems)
        w = rr.RunWriter(self.run)
        w.event("game_interrupted", game_id=gid[0], attempt=9)
        problems, _ = ra.audit(self.run)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
