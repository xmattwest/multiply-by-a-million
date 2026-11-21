"""test_vllm_client.py: offline tests for scripts/vllm_client.py.

Every test talks to a fake server on 127.0.0.1 run inside this process: a
raw socket server whose behaviour is scripted per test (a normal reply, an
error, a redirect, silence, a slow trickle, a hang-up, an oversized reply).
Each misbehaving fake gives up on its own after a few seconds, so even a
broken client can't hang the suite.

Usage: python3 tests/test_vllm_client.py      Standard library only, no network
beyond 127.0.0.1.
"""

import inspect
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
import game_engine as ge  # noqa: E402
import replay_audit as ra  # noqa: E402
import run_records as rr  # noqa: E402
import test_run_records as trr  # noqa: E402  (fake negotiator, fixtures, settings)
import vllm_client as vc  # noqa: E402

GIVE_UP = 3.0     # seconds after which every misbehaving fake closes the socket


def read_request(conn):
    """Read one HTTP/1.1 request (headers + Content-Length body)."""
    conn.settimeout(GIVE_UP)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(65536)
        if not chunk:
            return None
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    method, path, _ = lines[0].split(" ", 2)
    headers = {}
    for l in lines[1:]:
        k, _, v = l.partition(":")
        headers[k.strip().lower()] = v.strip()
    n = int(headers.get("content-length", "0"))
    while len(rest) < n:
        chunk = conn.recv(65536)
        if not chunk:
            break
        rest += chunk
    return {"method": method, "path": path, "headers": headers, "body": rest}


def http_reply(status, body, extra=""):
    reason = {200: "OK", 307: "Temporary Redirect", 503: "Service Unavailable"}.get(status, "X")
    return (f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n{extra}\r\n").encode() + body


class FakeServer:
    """behaviour(conn, request) writes the response (or misbehaves)."""

    def __init__(self, behaviour):
        self.behaviour, self.requests, self.connections = behaviour, [], 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except (socket.timeout, OSError):
                continue
            self.connections += 1
            try:
                req = read_request(conn)
                if req is not None:
                    self.requests.append(req)
                    self.behaviour(conn, req)
            except OSError:
                pass
            finally:
                conn.close()

    def close(self):
        self.stop.set()
        self.thread.join(5)
        self.sock.close()


def ok_reply(body):
    def b(conn, req):
        conn.sendall(http_reply(200, body))
    return b


def negotiator_server(conn, req):
    status, body, _ = trr.negotiator(req["body"])
    conn.sendall(http_reply(status, body))


def chat_body(**changes):
    g = trr.GAMES["F12-x1e5-S"]
    msgs = ge.render_prompt.render("seller", g["V"], g["F"], [], 1, trr.FX["f_pct_range"],
                                   trr.FX["c_pct_range"], 16, 500)
    body = ge.build_request(trr.MODEL, msgs, trr.ST, 12345)
    body.update(changes)
    return body


def enc(body):
    return rr.canonical(body).encode("utf-8")


class Base(unittest.TestCase):
    def serve(self, behaviour):
        s = FakeServer(behaviour)
        self.addCleanup(s.close)
        return s


class Wire(Base):
    def test_sends_exact_bytes_and_returns_raw_reply(self):
        reply = b'{"id":"x","choices":[]}'
        s = self.serve(ok_reply(reply))
        body = enc(chat_body())
        status, raw, err = vc.make_chat_transport(s.port, total_timeout=5)(body)
        self.assertEqual((status, raw, err), (200, reply, None))
        self.assertEqual(len(s.requests), 1)
        r = s.requests[0]
        self.assertEqual((r["method"], r["path"]), ("POST", "/v1/chat/completions"))
        self.assertEqual(r["body"], body)

    def test_headers_carry_no_credentials(self):
        s = self.serve(ok_reply(b"{}"))
        vc.request(s.port, "POST", "/tokenize", b"{}", total_timeout=5)
        h = s.requests[0]["headers"]
        for bad in ("authorization", "proxy-authorization", "cookie", "x-api-key"):
            self.assertNotIn(bad, h)
        self.assertEqual(h["host"], f"127.0.0.1:{s.port}")
        self.assertEqual(h["content-type"], "application/json")
        self.assertEqual(h["connection"], "close")

    def test_host_is_fixed_to_loopback(self):
        self.assertEqual(vc.HOST, "127.0.0.1")
        for f in (vc.request, vc.make_chat_transport):
            self.assertNotIn("host", inspect.signature(f).parameters)

    def test_proxy_settings_in_environment_are_ignored(self):
        proxy = self.serve(ok_reply(b'{"via":"proxy"}'))
        real = self.serve(ok_reply(b'{"via":"direct"}'))
        names = ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
        old = {n: os.environ.get(n) for n in names}
        try:
            for n in names:
                os.environ[n] = f"http://127.0.0.1:{proxy.port}"
            status, raw, err = vc.request(real.port, "GET", "/health", total_timeout=5)
        finally:
            for n, v in old.items():
                os.environ.pop(n, None) if v is None else os.environ.__setitem__(n, v)
        self.assertEqual(raw, b'{"via":"direct"}')
        self.assertEqual(proxy.connections, 0)

    def test_non_200_is_returned_raw_not_raised(self):
        s = self.serve(lambda c, r: c.sendall(http_reply(503, b'{"error":"busy"}')))
        self.assertEqual(vc.request(s.port, "GET", "/health", total_timeout=5),
                         (503, b'{"error":"busy"}', None))

    def test_redirect_is_not_followed(self):
        def redirect(conn, req):
            conn.sendall(http_reply(307, b"", "Location: /v1/chat/completions\r\n"))
        s = self.serve(redirect)
        status, _, err = vc.request(s.port, "GET", "/version", total_timeout=5)
        time.sleep(0.3)
        self.assertEqual((status, err), (307, None))
        self.assertEqual(len(s.requests), 1)


class Limits(Base):
    def assertQuickTimeout(self, result, t0, limit):
        status, raw, err = result
        self.assertLess(time.monotonic() - t0, limit + 1.0)
        self.assertIsNone(status)
        self.assertEqual(raw, b"")
        self.assertIn("timeout", err)

    def test_silent_server_times_out(self):
        s = self.serve(lambda c, r: time.sleep(GIVE_UP))
        t0 = time.monotonic()
        self.assertQuickTimeout(vc.request(s.port, "GET", "/health", total_timeout=0.5), t0, 0.5)

    def test_trickling_server_hits_the_overall_deadline(self):
        def trickle(conn, req):
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\nConnection: close\r\n\r\n")
            end = time.monotonic() + GIVE_UP
            while time.monotonic() < end:
                conn.sendall(b" ")
                time.sleep(0.05)
        s = self.serve(trickle)
        t0 = time.monotonic()
        self.assertQuickTimeout(vc.request(s.port, "GET", "/health", total_timeout=0.6), t0, 0.6)

    def test_oversized_reply_stops_at_1mb(self):
        big = b"x" * (2 * vc.MAX_RESPONSE)
        s = self.serve(ok_reply(big))
        status, raw, err = vc.request(s.port, "GET", "/health", total_timeout=10)
        self.assertEqual(status, 200)
        self.assertEqual(len(raw), vc.MAX_RESPONSE + 1)
        self.assertIn("1 MB", err)

    def test_refused_connection_is_reported_not_raised(self):
        t = socket.socket()
        t.bind(("127.0.0.1", 0))
        port = t.getsockname()[1]
        t.close()
        status, raw, err = vc.request(port, "GET", "/health", total_timeout=2)
        self.assertIsNone(status)
        self.assertIn("Connection", err)

    def test_server_hangup_is_reported_not_raised(self):
        s = self.serve(lambda c, r: None)          # reads the request, closes, says nothing
        status, raw, err = vc.request(s.port, "GET", "/health", total_timeout=5)
        self.assertIsNone(status)
        self.assertIsNotNone(err)


class Allowlists(Base):
    def test_only_listed_routes_can_be_called(self):
        s = self.serve(ok_reply(b"{}"))
        for m, p in (("POST", "/invocations"), ("GET", "/metrics"), ("GET", "/v1/chat/completions"),
                     ("POST", "/v1/completions"), ("GET", "/health/../invocations")):
            with self.assertRaises(ValueError, msg=f"{m} {p}"):
                vc.request(s.port, m, p, b"{}" if m == "POST" else None)
        self.assertEqual(s.connections, 0)

    def test_bad_ports_and_timeouts_refused(self):
        for p in (0, 65536, -1, True, "8000", 8000.0):
            with self.assertRaises(ValueError, msg=repr(p)):
                vc.request(p, "GET", "/health")
        for kw in ({"total_timeout": 0}, {"total_timeout": -1}, {"connect_timeout": 0}):
            with self.assertRaises(ValueError, msg=repr(kw)):
                vc.request(18000, "GET", "/health", **kw)

    def test_real_engine_request_passes_the_field_check(self):
        vc.check_chat_body(enc(chat_body()))
        vc.check_chat_body(enc(chat_body(return_prompt_text=True)))

    def test_field_check_rejects_before_anything_is_sent(self):
        s = self.serve(ok_reply(b"{}"))
        tr = vc.make_chat_transport(s.port, total_timeout=5)
        typo = chat_body()
        typo["repetiton_penalty"] = typo.pop("repetition_penalty")
        missing = chat_body()
        del missing["top_k"]
        dup = enc(chat_body()).replace(b'"seed":12345', b'"seed":12345,"seed":1', 1)
        cases = {
            "typo": enc(typo),
            "missing": enc(missing),
            "extra": enc(chat_body(reasoning_effort="low")),
            "duplicate": dup,
            "stream": enc(chat_body(stream=True)),
            "n=2": enc(chat_body(n=2)),
            "n=true": enc(chat_body(n=True)),
            "thinking on": enc(chat_body(chat_template_kwargs={"enable_thinking": True})),
            "not json": b"{oops",
            "not object": b"[1]",
        }
        self.assertNotEqual(dup, enc(chat_body()))
        for name, b in cases.items():
            with self.assertRaises(ValueError, msg=name):
                tr(b)
        self.assertEqual(s.connections, 0)


class EndToEnd(Base):
    def test_games_through_recorder_and_client_replay_exactly(self):
        s = self.serve(negotiator_server)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        run = os.path.join(tmp, "run")
        trr.run_games(run, trr.F12[:3], transport=vc.make_chat_transport(s.port, total_timeout=10))
        games = [json.loads(l) for l in trr.slurp(os.path.join(run, "games.jsonl")).splitlines()]
        self.assertEqual(len(games), 3)
        self.assertTrue(all(g["terminal"] == "AGREEMENT" for g in games), [g["terminal"] for g in games])
        problems, n = ra.audit(run)
        self.assertEqual((problems, n), ([], 3))
        self.assertEqual(len(s.requests), sum(g["n_messages"] for g in games))


if __name__ == "__main__":
    unittest.main(verbosity=1)
