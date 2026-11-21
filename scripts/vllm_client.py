#!/usr/bin/env python3
"""vllm_client.py: the only code that sends HTTP requests to the vLLM server.

WHAT IT DOES (docs/infra_plan.md D3, D5, D9; docs/api_map.md section 5)
  - request(): one HTTP request to vLLM through the SSH tunnel. It returns
    (http_status or None, response_bytes, transport_error or None) and never
    raises for network problems; it reports them.
  - make_chat_transport(): wraps request() into the `transport` function that
    run_records.recording_call() expects. Before anything goes on the wire,
    it checks the request body against an allowlist of fields (C3, C12):
    vLLM silently ignores fields it doesn't know, so a misspelled
    "repetiton_penalty" would otherwise just not apply.

SAFETY RULES BUILT IN
  - The host is fixed to 127.0.0.1 (the laptop end of the SSH tunnel). There
    is no way to pass another host.
  - Only a fixed list of (method, path) routes may be called. /invocations,
    /metrics and the rest are refused (C4).
  - Python's http.client never reads proxy settings from the environment and
    never follows redirects. No authorization header is ever sent: behind the
    tunnel there is no key (D3).
  - Explicit time limits: a connect limit, and one overall deadline for the
    whole request, enforced on every socket read, so a server that sends one
    byte at a time can't stretch a call past it.
  - The response is read up to 1 MB + 1 byte, then reading stops
    (data plan 6.2: larger is an infrastructure error).
  - A new connection per request, closed afterwards. No pooling, no reuse.

  Local mistakes (a disallowed route, a bad port, a body that fails the
  allowlist) raise ValueError: they are bugs in our code, not network
  problems, and the run must stop instead of retrying them.

Standard library only.
"""

import http.client
import json
import socket
import time

HOST = "127.0.0.1"
MAX_RESPONSE = 1_048_576
READ_CHUNK = 65_536

ROUTES = frozenset({
    ("POST", "/v1/chat/completions"),
    ("GET", "/health"),
    ("GET", "/version"),
    ("GET", "/v1/models"),
    ("POST", "/tokenize"),
    ("POST", "/detokenize"),
})

# Every field game_engine.build_request() sends; all must be present (C3).
CHAT_REQUIRED = ("model", "messages", "temperature", "top_p", "top_k", "min_p",
                 "presence_penalty", "frequency_penalty", "repetition_penalty",
                 "max_tokens", "seed", "n", "stream", "chat_template_kwargs",
                 "response_format")
# Allowed in addition, for Phase 7b checks (api_map 2.2, 2.3).
CHAT_OPTIONAL = ("return_prompt_text",)


class _Dup(Exception):
    pass


def _no_dupes(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _Dup()
    return dict(pairs)


def check_chat_body(body_bytes):
    """Raise ValueError unless the body is exactly the kind of request we send."""
    try:
        body = json.loads(body_bytes.decode("utf-8"), object_pairs_hook=_no_dupes)
    except _Dup:
        raise ValueError("request body has a duplicate field") from None
    except (UnicodeDecodeError, ValueError):
        raise ValueError("request body is not JSON") from None
    if not isinstance(body, dict):
        raise ValueError("request body is not a JSON object")
    unknown = sorted(k for k in body if k not in CHAT_REQUIRED + CHAT_OPTIONAL)
    if unknown:
        raise ValueError("request has fields vLLM would silently ignore or we never send: "
                         + ", ".join(unknown))
    missing = [k for k in CHAT_REQUIRED if k not in body]
    if missing:
        raise ValueError("request is missing fields (they would fall back to defaults): "
                         + ", ".join(missing))
    if body["stream"] is not False:
        raise ValueError("stream must be false")
    if body["n"] != 1 or isinstance(body["n"], bool):
        raise ValueError("n must be 1")
    if body["chat_template_kwargs"] != {"enable_thinking": False}:
        raise ValueError("chat_template_kwargs must be exactly {\"enable_thinking\": false}")
    return body


def _check_port(port):
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError(f"bad port: {port!r}")


def request(port, method, path, body=None, connect_timeout=5.0, total_timeout=120.0):
    """One HTTP request to 127.0.0.1:port. Returns (status, bytes, error)."""
    _check_port(port)
    if (method, path) not in ROUTES:
        raise ValueError(f"route not allowed: {method} {path}")
    if method == "POST" and not isinstance(body, bytes):
        raise ValueError("POST needs a bytes body")
    if method == "GET" and body is not None:
        raise ValueError("GET takes no body")
    if not (connect_timeout > 0 and total_timeout > 0):
        raise ValueError("time limits must be positive")

    deadline = time.monotonic() + total_timeout
    headers = {"Accept": "application/json", "Connection": "close"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    conn = http.client.HTTPConnection(HOST, port, timeout=min(connect_timeout, total_timeout))
    try:
        conn.connect()
        # Keep our own handle: http.client drops conn.sock once the
        # response owns the socket ("Connection: close").
        sock = conn.sock
        sock.settimeout(max(0.001, deadline - time.monotonic()))
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        status, data = resp.status, b""
        while not resp.isclosed():      # http.client closes it once the body is complete
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, b"", f"timeout: no complete reply within {total_timeout:g} s"
            sock.settimeout(remaining)
            # read1: at most one socket read, so the deadline is checked
            # between reads even if the server sends one byte at a time.
            chunk = resp.read1(READ_CHUNK)
            if not chunk:
                break
            data += chunk
            if len(data) > MAX_RESPONSE:
                return status, data[:MAX_RESPONSE + 1], "response larger than 1 MB (stopped reading)"
        return status, data, None
    except (socket.timeout, TimeoutError):
        return None, b"", f"timeout: no complete reply within {total_timeout:g} s"
    except (OSError, http.client.HTTPException) as e:
        return None, b"", f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def make_chat_transport(port, connect_timeout=5.0, total_timeout=120.0):
    """The transport for run_records.recording_call()."""
    _check_port(port)

    def transport(body_bytes):
        check_chat_body(body_bytes)
        return request(port, "POST", "/v1/chat/completions", body_bytes,
                       connect_timeout=connect_timeout, total_timeout=total_timeout)

    return transport
