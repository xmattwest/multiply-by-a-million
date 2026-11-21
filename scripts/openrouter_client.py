#!/usr/bin/env python3
"""openrouter_client.py: the only code that sends requests to OpenRouter (Addendum B).

Modelled on scripts/vllm_client.py. Standard library only.

WHAT IT DOES
  - request(): one HTTPS request to openrouter.ai (or, in preview, one HTTP
    request to a fake server on 127.0.0.1). Returns (status, bytes, error)
    and never raises for network problems; it reports them.
  - make_chat_transport(): the `transport` that run_addendum_b's recorder
    calls. Before anything goes on the wire it checks the body against a
    strict allowlist and against the arm it claims to be (model, pinned
    provider, no fallbacks, reasoning setting, seed rule).
  - key_status() / GET /api/v1/key: the key's own credit limit and usage
    (no prompt is sent), used before a paid run.

SECRETS
  - The API key comes only from the macOS Keychain item
    "scale-invariance-openrouter" (account $USER), read with
    `security find-generic-password -w` into memory. Never from the
    environment, a file or an argument. It is sent only in the
    Authorization header, never logged, never printed, and never part of
    a returned value. Every response is checked: if the key's text ever
    appears in it, the response is replaced by an error (it is not logged).
  - Preview mode never reads the Keychain and never leaves 127.0.0.1.

SAFETY RULES BUILT IN
  - Host fixed: "openrouter.ai":443 with TLS certificate and hostname
    checks (ssl.create_default_context). Preview: 127.0.0.1 only.
  - Only two routes: POST /api/v1/chat/completions, GET /api/v1/key.
  - http.client never reads proxy settings and never follows redirects
    (a 3xx is returned as a status, which the caller treats as a failure).
  - Connect limit and one overall deadline for the request, enforced on
    every socket read. The response is read up to 1 MB + 1 byte.
  - A new connection per request, closed afterwards.

  Local mistakes (bad route, a body that fails the allowlist, no key)
  raise ValueError / KeyError: they are bugs or setup problems, not
  network problems, and the run must stop rather than retry them.
"""
import http.client
import json
import os
import socket
import ssl
import subprocess
import time

HOST = "openrouter.ai"
PORT = 443
PREVIEW_HOST = "127.0.0.1"
KEYCHAIN_SERVICE = "scale-invariance-openrouter"
MAX_RESPONSE = 1_048_576
READ_CHUNK = 65_536
CHAT_PATH = "/api/v1/chat/completions"
KEY_PATH = "/api/v1/key"
ROUTES = frozenset({("POST", CHAT_PATH), ("GET", KEY_PATH)})

CHAT_REQUIRED = ("model", "messages", "max_tokens", "response_format", "provider", "reasoning", "stream")
CHAT_OPTIONAL = ("seed",)


class _Dup(Exception):
    pass


def _no_dupes(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _Dup()
    return dict(pairs)


def read_keychain_key(service=KEYCHAIN_SERVICE, account=None, runner=subprocess.run):
    """The key from the macOS Keychain, or KeyError. Nothing is printed."""
    account = account or os.environ.get("USER") or ""
    try:
        r = runner(["security", "find-generic-password", "-s", service, "-a", account, "-w"],
                   capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        raise KeyError(f"could not run `security` to read Keychain item {service}") from None
    key = (r.stdout or "").strip()
    if r.returncode != 0 or not key:
        raise KeyError(f"no Keychain item {service} for account {account!r}")
    if any(c.isspace() for c in key) or not key.isprintable():
        raise KeyError(f"Keychain item {service} does not look like one API key")
    return key


def check_chat_body(body_bytes, arms):
    """Raise ValueError unless the body is exactly a request for one known arm."""
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
        raise ValueError("request has fields we never send: " + ", ".join(unknown))
    missing = [k for k in CHAT_REQUIRED if k not in body]
    if missing:
        raise ValueError("request is missing fields: " + ", ".join(missing))
    arm = next((a for a in arms.values() if a["model"] == body["model"]), None)
    if arm is None:
        raise ValueError(f"model not in the registered panel: {body['model']!r}")
    if body["stream"] is not False:
        raise ValueError("stream must be false")
    if body["provider"] != provider_block(arm):
        raise ValueError("provider block must pin the vendor endpoint with fallbacks off")
    if body["reasoning"] != arm["reasoning"]:
        raise ValueError("reasoning setting differs from the registered arm")
    if ("seed" in body) != arm["send_seed"]:
        raise ValueError("seed must be sent exactly when the arm's endpoint supports it")
    if "seed" in body and (isinstance(body["seed"], bool) or not isinstance(body["seed"], int)):
        raise ValueError("seed must be an integer")
    if isinstance(body["max_tokens"], bool) or not isinstance(body["max_tokens"], int) \
            or not 1 <= body["max_tokens"] <= arm["max_tokens_cap"]:
        raise ValueError("max_tokens out of range")
    msgs = body["messages"]
    if not (isinstance(msgs, list) and len(msgs) == 2 and [m.get("role") for m in msgs] == ["system", "user"]
            and all(set(m) == {"role", "content"} and isinstance(m["content"], str) for m in msgs)):
        raise ValueError("messages must be exactly one system and one user message")
    rf = body["response_format"]
    if not (isinstance(rf, dict) and rf.get("type") == "json_schema"
            and isinstance(rf.get("json_schema"), dict) and rf["json_schema"].get("strict") is True):
        raise ValueError("response_format must be a strict json_schema")
    return body


def provider_block(arm):
    return {"only": [arm["provider"]], "allow_fallbacks": False, "require_parameters": True,
            "data_collection": "deny"}


class Client:
    """One client per run. preview_port=None means the real openrouter.ai."""

    def __init__(self, arms, preview_port=None, key_reader=read_keychain_key,
                 connect_timeout=10.0, total_timeout=120.0):
        if not (connect_timeout > 0 and total_timeout > 0):
            raise ValueError("time limits must be positive")
        self.arms, self.preview_port = arms, preview_port
        self.connect_timeout, self.total_timeout = connect_timeout, total_timeout
        if preview_port is not None:
            if isinstance(preview_port, bool) or not isinstance(preview_port, int) or not 1 <= preview_port <= 65535:
                raise ValueError(f"bad preview port: {preview_port!r}")
            self._key = None
        else:
            self._key = key_reader()

    def __repr__(self):                   # never show the key
        return f"Client(preview_port={self.preview_port!r})"

    def _conn(self, timeout):
        if self.preview_port is not None:
            return http.client.HTTPConnection(PREVIEW_HOST, self.preview_port, timeout=timeout)
        return http.client.HTTPSConnection(HOST, PORT, timeout=timeout, context=ssl.create_default_context())

    def request(self, method, path, body=None):
        """Returns (status or None, bytes, error or None)."""
        if (method, path) not in ROUTES:
            raise ValueError(f"route not allowed: {method} {path}")
        if method == "POST" and not isinstance(body, bytes):
            raise ValueError("POST needs a bytes body")
        if method == "GET" and body is not None:
            raise ValueError("GET takes no body")
        deadline = time.monotonic() + self.total_timeout
        headers = {"Accept": "application/json", "Connection": "close", "X-OpenRouter-Metadata": "enabled",
                   "X-Title": "scale-invariance addendum B"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._key is not None:
            headers["Authorization"] = "Bearer " + self._key
        conn = self._conn(min(self.connect_timeout, self.total_timeout))
        try:
            conn.connect()
            sock = conn.sock
            sock.settimeout(max(0.001, deadline - time.monotonic()))
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            status, data = resp.status, b""
            while not resp.isclosed():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, b"", f"timeout: no complete reply within {self.total_timeout:g} s"
                sock.settimeout(remaining)
                chunk = resp.read1(READ_CHUNK)
                if not chunk:
                    break
                data += chunk
                if len(data) > MAX_RESPONSE:
                    return status, data[:MAX_RESPONSE + 1], "response larger than 1 MB (stopped reading)"
            if self._key is not None and self._key.encode() in data:
                return None, b"", "response contained the API key; discarded unlogged"
            return status, data, None
        except (socket.timeout, TimeoutError):
            return None, b"", f"timeout: no complete reply within {self.total_timeout:g} s"
        except (OSError, http.client.HTTPException) as e:
            msg = f"{type(e).__name__}: {e}"
            if self._key is not None:
                msg = msg.replace(self._key, "[key]")
            return None, b"", msg
        finally:
            conn.close()

    def chat_transport(self):
        def transport(body_bytes):
            check_chat_body(body_bytes, self.arms)
            return self.request("POST", CHAT_PATH, body_bytes)
        return transport

    def key_status(self):
        """(limit, usage, limit_remaining) in dollars from GET /api/v1/key, or raises RuntimeError."""
        status, raw, err = self.request("GET", KEY_PATH)
        if err or status != 200:
            raise RuntimeError(f"key check failed: {err or 'HTTP ' + str(status)}")
        try:
            d = json.loads(raw.decode("utf-8"))["data"]
            return d.get("limit"), d.get("usage"), d.get("limit_remaining")
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            raise RuntimeError("key check: unexpected response") from None
