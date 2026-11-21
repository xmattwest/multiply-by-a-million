#!/usr/bin/env python3
"""run_records.py: write a run's files so they can be trusted later.

WHAT IT DOES (docs/data_plan.md sections 5, 6 and 10)
  - Chain: an append-only JSON Lines file where every line carries
    prev_sha256, the SHA-256 of the previous line's bytes (the first line
    uses 64 zeros). Editing, deleting or reordering any line breaks every
    hash after it; verify_chain() finds the first broken line. Every line
    is flushed to disk (fsync) before the next step.
  - RunWriter: creates data/runs/<run_id>/ with run_manifest.json (written
    once, never overwritten), calls.jsonl, games.jsonl, events.jsonl, and
    finally run_summary.json with the sha256 of every file.
  - recording_call(): wraps a transport (the function that actually sends
    an HTTP request) into the `call` function game_engine.play_game() uses.
    For each try it writes a "request" line BEFORE sending and a "result"
    line with the raw response bytes BEFORE anything parses them. Then it
    reads the OpenAI-style envelope. Infrastructure failures (no completed
    reply: transport error, non-200, malformed envelope) are retried up to
    2 times with the identical request (data plan E8).

THE TRANSPORT CONTRACT
  transport(body_bytes: bytes) -> (http_status or None, response_bytes,
                                   transport_error text or None)
  It must never raise for network problems; it reports them. It sends
  exactly the given bytes.

Standard library only. No network in this file.
"""

import datetime
import hashlib
import json
import os
import time

ZERO = "0" * 64
MAX_RESPONSE = 1_048_576      # data plan 6.2: anything larger is an infrastructure error
MAX_TRIES = 3                 # 1 try + 2 retries (data plan E8)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical(obj):
    """One line of JSON. Keys keep the order given; no spaces; UTF-8."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def verify_chain(path):
    """Return (ok, n_lines, problem or None, last_hash)."""
    prev, n = ZERO, 0
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except FileNotFoundError:
        return True, 0, None, ZERO
    if data and not data.endswith(b"\n"):
        return False, 0, "last line is incomplete (no newline): a write was cut off", None
    for raw in data.split(b"\n")[:-1] if data else []:
        n += 1
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return False, n, f"line {n} is not valid JSON", None
        if not isinstance(obj, dict) or obj.get("prev_sha256") != prev:
            return False, n, f"line {n}: chain broken (prev_sha256 does not match line {n - 1})", None
        prev = sha256_hex(raw)
    return True, n, None, prev


class Chain:
    def __init__(self, path):
        ok, _, problem, last = verify_chain(path)
        if not ok:
            raise ValueError(f"{os.path.basename(path)}: {problem}")
        self.path, self.last = path, last
        self.fh = open(path, "ab")

    def append(self, obj):
        if "prev_sha256" in obj:
            raise ValueError("records must not set prev_sha256 themselves")
        rec = dict(obj)
        rec["prev_sha256"] = self.last
        raw = canonical(rec).encode("utf-8")
        if b"\n" in raw:
            raise ValueError("a record would span lines")
        self.fh.write(raw + b"\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.last = sha256_hex(raw)
        return self.last

    def close(self):
        self.fh.close()


class RunWriter:
    FILES = ("calls.jsonl", "games.jsonl", "events.jsonl")

    def __init__(self, run_dir, manifest=None):
        """New run: pass the manifest. Resuming an existing run: pass None."""
        self.dir = run_dir
        mpath = os.path.join(run_dir, "run_manifest.json")
        if manifest is not None:
            os.makedirs(run_dir, exist_ok=False)          # never reuse a folder
            with open(mpath, "x", encoding="utf-8") as fh:  # "x": fail if it exists
                fh.write(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        elif not os.path.exists(mpath):
            raise FileNotFoundError("no run_manifest.json: not a run folder")
        if os.path.exists(os.path.join(run_dir, "run_summary.json")):
            raise ValueError("this run is already closed (run_summary.json exists)")
        self.calls, self.games, self.events = (Chain(os.path.join(run_dir, f)) for f in self.FILES)
        self.event("open", resumed=manifest is None)

    def event(self, kind, **fields):
        self.events.append({"schema": "event/1", "t": utc_now(), "kind": kind, **fields})

    def game(self, record):
        self.games.append(record)

    def close(self, **counts):
        self.event("close", **counts)
        for c in (self.calls, self.games, self.events):
            c.close()
        sums = {}
        for name in ("run_manifest.json",) + self.FILES:
            with open(os.path.join(self.dir, name), "rb") as fh:
                sums[name] = sha256_hex(fh.read())
        summary = {"schema": "run_summary/1", "t": utc_now(), "counts": counts, "sha256": sums}
        with open(os.path.join(self.dir, "run_summary.json"), "x", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, indent=1) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return summary


def read_envelope(status, raw):
    """Pull the fields we need out of a chat-completions response.
    Returns (fields, problem). A problem means 'no completed reply' = infrastructure."""
    if status != 200:
        return None, f"HTTP {status}"
    if len(raw) > MAX_RESPONSE:
        return None, "response larger than 1 MB"
    try:
        env = json.loads(raw.decode("utf-8"))
        choice = env["choices"][0]
        msg = choice["message"]
        content = msg.get("content")
        if content is not None and not isinstance(content, str):
            return None, "content is not text"
        usage = env.get("usage") or {}
        return {"content": content, "finish_reason": choice.get("finish_reason"),
                "reasoning": msg.get("reasoning"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "response_id": env.get("id")}, None
    except (UnicodeDecodeError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None, "not a chat-completion response"


def recording_call(writer, transport, run_id, sleep=time.sleep, backoff=(2.0, 5.0)):
    """Build the `call` function for game_engine.play_game()."""
    counter = [0]

    def call(body, info):
        req_bytes = canonical(body).encode("utf-8")
        req_sha = sha256_hex(req_bytes)
        problem = None
        for attempt_try in range(1, MAX_TRIES + 1):
            counter[0] += 1
            call_id = f"{info['game_id']}/a{info['attempt']}/m{info['msg_no']}/t{attempt_try}"
            writer.calls.append({"schema": "call-request/1", "run_id": run_id, "call_id": call_id,
                                 **info, "try": attempt_try, "t_sent": utc_now(),
                                 "request_sha256": req_sha, "request": body})
            t0 = time.monotonic()
            status, raw, terr = transport(req_bytes)
            latency = round((time.monotonic() - t0) * 1000)
            raw = raw or b""
            # Raw bytes first, before any parsing.
            writer.calls.append({"schema": "call-result/1", "run_id": run_id, "call_id": call_id,
                                 "t_received": utc_now(), "latency_ms": latency,
                                 "http_status": status, "transport_error": terr,
                                 "response_sha256": sha256_hex(raw),
                                 "response_text": raw[:MAX_RESPONSE + 1].decode("utf-8", errors="replace")})
            fields, problem = (None, f"transport: {terr}") if terr else read_envelope(status, raw)
            if fields is not None:
                return {"kind": "model_output", "call_id": call_id, **fields}
            writer.event("infrastructure_retry" if attempt_try < MAX_TRIES else "infrastructure_give_up",
                         call_id=call_id, problem=problem)
            if attempt_try < MAX_TRIES:
                sleep(backoff[min(attempt_try - 1, len(backoff) - 1)])
        return {"kind": "infrastructure", "call_id": call_id, "problem": problem}

    return call
