#!/usr/bin/env python3
"""run_job.py: run one model's job end to end, with a live progress panel.

THIS SPENDS MONEY (except with --preview). It rents one GPU machine through
scripts/launch_pod.sh, which always deletes it at the end.

    python3 scripts/run_job.py --model qwen3.8-27b-fp8 --pilot-then-run
    python3 scripts/run_job.py --model qwen3.8-27b-fp8 --resume data/runs/<run_id>
    python3 scripts/run_job.py --model qwen3.8-27b-fp8 --pilot
    ... add --slot N (1-9) to run a second job at the same time in another
    terminal: its own pod, session file, stop file and port 18000+N.
    ... add --preview to any of them for a free practice run on this laptop.

MODES
  --pilot-then-run  The normal job (pre-registration amendment A1): the
                    pilot (first 60 pilot games), then gate 5 decided
                    automatically from format counts, then, if it passes,
                    the real run (all 180 games of config/fixtures.json),
                    all on one machine, so the model downloads once.
  --resume DIR      Continue a real run that ended early for an
                    infrastructure reason (A1 change 4, E9): finished games
                    are kept, a game that was cut off is replayed from the
                    start as a new attempt.
  --pilot           The pilot alone (exploratory; --games N to shorten it).

WHAT EVERY JOB DOES
  1. Plan: refuses a real run unless the frozen inputs match
     prereg/inputs.sha256 and scripts/, config/ and prereg/ are committed.
     Then a free dry run shows the worst-case cost, and you type it back to
     launch; anything else cancels.
  2. Machine: the launcher creates the pod and waits for SSH, the weights
     download + fingerprint check, and vLLM. The panel follows each stage.
  3. Checks before any game (§8 gates 1-4, plus the A1-era watermark
     check): weights manifest hash, /version 0.30.0, exactly our model,
     thinking off in the rendered prompt, no watermark configured. After
     each phase's first reply: the settings vLLM logged equal
     prereg/settings.json. Any failure stops the job.
  4. Games one at a time (D11), every request and reply on disk before the
     next call (data plan §5). Voided games are replayed at the end, at most
     3 attempts (E8). No new game starts when machine time is nearly up.
  5. End of each phase: vLLM's logs copied into the run folder, the run
     closed (if complete), every game replayed from the raw records
     (replay_audit). Last: the machine is deleted.

DEAL OUTCOMES ARE NOT SHOWN, in any mode: the pilot judges format only
(§8), and a real run never stops because of outcomes (§8).

Ctrl-C, or closing the terminal, stops safely: the current game is marked
interrupted (E9), and the machine is deleted. A real run stopped this way
can be resumed. A second Ctrl-C skips the log copy.
Standard library only.
"""
import argparse
import hashlib
import http.server
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import game_engine as ge          # noqa: E402
import render_prompt as rp        # noqa: E402
import render_prompt_s2 as rp2    # noqa: E402
import replay_audit               # noqa: E402
import run_records as rr          # noqa: E402
import vllm_client as v           # noqa: E402

PORT = 18000
MAX_ATTEMPTS = 3                  # per game (data plan E8)
GATE_VALID = 0.95                 # pre-registration §8 gate 5
PILOT_GAMES = 60                  # amendment A1
# The rendered prompt must end exactly like this (thinking off / no reasoning block).
# mistral and phi have no thinking mode; their tails are the pinned chat templates'
# generation prompt (amendment 2026-10-04, prereg/AMENDMENT_WP2d.md).
TAILS = {"qwen": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
         "gemma": "<|turn>model\n<|channel>thought\n<channel|>",
         "mistral": "[/INST]",
         "phi": "<|im_start|>assistant<|im_sep|>"}


def family_of(model):
    """Chat-template family from the pinned model name; refuses unknown names."""
    for fam in TAILS:
        if model.startswith(fam):
            return fam
    raise ValueError(f"no chat-template family for model {model!r}")
P = lambda *a: os.path.join(ROOT, *a)          # noqa: E731
LAUNCHER = P("scripts", "launch_pod.sh")
POD_SSH = P("scripts", "lib", "pod_ssh.sh")
SESSION = P("logs", "pod_session.env")
STOPFILE = P("logs", "pod_session.stop")
PILOT_FX = "config/pilot_fixtures.json"
REAL_FX = "config/fixtures.json"
CHECKED = ["config/pins.json", PILOT_FX, REAL_FX, "prereg/settings.json", "prereg/inputs.sha256",
           "prereg/prompts/system.txt", "prereg/prompts/user.txt", "prereg/prompts/fragments.txt",
           "scripts/render_prompt.py", "scripts/game_engine.py", "scripts/run_records.py",
           "scripts/replay_audit.py", "scripts/vllm_client.py", "scripts/run_job.py",
           "scripts/launch_pod.sh", "scripts/pod/boot.sh"]
CHECKED_S2 = ["prereg/study2/inputs.sha256", "prereg/study2/prompts/system.txt", "prereg/study2/prompts/user.txt",
              "prereg/study2/prompts/fragments.txt", "scripts/render_prompt_s2.py"]


def use_slot(slot):
    """Point this process (and the launcher it starts) at one slot. 0 keeps the original names."""
    global PORT, SESSION, STOPFILE
    if not 0 <= slot <= 9:
        raise SystemExit("--slot must be 0..9")
    os.environ["SI_SLOT"] = str(slot)
    PORT = 18000 + slot
    SESSION = P("logs", "pod_session.env" if slot == 0 else f"pod_session.{slot}.env")
    STOPFILE = P("logs", "pod_session.stop" if slot == 0 else f"pod_session.{slot}.stop")


class JobStop(Exception):
    """Stop playing (Ctrl-C, time nearly up, machine gone, failed check)."""


def clock():
    return time.strftime("%H:%M:%S", time.localtime())


def sha_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def mmss(secs):
    secs = max(0, int(secs))
    return f"{secs // 3600}:{secs % 3600 // 60:02d}:{secs % 60:02d}" if secs >= 3600 else f"{secs // 60}:{secs % 60:02d}"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def frozen_inputs_ok(study=1):
    bad = []
    files = [P("prereg", "inputs.sha256")] + ([P("prereg", "study2", "inputs.sha256")] if study == 2 else [])
    for line in (l for f in files for l in open(f)):
        if line.strip():
            want, name = line.split(None, 1)
            name = name.strip().lstrip("*")
            if not os.path.exists(P(name)) or sha_file(P(name)) != want:
                bad.append(name)
    return bad


def uncommitted():
    r = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "scripts", "config", "prereg"],
                       capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.strip()] if r.returncode == 0 else ["(git status failed)"]


# --- the panel -------------------------------------------------------------

class Panel:
    """Redraws a fixed block of lines in place (TTY) or prints changes (pipe)."""

    def __init__(self, model, stages, tty, preview):
        self.model, self.stages, self.tty, self.preview = model, stages, tty, preview
        self.lock = threading.RLock()
        self.stage = {s: "todo" for s in stages}
        self.now = "starting"
        self.log = []
        self.phase = None
        self.t_created = None
        self.lifetime_s = None
        self.price = None
        self.worst = None
        self.cost_final = None
        self.drawn = 0
        self.last_plain = None
        self.reply_secs = []
        self.stop = threading.Event()
        self.reset("", 0)

    def reset(self, phase, total, done=0):
        with self.lock:
            self.phase, self.games_total, self.games_done = phase, total, done
            self.voided = self.replies = self.invalid = self.reasoning = self.truncated = self.retries = 0
            self.game_secs = []

    def note(self, text):
        with self.lock:
            self.log.append(f"{clock()}  {text}")
            self.log = self.log[-7:]
            if not self.tty:
                print(f"  {clock()}  {text}", flush=True)
        self.draw()

    def set_stage(self, name, state):
        with self.lock:
            if name in self.stage:
                self.stage[name] = state
        self.draw()

    def set_now(self, text):
        with self.lock:
            self.now = text
        self.draw()

    def lines(self):
        w = max(60, min(104, shutil.get_terminal_size((104, 30)).columns - 1))
        icon = {"todo": "·", "run": "▶", "ok": "✓", "fail": "✗"}
        out = [f" scale-invariance · {self.model}{' · PREVIEW (nothing rented)' if self.preview else ''}",
               " " + "─" * (w - 2),
               " Stages   " + "  ".join(f"{icon[self.stage[s]]} {s}" for s in self.stages),
               f" Now      {self.now}"]
        n, tot = self.games_done, self.games_total
        if tot:
            width = 28
            fill = int(width * n / tot)
            eta = ""
            if self.game_secs and n < tot:
                recent = self.game_secs[-15:]
                eta = f"   ~{math.ceil(sum(recent) / len(recent) * (tot - n) / 60)} min left"
            out.append(f" {self.phase:<8} [{'█' * fill}{'░' * (width - fill)}]  {n} / {tot} games"
                       f"   {100 * n // tot}%{eta}")
        else:
            out.append(" Games    (not started)")
        if self.replies:
            pct = 100 * (self.replies - self.invalid) / self.replies
            if self.phase == "Pilot":
                ok = "on track" if pct >= 100 * GATE_VALID and not self.reasoning else "BELOW THE GATE"
                tail = f" · gate 5 needs ≥ 95% and no reasoning: {ok}"
            else:
                tail = ""
            out.append(f" Replies  {self.replies - self.invalid} of {self.replies} valid ({pct:.1f}%){tail}")
        else:
            out.append(" Replies  (none yet)")
        out.append(f" Format   reasoning leaked: {self.reasoning} · cut off at token cap: {self.truncated}"
                   f" · server retries: {self.retries} · voided games: {self.voided}")
        if self.reply_secs:
            last = self.reply_secs[-20:]
            out.append(f" Speed    {sum(last) / len(last):.1f} s per reply (last {len(last)})"
                       f" · {len(self.reply_secs)} model calls this session")
        if self.t_created:
            up = time.time() - self.t_created
            left = (self.lifetime_s - up) if self.lifetime_s else 0
            cost = self.cost_final if self.cost_final is not None else (self.price or 0) * up / 3600
            out.append(f" Machine  up {mmss(up)} · auto-delete in {max(0, int(left // 60))} min"
                       f" · cost so far ≈ ${cost:.2f} (worst case ${self.worst})")
        out.append(" " + "─" * (w - 2))
        out += [" " + l for l in self.log] or [" "]
        out.append(" Deal outcomes are hidden on purpose (pre-registration §8).")
        out.append(" Ctrl-C stops safely and deletes the machine.")
        return [l[:w] for l in out]

    def draw(self):
        with self.lock:
            if self.tty:
                ls = self.lines()
                buf = f"\033[{self.drawn}F" if self.drawn else ""
                buf += "".join(f"\033[2K{l}\n" for l in ls)
                if self.drawn > len(ls):
                    buf += "\033[2K\n" * (self.drawn - len(ls)) + f"\033[{self.drawn - len(ls)}F"
                try:
                    sys.stdout.write(buf)
                    sys.stdout.flush()
                except OSError:          # the terminal went away; keep working
                    self.tty = False
                self.drawn = max(len(ls), self.drawn)
            else:
                key = (self.phase, self.games_done, tuple(self.stage.values()))
                if key != self.last_plain:
                    self.last_plain = key
                    done = "/".join(s for s in self.stages if self.stage[s] == "ok")
                    print(f"  [{done}] {self.phase or ''} {self.games_done}/{self.games_total} games,"
                          f" {self.replies} replies, {self.invalid} invalid, {self.reasoning} reasoning",
                          flush=True)

    def ticker(self):
        while not self.stop.wait(1.0):
            if self.tty:
                self.draw()


# --- the machine (real launcher, or a fake for --preview) -------------------

class Machine:
    """Runs launch_pod.sh in its own process group and turns its output into
    panel stages. Ctrl-C reaches only this program, which then ends the
    session through the stop file, so the launcher's teardown runs to the end."""

    def __init__(self, panel, logpath, slot=None, manifest_tsv=None):
        """slot: None = this process's slot (use_slot); a number = that slot's stop file, session and tunnel,
        so one process can hold two machines (cross-play). manifest_tsv: for the preview's weights line."""
        self.panel, self.logpath = panel, logpath
        self.slot, self.manifest_tsv = slot, manifest_tsv
        self.ready = threading.Event()
        self.session_ready = False
        self.ended = threading.Event()
        self.failed = None
        self.weights_line = None
        self.deleted = None
        self.proc = None

    def parse(self, line):
        p = self.panel
        s = line.strip()
        if s.startswith("Creating pod"):
            p.set_stage("machine", "run"); p.set_now("asking RunPod for a machine")
        elif s.startswith("created; billing"):
            p.t_created = time.time(); p.note("machine created; billing has started")
        elif s.startswith("status:"):
            p.set_now(f"machine {s.split()[1].lower()}; waiting for it to boot (image download)")
        elif s.startswith("Waiting for the pod's SSH"):
            p.set_now("machine booting; waiting for its SSH key")
        elif s.startswith("host key pinned"):
            p.set_stage("machine", "ok"); p.set_stage("model files", "run")
            p.note("machine up; SSH identity pinned")
            p.set_now("downloading the model and checking every file's fingerprint")
        elif "WEIGHTS_OK" in s:
            self.weights_line = s
            p.set_stage("model files", "ok"); p.set_stage("vLLM", "run")
            n = re.search(r"files=(\d+) bytes=(\d+)", s)
            p.note(f"model files OK: {n.group(1)} files, {int(n.group(2)) / 1e9:.1f} GB, all match" if n else s)
            p.set_now("vLLM is loading the model onto the GPU")
        elif s.startswith("SESSION_READY"):
            p.set_stage("vLLM", "ok"); p.note("vLLM ready")
            self.session_ready = True
            self.ready.set()
        elif s.startswith("download:") or s.startswith("note: placed in") or s.startswith("data centers:"):
            p.note(s)
        elif s.startswith("STOP:"):
            self.failed = s[5:].strip(); p.note("launcher: " + self.failed)
        elif s.startswith("✗") or s.startswith("(connection"):
            p.note(s)
        elif s == "Teardown":
            p.set_stage("teardown", "run"); p.set_now("deleting the machine")
        elif s.startswith("pod logs copied"):
            p.note("pod logs copied to logs/")
        elif "estimated cost $" in s:
            m = re.search(r"estimated cost \$([0-9.]+)", s)
            if m:
                p.cost_final = float(m.group(1))
        elif "pod deleted; pods of ours left on the account: 0" in s:
            self.deleted = True; p.set_stage("teardown", "ok"); p.note("machine deleted; 0 of our pods left")
        elif "COULD NOT CONFIRM" in s or s.startswith("!!!"):
            self.deleted = False; p.set_stage("teardown", "fail"); p.note(s)

    def stopfile(self):
        if self.slot is None:
            return STOPFILE
        return P("logs", "pod_session.stop" if self.slot == 0 else f"pod_session.{self.slot}.stop")

    def env(self):
        return os.environ if self.slot is None else {**os.environ, "SI_SLOT": str(self.slot)}

    def start(self, args):
        try:
            os.remove(self.stopfile())
        except FileNotFoundError:
            pass
        log = open(self.logpath, "w")
        self.proc = subprocess.Popen(["bash", LAUNCHER] + args, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     text=True, bufsize=1, start_new_session=True, env=self.env())

        def reader():
            for line in self.proc.stdout:
                log.write(line); log.flush()
                self.parse(line)
            self.proc.wait()
            log.close()
            self.ended.set()
            self.ready.set()          # unblock waiters
        threading.Thread(target=reader, daemon=True).start()

    def alive(self):
        return not self.ended.is_set()

    def finish(self, timeout=900):
        """Ask for teardown and wait. Before SESSION_READY the launcher doesn't
        watch the stop file, so it gets TERM (its trap tears down)."""
        if self.proc and not self.ended.is_set():
            if self.session_ready:
                open(self.stopfile(), "w").close()
            else:
                os.killpg(self.proc.pid, signal.SIGTERM)
            if not self.ended.wait(timeout):
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.ended.wait(120)
        try:
            os.remove(self.stopfile())
        except FileNotFoundError:
            pass

    def ssh(self, cmd, timeout=150):
        r = subprocess.run(["bash", "-c", '. "$1" && si_load_session && si_ssh "$2"', "x", POD_SSH, cmd],
                           capture_output=True, text=True, timeout=timeout, env=self.env())
        return r.returncode, r.stdout


class FakeMachine(Machine):
    """--preview: the same stages and lines, from a script, in seconds."""

    def start(self, args):
        def script():
            seq = [(0.5, "Creating pod si-preview (once; never retried)"), (0.5, "  created; billing has started"),
                   (1.0, "  status: RUNNING (0 min)"), (1.0, "Waiting for the pod's SSH host key in its log"),
                   (1.5, "  host key pinned (SHA256:preview)"),
                   (2.0, "  WEIGHTS_OK files=81 bytes=30890049597 manifest="
                    + sha_file(self.manifest_tsv or MANIFEST_TSV)),
                   (2.0, "SESSION_READY: (preview)")]
            for d, line in seq:
                time.sleep(d)
                self.parse(line)
            while not self.stop_asked.is_set():
                time.sleep(0.2)
            for d, line in [(0.3, "Teardown"), (0.8, "  estimated cost $0.00 (preview)"),
                            (0.4, "  ✓ pod deleted; pods of ours left on the account: 0")]:
                time.sleep(d)
                self.parse(line)
            self.ended.set()
        self.stop_asked = threading.Event()
        threading.Thread(target=script, daemon=True).start()

    def finish(self, timeout=60):
        self.stop_asked.set()
        self.ended.wait(timeout)

    def ssh(self, cmd, timeout=150):
        return 1, ""


FAKE = {"msg_no": 1, "V": 100}


def fake_server(port=None, model=None):
    """--preview: a stand-in vLLM on 127.0.0.1:18000 (or `port`) with plausible replies, serving `model`
    (default PREVIEW_MODEL)."""
    port, model = port or PORT, model or PREVIEW_MODEL

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, obj):
            d = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(d)))
            self.end_headers()
            self.wfile.write(d)

        def do_GET(self):
            if self.path == "/version":
                return self.reply(200, {"version": "0.30.0"})
            if self.path == "/v1/models":
                return self.reply(200, {"data": [{"id": model, "max_model_len": 8192}]})
            return self.reply(200, {})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fam = family_of(model)
            if self.path == "/tokenize":
                return self.reply(200, {"count": 660, "tokens": [1, 2, 3]})
            if self.path == "/detokenize":
                return self.reply(200, {"prompt": "...preview prompt..." + TAILS[fam]})
            time.sleep(random.uniform(0.03, 0.15) + FAKE.get("delay", 0))
            if random.random() < 0.01:
                return self.reply(503, {"error": "preview: simulated server hiccup"})
            r, V, msg_no = random.random(), FAKE["V"], FAKE["msg_no"]
            try:        # read this request's own limit and message number (FAKE is shared across threads)
                user = req["messages"][-1]["content"]
                lim = re.search(r"Your private (?:floor|ceiling) is \$?([0-9][0-9,]*(?:\.[0-9]+)?)", user)
                V = float(lim.group(1).replace(",", "")) if lim else V
                n = re.search(r"This is message ([0-9]+) of", user)
                msg_no = int(n.group(1)) if n else msg_no
            except (KeyError, TypeError, IndexError, ValueError):
                pass
            if msg_no > 1 and r < 0.25:
                mv = {"action": "ACCEPT", "price": None, "message": "Deal."}
            elif msg_no > 3 and r < 0.28:
                mv = {"action": "WALKAWAY", "price": None, "message": "No deal."}
            else:
                mv = {"action": "OFFER", "price": int(V * random.uniform(0.8, 1.2)), "message": "How about this?"}
                try:        # a schema asking for a priced string (new-game "biz" notation) gets "$1,234"
                    sch = req["response_format"]["json_schema"]["schema"]["properties"]["price"]["anyOf"][0]
                    if sch.get("type") == "string":
                        mv["price"] = f"${mv['price']:,}"
                except (KeyError, TypeError, IndexError):
                    pass
            content = json.dumps(mv) if random.random() > 0.005 else "Sure! " + json.dumps(mv)
            return self.reply(200, {"id": f"chatcmpl-preview{random.randrange(10**9)}",
                                    "choices": [{"finish_reason": "stop",
                                                 "message": {"content": content, "reasoning": None}}],
                                    "usage": {"prompt_tokens": 660, "completion_tokens": random.randint(40, 90)}})

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# --- checks ----------------------------------------------------------------

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "frequency_penalty", "repetition_penalty", "max_tokens", "seed")


def logged_settings_match(machine, response_id, body):
    """Gate 3: the SamplingParams vLLM logged for this request equal what we sent."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", response_id or ""):
        return False, "odd response id"
    rc, out = machine.ssh(f"grep -F '{response_id}' /var/log/si/vllm.log | grep -o 'SamplingParams(.*' | head -1")
    if rc != 0 or "SamplingParams(" not in out:
        return False, "no request-log entry for the first reply"
    got = dict(re.findall(r"\b(" + "|".join(SAMPLING_KEYS) + r")=([-0-9.e+]+)", out))
    bad = [k for k in SAMPLING_KEYS if k not in got or float(got[k]) != float(body[k])]
    return (not bad), ("differs on " + ", ".join(bad)) if bad else "all 9 fields equal"


def server_checks(ctx):
    """Gates 1, 2, 4 and the watermark check. Returns (problems, info)."""
    machine, model, a = ctx["machine"], ctx["model"], ctx["args"]
    problems, info = [], {}
    if f"manifest={sha_file(MANIFEST_TSV)}" not in (machine.weights_line or ""):
        problems.append("weights were not checked against our manifest")
    st, raw, err = v.request(PORT, "GET", "/version", total_timeout=10)
    info["version"] = json.loads(raw).get("version") if st == 200 else None
    if info["version"] != "0.30.0":
        problems.append(f"vLLM version {info['version']!r}, not 0.30.0")
    st, raw, err = v.request(PORT, "GET", "/v1/models", total_timeout=10)
    served = [m.get("id") for m in json.loads(raw).get("data", [])] if st == 200 else []
    if served != [model]:
        problems.append(f"served models {served}")
    fx, settings = ctx["pilot_fx"], ctx["settings"]
    g0 = sorted(fx["games"], key=lambda g: g["run_order"])[0]
    msgs0 = ctx["render"](g0["opener"], g0["V"], g0["F"] if g0["opener"] == "seller" else g0["C"], [], 1,
                      fx["f_pct_range"], fx["c_pct_range"], settings["max_messages"]["value"],
                      settings["max_message_chars"]["value"])
    st, raw, err = v.request(PORT, "POST", "/tokenize", json.dumps(
        {"model": model, "messages": msgs0, "add_generation_prompt": True,
         "chat_template_kwargs": {"enable_thinking": False}}).encode(), total_timeout=30)
    toks = json.loads(raw).get("tokens", []) if st == 200 else []
    st, raw, err = v.request(PORT, "POST", "/detokenize",
                             json.dumps({"model": model, "tokens": toks}).encode(), total_timeout=30)
    info["rendered"] = json.loads(raw).get("prompt", "") if st == 200 else ""
    if not info["rendered"].endswith(TAILS[ctx["family"]]):
        problems.append("rendered prompt does not end with the thinking-off marker")
    info["nondefault"], info["hw"] = None, {}
    if not a.preview:
        rc, out = machine.ssh("grep -o 'non-default args: .*' /var/log/si/vllm.log | head -1")
        info["nondefault"] = out.strip() or None
        if not info["nondefault"]:
            problems.append("could not read vLLM's startup arguments")
        elif "watermark" in info["nondefault"].lower():
            problems.append("a watermark is configured on the server")
        rc, out = machine.ssh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
                              " | head -1; nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1;"
                              " grep -o 'pod: dc=[A-Z0-9-]*' /var/log/si/boot.log | tail -1")
        parts = out.strip().splitlines()
        info["hw"] = {"gpu": parts[0] if parts else None, "cuda": parts[1] if len(parts) > 1 else None,
                      "data_center": parts[2].split("=", 1)[1] if len(parts) > 2 else None}
    sess = {}
    if os.path.exists(SESSION):
        for k, val in re.findall(r"^([A-Z_]+)=(.*)$", open(SESSION).read(), re.M):
            sess[k] = shlex.split(val)[0] if val else ""
    info["pod"] = {"pod_id": sess.get("POD_ID"), "pod_name": sess.get("POD_NAME"), **info["hw"]}
    return problems, info


def session_record(ctx, info):
    git = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"vllm_version": info["version"], "server_args_logged": info["nondefault"],
            "hardware": info["pod"], "weights_check": (ctx["machine"].weights_line or "").split("] ", 1)[-1] or None,
            "git_commit": git.stdout.strip() or None, "git_uncommitted": uncommitted(),
            "inputs_sha256": {f: sha_file(P(f)) for f in CHECKED + (CHECKED_S2 if ctx["study"] == 2 else [])
                              + [f"config/manifests/{ctx['model']}.tsv"]}}


def make_manifest(ctx, kind, run_id, fixtures, games, info, extra):
    model, settings, pins, mp = ctx["model"], ctx["settings"], ctx["pins"], ctx["mp"]
    sess = session_record(ctx, info)
    return {
        "schema": "run_manifest/1", "run_id": run_id, "kind": kind, "preview": ctx["args"].preview,
        "study": ctx["study"], "prompts": "prereg/study2/prompts" if ctx["study"] == 2 else "prereg/prompts",
        "model": model, "repo": mp[model]["repo"], "revision": mp[model]["revision"],
        "image": "vllm/vllm-openai@" + pins["vllm_image"]["index_digest"],
        "vllm_version": sess["vllm_version"], "inputs_sha256": sess["inputs_sha256"],
        "git_commit": sess["git_commit"], "git_uncommitted": sess["git_uncommitted"],
        "server_args_logged": sess["server_args_logged"],
        "sampling_sent": {**settings["sampling"][model],
                          "presence_penalty": settings["presence_penalty"]["value"],
                          "frequency_penalty": settings["frequency_penalty"]["value"],
                          "repetition_penalty": settings["repetition_penalty"]["value"],
                          "max_tokens": settings["max_tokens"]["value"], "seed_rule": settings["seed"]["rule"],
                          "response_format": ge.response_schema(settings["max_message_chars"]["value"])},
        "rules": {"tries_per_call": rr.MAX_TRIES, "attempts_per_game": MAX_ATTEMPTS, "concurrency": 1,
                  "max_messages": settings["max_messages"]["value"],
                  "amendments": ["A1"] if ctx["study"] == 1 else [], "preregistration":
                  "prereg/PREREGISTRATION.md" if ctx["study"] == 1 else "prereg/study2/PREREGISTRATION.md"},
        "hardware": sess["hardware"], "weights_check": sess["weights_check"],
        "fixtures": fixtures, "games_planned": [g["game_id"] for g in games],
        "started_utc": rr.utc_now(), **extra,
    }


# --- one phase of games ----------------------------------------------------

def move_cut_off_lines(run_dir):
    """A crash mid-write can leave a last line without its newline; the hash
    chain then refuses the file. Move those bytes to a sidecar file (kept,
    never deleted) so the run can resume. Returns [(file, n_bytes, sha256)]."""
    moved = []
    for name in rr.RunWriter.FILES:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            data = fh.read()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            tail = data[cut:]
            side = f"{path}.cutoff-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            with open(side, "xb") as fh:
                fh.write(tail)
                fh.flush()
                os.fsync(fh.fileno())
            with open(path, "r+b") as fh:
                fh.truncate(cut)
                fh.flush()
                os.fsync(fh.fileno())
            moved.append((name, len(tail), rr.sha256_hex(tail)))
    return moved


def resume_state(run_dir, games):
    """For --resume: which games are finished, which attempts were cut off,
    and the queue: unstarted games in run order, then replays (E8, E9)."""
    recs = load_jsonl(os.path.join(run_dir, "games.jsonl"))
    calls = [c for c in load_jsonl(os.path.join(run_dir, "calls.jsonl"))
             if c.get("schema") == "call-request/1"]          # result lines carry no game_id
    events = load_jsonl(os.path.join(run_dir, "events.jsonl"))
    marked = {(e["game_id"], e["attempt"]) for e in events if e.get("kind") == "game_interrupted"}
    recorded = {(r["game_id"], r["attempt"]) for r in recs}
    finished = {r["game_id"] for r in recs if r["terminal"] != "INFRASTRUCTURE_VOID"}
    tried = {}
    for c in calls:
        tried[c["game_id"]] = max(tried.get(c["game_id"], 0), c["attempt"])
    orphans = sorted({(c["game_id"], c["attempt"]) for c in calls} - recorded - marked)
    fresh, replays, given_up = [], [], []
    for g in games:
        gid = g["game_id"]
        if gid in finished:
            continue
        n = tried.get(gid, 0)
        if n == 0:
            fresh.append((g, 1))
        elif n < MAX_ATTEMPTS:
            replays.append((g, n + 1))
        else:
            given_up.append(gid)
    return {"finished": len(finished), "orphans": orphans, "queue": fresh + replays,
            "given_up": given_up, "voided": sum(1 for r in recs if r["terminal"] == "INFRASTRUCTURE_VOID")}


def play_phase(ctx, writer, run_id, fx, queue, deadline_s, phase_name):
    """Play the queue. Returns 'complete'; raises JobStop to stop early."""
    panel, machine, a, settings, model = ctx["panel"], ctx["machine"], ctx["args"], ctx["settings"], ctx["model"]
    transport = v.make_chat_transport(PORT, total_timeout=120)
    base_call = rr.recording_call(writer, transport, run_id,
                                  sleep=(lambda s: time.sleep(0.1)) if a.preview else time.sleep)
    first = {}
    by_id = {g["game_id"]: g for g, _ in queue}

    def call(body, info):
        if ctx["interrupts"][0]:
            raise JobStop("stopped by you")
        if not machine.alive():
            raise JobStop("the machine session ended")
        FAKE["msg_no"], FAKE["V"] = info["msg_no"], by_id[info["game_id"]]["V"]
        panel.set_now(f"{phase_name.lower()}: {info['game_id']} (attempt {info['attempt']}),"
                      f" message {info['msg_no']} ({info['role']})")
        t0 = time.monotonic()
        reply = base_call(body, info)
        with panel.lock:
            panel.reply_secs.append(time.monotonic() - t0)
            panel.retries += int(reply["call_id"].rsplit("/t", 1)[1]) - 1
        if not first and reply.get("kind") == "model_output":
            first.update(rid=reply.get("response_id"))
            if not a.preview:
                ok, why = logged_settings_match(machine, first["rid"], body)
                writer.event("gate_effective_settings", ok=ok, detail=why, response_id=first["rid"])
                if not ok:
                    raise JobStop("check failed: logged settings " + why)
                panel.note(f"{phase_name}: settings vLLM logged for the first reply equal settings.json")
        return reply

    while queue:
        if ctx["interrupts"][0]:
            raise JobStop("stopped by you")
        if not a.preview and panel.t_created:
            left = panel.t_created + deadline_s - time.time()
            recent = panel.game_secs[-15:]
            avg = sum(recent) / len(recent) if recent else 120
            if left < max(300, 2 * avg):
                raise JobStop("machine time nearly up; no new game started")
        g, attempt = queue.pop(0)
        t0 = time.monotonic()
        try:
            rec = ge.play_game(g, model, settings, fx["f_pct_range"], fx["c_pct_range"], call, attempt=attempt,
                               render=None if ctx["study"] == 1 else rp2.render)
        except JobStop:
            writer.event("game_interrupted", game_id=g["game_id"], attempt=attempt, why="stopped mid-game")  # E9
            raise
        writer.game(rec)
        with panel.lock:
            d = rec["diagnostics"]
            panel.reasoning += sum(1 for x in d if x.startswith("REASONING_PRESENT"))
            panel.truncated += sum(1 for x in d if x.startswith("INVALID:") and "TRUNCATED" in x)
            if rec["terminal"] == "INFRASTRUCTURE_VOID":
                panel.voided += 1
                panel.replies += rec["n_messages"] - 1
                if attempt < MAX_ATTEMPTS:
                    queue.append((g, attempt + 1))
                    panel.note(f"{g['game_id']} voided (server trouble); replayed at the end")
                else:
                    panel.note(f"{g['game_id']} voided {MAX_ATTEMPTS} times; given up (E8)")
                    panel.games_done += 1
            else:
                panel.replies += rec["n_messages"]
                panel.invalid += int(rec["terminal"] == "PROTOCOL_FORFEIT")
                panel.games_done += 1
                panel.game_secs.append(time.monotonic() - t0)
        panel.draw()
    return "complete"


def copy_server_logs(ctx, run_dir):
    machine, panel = ctx["machine"], ctx["panel"]
    if ctx["args"].preview or not machine.alive() or ctx["interrupts"][0] >= 2:
        return
    panel.set_now("copying vLLM's logs into the run folder")
    os.makedirs(os.path.join(run_dir, "server"), exist_ok=True)
    rc, out = machine.ssh("cat /var/log/si/vllm.log", timeout=200)
    if rc == 0:
        with open(os.path.join(run_dir, "server", "vllm.log"), "w") as fh:
            fh.write(out)
    rc, out = machine.ssh("sed -E 's/public_ip=[^ ]*/public_ip=<masked>/; s/ssh_port=[0-9]*/ssh_port=<masked>/'"
                          " /var/log/si/boot.log", timeout=60)
    if rc == 0:
        with open(os.path.join(run_dir, "server", "boot.log"), "w") as fh:
            fh.write(out)


def finish_phase(ctx, writer, run_dir, status, complete):
    """Copy logs, close the run if complete, audit. Returns audit problems."""
    copy_server_logs(ctx, run_dir)
    p = ctx["panel"]
    writer.event("session_end", status=status)
    if complete:
        writer.close(games=p.games_done, voided=p.voided, replies=p.replies, invalid=p.invalid,
                     reasoning=p.reasoning)
    else:
        for c in (writer.calls, writer.games, writer.events):
            c.close()
    p.set_now("replaying every game from the raw records (audit)")
    problems, _ = replay_audit.audit(run_dir)
    return problems


def gate5(panel):
    pct = (panel.replies - panel.invalid) / panel.replies if panel.replies else 0
    return pct >= GATE_VALID and panel.reasoning == 0, pct


# --- the job ----------------------------------------------------------------

def dry_run(model, lifetime, extra=(), slot=None):
    env = os.environ if slot is None else {**os.environ, "SI_SLOT": str(slot)}
    r = subprocess.run(["bash", LAUNCHER, "--mode", "serve", "--model", model, "--lifetime-min", str(lifetime)]
                       + list(extra), capture_output=True, text=True, timeout=180, env=env)
    return r.returncode, r.stdout + r.stderr


def main(argv):
    global PREVIEW_MODEL, MANIFEST_TSV
    ap = argparse.ArgumentParser(description="Run one model's job (see the top of this file).")
    ap.add_argument("--model", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot-then-run", action="store_true")
    mode.add_argument("--resume", metavar="RUN_DIR")
    mode.add_argument("--pilot", action="store_true")
    ap.add_argument("--games", type=int, help="--pilot only: play the first N pilot games")
    ap.add_argument("--lifetime-min", type=int)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--study", type=int, choices=(1, 2), default=2,
                    help="2 (default since 2026-10-03): no appraisal; 1: the original registered prompt")
    ap.add_argument("--slot", type=int, default=int(os.environ.get("SI_SLOT", "0") or 0),
                    help="0-9: run several jobs at once, one slot each (default 0)")
    ap.add_argument("--preview-games", type=int, default=12, help=argparse.SUPPRESS)
    ap.add_argument("--data-root", help=argparse.SUPPRESS)       # preview/tests: keep runs here
    ap.add_argument("--preview-delay", type=float, default=0.0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    use_slot(a.slot)

    pins = json.load(open(P("config", "pins.json")))
    mp = {m["name"]: m for m in pins["models"]}
    if a.model not in mp:
        print("--model must be one of: " + ", ".join(mp))
        return 2
    if a.games and not a.pilot:
        print("--games only applies to --pilot (amendment A1 fixes the pilot at 60 and the real run at 180).")
        return 2
    if a.data_root and not a.preview:
        print("--data-root is for previews only.")
        return 2
    model, PREVIEW_MODEL = a.model, a.model
    FAKE["delay"] = a.preview_delay
    MANIFEST_TSV = P("config", "manifests", model + ".tsv")
    settings = json.load(open(P("prereg", "settings.json")))
    pilot_fx = json.load(open(P(PILOT_FX)))
    real_fx = json.load(open(P(REAL_FX)))
    if pilot_fx.get("seed_phrase") != "scale-invariance/fixtures/pilot" or pilot_fx.get("id_prefix") != "P-":
        print(f"{PILOT_FX} is not the pilot set")
        return 2
    pick = lambda fx, n: sorted(fx["games"], key=lambda g: g["run_order"])[:n]   # noqa: E731
    n_pilot = a.games or (min(a.preview_games, PILOT_GAMES) if a.preview else PILOT_GAMES)
    n_real = a.preview_games if a.preview else len(real_fx["games"])
    pilot_games = pick(pilot_fx, n_pilot)
    real_games = pick(real_fx, n_real)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    data_root = a.data_root or (tempfile.mkdtemp(prefix="si-preview-") if a.preview else P("data"))

    # What this job will do
    resume = None
    if a.resume:
        run_dir = os.path.abspath(a.resume)
        try:
            man = json.load(open(os.path.join(run_dir, "run_manifest.json")))
        except (OSError, ValueError):
            print(f"{a.resume}: not a run folder")
            return 2
        if (man.get("kind") != "real" or man.get("model") != model or bool(man.get("preview")) != a.preview
                or man.get("study", 1) != a.study):
            print(f"{a.resume}: not a {'preview ' if a.preview else ''}real run of {model}")
            return 2
        if os.path.exists(os.path.join(run_dir, "run_summary.json")):
            print(f"{a.resume}: this run is already complete")
            return 2
        cut_off = move_cut_off_lines(run_dir)
        planned = {gid: True for gid in man["games_planned"]}
        real_games = [g for g in pick(real_fx, len(real_fx["games"])) if g["game_id"] in planned]
        resume = resume_state(run_dir, real_games)
        stages = ("machine", "model files", "vLLM", "checks", "real run", "teardown")
        n_left = len(resume["queue"])
        plan = (f"Resume {os.path.basename(run_dir)}: {resume['finished']} of {len(real_games)} games done,"
                f" {n_left} to play" + (f" ({len(resume['orphans'])} cut-off game(s) replayed from the start)"
                                        if resume["orphans"] else ""))
        est_games = n_left
    elif a.pilot_then_run:
        stages = ("machine", "model files", "vLLM", "checks", "pilot", "gate 5", "real run", "teardown")
        plan = (f"Pilot ({n_pilot} pilot games), then gate 5 (automatic, format only), then the real run"
                f" ({n_real} games), one machine, one model download")
        est_games = n_pilot + n_real
    else:
        stages = ("machine", "model files", "vLLM", "checks", "pilot", "teardown")
        plan = f"Pilot only ({n_pilot} pilot games)"
        est_games = n_pilot
    lifetime = a.lifetime_min or min(360, 20 + math.ceil(0.8 * est_games))
    if lifetime < 30:
        lifetime = 30

    # 1. Safety before money
    real_mode = bool(a.pilot_then_run or a.resume)
    bad = frozen_inputs_ok(a.study)
    if bad:
        print("Frozen inputs differ from prereg/inputs.sha256: " + ", ".join(bad))
        return 3
    if real_mode and not a.preview and uncommitted():
        print("A real run needs committed code, so its record names the exact version. Uncommitted:")
        print("\n".join("  " + l for l in uncommitted()[:20]))
        return 3
    if a.preview:
        worst, price = "0.00", 0.0
        print(f"PREVIEW: fake machine and fake model server on this laptop. Nothing is rented.\n"
              f"  {plan}.\n  A real job would show its worst-case cost and ask you to type it.\n")
    else:
        rc, out = dry_run(model, lifetime)
        print("\n".join(l for l in out.rstrip().splitlines() if not l.startswith("Dry run:")).rstrip())
        m = re.search(r"--confirm-cost ([0-9.]+)", out)
        if rc != 0 or not m:
            print("\nNot launching (the launcher's checks above say why).")
            return 3
        worst = m.group(1)
        price = float(re.search(r"\$([0-9.]+)/hr", out).group(1))
        print(f"\nThis job: {plan}.")
        print(f"The machine deletes itself after {lifetime} min at the latest."
              + (" A real run that doesn't finish in time can be resumed." if real_mode else ""))
        try:
            typed = input(f"Type the worst-case cost ({worst}) to launch, or anything else to cancel: ").strip()
        except (EOFError, KeyboardInterrupt):
            typed = ""
        if typed.lstrip("$") != worst:
            print("Cancelled. Nothing was created.")
            return 0
        print()
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    tty = sys.stdout.isatty()
    panel = Panel(f"STUDY {a.study} · {model}" + (f" · slot {a.slot}" if a.slot else ""), stages, tty, a.preview)
    panel.lifetime_s, panel.price, panel.worst = lifetime * 60, price, worst
    os.makedirs(P("logs"), exist_ok=True)
    machine = (FakeMachine if a.preview else Machine)(panel, P("logs", f"job_{model}_{stamp}.launcher.out"))
    srv = fake_server() if a.preview else None
    if a.preview:
        panel.t_created = time.time()
    interrupts = [0]
    ctx = {"panel": panel, "machine": machine, "args": a, "model": model, "settings": settings, "pins": pins,
           "mp": mp, "pilot_fx": pilot_fx, "family": family_of(model),
           "interrupts": interrupts, "study": a.study, "render": rp.render if a.study == 1 else rp2.render}

    def on_int(signum, frame):
        interrupts[0] += 1
        panel.note("stop requested: finishing this move, then deleting the machine"
                   if interrupts[0] == 1 else "stop requested again: skipping the log copy")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_int)
    threading.Thread(target=panel.ticker, daemon=True).start()
    panel.draw()

    report = []          # lines for the end
    result = {"status": "not started"}
    open_writer, open_dir, open_complete = None, None, False
    try:
        machine.start(["--mode", "serve", "--model", model, "--lifetime-min", str(lifetime), "--confirm-cost", worst])
        while not machine.ready.wait(1.0):
            if interrupts[0]:
                raise JobStop("stopped by you before the games began")
        if not machine.alive() or machine.failed:
            raise JobStop(machine.failed or "the launcher ended before vLLM was ready")

        panel.set_stage("checks", "run"); panel.set_now("checking the server before any game")
        problems, info = server_checks(ctx)
        if problems:
            raise JobStop("check failed: " + "; ".join(problems))
        panel.set_stage("checks", "ok")
        panel.note("checks passed: weights, version 0.30.0, one model, thinking off"
                   + ("" if a.preview else ", no watermark"))
        deadline_s = lifetime * 60

        # Pilot
        pilot_id = None
        if a.pilot or a.pilot_then_run:
            pilot_id = f"{'s2-' if a.study == 2 else ''}pilot-{model}-{stamp}"
            pdir = os.path.join(data_root, "pilot", pilot_id)
            w = rr.RunWriter(pdir, make_manifest(ctx, "pilot", pilot_id, PILOT_FX, pilot_games, info, {}))
            os.makedirs(os.path.join(pdir, "server"))
            with open(os.path.join(pdir, "server", "rendered_prompt_first.txt"), "w") as fh:
                fh.write(info["rendered"])
            open_writer, open_dir, open_complete = w, pdir, False
            panel.set_stage("pilot", "run"); panel.reset("Pilot", len(pilot_games))
            panel.note(f"pilot started: {len(pilot_games)} pilot games")
            play_phase(ctx, w, pilot_id, pilot_fx, [(g, 1) for g in pilot_games], deadline_s, "Pilot")
            panel.set_stage("pilot", "ok")
            passed, pct = gate5(panel)
            w.event("gate5", passed=passed, valid_fraction=round(pct, 6), replies=panel.replies,
                    invalid=panel.invalid, reasoning=panel.reasoning)
            probs = finish_phase(ctx, w, pdir, "complete", True)
            open_writer = None
            pilot_summary = (f"{panel.replies - panel.invalid} of {panel.replies} replies valid ({100 * pct:.1f}%,"
                             f" needs ≥ 95%); reasoning in {panel.reasoning} (needs 0)")
            report.append(f"Pilot: {panel.games_done} of {len(pilot_games)} games. Gate 5"
                          f" {'PASSED' if passed else 'FAILED'}: {pilot_summary}.")
            report.append("  Pilot audit: " + ("every game replays exactly from the raw records." if not probs
                                               else f"{len(probs)} problem(s); first: {probs[0]}"))
            report.append(f"  Pilot data: {os.path.relpath(pdir, ROOT) if not a.preview else '(preview)'}")
            if probs:
                raise JobStop("the pilot's records did not pass the audit; not starting the real run")
            if a.pilot_then_run:
                panel.set_stage("gate 5", "ok" if passed else "fail")
                panel.note(f"gate 5 {'passed' if passed else 'FAILED'}: {pilot_summary}")
                if not passed:
                    raise JobStop("gate 5 failed; no real run (pre-registration §8: amend format, rerun the pilot)")
                gate_info = {"pilot_run_id": pilot_id, "gate5": {"passed": True, "valid_fraction": round(pct, 6),
                                                                  "reasoning": panel.reasoning}}

        # Real run
        if a.pilot_then_run or a.resume:
            if a.resume:
                w = rr.RunWriter(run_dir)
                w.event("session_start", **session_record(ctx, info))
                for name, nb, sh in cut_off:
                    w.event("cut_off_line_moved", file=name, bytes=nb, sha256=sh)
                for gid, att in resume["orphans"]:
                    w.event("game_interrupted", game_id=gid, attempt=att, why="found cut off on resume")
                queue = resume["queue"]
                rid = man["run_id"]
                panel.reset("Real run", len(real_games), resume["finished"] + len(resume["given_up"]))
                panel.voided = resume["voided"]
            else:
                rid = f"{'s2-' if a.study == 2 else ''}{model}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                run_dir = os.path.join(data_root, "runs", rid)
                w = rr.RunWriter(run_dir, make_manifest(ctx, "real", rid, REAL_FX, real_games, info, gate_info))
                queue = [(g, 1) for g in real_games]
                panel.reset("Real run", len(real_games))
            open_writer, open_dir, open_complete = w, run_dir, False
            panel.set_stage("real run", "run")
            panel.note(f"real run {'resumed' if a.resume else 'started'}: {len(queue)} games to play")
            play_phase(ctx, w, rid, real_fx, queue, deadline_s, "Real run")
            panel.set_stage("real run", "ok")
            open_complete = True
        result = {"status": "complete"}
    except JobStop as e:
        result = {"status": "stopped", "why": str(e)}
        panel.note("stopped: " + str(e))
    except Exception as e:                  # anything unexpected: still tear down
        result = {"status": "error", "why": f"{type(e).__name__}: {e}"}
        panel.note(f"error: {type(e).__name__}: {e}")
    finally:
        for s in ("pilot", "real run", "checks"):
            if panel.stage.get(s) == "run":
                panel.set_stage(s, "fail")
        audit_problems = None
        if open_writer is not None:
            audit_problems = finish_phase(ctx, open_writer, open_dir, result["status"], open_complete)
        panel.set_now("deleting the machine")
        machine.finish()
        if srv:
            srv.shutdown()
        panel.set_now("done" if result["status"] == "complete" else "stopped")
        panel.stop.set()
        panel.draw()

    # Report
    print()
    print(f"Job {result['status']}" + (f": {result['why']}" if result.get("why") else "") + ".")
    for line in report:
        print("  " + line)
    if open_dir and os.path.basename(os.path.dirname(open_dir)) == "runs":
        n_done = panel.games_done
        print(f"  Real run: {n_done} of {panel.games_total} games done"
              + ("; complete and closed." if open_complete else
                 f"; NOT complete. Resume it with:\n      python3 scripts/run_job.py --model {model}"
                 f" --resume {os.path.relpath(open_dir, ROOT) if not a.preview else open_dir}"
                 + (" --preview" + (f" --data-root {a.data_root}" if a.data_root else "") if a.preview else "")))
        if audit_problems is not None:
            print("  Real-run audit: " + ("every game so far replays exactly from the raw records." if not audit_problems
                                          else f"{len(audit_problems)} problem(s); first: {audit_problems[0]}"))
        print("  Real-run data: " + (os.path.relpath(open_dir, ROOT) if not a.preview else "(preview)"))
    elif open_dir and audit_problems:
        print(f"  Pilot audit: {len(audit_problems)} problem(s); first: {audit_problems[0]}")
    ok_teardown = machine.deleted is True
    if ok_teardown:
        cost = f"${panel.cost_final:.2f}" if panel.cost_final is not None else "see logs/budget.tsv"
        print(f"  Machine deleted; 0 of our pods left. Estimated cost {cost} (billing confirms later).")
    else:
        print("  !!! COULD NOT CONFIRM THE MACHINE IS GONE. Check the RunPod console now. !!!")
    if a.preview and not a.data_root:
        shutil.rmtree(data_root, ignore_errors=True)
        print("  Preview: nothing was rented; the practice run folders have been deleted.")
    if not ok_teardown:
        return 9
    return 0 if result["status"] == "complete" and not audit_problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
