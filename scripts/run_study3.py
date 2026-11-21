#!/usr/bin/env python3
"""run_study3.py: play Study 3's games (prereg/study3/PREREGISTRATION.md, DRAFT until frozen).

THIS SPENDS MONEY unless --preview.
  --backend api   the three API arms together, through OpenRouter (prompts only leave this laptop);
  --backend vllm  one open-weight model on one rented GPU pod (scripts/launch_pod.sh, deleted at the end).

    python3 scripts/run_study3.py --backend api --pilot --preview                 free rehearsal (fake server)
    python3 scripts/run_study3.py --backend api --pilot                           dry run: prints the cap, sends nothing
    python3 scripts/run_study3.py --backend api --pilot --confirm-cost X.XX       PAID format pilot (Q01-Q05)
    python3 scripts/run_study3.py --backend api --session 1 --confirm-cost key-limit    PAID registered session
    python3 scripts/run_study3.py --backend vllm --model gemma-4-31b-it --pilot --preview
    python3 scripts/run_study3.py --backend vllm --model gemma-4-31b-it --pilot   PAID: rents a pod (type the worst case)
    python3 scripts/run_study3.py --backend vllm --model gemma-4-31b-it --session 1 [--slot N]
    python3 scripts/run_study3.py --backend {api|vllm} [--model M] --resume data/runs/<s3 run>

WHAT A RUN IS. One backend's arms (all three API arms, or one vLLM model), and either the format pilot
(Q01-Q05, k = 1, five cells: A0 and A1 at $100 and $100M, and K-x1e8-o1e2) or one registered session
(G01-G40, all ten cells, samples 1-2 in session 1 and 3-4 in session 2). Every game of the run is
interleaved in one fixed shuffle. Games are played concurrently (--workers: API 8, vLLM 16 by default);
the number is recorded in the manifest. A game is scripts/study3.play (see that file): Study 2's engine,
unchanged, with the clamp's two scripted moves answered by the call function itself.

RETRIES AND VOIDS (as Studies 1-2 and Addendum C). A call is retried with the identical request (3 tries)
only when no completed reply came back. A game whose call still fails is INFRASTRUCTURE_VOID and is
replayed at once as attempt 2, then 3 (the attempt is in the seed); after 3 it is missing. On the API, a
4xx other than 408/429 stops the run. Invalid replies forfeit; nothing is repaired, except that text after
the first JSON object is ignored and counted (study3.first_json_only, decided 2026-10-14).

STOPPING. Ctrl-C once: no new game starts; games in flight finish. Twice: games in flight are marked
interrupted at their next call (E9) and replayed from the start on --resume. On vLLM no new game starts
when the pod's time is nearly up. A registered session that did not finish is completed with --resume.

GATES. A registered session refuses to start unless prereg/study3/inputs.sha256 is committed and every
file it lists matches (and, on vLLM, scripts/, config/ and prereg/ are committed). vLLM pods pass
run_job.py's server checks first (weights manifest, vLLM 0.30.0, one model, thinking off, no
watermark) and the first reply's logged sampling settings must equal the request. The pilot judges
format only (§6): no split, opening or midpoint number is computed from it.

Standard library only.
"""
import argparse
import json
import math
import os
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import openrouter_client as oc      # noqa: E402
import render_prompt_s2 as rp2      # noqa: E402
import run_addendum_b as rb         # noqa: E402
import run_addendum_c as rac        # noqa: E402
import run_job as rj                # noqa: E402
import run_records as rr            # noqa: E402
import study3 as S                  # noqa: E402
import vllm_client as v             # noqa: E402

WORKERS = {"api": 8, "vllm": 16}
PILOT_CAP = 2.0                     # dollars, API pilot only
ASSUMED_GAMES_PER_HOUR = 1000       # vLLM lifetime: Qwen's pilot measured 2,189/h at 16 concurrent (10-14)
SLOW_DOWNLOAD_BPS = 20e6            # bytes/s: below the slowest download seen (EUR-IS-2, ~26 MB/s, 10-14)
SETUP_FIXED_MIN = 10                # boot, SSH, file check, vLLM load, server checks
GATE_VALID = 0.95


def default_lifetime(model, n_games):
    """Pod lifetime in minutes: a setup allowance sized to the model's weights at a slow download speed, plus
    the games at a conservative rate. The pod deletes itself when the run ends, so this only sets the worst case."""
    with open(os.path.join(ROOT, "config", "manifests", model + ".tsv"), encoding="utf-8") as fh:
        size = sum(int(l.split("\t")[1]) for l in fh if l.strip() and not l.startswith("#"))
    setup = SETUP_FIXED_MIN + math.ceil(size / SLOW_DOWNLOAD_BPS / 60)
    return min(360, max(40, setup + math.ceil(60 * n_games / ASSUMED_GAMES_PER_HOUR)))

FROZEN_FILES = ("scripts/study3.py", "scripts/run_study3.py", "scripts/analyze_study3.py",
                "scripts/render_prompt_s3.py", "scripts/study3_clamp_values.py", "scripts/power_study3.py",
                "prereg/study3/PREREGISTRATION.md", "prereg/study3/clamp_values.json",
                "prereg/study3/prompts/system_a1.txt", "prereg/study3/prompts/clamp.txt", "tests/test_study3.py")
# Everything else a Study 3 game, launch or analysis executes or reads, hashed too (§10): the earlier hash lists
# don't cover all of it, and launch_pod.sh / run_job.py changed on 10-14 (data centers, download log).
REUSED_FILES = ("scripts/game_engine.py", "scripts/run_records.py", "scripts/render_prompt.py",
                "scripts/render_prompt_s2.py", "scripts/make_fixtures.py", "scripts/openrouter_client.py",
                "scripts/vllm_client.py", "scripts/run_addendum_b.py", "scripts/run_addendum_c.py",
                "scripts/run_job.py", "scripts/replay_audit.py", "scripts/analyze.py", "scripts/analyze_addendum_c.py",
                "scripts/launch_pod.sh", "scripts/lib/pod_ssh.sh", "scripts/lib/pack_interp.sh", "scripts/pod/boot.sh",
                "scripts/connections/03_pod.sh", "scripts/connections/04_vllm.sh",
                "prereg/study2/prompts/system.txt", "prereg/study2/prompts/user.txt",
                "prereg/study2/prompts/fragments.txt", "prereg/settings.json", "config/pins.json",
                "config/manifests/gemma-4-31b-it.tsv", "config/manifests/qwen3.8-27b-fp8.tsv")
FROZEN_FILES = FROZEN_FILES + REUSED_FILES


def inputs_sha256():
    out = {}
    for p in FROZEN_FILES + ("prereg/study3/inputs.sha256",):
        full = os.path.join(ROOT, p)
        if os.path.exists(full):
            with open(full, "rb") as fh:
                out[p] = rr.sha256_hex(fh.read())
    return out


def arms_info(backend, arms):
    if backend == "api":
        return {a: S.API_ARMS[a] for a in arms}
    with open(os.path.join(ROOT, "config", "pins.json"), encoding="utf-8") as fh:
        pins = {m["name"]: m for m in json.load(fh)["models"]}
    st = S.vllm_settings()
    return {a: {"repo": pins[a]["repo"], "revision": pins[a]["revision"], "sampling": st["sampling"][a],
                "presence_penalty": st["presence_penalty"]["value"],
                "frequency_penalty": st["frequency_penalty"]["value"],
                "repetition_penalty": st["repetition_penalty"]["value"],
                "thinking_off": "chat_template_kwargs {enable_thinking: false}"} for a in arms}


def worst_case_api(games):
    one = sum(rac.MAX_MSGS * rac.worst_per_call(S.API_ARMS[g["arm"]]) for g in games if g["backend"] == "api")
    return one, one * rb.MAX_TRIES * S.MAX_ATTEMPTS


def resume_queue(run_dir, games):
    """Finished games, cut-off attempts, and the queue [(game, next attempt)] for --resume."""
    recs = S._load_jsonl(os.path.join(run_dir, "games.jsonl"))
    calls = S._load_jsonl(os.path.join(run_dir, "calls.jsonl"))
    events = S._load_jsonl(os.path.join(run_dir, "events.jsonl"))
    finished = {r["game_id"] for r in recs if r["terminal"] != "INFRASTRUCTURE_VOID"}
    seen = {}
    for r in recs:
        seen[r["game_id"]] = max(seen.get(r["game_id"], 0), r["attempt"])
    marked = {(e["game_id"], e["attempt"]) for e in events if e.get("kind") == "game_interrupted"}
    for gid, a in marked:
        seen[gid] = max(seen.get(gid, 0), a)
    recorded = {(r["game_id"], r["attempt"]) for r in recs}
    tried = set()
    for c in calls:
        if c["schema"].endswith("call-request/1"):
            gid, a, _, _ = S.split_call_id(c["call_id"])
            tried.add((gid, a))
            seen[gid] = max(seen.get(gid, 0), a)
    orphans = sorted(tried - recorded - marked)
    q, given_up = [], []
    for g in games:
        if g["game_id"] in finished:
            continue
        n = seen.get(g["game_id"], 0)
        (q if n < S.MAX_ATTEMPTS else given_up).append((g, n + 1) if n < S.MAX_ATTEMPTS else g["game_id"])
    return {"finished": len(finished), "orphans": orphans, "queue": q, "given_up": given_up,
            "voids": sum(1 for r in recs if r["terminal"] == "INFRASTRUCTURE_VOID")}


def pilot_gate(records, backend, arms):
    """Format-only verdict per arm (§6). Also the throughput, which sets the vLLM sessions' lifetime."""
    out = {}
    for name in arms:
        rs = [r for r in records if r["arm"] == name]
        calls = [c for r in rs for c in r["calls"]]
        done = [c for c in calls if c["kind"] == "model_output"]
        forfeits = sum(1 for r in rs if r["terminal"] == "PROTOCOL_FORFEIT")
        errs = {}
        for r in rs:
            for d in r["diagnostics"]:
                if d.startswith("INVALID:"):
                    for e in d.split(":", 2)[2].split(","):
                        errs[e] = errs.get(e, 0) + 1
        frac = 1 - forfeits / len(done) if done else 0.0
        g = {"games": len(rs), "calls": len(calls), "replies": len(done), "valid_share": round(frac, 4),
             "valid_ok": frac >= GATE_VALID and len(done) == len(calls), "invalid_by_error": errs,
             "void_attempts": sum(1 for r in rs if r["terminal"] == "INFRASTRUCTURE_VOID"),
             "replies_with_reasoning_text": sum(1 for c in done if c["reasoning_text_present"]),
             "replies_with_trailing_text_ignored": sum(1 for c in done if c.get("trailing_text_ignored")),
             "clamp_games": sum(1 for r in rs if r["scripted_msgs"]),
             "mean_calls_per_game": round(statistics.fmean(len(r["calls"]) for r in rs), 2) if rs else None}
        if backend == "api":
            rtok = sum(1 for c in done if (c.get("reasoning_tokens") or 0) > 0)
            prov = sorted({str(c.get("provider_returned")) for c in done})
            g.update({"replies_with_reasoning_tokens": rtok, "providers_returned": prov,
                      "provider_ok": prov == [S.API_ARMS[name]["vendor"]],
                      "models_returned": sorted({f"{c.get('model_returned')} = {c.get('model_version')}" for c in done}),
                      "expected_version": S.API_ARMS[name]["canonical"],
                      "mean_cost_per_game": round(statistics.fmean(r["cost"] for r in rs), 6) if rs else None})
            g["reasoning_ok"] = name == "gemini-3.5-flash-lite" or rtok == 0
            g["pass"] = g["valid_ok"] and g["reasoning_ok"] and g["provider_ok"]
        else:
            g["reasoning_ok"] = g["replies_with_reasoning_text"] == 0
            g["pass"] = g["valid_ok"] and g["reasoning_ok"]
        out[name] = g
    return out


def main(argv, stdout=sys.stdout, client_factory=None, sleep=time.sleep):
    def say(*a):
        print(*a, file=stdout, flush=True)

    ap = argparse.ArgumentParser(description="Study 3 games (see the top of this file).")
    ap.add_argument("--backend", required=True, choices=("api", "vllm"))
    ap.add_argument("--model", help="vLLM only: " + ", ".join(S.VLLM_ARMS))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--session", type=int, choices=(1, 2))
    mode.add_argument("--resume", metavar="RUN_DIR")
    ap.add_argument("--preview", action="store_true", help="fake server(s) on this laptop; nothing is sent or rented")
    ap.add_argument("--confirm-cost", default=None,
                    help="API pilot: the printed cap; API session: key-limit; vLLM: the printed worst case")
    ap.add_argument("--spend-cap", type=float, default=None, help=f"API pilot only (default {PILOT_CAP:g})")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lifetime-min", type=int, default=None, help="vLLM pod lifetime (default from the game count)")
    ap.add_argument("--dc", default=None, help="vLLM: data centers, us|any|ID,ID (launcher default: us)")
    ap.add_argument("--slot", type=int, default=0, help="vLLM: 0-9, one pod per slot")
    ap.add_argument("--games", type=int, default=None, help=argparse.SUPPRESS)      # preview/tests: first N games
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)                  # preview/tests: data root
    a = ap.parse_args(argv)

    be = a.backend
    if be == "vllm":
        if a.model not in S.VLLM_ARMS:
            say("--backend vllm needs --model " + " or ".join(S.VLLM_ARMS))
            return 2
        arms = [a.model]
    else:
        if a.model:
            say("--model is for --backend vllm only")
            return 2
        arms = list(S.API_ARMS)
    workers = a.workers or WORKERS[be]
    if not 1 <= workers <= 64:
        say("--workers must be 1-64")
        return 2
    if a.games is not None and not a.preview:
        say("--games is for previews only")
        return 2
    if a.out and not a.preview:
        say("--out is for previews only")
        return 2

    # ---------------------------------------------------------------- what this run is
    resume = None
    if a.resume:
        run_dir = os.path.abspath(a.resume)
        try:
            with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as fh:
                man = json.load(fh)
        except (OSError, ValueError):
            say(f"{a.resume}: not a run folder")
            return 2
        if (not str(man.get("schema", "")).startswith("run_manifest") or man.get("study") != 3
                or man.get("backend") != be or sorted(man["arms"]) != sorted(arms)
                or bool(man.get("preview")) != a.preview):
            say(f"{a.resume}: not a {'preview ' if a.preview else ''}Study 3 {be} run of {', '.join(arms)}")
            return 2
        if os.path.exists(os.path.join(run_dir, "run_summary.json")):
            say(f"{a.resume}: this run is already complete")
            return 2
        pilot, session = man["kind"] == "study3_pilot", man.get("session")
        games = man["games"]
        clamp_values = man["clamp_values"]
        cut = rj.move_cut_off_lines(run_dir)
        resume = resume_queue(run_dir, games)
        work = resume["queue"]
        run_id = man["run_id"]
    else:
        pilot, session = a.pilot, a.session
        games, fmeta = S.build_games(arms, pilot=pilot, session=session)
        if a.games:
            games = games[:a.games]
        clamp_values = S.load_clamp_values()
        work = [(g, 1) for g in games]
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        who = "api" if be == "api" else a.model
        run_id = (f"s3-{who}-{'pilot' if pilot else f's{session}'}-{'preview-' if a.preview else ''}{stamp}")
    registered = not pilot and not a.preview
    for g, _ in work:            # every scripted price must be usable before anything is sent
        if g["condition"] == "K":
            S.clamp_prices(g, clamp_values)

    kind = "pilot" if pilot else f"session {session}"
    say(f"Study 3 {be} {kind}{' (RESUME)' if resume else ''}{' (PREVIEW: nothing sent or rented)' if a.preview else ''}")
    say(f"  arms: {', '.join(arms)}; {len(games)} games planned, {len(work)} to play; {workers} concurrent")
    if resume:
        say(f"  {resume['finished']} finished, {len(resume['given_up'])} given up after {S.MAX_ATTEMPTS} void attempts,"
            f" {len(resume['orphans'])} cut-off attempt(s) replayed from the start")
    cells = sorted({g["cell"] for g in games})
    say(f"  cells: {', '.join(cells)}")

    # ---------------------------------------------------------------- safety before money
    if registered:
        problem = S.check_frozen()
        if problem:
            say("Not running: " + problem)
            return 3
        if be == "vllm" and rj.uncommitted():
            say("Not running: a registered vLLM session needs scripts/, config/ and prereg/ committed:")
            say("\n".join("  " + l for l in rj.uncommitted()[:20]))
            return 3

    # ---------------------------------------------------------------- data folder and manifest
    if resume is None:
        data_root = a.out or (tempfile.mkdtemp(prefix="si-s3-preview-") if a.preview else
                              os.path.join(ROOT, "data", "pilot" if pilot else "runs"))
        run_dir = os.path.join(data_root, run_id)
    stop, hard, stop_why, lock = threading.Event(), threading.Event(), [], threading.Lock()
    interrupts = [0]

    def on_int(signum, frame):
        interrupts[0] += 1
        stop_why.append("stopped by you")
        stop.set()
        if interrupts[0] >= 2:
            hard.set()
    old = {s: signal.signal(s, on_int) for s in (signal.SIGINT, signal.SIGTERM)} \
        if threading.current_thread() is threading.main_thread() else {}

    try:
        if be == "api":
            return _run_api(a, say, arms, games, work, clamp_values, run_id, run_dir, resume, pilot, session,
                            registered, workers, stop, hard, stop_why, lock, client_factory, sleep,
                            None if resume else fmeta, cut if resume else [])
        return _run_vllm(a, say, arms, games, work, clamp_values, run_id, run_dir, resume, pilot, session,
                         registered, workers, stop, hard, stop_why, lock, interrupts,
                         None if resume else fmeta, cut if resume else [])
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def base_manifest(a, be, arms, games, clamp_values, run_id, pilot, session, registered, workers, fmeta):
    return {"schema": "run_manifest/1", "study": 3, "run_id": run_id,
            "kind": "study3_pilot" if pilot else "study3_session", "session": session, "preview": a.preview,
            "registered": registered, "prereg": "prereg/study3/PREREGISTRATION.md", "backend": be,
            "arms": arms_info(be, arms), "concurrency": workers, "max_tokens": S.MAX_TOKENS,
            "max_messages": S.MAX_MSGS, "max_message_chars": S.MAX_CHARS, "response_schema": S.response_schema(),
            "renderers": {"A0": "scripts/render_prompt_s2.py", "K": "scripts/render_prompt_s2.py",
                          "A1": "scripts/render_prompt_s3.py"},
            "seed_rule": f"sha256('{S.SEED_LABEL}|fixture|opener|sample|msg_no|attempt')[:8] & 0x7fffffff; "
                         "scale, appraisal and clamp cell are not inputs"
                         + ("; sent only where send_seed" if be == "api" else ""),
            "fixture_meta": fmeta, "clamp_values": clamp_values,
            "vllm_settings": S.vllm_settings() if be == "vllm" else None,
            "games": games, "games_planned": [g["game_id"] for g in games],
            "git": rb.git_state(), "inputs_sha256": inputs_sha256(), "started_utc": rr.utc_now()}


def play_queue(work, play_one, workers, stop, hard, stop_why, writer, say, total, can_start, extra=None):
    """Play [(game, first attempt)] with `workers` threads. Returns (records, counts)."""
    q = queue.Queue()
    for item in work:
        q.put(item)
    lock = threading.Lock()
    records, counts = [], {"games_done": 0, "games_missing": 0, "void_attempts": 0, "forfeits": 0}
    t_start, secs = time.monotonic(), []

    def worker():
        while not stop.is_set():
            why = can_start(secs)
            if why:
                with lock:
                    if not stop.is_set():
                        stop_why.append(why)
                        stop.set()
                return
            try:
                g, first = q.get_nowait()
            except queue.Empty:
                return
            for attempt in range(first, S.MAX_ATTEMPTS + 1):
                t0 = time.monotonic()
                try:
                    rec = play_one(g, attempt)
                except (S.Stop, rb.CapReached, rb.RequestRejected) as e:
                    with lock:
                        stop_why.append(str(e))
                    stop.set()
                    hard.set()
                    writer.event("game_interrupted", game_id=g["game_id"], attempt=attempt, why=str(e))
                    return
                writer.game(rec)
                with lock:
                    records.append(rec)
                    if rec["terminal"] == "INFRASTRUCTURE_VOID":
                        counts["void_attempts"] += 1
                        if attempt == S.MAX_ATTEMPTS:
                            counts["games_missing"] += 1
                        continue
                    secs.append(time.monotonic() - t0)
                    counts["games_done"] += 1
                    counts["forfeits"] += rec["terminal"] == "PROTOCOL_FORFEIT"
                    n = counts["games_done"] + counts["games_missing"]
                    if n % 500 == 0 or n == total:
                        rate = n / max(1e-9, time.monotonic() - t_start) * 3600
                        say(f"  {n}/{total} games done ({rate:.0f} games/h)" + (extra() if extra else ""))
                break

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(0.5)
    counts["wall_seconds"] = round(time.monotonic() - t_start, 1)
    return records, counts


def finish(say, w, run_dir, games, records, counts, resume, stop_why, pilot, be, arms, extra_close=None):
    planned = len(games)
    finished = counts["games_done"] + counts["games_missing"] + (
        (resume["finished"] + len(resume["given_up"])) if resume else 0)
    status = "complete" if finished == planned and not stop_why else "stopped"
    gate = pilot_gate(records, be, arms) if pilot else None
    if gate:
        w.event("pilot_format_gate", result=gate)
    if status == "complete":
        w.close(status=status, games=planned, **counts, **(extra_close or {}))
    else:
        w.event("session_end", status=status, why=stop_why[0] if stop_why else None, **counts)
        for c in (w.calls, w.games, w.events):
            c.close()
    problems, n = S.audit(run_dir)
    say(f"Study 3 run {status}" + (f": {stop_why[0]}" if stop_why else "") + ".")
    say(f"  {counts['games_done']} games scored this session ({counts['forfeits']} forfeits),"
        f" {counts['games_missing']} missing, {counts['void_attempts']} void attempts;"
        f" {counts['wall_seconds'] / 60:.1f} min wall clock")
    say("  Replay audit: " + ("every game reproduces exactly from the raw records." if not problems
                              else f"{len(problems)} problem(s); first: {problems[0]}"))
    if gate:
        for name, g in gate.items():
            say(f"  pilot gate {name:22s} {'PASS' if g['pass'] else 'FAIL'}: valid {g['valid_share']:.1%} "
                f"({g['replies']}/{g['calls']} calls answered), reasoning text {g['replies_with_reasoning_text']}"
                f", trailing text ignored {g['replies_with_trailing_text_ignored']}"
                + (f", reasoning tokens {g['replies_with_reasoning_tokens']}, providers {g['providers_returned']},"
                   f" models {g['models_returned']}, ${g['mean_cost_per_game']}/game" if be == "api" else "")
                + f"; {g['clamp_games']} clamp games; {g['mean_calls_per_game']} calls/game"
                + (f"; errors {g['invalid_by_error']}" if g["invalid_by_error"] else ""))
        say("  The pilot is format only (§6): no split, opening or midpoint number is computed from it.")
    say(f"  Data: {run_dir}")
    if status != "complete":
        say(f"  Resume with: python3 scripts/run_study3.py --backend {be}"
            + (f" --model {arms[0]}" if be == "vllm" else "") + f" --resume {run_dir}")
    return status, problems


# ---------------------------------------------------------------- API backend

def _run_api(a, say, arms, games, work, clamp_values, run_id, run_dir, resume, pilot, session, registered,
             workers, stop, hard, stop_why, lock, client_factory, sleep, fmeta, cut):
    todo = [g for g, _ in work]
    one, worst = worst_case_api(todo)
    if pilot:
        cap_amt = a.spend_cap if a.spend_cap is not None else PILOT_CAP
        if not 0 < cap_amt <= 25:
            say("bad --spend-cap (0-25)")
            return 2
        confirm = rb.money(min(worst, cap_amt))
        say(f"  worst case ${one:.2f} (${worst:.2f} with every retry and void replay); hard cap ${cap_amt:.2f}")
    else:
        if a.spend_cap is not None:
            say("--spend-cap is pilot-only; sessions use the OpenRouter key's cap")
            return 2
        cap_amt, confirm = math.inf, "key-limit"
        say(f"  worst case ${one:.2f} (${worst:.2f} with every retry and void replay); OpenRouter key cap only")
    if a.confirm_cost is None and not a.preview:
        say(f"Dry run: nothing sent. To run, add:  --confirm-cost {confirm}")
        return 0
    if not a.preview and a.confirm_cost.lstrip("$") != confirm:
        say(f"--confirm-cost {a.confirm_cost} does not match {confirm}. Nothing sent.")
        return 2

    srv = rac.fake_server(key_limit=1000.0) if a.preview else None
    try:
        client = client_factory(S.API_ARMS, srv) if client_factory else \
            oc.Client(S.API_ARMS, preview_port=srv.server_address[1] if srv else None)
        limit, usage, remaining = client.key_status()
    except (KeyError, RuntimeError) as e:
        say(f"Not running: {e}")
        return 3
    say(f"  OpenRouter key: limit {limit}, used {usage}, remaining {remaining}")
    if pilot and remaining is not None and remaining < min(worst, cap_amt):
        say(f"Not running: the key's remaining credit ({remaining}) is below the pilot cap.")
        return 3
    if not pilot and not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0
                             for x in (limit, remaining)):
        say("Not running: a session needs a capped OpenRouter key with positive remaining credit.")
        return 3

    if resume:
        w = rb.locked_writer(rr.RunWriter(run_dir))
        for name, nb, sh in cut:
            w.event("cut_off_line_moved", file=name, bytes=nb, sha256=sh)
        for gid, att in resume["orphans"]:
            w.event("game_interrupted", game_id=gid, attempt=att, why="found cut off on resume")
    else:
        man = base_manifest(a, "api", arms, games, clamp_values, run_id, pilot, session, registered, workers, fmeta)
        man.update({"spend_cap": None if math.isinf(cap_amt) else cap_amt,
                    "spend_policy": "client_and_key_caps" if pilot else "openrouter_key_cap_only",
                    "provider_block_rule": "only [provider], allow_fallbacks false, require_parameters true, "
                                           "data_collection deny",
                    "sampling": "vendor defaults (no temperature/top_p/top_k sent)"})
        w = rb.locked_writer(rr.RunWriter(run_dir, man))
    w.event("session_start", key_credit={"limit": limit, "usage": usage, "remaining": remaining},
            git=rb.git_state(), inputs_sha256=inputs_sha256(), workers=workers)
    cap = rb.SpendCap(cap_amt)
    api_call = rac.recording_call(w, client.chat_transport(), run_id, S.API_ARMS, cap, sleep=sleep)

    def send(body, key, game, info):
        if hard.is_set():
            raise S.Stop(stop_why[0] if stop_why else "stopped")
        return api_call(body, key, game["arm"])

    records, counts = play_queue(work, lambda g, att: S.play(g, att, send, clamp_values), workers, stop, hard,
                                 stop_why, w, say, len(work), lambda secs: None,
                                 extra=lambda: f"; spent ${cap.spent:.4f}")
    status, problems = finish(say, w, run_dir, games, records, counts, resume, stop_why, pilot, "api", arms,
                              extra_close={"spent": round(cap.spent, 6), "unpriced_replies": cap.unpriced})
    say(f"  Spent ${cap.spent:.4f} this session (from usage.cost).")
    if srv:
        assert all(h is None for h in srv.auth_seen), "preview sent an Authorization header"
        srv.shutdown()
        srv.server_close()
    return 0 if status == "complete" and not problems else 1


# ---------------------------------------------------------------- vLLM backend

def _run_vllm(a, say, arms, games, work, clamp_values, run_id, run_dir, resume, pilot, session, registered,
              workers, stop, hard, stop_why, lock, interrupts, fmeta, cut):
    model = arms[0]
    rj.use_slot(a.slot)
    rj.PREVIEW_MODEL = model
    rj.MANIFEST_TSV = os.path.join(ROOT, "config", "manifests", model + ".tsv")
    lifetime = a.lifetime_min or default_lifetime(model, len(work))
    dc = ["--dc", a.dc] if a.dc else []
    if a.preview:
        worst, price = "0.00", 0.0
    else:
        rc, out = rj.dry_run(model, lifetime, extra=dc, slot=a.slot)
        say("\n".join(l for l in out.rstrip().splitlines() if not l.startswith("Dry run:")).rstrip())
        m = re.search(r"--confirm-cost ([0-9.]+)", out)
        if rc != 0 or not m:
            say("Not launching (the launcher's checks above say why).")
            return 3
        worst = m.group(1)
        price = float(re.search(r"\$([0-9.]+)/hr", out).group(1))
        say(f"  The pod deletes itself after {lifetime} min at the latest; an unfinished session can be resumed.")
        typed = a.confirm_cost
        if typed is None:
            try:
                typed = input(f"Type the worst-case cost ({worst}) to launch, or anything else to cancel: ").strip()
            except (EOFError, KeyboardInterrupt):
                typed = ""
        if typed.lstrip("$") != worst:
            say("Cancelled. Nothing was created.")
            return 0
    if shutil.which("caffeinate") and not a.preview:
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    stages = ("machine", "model files", "vLLM", "checks", "games", "teardown")
    panel = rj.Panel(f"STUDY 3 · {model}" + (f" · slot {a.slot}" if a.slot else ""), stages, False, a.preview)
    panel.lifetime_s, panel.price, panel.worst = lifetime * 60, price, worst
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    logpath = os.path.join(tempfile.gettempdir() if a.preview else os.path.join(ROOT, "logs"),
                           f"study3_{model}_{stamp}.launcher.out")
    machine = (rj.FakeMachine if a.preview else rj.Machine)(panel, logpath, manifest_tsv=rj.MANIFEST_TSV)
    srv = rj.fake_server(port=rj.PORT, model=model) if a.preview else None
    if a.preview:
        panel.t_created = time.time()
    settings = S.vllm_settings()
    g0 = {"run_order": 1, "V": 100, "F": 83, "C": 118, "opener": "seller"}
    ctx = {"panel": panel, "machine": machine, "args": a, "model": model, "settings": settings,
           "pilot_fx": {"games": [g0], "f_pct_range": list(S.F_RANGE), "c_pct_range": list(S.C_RANGE)},
           "family": rj.family_of(model), "interrupts": interrupts, "study": 2, "render": rp2.render}
    w, status, problems, counts = None, "not started", None, None
    try:
        machine.start(["--mode", "serve", "--model", model, "--lifetime-min", str(lifetime), "--confirm-cost", worst]
                      + dc)
        while not machine.ready.wait(1.0):
            if stop.is_set():
                raise S.Stop("stopped by you before the games began")
        if not machine.alive() or machine.failed:
            raise S.Stop(machine.failed or "the launcher ended before vLLM was ready")
        panel.set_stage("checks", "run")
        server_problems, info = rj.server_checks(ctx)
        if server_problems:
            raise S.Stop("check failed: " + "; ".join(server_problems))
        panel.set_stage("checks", "ok")
        session_info = {"vllm_version": info["version"], "server_args_logged": info["nondefault"],
                        "hardware": info["pod"], "weights_check": (machine.weights_line or "").split("] ", 1)[-1] or None,
                        "git": rb.git_state(), "inputs_sha256": inputs_sha256(), "workers": workers}
        if resume:
            w = rb.locked_writer(rr.RunWriter(run_dir))
            for name, nb, sh in cut:
                w.event("cut_off_line_moved", file=name, bytes=nb, sha256=sh)
            for gid, att in resume["orphans"]:
                w.event("game_interrupted", game_id=gid, attempt=att, why="found cut off on resume")
        else:
            with open(os.path.join(ROOT, "config", "pins.json"), encoding="utf-8") as fh:
                pins = json.load(fh)
            man = base_manifest(a, "vllm", arms, games, clamp_values, run_id, pilot, session, registered, workers,
                                fmeta)
            man.update({"image": "vllm/vllm-openai@" + pins["vllm_image"]["index_digest"], "slot": a.slot,
                        "lifetime_min": lifetime})
            w = rb.locked_writer(rr.RunWriter(run_dir, man))
            os.makedirs(os.path.join(run_dir, "server"))
            with open(os.path.join(run_dir, "server", "rendered_prompt_first.txt"), "w") as fh:
                fh.write(info["rendered"])
        w.event("session_start", **session_info)
        transport = v.make_chat_transport(rj.PORT, total_timeout=120)
        vcall = rr.recording_call(w, transport, run_id, sleep=(lambda s: time.sleep(0.05)) if a.preview else time.sleep)
        first, first_lock = {}, threading.Lock()

        def send(body, key, game, info_):
            if hard.is_set():
                raise S.Stop(stop_why[0] if stop_why else "stopped")
            if not machine.alive():
                raise S.Stop("the machine session ended")
            reply = vcall(body, {"game_id": game["game_id"], "attempt": info_["attempt"],
                                 "msg_no": info_["msg_no"], "role": info_["role"]})
            if reply.get("kind") == "model_output" and not a.preview:
                with first_lock:
                    if not first:
                        first["rid"] = reply.get("response_id")
                        ok, why = rj.logged_settings_match(machine, first["rid"], body)
                        w.event("gate_effective_settings", ok=ok, detail=why, response_id=first["rid"])
                        if not ok:
                            stop_why.append("check failed: logged settings " + why)
                            stop.set()
                            hard.set()
                            raise S.Stop("check failed: logged settings " + why)
            return reply

        deadline = (panel.t_created or time.time()) + lifetime * 60

        def can_start(secs):
            if a.preview:
                return None
            if not machine.alive():
                return "the machine session ended"
            recent = secs[-50:]
            avg = sum(recent) / len(recent) if recent else 300
            if deadline - time.time() < max(600, 3 * avg):
                return "machine time nearly up; no new game started"
            return None

        panel.set_stage("games", "run")
        records, counts = play_queue(work, lambda g, att: S.play(g, att, send, clamp_values, settings), workers,
                                     stop, hard, stop_why, w, say, len(work), can_start)
        panel.set_stage("games", "ok" if not stop_why else "fail")
        rj.copy_server_logs(ctx, run_dir)
        status, problems = finish(say, w, run_dir, games, records, counts, resume, stop_why, pilot, "vllm", arms)
        w = None
    except S.Stop as e:
        say("stopped: " + str(e))
        status = "stopped"
        if w is not None:
            w.event("session_end", status="stopped", why=str(e))
            for c in (w.calls, w.games, w.events):
                c.close()
    finally:
        machine.finish()
        if srv:
            srv.shutdown()
            srv.server_close()
        panel.stop.set()
    if machine.deleted is True:
        cost = f"${panel.cost_final:.2f}" if panel.cost_final is not None else "see logs/budget.tsv"
        say(f"  Machine deleted; 0 of our pods left. Estimated cost {cost} (billing confirms later).")
    else:
        say("  !!! COULD NOT CONFIRM THE MACHINE IS GONE. Check the RunPod console now. !!!")
        return 9
    return 0 if status == "complete" and not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
