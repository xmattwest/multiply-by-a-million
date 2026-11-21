#!/usr/bin/env python3
"""run_probe.py: exploratory probe session (STORY §12). NOT a pre-registered run.

THIS SPENDS MONEY (except with --preview). One pod, several conditions,
games played concurrently (vLLM batches them), fresh exploration fixtures.

    python3 scripts/run_probe.py --model qwen3.8-27b-fp8 --name q1q3
    python3 scripts/run_probe.py --model qwen3.8-27b-fp8 --name q1q3 --preview     (free, fake server)

CONDITIONS (pick with --conditions, default A,N,LS,LB)
  A   Study 1 prompt: public appraisal and public ranges
  N   Study 2 prompt: no appraisal; each side knows only its own limit
  LS  N + the seller is told the buyer's ceiling (from message --leak-from on)
  LB  N + the buyer is told the seller's floor (from message --leak-from on)

What is reused from the confirmatory runner (scripts/run_job.py): the pod
launcher and teardown, the live panel, the server checks (weights hash,
vLLM 0.30.0, one model, thinking off, no watermark), the first-reply logged
settings check, and the hash-chained run records. What is lighter: no
pre-registration or commit lock (the git state is recorded instead), no
replay audit, games run concurrently, data goes to data/explore/.
Fixtures: make_fixtures.build() with the phrase scale-invariance/fixtures/explore-v1,
renamed X01.., so they share nothing with the confirmatory fixtures.
Standard library only.
"""
import argparse
import json
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import game_engine as ge          # noqa: E402
import make_fixtures as mf        # noqa: E402
import render_prompt as rp        # noqa: E402
import render_prompt_s2 as rp2    # noqa: E402
import run_job as J               # noqa: E402
import run_records as rr          # noqa: E402
import vllm_client as v           # noqa: E402

P = J.P
PHRASE = "scale-invariance/fixtures/explore-v1"
CONDITIONS = {
    "A": ("Study 1 prompt (public appraisal and ranges)", "s1", None),
    "N": ("Study 2 prompt (no appraisal)", "s2", None),
    "LS": ("Study 2 prompt; the seller is told the buyer's ceiling", "s2", "seller"),
    "LB": ("Study 2 prompt; the buyer is told the seller's floor", "s2", "buyer"),
}
LEAK_TEXT = {
    "seller": "You have also obtained reliable information about the buyer: the buyer's private ceiling is {X}."
              " The buyer does not know that you know this.",
    "buyer": "You have also obtained reliable information about the seller: the seller's private floor is {X}."
             " The seller does not know that you know this.",
}
ANCHOR = "does not know it.\n"


def make_render(cond, game, leak_from):
    base = rp.render if CONDITIONS[cond][1] == "s1" else rp2.render
    informed = CONDITIONS[cond][2]
    if informed is None:
        return base
    other_limit = game["C"] if informed == "seller" else game["F"]
    line = LEAK_TEXT[informed].replace("{X}", rp.money(other_limit))

    def render(role, V, limit, history, msg_no, *rest, **kw):
        msgs = base(role, V, limit, history, msg_no, *rest, **kw)
        if role == informed and msg_no >= leak_from:
            u = msgs[-1]
            assert u["role"] == "user" and u["content"].count(ANCHOR) == 1, "leak anchor not found"
            msgs[-1] = {**u, "content": u["content"].replace(ANCHOR, ANCHOR + line + "\n", 1)}
        return msgs
    return render


def build_games(conds, n_fix, scales):
    doc = mf.build(PHRASE)
    fx = {f["id"]: f"X{f['id'][1:]}" for f in doc["fixtures"][:n_fix]}
    games = []
    for g in doc["games"]:
        if g["fixture"] in fx and g["scale"] in scales:
            for c in conds:
                gid = f"{fx[g['fixture']]}-{g['scale']}-{g['opener'][0].upper()}-{c}"
                games.append({**g, "fixture": fx[g["fixture"]], "game_id": gid, "condition": c})
    rng = random.Random(rr.sha256_hex(PHRASE.encode()))      # interleave conditions in time
    rng.shuffle(games)
    for i, g in enumerate(games, 1):
        g["run_order"], g["block"] = i, None
    meta = {"seed_phrase": PHRASE, "f_pct_range": doc["f_pct_range"], "c_pct_range": doc["c_pct_range"],
            "fixtures": [{**f, "id": fx[f["id"]]} for f in doc["fixtures"][:n_fix]]}
    return meta, games


def locked_writer(w):
    lock = threading.Lock()
    for ch in (w.calls, w.games, w.events):
        orig = ch.append

        def append(obj, _orig=orig):
            with lock:
                return _orig(obj)
        ch.append = append
    return w


def main(argv):
    ap = argparse.ArgumentParser(description="Exploratory probe session (see the top of this file).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="short probe label, e.g. q1q3")
    ap.add_argument("--conditions", default="A,N,LS,LB")
    ap.add_argument("--fixtures", type=int, default=12)
    ap.add_argument("--scales", default="x1e2,x1e8")
    ap.add_argument("--leak-from", type=int, default=1, help="message number from which the informed side knows")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lifetime-min", type=int, default=60)
    ap.add_argument("--slot", type=int, default=int(os.environ.get("SI_SLOT", "0") or 0),
                    help="0-9: run alongside other jobs, one slot each (default 0)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview-games", type=int, default=16, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    J.use_slot(a.slot)

    pins = json.load(open(P("config", "pins.json")))
    mp = {m["name"]: m for m in pins["models"]}
    conds = a.conditions.split(",")
    scales = a.scales.split(",")
    if a.model not in mp or any(c not in CONDITIONS for c in conds) or not re.fullmatch(r"[a-z0-9-]{1,20}", a.name):
        print("bad --model, --conditions or --name")
        return 2
    model = a.model
    J.PREVIEW_MODEL, J.MANIFEST_TSV = model, P("config", "manifests", model + ".tsv")
    settings = json.load(open(P("prereg", "settings.json")))
    meta, games = build_games(conds, a.fixtures, scales)
    if a.preview:
        games = games[:a.preview_games]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    data_root = tempfile.mkdtemp(prefix="si-probe-preview-") if a.preview else P("data", "explore")
    run_id = f"X-{a.name}-{model}-{stamp}"
    run_dir = os.path.join(data_root, run_id)
    plan = (f"Probe {a.name}: conditions {','.join(conds)}, {a.fixtures} fixtures x 2 openers x {len(scales)}"
            f" scales = {len(games)} games, {a.workers} at a time; leak from message {a.leak_from}")

    # Cost confirmation (same launcher dry run as the confirmatory jobs)
    if a.preview:
        worst, price = "0.00", 0.0
        print(f"PREVIEW: fake machine and server on this laptop. Nothing is rented.\n  {plan}.\n")
    else:
        if J.uncommitted():
            print("Note: uncommitted changes are recorded in the manifest (probe runs don't require a commit).")
        rc, out = J.dry_run(model, a.lifetime_min)
        print("\n".join(l for l in out.rstrip().splitlines() if not l.startswith("Dry run:")).rstrip())
        m = re.search(r"--confirm-cost ([0-9.]+)", out)
        if rc != 0 or not m:
            print("\nNot launching (the launcher's checks above say why).")
            return 3
        worst, price = m.group(1), float(re.search(r"\$([0-9.]+)/hr", out).group(1))
        print(f"\nThis probe: {plan}.\nThe machine deletes itself after {a.lifetime_min} min at the latest.")
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

    panel = J.Panel(f"PROBE {a.name} · {model}" + (f" · slot {a.slot}" if a.slot else ""), ("machine", "model files", "vLLM", "checks", "probe", "teardown"),
                    sys.stdout.isatty(), a.preview)
    panel.lifetime_s, panel.price, panel.worst = a.lifetime_min * 60, price, worst
    os.makedirs(P("logs"), exist_ok=True)
    machine = (J.FakeMachine if a.preview else J.Machine)(panel, P("logs", f"probe_{a.name}_{stamp}.launcher.out"))
    srv = J.fake_server() if a.preview else None
    if a.preview:
        panel.t_created = time.time()
    interrupts = [0]
    probe_fx = {"games": games, "f_pct_range": meta["f_pct_range"], "c_pct_range": meta["c_pct_range"]}
    ctx = {"panel": panel, "machine": machine, "args": a, "model": model, "settings": settings, "pins": pins,
           "mp": mp, "pilot_fx": probe_fx, "family": J.family_of(model),
           "interrupts": interrupts, "study": 2, "render": rp2.render}

    def on_int(signum, frame):
        interrupts[0] += 1
        panel.note("stop requested: finishing current moves, then deleting the machine"
                   if interrupts[0] == 1 else "stop requested again: skipping the log copy")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_int)
    threading.Thread(target=panel.ticker, daemon=True).start()
    panel.draw()

    result, w = {"status": "not started"}, None
    try:
        machine.start(["--mode", "serve", "--model", model, "--lifetime-min", str(a.lifetime_min),
                       "--confirm-cost", worst])
        while not machine.ready.wait(1.0):
            if interrupts[0]:
                raise J.JobStop("stopped by you before the games began")
        if not machine.alive() or machine.failed:
            raise J.JobStop(machine.failed or "the launcher ended before vLLM was ready")
        panel.set_stage("checks", "run"); panel.set_now("checking the server before any game")
        problems, info = J.server_checks(ctx)
        if problems:
            raise J.JobStop("check failed: " + "; ".join(problems))
        panel.set_stage("checks", "ok")
        panel.note("checks passed: weights, version 0.30.0, one model, thinking off"
                   + ("" if a.preview else ", no watermark"))

        sess = J.session_record(ctx, info)
        manifest = {
            "schema": "run_manifest/1", "run_id": run_id, "kind": "probe", "preview": a.preview,
            "study": "probe", "probe": a.name, "not_preregistered": True, "model": model,
            "repo": mp[model]["repo"], "revision": mp[model]["revision"],
            "image": "vllm/vllm-openai@" + pins["vllm_image"]["index_digest"], **sess,
            "conditions": {c: CONDITIONS[c][0] for c in conds}, "leak_text": LEAK_TEXT, "leak_from_msg": a.leak_from,
            "workers": a.workers, "scales": scales, "fixture_meta": meta,
            "games_planned": [g["game_id"] for g in games], "started_utc": rr.utc_now(),
        }
        w = locked_writer(rr.RunWriter(run_dir, manifest))
        with open(os.path.join(run_dir, "games_planned.json"), "w") as fh:
            json.dump(games, fh, indent=0)
        panel.set_stage("probe", "run"); panel.reset("Probe", len(games))
        panel.note(f"probe started: {len(games)} games, {a.workers} at a time")

        transport = v.make_chat_transport(J.PORT, total_timeout=300)
        base_call = rr.recording_call(w, transport, run_id,
                                      sleep=(lambda s: time.sleep(0.1)) if a.preview else time.sleep)
        first, first_lock, stop = {}, threading.Lock(), threading.Event()
        by_id = {g["game_id"]: g for g in games}

        def call(body, info_):
            if interrupts[0] or stop.is_set():
                raise J.JobStop("stopped")
            if not machine.alive():
                raise J.JobStop("the machine session ended")
            J.FAKE["msg_no"], J.FAKE["V"] = info_["msg_no"], by_id[info_["game_id"]]["V"]
            t0 = time.monotonic()
            reply = base_call(body, info_)
            with panel.lock:
                panel.reply_secs.append(time.monotonic() - t0)
                panel.retries += int(reply["call_id"].rsplit("/t", 1)[1]) - 1
            if reply.get("kind") == "model_output" and not a.preview:
                with first_lock:
                    if not first:
                        first["rid"] = reply.get("response_id")
                        ok, why = J.logged_settings_match(machine, first["rid"], body)
                        w.event("gate_effective_settings", ok=ok, detail=why, response_id=first["rid"])
                        if not ok:
                            stop.set()
                            raise J.JobStop("check failed: logged settings " + why)
                        panel.note("settings vLLM logged for the first reply equal settings.json")
            return reply

        q = queue.Queue()
        for g in games:
            q.put((g, 1))
        stop_why = []

        def worker():
            while not stop.is_set() and not interrupts[0]:
                if not a.preview and panel.t_created:
                    recent = panel.game_secs[-15:]
                    avg = sum(recent) / len(recent) if recent else 120
                    if panel.t_created + a.lifetime_min * 60 - time.time() < max(300, 2 * avg):
                        stop_why.append("machine time nearly up; no new game started")
                        stop.set()
                        return
                try:
                    g, attempt = q.get_nowait()
                except queue.Empty:
                    return
                t0 = time.monotonic()
                try:
                    rec = ge.play_game(g, model, settings, meta["f_pct_range"], meta["c_pct_range"], call,
                                       attempt=attempt, render=make_render(g["condition"], g, a.leak_from))
                except J.JobStop as e:
                    w.event("game_interrupted", game_id=g["game_id"], attempt=attempt, why=str(e))
                    stop_why.append(str(e))
                    stop.set()
                    return
                rec["condition"] = g["condition"]
                w.game(rec)
                with panel.lock:
                    d = rec["diagnostics"]
                    panel.reasoning += sum(1 for x in d if x.startswith("REASONING_PRESENT"))
                    panel.truncated += sum(1 for x in d if x.startswith("INVALID:") and "TRUNCATED" in x)
                    if rec["terminal"] == "INFRASTRUCTURE_VOID":
                        panel.voided += 1
                        panel.replies += rec["n_messages"] - 1
                        if attempt < J.MAX_ATTEMPTS:
                            q.put((g, attempt + 1))
                        else:
                            panel.games_done += 1
                    else:
                        panel.replies += rec["n_messages"]
                        panel.invalid += int(rec["terminal"] == "PROTOCOL_FORFEIT")
                        panel.games_done += 1
                        panel.game_secs.append((time.monotonic() - t0) / max(1, a.workers))
                panel.set_now(f"probe: {panel.games_done} of {len(games)} games done")
                panel.draw()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(1.0)
        if interrupts[0]:
            raise J.JobStop("stopped by you")
        if stop_why:
            raise J.JobStop(stop_why[0])
        panel.set_stage("probe", "ok")
        result = {"status": "complete"}
    except J.JobStop as e:
        result = {"status": "stopped", "why": str(e)}
        panel.note("stopped: " + str(e))
    except Exception as e:
        result = {"status": "error", "why": f"{type(e).__name__}: {e}"}
        panel.note(f"error: {type(e).__name__}: {e}")
    finally:
        if panel.stage.get("probe") == "run":
            panel.set_stage("probe", "fail")
        if w is not None:
            J.copy_server_logs(ctx, run_dir)
            w.close(status=result["status"], games_done=panel.games_done, replies=panel.replies,
                    forfeits=panel.invalid, voided=panel.voided, reasoning=panel.reasoning)
        panel.set_now("deleting the machine")
        machine.finish()
        if srv:
            srv.shutdown()
        panel.set_now("done" if result["status"] == "complete" else "stopped")
        panel.stop.set()
        panel.draw()

    print()
    print(f"Probe {result['status']}" + (f": {result['why']}" if result.get("why") else "") + ".")
    print(f"  {panel.games_done} of {len(games)} games; {panel.replies - panel.invalid} of {panel.replies} replies"
          f" valid; voided {panel.voided}; reasoning {panel.reasoning}")
    print("  Data: " + (os.path.relpath(run_dir, J.ROOT) if not a.preview else run_dir + " (preview)"))
    ok = machine.deleted is True
    if ok:
        cost = f"${panel.cost_final:.2f}" if panel.cost_final is not None else "see logs/budget.tsv"
        print(f"  Machine deleted; 0 of our pods left. Estimated cost {cost}.")
    else:
        print("  !!! COULD NOT CONFIRM THE MACHINE IS GONE. Check the RunPod console now. !!!")
    if a.preview:
        print("  Preview folder kept for inspection: " + run_dir)
    return 9 if not ok else (0 if result["status"] == "complete" else 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
