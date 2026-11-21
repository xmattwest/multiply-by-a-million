#!/usr/bin/env python3
"""run_digits_probe.py: the digits x stakes first-move probe (STORY §12, Q2). NOT pre-registered.

THIS SPENDS MONEY (except with --preview). One pod, first moves only.

    python3 scripts/run_digits_probe.py --model gemma-4-31b-it --preview      (free: fake server)
    python3 scripts/run_digits_probe.py --model gemma-4-31b-it                (paid)

WHY. In Studies 1-2 the number of digits and the stakes always moved together
($100 = 3 digits, $100M = 9 digits). This probe pulls them apart on the
opener's first offer (Addendum A's HA gap), the move that drives Gemma 31B's
shift. Study 2 prompt (no appraisal), message 1, no history, both roles.

CONDITIONS. b is the fixture's limit at V = 100 (e.g. floor 81, ceiling 124).
  code  limit as written       stakes      reply unit           what it isolates
  D2    "81 dollars"           $81         dollars              = Study 2 at $100
  D5    "81000 dollars"        $81K        dollars              = Study 2 at $100K
  D8    "81000000 dollars"     $81M        dollars              = Study 2 at $100M
  C8    "81,000,000 dollars"   $81M        dollars              same stakes, commas
  W8    "81 million dollars"   $81M        dollars              same stakes, short digits in
  U5    "81 thousand dollars"  $81K        thousands of dollars same digits in and out, mid stakes
  U8    "81 million dollars"   $81M        millions of dollars  same digits in and out, high stakes
  K2    "8100 cents"           $81         cents                more digits, same small stakes
  S2    "$81"                  $81         dollars              dollar sign, small
  SN    "$81000000"            $81M        dollars              dollar sign, no commas
  S8    "$81,000,000"          $81M        dollars              dollar sign and commas
  SM    "$81 million"          $81M        dollars              dollar sign and a word
  X2    "81.00 dollars"        $81         dollars              cents shown (round-number test)
  The default set is the first run (D2..K2); the follow-up is
  --conditions D2,D8,S2,SN,S8,SM,X2 (D2 and D8 re-run as same-session anchors).
  D2/D5/D8 is the gradient the studies used (digits and stakes together);
  D2/U5/U8 is a stakes-only gradient (the digits the model reads and writes
  stay those of D2). For U5, U8 and K2 the system message's two price lines
  name the unit ("a whole number of millions of dollars"); everything else
  is the frozen Study 2 prompt word for word.

DESIGN. --fixtures fresh fixtures (phrase scale-invariance/fixtures/explore-v2,
IDs Y01..; never used before) x 2 roles x 8 conditions x --samples samples.
The seed depends on (fixture, role, sample), not the condition, so each
condition gets the same random draws (common random numbers). Calls are
shuffled so conditions interleave in time. Outcome per call: the first move
(OFFER price or WALKAWAY) and the gap from the own limit, in the reply's
unit: seller P/L - 1, buyer 1 - P/L.

Reused from scripts/run_job.py: launcher, teardown, panel, server checks
(weights hash, vLLM 0.30.0, one model, thinking off, no watermark), the
first-reply logged-settings check, hash-chained records. Data goes to
data/explore/X-digits-<model>-<stamp>/; one record per call in games.jsonl.
Analyse with scripts/analyze_digits_probe.py. Standard library only.
"""
import argparse
import hashlib
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
PHRASE = "scale-invariance/fixtures/explore-v2"
PROBE = "digits"
# code: (limit text from b, stakes multiplier in dollars, reply unit or None = dollars,
#        reply units per base unit b, description)
CONDITIONS = {
    "D2": (lambda b: f"{b} dollars", 1, None, 1, "digits, $100 scale (Study 2 wording)"),
    "D5": (lambda b: f"{b * 10**3} dollars", 10**3, None, 10**3, "digits, $100K scale (Study 2 wording)"),
    "D8": (lambda b: f"{b * 10**6} dollars", 10**6, None, 10**6, "digits, $100M scale (Study 2 wording)"),
    "C8": (lambda b: f"{b * 10**6:,} dollars", 10**6, None, 10**6, "$100M scale written with commas"),
    "W8": (lambda b: f"{b} million dollars", 10**6, None, 10**6, "$100M scale in words; reply in dollars"),
    "U5": (lambda b: f"{b} thousand dollars", 10**3, "thousands of dollars", 1,
           "$100K scale in words; reply in thousands"),
    "U8": (lambda b: f"{b} million dollars", 10**6, "millions of dollars", 1,
           "$100M scale in words; reply in millions"),
    "K2": (lambda b: f"{b * 100} cents", 1, "cents", 100, "$100 scale in cents; reply in cents"),
    # Follow-up (2026-10-03 22:25): the dollar sign and decimals; replies stay plain whole dollars.
    "S2": (lambda b: f"${b}", 1, None, 1, "$100 scale with a dollar sign"),
    "SN": (lambda b: f"${b * 10**6}", 10**6, None, 10**6, "$100M scale with a dollar sign, no commas"),
    "S8": (lambda b: f"${b * 10**6:,}", 10**6, None, 10**6, "$100M scale with a dollar sign and commas"),
    "SM": (lambda b: f"${b} million", 10**6, None, 10**6, "$100M scale as '$81 million'; reply in dollars"),
    "X2": (lambda b: f"{b}.00 dollars", 1, None, 1, "$100 scale with cents shown ('81.00 dollars')"),
}
FIRST_SET = "D2,D5,D8,C8,W8,U5,U8,K2"            # the first run (X15)
DOLLAR_SET = "D2,D8,S2,SN,S8,SM,X2"              # the dollar-sign follow-up
SYS_PRICE = '"price": a whole number of dollars for an OFFER'
SYS_PLAIN = "Write the price as a plain whole number: no dollar sign, no commas, no decimals."


def item_seed(fixture, role, sample):
    """Common random numbers: the condition is deliberately not an input."""
    label = f"scale-invariance/digits-probe/v1|{fixture}|{role}|{sample}"
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def render_item(item, settings, templates=None):
    """The opener's first-turn messages for one item, in its condition's wording."""
    text_of, _, unit, _, _ = CONDITIONS[item["condition"]]
    b, role = item["base_limit"], item["role"]
    msgs = rp2.render(role, 100, b, [], 1, (75, 95), (105, 125), settings["max_messages"]["value"],
                      settings["max_message_chars"]["value"], templates=templates)
    sysm, user = msgs[0]["content"], msgs[1]["content"]
    name = "floor" if role == "seller" else "ceiling"
    anchor = f"Your private {name} is {rp.money(b)}."
    assert user.count(anchor) == 1, "limit anchor not found"
    user = user.replace(anchor, f"Your private {name} is {text_of(b)}.", 1)
    if unit:
        assert sysm.count(SYS_PRICE) == 1 and sysm.count(SYS_PLAIN) == 1, "system price lines not found"
        sysm = sysm.replace(SYS_PRICE, SYS_PRICE.replace("of dollars", f"of {unit}"), 1)
        sysm = sysm.replace(SYS_PLAIN, SYS_PLAIN.replace("whole number:", f"whole number of {unit}:"), 1)
    return [{"role": "system", "content": sysm}, {"role": "user", "content": user}]


def gap(role, price, limit_units):
    """Opener's distance from its own limit, as a fraction (HA's g)."""
    return price / limit_units - 1 if role == "seller" else 1 - price / limit_units


def build_items(n_fix, samples, conds):
    doc = mf.build(PHRASE)
    items = []
    for f in doc["fixtures"][:n_fix]:
        fid = f"Y{f['id'][1:]}"
        for role, b in (("seller", f["f_pct"]), ("buyer", f["c_pct"])):
            for s in range(1, samples + 1):
                for c in conds:
                    _, stakes, unit, per, _ = CONDITIONS[c]
                    items.append({"item_id": f"{fid}-{role[0].upper()}-{c}-s{s}", "fixture": fid, "role": role,
                                  "condition": c, "sample": s, "base_limit": b, "limit_text": CONDITIONS[c][0](b),
                                  "limit_units": b * per, "reply_unit": unit or "dollars",
                                  "stakes_dollars": b * stakes, "seed": item_seed(fid, role, s)})
    random.Random(rr.sha256_hex((PHRASE + "|digits-order").encode())).shuffle(items)
    for i, it in enumerate(items, 1):
        it["run_order"] = i
    meta = {"seed_phrase": PHRASE, "fixtures": [{**f, "id": f"Y{f['id'][1:]}"} for f in doc["fixtures"][:n_fix]]}
    return meta, items


def score(item, reply, max_chars):
    """One record per call (schema firstmove/1)."""
    rec = {"schema": "firstmove/1", **{k: item[k] for k in (
        "item_id", "fixture", "role", "condition", "sample", "base_limit", "limit_text", "limit_units",
        "reply_unit", "stakes_dollars", "seed", "run_order")},
        "call_id": reply.get("call_id"), "action": None, "price": None, "gap": None, "errors": [],
        "message_chars": None, "completion_tokens": reply.get("completion_tokens"),
        "reasoning_present": reply.get("reasoning") not in (None, "")}
    if reply.get("kind") != "model_output":
        rec["errors"] = ["INFRASTRUCTURE"]
        return rec
    move, errors = ge.validate_reply(reply.get("content"), reply.get("finish_reason"), False, max_chars)
    if errors:
        rec["errors"] = errors
        return rec
    rec["action"], rec["message_chars"] = move["action"], len(move["message"])
    if move["action"] == "OFFER":
        rec["price"] = move["price"]
        rec["gap"] = gap(item["role"], move["price"], item["limit_units"])
    return rec


def main(argv):
    ap = argparse.ArgumentParser(description="Digits x stakes first-move probe (see the top of this file).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", default=FIRST_SET, help=f"default {FIRST_SET}; follow-up: {DOLLAR_SET}")
    ap.add_argument("--fixtures", type=int, default=30)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--lifetime-min", type=int, default=50)
    ap.add_argument("--slot", type=int, default=int(os.environ.get("SI_SLOT", "0") or 0),
                    help="0-9: run alongside other jobs, one slot each (default 0)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview-items", type=int, default=64, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    J.use_slot(a.slot)

    pins = json.load(open(P("config", "pins.json")))
    mp = {m["name"]: m for m in pins["models"]}
    conds = a.conditions.split(",")
    if a.model not in mp or any(c not in CONDITIONS for c in conds) or not 1 <= a.fixtures <= 30 \
            or not 1 <= a.samples <= 20:
        print("bad --model, --conditions, --fixtures (1-30) or --samples (1-20)")
        return 2
    model = a.model
    J.PREVIEW_MODEL, J.MANIFEST_TSV = model, P("config", "manifests", model + ".tsv")
    settings = json.load(open(P("prereg", "settings.json")))
    max_chars = settings["max_message_chars"]["value"]
    meta, items = build_items(a.fixtures, a.samples, conds)
    if a.preview:
        items = items[:a.preview_items]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    data_root = tempfile.mkdtemp(prefix="si-digits-preview-") if a.preview else P("data", "explore")
    run_id = f"X-{PROBE}-{model}-{stamp}"
    run_dir = os.path.join(data_root, run_id)
    plan = (f"Probe {PROBE}: {len(conds)} conditions x {a.fixtures} fixtures x 2 roles x {a.samples} samples"
            f" = {len(items)} first moves, {a.workers} at a time")

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

    panel = J.Panel(f"PROBE {PROBE} · {model}" + (f" · slot {a.slot}" if a.slot else ""),
                    ("machine", "model files", "vLLM", "checks", "probe", "teardown"), sys.stdout.isatty(), a.preview)
    panel.lifetime_s, panel.price, panel.worst = a.lifetime_min * 60, price, worst
    os.makedirs(P("logs"), exist_ok=True)
    machine = (J.FakeMachine if a.preview else J.Machine)(panel, P("logs", f"probe_{PROBE}_{stamp}.launcher.out"))
    srv = J.fake_server() if a.preview else None
    if a.preview:
        panel.t_created = time.time()
    interrupts = [0]
    # server_checks() renders one Study 2 first move to confirm the thinking-off marker.
    check_game = {"game_id": "check", "opener": "seller", "V": 100, "F": 81, "C": 124, "run_order": 1}
    ctx = {"panel": panel, "machine": machine, "args": a, "model": model, "settings": settings, "pins": pins,
           "mp": mp, "pilot_fx": {"games": [check_game], "f_pct_range": (75, 95), "c_pct_range": (105, 125)},
           "family": J.family_of(model), "interrupts": interrupts, "study": 2,
           "render": rp2.render}

    def on_int(signum, frame):
        interrupts[0] += 1
        panel.note("stop requested: finishing current calls, then deleting the machine"
                   if interrupts[0] == 1 else "stop requested again: skipping the log copy")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_int)
    threading.Thread(target=panel.ticker, daemon=True).start()
    panel.draw()

    result, w, counts = {"status": "not started"}, None, {"offers": 0, "walkaways": 0, "invalid": 0}
    try:
        machine.start(["--mode", "serve", "--model", model, "--lifetime-min", str(a.lifetime_min),
                       "--confirm-cost", worst])
        while not machine.ready.wait(1.0):
            if interrupts[0]:
                raise J.JobStop("stopped by you before the probe began")
        if not machine.alive() or machine.failed:
            raise J.JobStop(machine.failed or "the launcher ended before vLLM was ready")
        panel.set_stage("checks", "run"); panel.set_now("checking the server before any call")
        problems, info = J.server_checks(ctx)
        if problems:
            raise J.JobStop("check failed: " + "; ".join(problems))
        panel.set_stage("checks", "ok")
        panel.note("checks passed: weights, version 0.30.0, one model, thinking off"
                   + ("" if a.preview else ", no watermark"))

        sess = J.session_record(ctx, info)
        example = {c: render_item({"condition": c, "role": "seller", "base_limit": 81}, settings) for c in conds}
        manifest = {
            "schema": "run_manifest/1", "run_id": run_id, "kind": "probe", "preview": a.preview,
            "study": "probe", "probe": PROBE, "not_preregistered": True, "model": model,
            "repo": mp[model]["repo"], "revision": mp[model]["revision"],
            "image": "vllm/vllm-openai@" + pins["vllm_image"]["index_digest"], **sess,
            "base_prompt": "prereg/study2/prompts (first move, no history)",
            "conditions": {c: CONDITIONS[c][4] for c in conds},
            "example_prompts_seller_floor_81": example,
            "seed_rule": "sha256('scale-invariance/digits-probe/v1|fixture|role|sample')[:8]; same across conditions",
            "workers": a.workers, "samples": a.samples, "fixture_meta": meta,
            "games_planned": [it["item_id"] for it in items], "started_utc": rr.utc_now(),
        }
        w = locked_writer(rr.RunWriter(run_dir, manifest))
        with open(os.path.join(run_dir, "items_planned.json"), "w") as fh:
            json.dump(items, fh, indent=0)
        panel.set_stage("probe", "run"); panel.reset("Probe", len(items))
        panel.note(f"probe started: {len(items)} first moves, {a.workers} at a time")

        transport = v.make_chat_transport(J.PORT, total_timeout=300)
        base_call = rr.recording_call(w, transport, run_id,
                                      sleep=(lambda s: time.sleep(0.1)) if a.preview else time.sleep)
        first, first_lock, stop = {}, threading.Lock(), threading.Event()

        def call(body, info_, item):
            if interrupts[0] or stop.is_set():
                raise J.JobStop("stopped")
            if not machine.alive():
                raise J.JobStop("the machine session ended")
            J.FAKE["msg_no"], J.FAKE["V"] = 1, item["limit_units"]
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
        for it in items:
            q.put((it, 1))
        stop_why = []

        def worker():
            while not stop.is_set() and not interrupts[0]:
                if not a.preview and panel.t_created and \
                        panel.t_created + a.lifetime_min * 60 - time.time() < 240:
                    stop_why.append("machine time nearly up; no new call started")
                    stop.set()
                    return
                try:
                    it, attempt = q.get_nowait()
                except queue.Empty:
                    return
                t0 = time.monotonic()
                body = ge.build_request(model, render_item(it, settings), settings, it["seed"])
                info_ = {"game_id": it["item_id"], "attempt": attempt, "msg_no": 1, "role": it["role"]}
                try:
                    reply = call(body, info_, it)
                except J.JobStop as e:
                    w.event("item_interrupted", item_id=it["item_id"], attempt=attempt, why=str(e))
                    stop_why.append(str(e))
                    stop.set()
                    return
                rec = score(it, reply, max_chars)
                rec["attempt"] = attempt
                w.game(rec)
                with panel.lock:
                    panel.reasoning += int(rec["reasoning_present"])
                    panel.truncated += int("TRUNCATED" in rec["errors"])
                    if rec["errors"] == ["INFRASTRUCTURE"]:
                        panel.voided += 1
                        if attempt < J.MAX_ATTEMPTS:
                            q.put((it, attempt + 1))
                        else:
                            panel.games_done += 1
                    else:
                        panel.replies += 1
                        panel.games_done += 1
                        if rec["errors"]:
                            panel.invalid += 1
                            counts["invalid"] += 1
                        elif rec["action"] == "OFFER":
                            counts["offers"] += 1
                        else:
                            counts["walkaways"] += 1
                        panel.game_secs.append((time.monotonic() - t0) / max(1, a.workers))
                panel.set_now(f"probe: {panel.games_done} of {len(items)} first moves done")
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
            try:
                J.copy_server_logs(ctx, run_dir)
                w.close(status=result["status"], items_done=panel.games_done, replies=panel.replies,
                        voided=panel.voided, reasoning=panel.reasoning, **counts)
            except Exception as e:          # never let record-keeping block the teardown below
                panel.note(f"could not close the run records: {type(e).__name__}: {e}")
        panel.set_now("deleting the machine")
        machine.finish()
        if srv:
            srv.shutdown()
        panel.set_now("done" if result["status"] == "complete" else "stopped")
        panel.stop.set()
        panel.draw()

    print()
    print(f"Probe {result['status']}" + (f": {result['why']}" if result.get("why") else "") + ".")
    print(f"  {panel.games_done} of {len(items)} first moves; {panel.replies - panel.invalid} of {panel.replies}"
          f" replies valid; voided {panel.voided}; reasoning {panel.reasoning}")
    print("  Data: " + (os.path.relpath(run_dir, J.ROOT) if not a.preview else run_dir + " (preview)"))
    print("  Next: python3 scripts/analyze_digits_probe.py " +
          (os.path.relpath(run_dir, J.ROOT) if not a.preview else run_dir))
    ok = machine.deleted is True
    if ok:
        cost = f"${panel.cost_final:.2f}" if panel.cost_final is not None else "see logs/budget.tsv"
        print(f"  Machine deleted; 0 of our pods left. Estimated cost {cost}.")
    else:
        print("  !!! COULD NOT CONFIRM THE MACHINE IS GONE. Check the RunPod console now. !!!")
    return 9 if not ok else (0 if result["status"] == "complete" else 1)


def locked_writer(w):
    """Thread-safe appends to the three hash chains (same as run_probe.locked_writer)."""
    lock = threading.Lock()
    for ch in (w.calls, w.games, w.events):
        orig = ch.append

        def append(obj, _orig=orig):
            with lock:
                return _orig(obj)
        ch.append = append
    return w


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
