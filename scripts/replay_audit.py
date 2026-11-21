#!/usr/bin/env python3
"""replay_audit.py: re-derive every game of a run from its raw call records.

WHAT IT DOES (docs/data_plan.md section 10, step 5; required before analysis)
  1. Checks the hash chain of calls.jsonl, games.jsonl and events.jsonl,
     and, if the run is closed, every sha256 in run_summary.json.
  2. For every game record, replays the game with game_engine.play_game(),
     feeding it the recorded replies instead of a model. At each move the
     engine rebuilds the request it would send; that must equal the
     recorded request byte for byte. That proves the prompt, the private
     value, the public history, the seed and every setting were exactly as
     registered, and that nothing was sent that the engine wouldn't send.
  3. The replayed game record must equal the recorded one exactly.
  4. Every request line has a result line, and no call is left unused.

USAGE
  python3 scripts/replay_audit.py data/runs/<run_id>
  Exit 0: everything reproduces. 1: a mismatch (listed). 2: usage problem.

Standard library only. Reads files only; never contacts a server.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import game_engine  # noqa: E402
import render_prompt_s2  # noqa: E402
import run_records  # noqa: E402

ROOT = os.path.join(HERE, "..")


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def audit(run_dir, fixtures_path=None, settings_path=None):
    problems = []
    settings_path = settings_path or os.path.join(ROOT, "prereg", "settings.json")
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    # Pilot runs name their own fixtures file (data plan E4); real runs use the default.
    fixtures_path = fixtures_path or os.path.join(ROOT, manifest.get("fixtures") or "config/fixtures.json")
    with open(fixtures_path, encoding="utf-8") as fh:
        fx = json.load(fh)
    with open(settings_path, encoding="utf-8") as fh:
        settings = json.load(fh)
    games_by_id = {g["game_id"]: g for g in fx["games"]}
    render = render_prompt_s2.render if manifest.get("study") == 2 else None   # Study 2: no appraisal

    for name in run_records.RunWriter.FILES:
        ok, n, problem, _ = run_records.verify_chain(os.path.join(run_dir, name))
        if not ok:
            problems.append(f"{name}: {problem}")
    spath = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(spath):
        with open(spath, encoding="utf-8") as fh:
            summary = json.load(fh)
        for name, want in summary["sha256"].items():
            with open(os.path.join(run_dir, name), "rb") as fh:
                if run_records.sha256_hex(fh.read()) != want:
                    problems.append(f"{name}: sha256 differs from run_summary.json")
    if problems:
        return problems, 0      # don't replay on top of broken files

    calls = load_jsonl(os.path.join(run_dir, "calls.jsonl"))
    requests = {c["call_id"]: c for c in calls if c["schema"] == "call-request/1"}
    results = {c["call_id"]: c for c in calls if c["schema"] == "call-result/1"}
    # E9: an attempt cut off mid-game is recorded as a game_interrupted event
    # instead of a game record. Its calls are kept, and it is replayed later
    # as a new attempt, so its calls belong to no game and may lack a result.
    events = load_jsonl(os.path.join(run_dir, "events.jsonl"))
    interrupted = {(e["game_id"], e["attempt"]) for e in events if e.get("kind") == "game_interrupted"}
    cut = lambda c: (c["game_id"], c["attempt"]) in interrupted   # noqa: E731
    for cid in requests:
        if cid not in results and not cut(requests[cid]):
            problems.append(f"{cid}: request without a result line")
    used = set()

    records = load_jsonl(os.path.join(run_dir, "games.jsonl"))
    for rec in records:
        gid, attempt = rec["game_id"], rec["attempt"]
        if (gid, attempt) in interrupted:
            problems.append(f"{gid} attempt {attempt}: has a game record but is also marked interrupted")
        g = games_by_id.get(gid)
        if g is None:
            problems.append(f"{gid}: not a game in config/fixtures.json")
            continue

        def replay_call(body, info, _gid=gid):
            # The successful (last) try for this move is the reply the game saw.
            tries = sorted((c for cid, c in requests.items()
                            if c["game_id"] == _gid and c["attempt"] == info["attempt"]
                            and c["msg_no"] == info["msg_no"]), key=lambda c: c["try"])
            if not tries:
                raise AssertionError(f"{_gid} a{info['attempt']} m{info['msg_no']}: no recorded call")
            want = run_records.canonical(body).encode("utf-8")
            for t in tries:
                used.add(t["call_id"])
                if run_records.canonical(t["request"]).encode("utf-8") != want:
                    raise AssertionError(f"{t['call_id']}: recorded request differs from what the "
                                         "engine builds for this move")
                if run_records.sha256_hex(want) != t["request_sha256"]:
                    raise AssertionError(f"{t['call_id']}: request_sha256 does not match")
            last = results[tries[-1]["call_id"]]
            raw = last["response_text"].encode("utf-8")
            if run_records.sha256_hex(raw) != last["response_sha256"]:
                raise AssertionError(f"{last['call_id']}: response_text does not match its sha256")
            if last["transport_error"]:
                return {"kind": "infrastructure", "call_id": last["call_id"]}
            fields, problem = run_records.read_envelope(last["http_status"], raw)
            if fields is None:
                return {"kind": "infrastructure", "call_id": last["call_id"]}
            return {"kind": "model_output", "call_id": last["call_id"], **fields}

        try:
            again = game_engine.play_game(g, manifest["model"], settings, fx["f_pct_range"],
                                          fx["c_pct_range"], replay_call, attempt=attempt,
                                          render=render)
        except AssertionError as e:
            problems.append(str(e))
            continue
        again = json.loads(run_records.canonical(again))
        stored = {k: v for k, v in rec.items() if k != "prev_sha256"}
        if again != stored:
            diff = sorted(k for k in set(again) | set(stored) if again.get(k) != stored.get(k))
            problems.append(f"{gid} attempt {attempt}: game record differs on {', '.join(diff)}")

    for cid in requests:
        if cid not in used and not cut(requests[cid]):
            problems.append(f"{cid}: call not part of any recorded game")
    return problems, len(records)


def main(argv):
    if len(argv) != 1 or not os.path.isdir(argv[0]):
        print("usage: python3 scripts/replay_audit.py data/runs/<run_id>", file=sys.stderr)
        return 2
    problems, n = audit(argv[0])
    if problems:
        print(f"REPLAY AUDIT FAILED: {len(problems)} problem(s)")
        for p in problems[:50]:
            print("  ✗ " + p)
        return 1
    print(f"replay audit passed: {n} game records reproduce exactly from the raw calls")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
