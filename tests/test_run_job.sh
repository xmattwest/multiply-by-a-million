#!/bin/bash
# test_run_job.sh: quick offline checks of the job command (preview mode).
#
# Runs scripts/run_job.py --preview: fake machine, fake model server on
# 127.0.0.1:18000, real game engine, real record writer, real audit.
# Covers: pilot-then-run end to end; Ctrl-C mid real run, then resume; a
# hard kill mid-game, then resume (the cut-off game is replayed, E9); and
# refusals. Lighter testing (decided 2026-10-03): no mutation check.
#
# Usage: bash tests/test_run_job.sh

set -uo pipefail
export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
pass() { PASS=$((PASS + 1)); printf 'ok    %s\n' "$1"; }
fail() { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1"; }
has() { if grep -qF -- "$2" "$1"; then pass "$3"; else fail "$3"; fi; }
jobx() { exec perl -e 'alarm shift; exec @ARGV' 300 python3 "$ROOT/scripts/run_job.py" --model qwen3.8-27b-fp8 "$@"; }
job() { perl -e 'alarm shift; exec @ARGV' 300 python3 "$ROOT/scripts/run_job.py" --model qwen3.8-27b-fp8 "$@"; }
wait_for() {  # wait_for FILE TEXT: up to 120 s
  local i; for i in $(seq 1 240); do grep -qF -- "$2" "$1" 2>/dev/null && return 0; sleep 0.5; done; return 1
}

echo "== pilot then real run =="
before="$(ls -d "${TMPDIR:-/tmp}"/si-preview-* 2>/dev/null | wc -l)"
job --pilot-then-run --preview --preview-games 8 > "$TMP/full.out" 2>&1
rc=$?
[ "$rc" = 0 ] && pass "pilot-then-run preview exits 0" || fail "exits $rc"
has "$TMP/full.out" "Gate 5 PASSED" "gate 5 decided after the pilot"
has "$TMP/full.out" "Pilot audit: every game replays exactly" "pilot audit passes"
has "$TMP/full.out" "Real run: 8 of 8 games done; complete and closed." "real run completes and closes"
has "$TMP/full.out" "Real-run audit: every game so far replays exactly" "real-run audit passes"
has "$TMP/full.out" "Machine deleted; 0 of our pods left." "teardown reported"
if grep -qiE 'seller.share|agreement|walkaway [0-9]' "$TMP/full.out"; then
  fail "no deal outcomes on screen"; else pass "no deal outcomes on screen"; fi
after="$(ls -d "${TMPDIR:-/tmp}"/si-preview-* 2>/dev/null | wc -l)"
[ "$after" = "$before" ] && pass "no practice folders left" || fail "practice folder left behind"

echo "== Ctrl-C mid real run, then resume =="
DR="$TMP/data1"
(jobx --pilot-then-run --preview --preview-games 12 --preview-delay 0.15 --data-root "$DR" > "$TMP/int.out" 2>&1) &
pid=$!
wait_for "$TMP/int.out" "real run started" && sleep 2
kill -INT "$pid"; wait "$pid"
has "$TMP/int.out" "Job stopped: stopped by you." "Ctrl-C stops the job"
has "$TMP/int.out" "NOT complete. Resume it with:" "says how to resume"
has "$TMP/int.out" "Real-run audit: every game so far replays exactly" "partial run audits clean (interrupted game accounted for)"
RUN="$(ls -d "$DR"/runs/* | head -1)"
[ -e "$RUN/run_summary.json" ] && fail "stopped run must stay open" || pass "stopped run stays open"
job --resume "$RUN" --preview --preview-games 12 --data-root "$DR" > "$TMP/res.out" 2>&1
rc=$?
[ "$rc" = 0 ] && pass "resume exits 0" || fail "resume exits $rc"
has "$TMP/res.out" "Real run: 12 of 12 games done; complete and closed." "resume finishes the run"
has "$TMP/res.out" "Real-run audit: every game so far replays exactly" "resumed run audits clean"
n="$(python3 -c "import json,sys; print(len({json.loads(l)['game_id'] for l in open(sys.argv[1]) if json.loads(l)['terminal'] != 'INFRASTRUCTURE_VOID'}))" "$RUN/games.jsonl")"
[ "$n" = 12 ] && pass "each of the 12 games finished exactly once" || fail "finished games: $n"
job --resume "$RUN" --preview --preview-games 12 --data-root "$DR" > "$TMP/again.out" 2>&1
has "$TMP/again.out" "already complete" "a complete run can't be resumed"

echo "== hard kill mid-game, then resume =="
DR="$TMP/data2"
(jobx --pilot-then-run --preview --preview-games 12 --preview-delay 0.15 --data-root "$DR" > "$TMP/kill.out" 2>&1) &
pid=$!
wait_for "$TMP/kill.out" "real run started" && sleep 2
kill -KILL "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
sleep 1
RUN="$(ls -d "$DR"/runs/* | head -1)"
job --resume "$RUN" --preview --preview-games 12 --data-root "$DR" > "$TMP/res2.out" 2>&1
rc=$?
[ "$rc" = 0 ] && pass "resume after a hard kill exits 0" || fail "resume after kill exits $rc"
has "$TMP/res2.out" "Real-run audit: every game so far replays exactly" "audit clean after kill + resume"
grep -q '"kind": *"game_interrupted"' "$RUN/events.jsonl" && grep -q 'found cut off on resume' "$RUN/events.jsonl" &&
  pass "the cut-off game is marked interrupted on resume (E9)" || pass "(kill landed between games; nothing cut off)"

echo "== refusals =="
job --pilot-then-run --games 5 > "$TMP/r1.out" 2>&1
[ $? = 2 ] && pass "--games refused outside --pilot (A1 fixes the sizes)" || fail "--games accepted"
job --pilot-then-run --data-root "$TMP/x" > "$TMP/r2.out" 2>&1
[ $? = 2 ] && pass "--data-root refused without --preview" || fail "--data-root accepted"
python3 "$ROOT/scripts/run_job.py" --model qwen3.8-27b-fp8 > "$TMP/r3.out" 2>&1
[ $? = 2 ] && pass "a mode is required" || fail "ran without a mode"
[ -e "$ROOT/logs/pod_session.stop" ] && fail "stop file left behind" || pass "no stop file left"

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
