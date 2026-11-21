# mutation_check_run_records.sh: proves the recorder and replay-audit tests can fail.
#
# For each mutant, copies scripts/, config/, prereg/ and the test into a
# temporary folder with one deliberate flaw planted in run_records.py or
# replay_audit.py, runs tests/test_run_records.py there, and checks that the
# matching test fails or crashes. An unmodified copy runs first and must
# pass. The real files are never modified. No network.
#
# Usage: bash tests/mutation_check_run_records.sh

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
caught=0
total=0

layout() {
  mkdir -p "$1/scripts" "$1/tests"
  cp "$ROOT"/scripts/*.py "$1/scripts/"
  cp "$HERE/test_run_records.py" "$1/tests/"
  ln -s "$ROOT/config" "$1/config"
  ln -s "$ROOT/prereg" "$1/prereg"
}

layout "$TMP/baseline"
if base="$(python3 "$TMP/baseline/tests/test_run_records.py" 2>&1)"; then
  printf 'baseline: unmodified copy passes (%s)\n\n' "$(printf '%s\n' "$base" | grep -E '^Ran ')"
else
  printf 'baseline FAILED; fix the tests before trusting this check:\n%s\n' \
    "$(printf '%s\n' "$base" | grep -E '^(FAIL|ERROR):')"
  exit 1
fi

# mutant NAME FILE "TEST THAT MUST FAIL" SED_PROGRAM
mutant() {
  local name="$1" file="$2" expect="$3" prog="$4" d="$TMP/$1" out rc
  total=$((total + 1))
  layout "$d"
  if ! sed -e "$prog" "$ROOT/scripts/$file" > "$d/scripts/$file" || [ ! -s "$d/scripts/$file" ]; then
    printf 'BROKEN  %-16s sed failed or produced an empty file; update this mutant\n' "$name"
    return
  fi
  if cmp -s "$ROOT/scripts/$file" "$d/scripts/$file"; then
    printf 'BROKEN  %-16s the planted change did not apply; update this mutant\n' "$name"
    return
  fi
  out="$(python3 "$d/tests/test_run_records.py" 2>&1)"
  rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qE "^(FAIL|ERROR): $expect "; then
    caught=$((caught + 1))
    printf 'caught  %-16s -> %s\n' "$name" "$expect"
  else
    printf 'MISSED  %-16s expected: %s\n' "$name" "$expect"
  fi
}

# The chain stops linking lines, so edits and deletions go unnoticed.
mutant no_chain_check  run_records.py "test_chain_detects_edits_deletions_reordering_and_cut_writes" \
  's/^        if not isinstance(obj, dict) or obj.get("prev_sha256") != prev:$/        if not isinstance(obj, dict):/'
# A cut-off last line is accepted.
mutant cut_write_ok    run_records.py "test_chain_detects_edits_deletions_reordering_and_cut_writes" \
  's/^    if data and not data.endswith(b"\\n"):$/    if False:/'
# The manifest may be overwritten.
mutant manifest_reuse  run_records.py "test_run_folder_and_manifest_are_never_reused" \
  's/os.makedirs(run_dir, exist_ok=False)/os.makedirs(run_dir, exist_ok=True)/; s/open(mpath, "x", encoding="utf-8")/open(mpath, "w", encoding="utf-8")/'
# The request is written only after the reply arrives.
mutant request_late    run_records.py "test_request_is_on_disk_before_sending_and_raw_reply_kept" \
  's/^            status, raw, terr = transport(req_bytes)$/            status, raw, terr = (None, b"", "skipped")/'
# No retries: one hiccup voids the game.
mutant no_retries      run_records.py "test_retries_identical_bytes_then_gives_up" \
  's/^MAX_TRIES = 3 /MAX_TRIES = 1 /'
# The HTTP status is ignored, so an error response that looks like a reply counts as a move.
mutant status_ignored   run_records.py "test_what_counts_as_infrastructure" \
  's/^    if status != 200:$/    if False:/'
# The audit skips the summary hashes.
mutant no_summary      replay_audit.py "test_careful_tampering_is_caught_by_the_summary" \
  's/^    if os.path.exists(spath):$/    if False:/'
# The audit doesn't compare requests, so changed settings slip through.
mutant no_request_cmp  replay_audit.py "test_changed_settings_are_caught_by_replay" \
  's/^                if run_records.canonical(t\["request"\]).encode("utf-8") != want:$/                if False:/'
# The audit trusts response_text without its hash.
mutant no_reply_hash   replay_audit.py "test_edited_reply_is_caught" \
  's/^            if run_records.sha256_hex(raw) != last\["response_sha256"\]:$/            if False:/'
# The audit doesn't compare the replayed game with the stored one.
mutant no_game_cmp     replay_audit.py "test_edited_game_record_is_caught_by_replay" \
  's/^        if again != stored:$/        if False:/'

printf '\n%d of %d planted flaws caught\n' "$caught" "$total"
[ "$caught" -eq "$total" ]
