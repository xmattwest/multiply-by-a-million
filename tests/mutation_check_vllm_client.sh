# mutation_check_vllm_client.sh: proves the vLLM client tests can fail.
#
# For each mutant, copies scripts/ and the tests into a temporary folder
# with one deliberate flaw planted in scripts/vllm_client.py, runs
# tests/test_vllm_client.py there, and checks that the matching test fails
# or crashes. An unmodified copy runs first and must pass. The real files
# are never modified. No network beyond 127.0.0.1 (the tests' fake server).
# Every test run has a hard 90-second limit, so a mutant that hangs counts
# as caught instead of hanging this script.
#
# Usage: bash tests/mutation_check_vllm_client.sh

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
caught=0
total=0
readonly FILE="vllm_client.py"

layout() {
  mkdir -p "$1/scripts" "$1/tests"
  cp "$ROOT"/scripts/*.py "$1/scripts/"
  cp "$HERE/test_vllm_client.py" "$HERE/test_run_records.py" "$1/tests/"
  ln -s "$ROOT/config" "$1/config"
  ln -s "$ROOT/prereg" "$1/prereg"
}

run_tests() {   # hard time limit; macOS has no `timeout`, perl's alarm does the job
  perl -e 'alarm shift; exec @ARGV' 90 python3 "$1/tests/test_vllm_client.py" 2>&1
}

layout "$TMP/baseline"
if base="$(run_tests "$TMP/baseline")"; then
  printf 'baseline: unmodified copy passes (%s)\n\n' "$(printf '%s\n' "$base" | grep -E '^Ran ')"
else
  printf 'baseline FAILED; fix the tests before trusting this check:\n%s\n' \
    "$(printf '%s\n' "$base" | grep -E '^(FAIL|ERROR):')"
  exit 1
fi

# mutant NAME "TEST THAT MUST FAIL" SED_PROGRAM
mutant() {
  local name="$1" expect="$2" prog="$3" d="$TMP/$1" out rc
  total=$((total + 1))
  layout "$d"
  if ! sed -e "$prog" "$ROOT/scripts/$FILE" > "$d/scripts/$FILE" || [ ! -s "$d/scripts/$FILE" ]; then
    printf 'BROKEN  %-18s sed failed or produced an empty file; update this mutant\n' "$name"
    return
  fi
  if cmp -s "$ROOT/scripts/$FILE" "$d/scripts/$FILE"; then
    printf 'BROKEN  %-18s the planted change did not apply; update this mutant\n' "$name"
    return
  fi
  out="$(run_tests "$d")"
  rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qE "^(FAIL|ERROR): $expect "; then
    caught=$((caught + 1))
    printf 'caught  %-18s -> %s\n' "$name" "$expect"
  elif [ "$rc" -ge 128 ]; then
    caught=$((caught + 1))
    printf 'caught  %-18s -> hit the 90 s hard limit (hung)\n' "$name"
  else
    printf 'MISSED  %-18s expected: %s\n' "$name" "$expect"
  fi
}

# The client may talk to a host other than the tunnel's loopback end.
mutant host_not_loopback "test_host_is_fixed_to_loopback" \
  's/^HOST = "127.0.0.1"$/HOST = "localhost"/'
# A credential header is sent.
mutant auth_header       "test_headers_carry_no_credentials" \
  's/^    headers = {"Accept": "application\/json", "Connection": "close"}$/    headers = {"Accept": "application\/json", "Connection": "close", "Authorization": "Bearer x"}/'
# Any route may be called.
mutant no_route_check    "test_only_listed_routes_can_be_called" \
  's/^    if (method, path) not in ROUTES:$/    if False:/'
# The unprotected inference route (C4) is allowed.
mutant invocations_ok    "test_only_listed_routes_can_be_called" \
  's/^    ("POST", "\/detokenize"),$/    ("POST", "\/detokenize"), ("POST", "\/invocations"),/'
# A misspelled or extra field goes through (C12).
mutant unknown_fields_ok "test_field_check_rejects_before_anything_is_sent" \
  's/^    if unknown:$/    if False:/'
# A missing sampling field goes through, so it falls back to a default (C3).
mutant missing_fields_ok "test_field_check_rejects_before_anything_is_sent" \
  's/^    if missing:$/    if False:/'
# Duplicate fields are silently merged.
mutant dupes_ok          "test_field_check_rejects_before_anything_is_sent" \
  's/object_pairs_hook=_no_dupes/object_pairs_hook=dict/'
# Thinking switched on goes through.
mutant thinking_ok       "test_field_check_rejects_before_anything_is_sent" \
  's/^    if body\["chat_template_kwargs"\] != {"enable_thinking": False}:$/    if False:/'
# n=true slips through because True == 1 in Python.
mutant n_bool_ok         "test_field_check_rejects_before_anything_is_sent" \
  's/^    if body\["n"\] != 1 or isinstance(body\["n"\], bool):$/    if body["n"] != 1:/'
# The transport skips the field check entirely.
mutant no_body_check     "test_field_check_rejects_before_anything_is_sent" \
  's/^        check_chat_body(body_bytes)$/        pass/'
# No time limit while waiting for the reply headers.
mutant no_wait_timeout   "test_silent_server_times_out" \
  's/^        sock.settimeout(max(0.001, deadline - time.monotonic()))$/        sock.settimeout(None)/'
# No overall deadline: each read gets a fresh, generous limit.
mutant no_deadline       "test_trickling_server_hits_the_overall_deadline" \
  's/^            if remaining <= 0:$/            if False:/; s/^            sock.settimeout(remaining)$/            sock.settimeout(30)/'
# Buffered read: one call keeps reading past the deadline.
mutant buffered_read     "test_trickling_server_hits_the_overall_deadline" \
  's/resp.read1(READ_CHUNK)/resp.read(READ_CHUNK)/'
# No 1 MB cap: an oversized reply is read in full.
mutant no_size_cap       "test_oversized_reply_stops_at_1mb" \
  's/^            if len(data) > MAX_RESPONSE:$/            if False:/'
# Network errors raise instead of being reported.
mutant raises_on_network "test_refused_connection_is_reported_not_raised" \
  's/^    except (OSError, http.client.HTTPException) as e:$/    except ZeroDivisionError as e:/'

printf '\n%d of %d planted flaws caught\n' "$caught" "$total"
[ "$caught" -eq "$total" ]
