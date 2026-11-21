#!/bin/bash
# mutation_check_runpod_auth.sh: proves the auth-check safety tests can fail.
#
# For each mutant, makes a temporary copy of scripts/runpod_auth_check.sh with
# one deliberate flaw planted (most are ways to leak the key), runs the
# offline test suite against that copy, and checks that the matching test
# fails. An unmodified copy runs first and must pass everything. The real
# script is never modified. No network.
#
# Usage: bash tests/mutation_check_runpod_auth.sh

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC_SCRIPT="$HERE/../scripts/runpod_auth_check.sh"
SRC_CONN="$HERE/../scripts/connections"
SRC_TESTS="$HERE/test_runpod_auth_check.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
caught=0
total=0

# layout DIR: copy the scripts and tests into DIR with the real folder layout.
layout() {
  mkdir -p "$1/scripts/connections" "$1/tests"
  cp "$SRC_TESTS" "$1/tests/"
  cp "$SRC_CONN"/*.sh "$1/scripts/connections/"
}

# Baseline: the unmodified copy must pass every test, so a "caught" below
# means the planted flaw broke a test, not that the copy was broken.
layout "$TMP/baseline"
cp "$SRC_SCRIPT" "$TMP/baseline/scripts/"
if base="$(bash "$TMP/baseline/tests/test_runpod_auth_check.sh" 2>&1)"; then
  printf 'baseline: unmodified copy passes (%s)\n\n' "$(printf '%s\n' "$base" | tail -1)"
else
  printf 'baseline FAILED; fix the tests before trusting this check:\n%s\n' \
    "$(printf '%s\n' "$base" | grep FAIL)"
  exit 1
fi

# mutant NAME "LABEL OF THE TEST THAT MUST FAIL" SED_PROGRAM
mutant() {
  local name="$1" expect="$2" prog="$3" d="$TMP/$1" out rc
  total=$((total + 1))
  layout "$d"
  if ! sed -e "$prog" "$SRC_SCRIPT" > "$d/scripts/runpod_auth_check.sh" || [ ! -s "$d/scripts/runpod_auth_check.sh" ]; then
    printf 'BROKEN  %-15s sed failed or produced an empty file; update this mutant\n' "$name"
    return
  fi
  if cmp -s "$SRC_SCRIPT" "$d/scripts/runpod_auth_check.sh"; then
    printf 'BROKEN  %-14s the planted change did not apply; update this mutant\n' "$name"
    return
  fi
  out="$(bash "$d/tests/test_runpod_auth_check.sh" 2>&1)"
  rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qF "FAIL  $expect"; then
    caught=$((caught + 1))
    printf 'caught  %-14s -> FAIL  %s\n' "$name" "$expect"
  else
    printf 'MISSED  %-14s expected: FAIL  %s\n' "$name" "$expect"
  fi
}

# Puts the key on curl's command line, where `ps` can see it.
mutant key_in_argv   "key absent from curl's command line" \
  's/-H @- -H/-H "Authorization: Bearer $RUNPOD_KEY" -H/'
# Leaves bash tracing on, so `bash -x` prints the key.
mutant xtrace_on     "key absent from xtrace output" \
  '/^set +x +v +a$/d'
# Prints each pod's environment variables (which can hold secrets).
mutant prints_env    "pod env secret absent from output" \
  's/+ "  created=/+ "  env=\\(.env)" + "  created=/'
# Stops scrubbing the key from error messages a server echoes back.
mutant no_redaction  "echoed key is redacted" \
  '/REDACTED/d'
# Lets ~/.curlrc settings (verbose/trace) apply.
mutant reads_curlrc  "-q is not first on every curl call" \
  's/curl -q -sS/curl -sS/'
# Leaves the key in the environment that curl inherits.
mutant env_inherited "RUNPOD_API_KEY not inherited by curl" \
  '/^unset RUNPOD_API_KEY$/d'
# The 2026-10-02 bug: adds stopped pods' rates into "what is billing now".
mutant sums_stopped  "stopped pod's rate excluded from the live total" \
  's/select(.status != "EXITED" and .status != "TERMINATED")/select(true)/'

printf '\n%d of %d planted flaws caught\n' "$caught" "$total"
[ "$caught" -eq "$total" ]
