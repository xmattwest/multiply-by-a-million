#!/bin/bash
# mutation_check_connections.sh: proves the step-02 tests can fail.
#
# For each mutant, makes a temporary copy of scripts/connections/
# 02_huggingface.sh with one deliberate flaw planted, runs the offline
# suite tests/test_connections.sh against that copy, and checks that the
# matching test fails. An unmodified copy runs first and must pass. The
# real script is never modified. No network.
#
# Usage: bash tests/mutation_check_connections.sh

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/scripts/connections/02_huggingface.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
caught=0
total=0

# layout DIR: copy scripts and tests into DIR with the real folder layout.
layout() {
  mkdir -p "$1/scripts/connections" "$1/tests"
  cp "$ROOT/scripts/runpod_auth_check.sh" "$1/scripts/"
  cp "$ROOT/scripts/connections/"*.sh "$1/scripts/connections/"
  cp "$HERE/test_connections.sh" "$1/tests/"
  mkdir -p "$1/config"
  cp "$ROOT/config/pins.json" "$1/config/"
}

layout "$TMP/baseline"
if base="$(bash "$TMP/baseline/tests/test_connections.sh" 2>&1)"; then
  printf 'baseline: unmodified copy passes (%s)\n\n' "$(printf '%s\n' "$base" | tail -1)"
else
  printf 'baseline FAILED; fix the tests before trusting this check:\n%s\n' \
    "$(printf '%s\n' "$base" | grep FAIL)"
  exit 1
fi

# mutant NAME "LABEL OF THE TEST THAT MUST FAIL" SED_PROGRAM
mutant() {
  local name="$1" expect="$2" prog="$3" d="$TMP/$1" target out rc
  total=$((total + 1))
  layout "$d"
  target="$d/scripts/connections/02_huggingface.sh"
  if ! sed -e "$prog" "$SRC" > "$target" || [ ! -s "$target" ]; then
    printf 'BROKEN  %-15s sed failed or produced an empty file; update this mutant\n' "$name"
    return
  fi
  if cmp -s "$SRC" "$target"; then
    printf 'BROKEN  %-15s the planted change did not apply; update this mutant\n' "$name"
    return
  fi
  out="$(bash "$d/tests/test_connections.sh" 2>&1)"
  rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qF "FAIL  $expect"; then
    caught=$((caught + 1))
    printf 'caught  %-15s -> FAIL  %s\n' "$name" "$expect"
  else
    printf 'MISSED  %-15s expected: FAIL  %s\n' "$name" "$expect"
  fi
}

# Leaves Hugging Face token variables in the environment curl inherits.
mutant token_inherited "HF token variables not inherited by curl" \
  '/^unset HF_TOKEN /d'
# Sends a token, i.e. the check is no longer anonymous.
mutant sends_token     "an Authorization header was sent" \
  "s/-H 'Accept: application\\/json'/-H 'Accept: application\\/json' -H \"Authorization: Bearer \${HF_TOKEN:-}\"/"
# Follows redirects, so a wrong-case name would quietly pass.
mutant follows_redir   "wrong-case name (307) -> exit 1" \
  's/curl -q -sS -g/curl -q -sS -g -L/'
# Lets ~/.curlrc settings apply.
mutant reads_curlrc    "-q is not first on every curl call" \
  's/curl -q -sS -g/curl -sS -g/'
# Treats a gated repo as a plain pass, hiding that a token would be needed.
mutant hides_gating    "gated repo flagged with its approval type" \
  's/GATED (\$gated approval)/not gated/'
# Trusts the server's ID without checking it matches what we asked for.
mutant skips_id_check  "answer for a different repo -> exit 3" \
  's/if \[ "\$id" != "\$repo" \]; then/if false; then/'
# Prints the server-supplied redirect target without validating it.
mutant raw_redirect    "odd redirect target not repeated" \
  's/\*\/\*\/\*|\/\*|\*\/|'"''"') suggestion="" ;;/*\/*\/*|\/*|*\/) ;;/'
# Prints curl's error message without stripping escape codes.
mutant no_sanitize     "escape code from curl's message reached the screen" \
  "s/tr -cd '\\[:print:\\]'/cat/"
# Lets a missing, misnamed, or disabled repo count as success.
mutant missing_ok      "missing repo (401) -> exit 1" \
  's/^      worse 1$/      : worse 1/'

# Treats any current commit as the pinned one, so an updated model slips by.
mutant ignores_pin     "moved repo gets a warning mark" \
  's/if \[ "\$sha" = "\$pin" \]; then/if true; then/'
# A deleted pinned commit counts as success.
mutant gone_pin_ok     "pinned commit gone -> exit 1" \
  's/^        worse 1$/        : worse 1/'
# Skips the 40-character check, so a short or partial pin is accepted.
mutant short_pin_ok    "short pinned commit -> exit 2" \
  '/must be 40 hex characters/d'
# Lets a repo ID carry URL characters from the pins file into the request.
mutant repo_unchecked  "repo ID with URL characters -> exit 2" \
  's/\*\/\*\/\*|\*\[!A-Za-z0-9._\/-\]\*|\/\*|\*\/) die 2/*\/*\/*) die 2/'

printf '\n%d of %d planted flaws caught\n' "$caught" "$total"
[ "$caught" -eq "$total" ]
