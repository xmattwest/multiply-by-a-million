# mutation_check_make_manifests.sh: proves the make_manifests.sh tests can fail.
#
# For each mutant, writes a temporary copy of scripts/make_manifests.sh with
# one deliberate flaw planted, runs tests/test_make_manifests.sh against
# that copy (via MM_SCRIPT), and checks that the matching test fails. An
# unmodified copy runs first and must pass. The real script is never
# modified. No network.
#
# Usage: bash tests/mutation_check_make_manifests.sh

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../scripts/make_manifests.sh"
SUITE="$HERE/test_make_manifests.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
caught=0
total=0

cp "$SRC" "$TMP/baseline.sh"
if base="$(MM_SCRIPT="$TMP/baseline.sh" bash "$SUITE" 2>&1)"; then
  printf 'baseline: unmodified copy passes (%s)\n\n' "$(printf '%s\n' "$base" | tail -1)"
else
  printf 'baseline FAILED; fix the tests before trusting this check:\n%s\n' \
    "$(printf '%s\n' "$base" | grep FAIL)"
  exit 1
fi

# mutant NAME "LABEL OF THE TEST THAT MUST FAIL" SED_PROGRAM
mutant() {
  local name="$1" expect="$2" prog="$3" target="$TMP/$1.sh" out rc
  total=$((total + 1))
  if ! sed -e "$prog" "$SRC" > "$target" || [ ! -s "$target" ]; then
    printf 'BROKEN  %-15s sed failed or produced an empty file; update this mutant\n' "$name"
    return
  fi
  if cmp -s "$SRC" "$target"; then
    printf 'BROKEN  %-15s the planted change did not apply; update this mutant\n' "$name"
    return
  fi
  out="$(MM_SCRIPT="$target" bash "$SUITE" 2>&1)"
  rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qF "FAIL  $expect"; then
    caught=$((caught + 1))
    printf 'caught  %-15s -> FAIL  %s\n' "$name" "$expect"
  else
    printf 'MISSED  %-15s expected: FAIL  %s\n' "$name" "$expect"
  fi
}

# Ignores paging, so a long file list would be silently cut short.
mutant ignores_pages  "file list split into pages -> exit 3" \
  '/split into pages; this script reads only one page/d'
# Lets a ../ path from the server into the manifest.
mutant path_unchecked "big file with a ../ path -> exit 3" \
  's/^    safe_path "\$path" || die 3 "unexpected file name in \$repo"$/    :/'
# Accepts a fingerprint of any length.
mutant short_oid_ok   "short fingerprint -> exit 3" \
  '/fingerprint is not 64 hex characters/d'
# Doesn't check that the downloaded small file has the listed size.
mutant no_size_check  "small file downloaded short -> exit 3" \
  '/downloaded \$got bytes, file list says \$size/d'
# Writes each manifest as soon as it's built, so a later failure leaves a mix.
mutant not_atomic     "a failure leaves every old manifest untouched" \
  's|} > "\$TMP/out/\$name.tsv"|} > "\$TMP/out/\$name.tsv"; mkdir -p "\$OUTDIR"; cp "\$TMP/out/\$name.tsv" "\$OUTDIR/"|'
# Leaves Hugging Face token variables in the environment curl inherits.
mutant token_inherited "HF token variables not inherited by curl" \
  '/^unset HF_TOKEN /d'
# Lets ~/.curlrc settings apply.
mutant reads_curlrc   "-q is first on every curl call" \
  's/curl -q -sS/curl -sS/'
# Validates pins lazily, so a bad later pin is found after requests were made.
mutant lazy_pins      "bad repo in a later model: no request made" \
  's/^  case "\$repo" in \*\/\*\/\*|.*die 2 "bad repo in pins for \$name" ;; esac$/  :/'
# Drops the sort, so file order depends on the server's answer.
mutant unsorted       "lines sorted by path" \
  '/^  sort -o "\$out" "\$out"$/d'
# Allows the same file to be listed twice.
mutant dupes_ok       "same path listed twice -> exit 3" \
  's/\[ -z "\$dups" \] || die 3/: || die 3/'

printf '\n%d of %d planted flaws caught\n' "$caught" "$total"
[ "$caught" -eq "$total" ]
