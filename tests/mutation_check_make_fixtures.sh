# mutation_check_make_fixtures.sh: proves the make_fixtures.py tests can fail.
#
# For each mutant, writes a temporary copy of scripts/make_fixtures.py with
# one deliberate flaw planted, runs tests/test_make_fixtures.sh against that
# copy (via FX_SCRIPT), and checks that the matching test fails. An
# unmodified copy runs first and must pass. The real script is never
# modified. No network.
#
# Usage: bash tests/mutation_check_make_fixtures.sh

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../scripts/make_fixtures.py"
SUITE="$HERE/test_make_fixtures.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
caught=0
total=0

cp "$SRC" "$TMP/baseline.py"
if base="$(FX_SCRIPT="$TMP/baseline.py" bash "$SUITE" 2>&1)"; then
  printf 'baseline: unmodified copy passes (%s)\n\n' "$(printf '%s\n' "$base" | tail -1)"
else
  printf 'baseline FAILED; fix the tests before trusting this check:\n%s\n' \
    "$(printf '%s\n' "$base" | grep FAIL)"
  exit 1
fi

# mutant NAME "LABEL OF THE TEST THAT MUST FAIL" SED_PROGRAM
mutant() {
  local name="$1" expect="$2" prog="$3" target="$TMP/$1.py" out rc
  total=$((total + 1))
  if ! sed -e "$prog" "$SRC" > "$target" || [ ! -s "$target" ]; then
    printf 'BROKEN  %-15s sed failed or produced an empty file; update this mutant\n' "$name"
    return
  fi
  if cmp -s "$SRC" "$target"; then
    printf 'BROKEN  %-15s the planted change did not apply; update this mutant\n' "$name"
    return
  fi
  out="$(FX_SCRIPT="$target" bash "$SUITE" 2>&1)"
  rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qF "FAIL  $expect"; then
    caught=$((caught + 1))
    printf 'caught  %-15s -> FAIL  %s\n' "$name" "$expect"
  else
    printf 'MISSED  %-15s expected: FAIL  %s\n' "$name" "$expect"
  fi
}

# Takes x mod n with no rejection, so small outcomes come up slightly more often.
mutant no_rejection   "derive() agrees with bash where rejection matters" \
  's/^        if x < limit:$/        if True:/'
# Floor range one wider than the ZOPA spec.
mutant wide_range     "all 30 fixtures match the bash re-derivation" \
  's/^F_RANGE = (75, 95)/F_RANGE = (75, 96)/'
# Draws the floor with the ceiling's label, so the two are correlated.
mutant shared_label   "all 30 fixtures match the bash re-derivation" \
  's|f"fixture/{i}/f"|f"fixture/{i}/c"|'
# Skips the shuffles: games run in fixture order, scales always in the same order.
mutant no_shuffle     "run order and every F, C, opener match the bash re-derivation" \
  's/^        items\[j\], items\[r\] = items\[r\], items\[j\]$/        pass/'
# Plain shuffle of all 180 games: paired games no longer run back to back.
mutant no_blocks      "each block's three games run back to back" \
  's/^    for pos, g in enumerate(games, start=1):$/    shuffle(games, seed, "flat")\
    for pos, g in enumerate(games, start=1):/'
# Breaks the pairing: the floor creeps up 1 point at the bigger scales.
mutant broken_pair    "F and C are the same percentages of V at every scale (the pairing)" \
  's|^                floor = fx\["f_pct"\] \* v // 100$|                floor = (fx["f_pct"] + (v > 1000)) * v // 100|'
# Drops the buyer-opens games, leaving each seller game twice.
mutant one_opener     "each fixture x scale x opener exactly once" \
  's/^OPENERS = \[("S", "seller"), ("B", "buyer")\]/OPENERS = [("S", "seller"), ("B", "seller")]/'
# --check always says "matches".
mutant check_lies     "--check catches a hand-edited fixture" \
  's/^        if on_disk == text:$/        if True:/'
# Seed taken from a fixed constant instead of the phrase.
mutant fixed_seed     "a different phrase gives different fixtures" \
  's/^    return hashlib.sha256(phrase.encode("utf-8")).hexdigest()$/    return hashlib.sha256(DEFAULT_PHRASE.encode("utf-8")).hexdigest()/'
# Accepts an empty phrase.
mutant empty_phrase   "empty phrase -> exit 2" \
  's/^    if not args.phrase or not args.phrase.isprintable():$/    if False:/'

printf '\n%d of %d planted flaws caught\n' "$caught" "$total"
[ "$caught" -eq "$total" ]
