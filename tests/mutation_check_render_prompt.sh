# mutation_check_render_prompt.sh: proves the render_prompt.py tests can fail.
#
# For each mutant, writes a temporary copy of scripts/render_prompt.py with
# one deliberate flaw planted, runs tests/test_render_prompt.py against that
# copy (via RP_SCRIPT), and checks that the matching test fails or crashes. An
# unmodified copy runs first and must pass. The real script is never
# modified. No network.
#
# Usage: bash tests/mutation_check_render_prompt.sh

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../scripts/render_prompt.py"
SUITE="$HERE/test_render_prompt.py"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
caught=0
total=0

# The script finds its templates at ../prereg/prompts, so copies live in
# $TMP/scripts/ next to a link to the real prereg/ folder.
mkdir -p "$TMP/scripts"
ln -s "$(cd "$HERE/.." && pwd)/prereg" "$TMP/prereg"
cp "$SRC" "$TMP/scripts/baseline.py"
if base="$(RP_SCRIPT="$TMP/scripts/baseline.py" python3 "$SUITE" 2>&1)"; then
  printf 'baseline: unmodified copy passes (%s)\n\n' "$(printf '%s\n' "$base" | grep -E '^Ran ')"
else
  printf 'baseline FAILED; fix the tests before trusting this check:\n%s\n' \
    "$(printf '%s\n' "$base" | grep -E '^(FAIL|ERROR):')"
  exit 1
fi

# mutant NAME "LABEL OF THE TEST THAT MUST FAIL" SED_PROGRAM
mutant() {
  local name="$1" expect="$2" prog="$3" target="$TMP/scripts/$1.py" out rc
  total=$((total + 1))
  if ! sed -e "$prog" "$SRC" > "$target" || [ ! -s "$target" ]; then
    printf 'BROKEN  %-15s sed failed or produced an empty file; update this mutant\n' "$name"
    return
  fi
  if cmp -s "$SRC" "$target"; then
    printf 'BROKEN  %-15s the planted change did not apply; update this mutant\n' "$name"
    return
  fi
  out="$(RP_SCRIPT="$target" python3 "$SUITE" 2>&1)"
  rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qE "^(FAIL|ERROR): $expect "; then
    caught=$((caught + 1))
    printf 'caught  %-15s -> FAIL: %s\n' "$name" "$expect"
  else
    printf 'MISSED  %-15s expected: FAIL: %s\n' "$name" "$expect"
  fi
}

# Writes money with thousands separators: the scales then differ in more than digits.
mutant commas         "test_same_words_at_every_scale" \
  's/^    return f"{x} dollars"$/    return f"{x:,} dollars"/'
# Calls big amounts "USD" and small ones "dollars": a wording change by scale.
mutant word_by_scale  "test_same_words_at_every_scale" \
  's/^    return f"{x} dollars"$/    return f"{x} {chr(85)*0}{\x27dollars\x27 if x < 10**6 else \x27USD\x27}"/'
# Pastes the other model's message in raw, so it can pose as a new prompt line.
mutant raw_text       "test_injected_text_stays_literal" \
  's/^                "TEXT": json.dumps(text, ensure_ascii=False),$/                "TEXT": text,/'
# Shows the side's first offer as the one on the table instead of the latest.
mutant stale_pending  "test_pending_offer_is_the_other_sides_last" \
  's/money(history\[-1\]\[1\])})/money(history[0][1])})/'
# Labels the side's own earlier offers with its role name instead of "you".
mutant no_you         "test_pending_offer_is_the_other_sides_last" \
  's/"WHO": "you" if who == role else f"the {who}",/"WHO": f"the {who}",/'
# Stops checking that turns alternate.
mutant no_alternation "test_bad_histories_refused" \
  's/^        if who != expect:$/        if False:/'
# Lets a value be passed and silently not used.
mutant unused_ok      "test_unknown_or_unused_placeholder_is_an_error" \
  's/^    if unused:$/    if False:/'
# Accepts True as a price (Python treats it as the number 1).
mutant bool_money     "test_money_is_strict" \
  's/^    if isinstance(x, bool) or not isinstance(x, int) or x < 1:$/    if not isinstance(x, int) or x < 1:/'
# Gives render() the other side's limit as an extra input.
mutant sees_other     "test_render_cannot_see_the_other_limit" \
  's/^def render(role, V, limit, history,/def render(role, V, limit, C=None, *, history=None, _h=None, **kw): pass\ndef _unused(role, V, limit, history,/'

printf '\n%d of %d planted flaws caught\n' "$caught" "$total"
[ "$caught" -eq "$total" ]
