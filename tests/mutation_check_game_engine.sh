# mutation_check_game_engine.sh: proves the game_engine.py tests can fail.
#
# For each mutant, writes a temporary copy of scripts/game_engine.py with
# one deliberate flaw planted, runs tests/test_game_engine.py against that
# copy (via GE_SCRIPT), and checks that the matching test fails or crashes. An
# unmodified copy runs first and must pass. The real script is never
# modified. No network.
#
# Usage: bash tests/mutation_check_game_engine.sh

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../scripts/game_engine.py"
SUITE="$HERE/test_game_engine.py"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
caught=0
total=0

# The script finds its templates at ../prereg/prompts, so copies live in
# $TMP/scripts/ next to a link to the real prereg/ folder.
mkdir -p "$TMP/scripts"
ln -s "$(cd "$HERE/.." && pwd)/prereg" "$TMP/prereg"
cp "$HERE/../scripts/render_prompt.py" "$TMP/scripts/"
cp "$SRC" "$TMP/scripts/baseline.py"
if base="$(GE_SCRIPT="$TMP/scripts/baseline.py" python3 "$SUITE" 2>&1)"; then
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
  out="$(GE_SCRIPT="$target" python3 "$SUITE" 2>&1)"
  rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qE "^(FAIL|ERROR): $expect "; then
    caught=$((caught + 1))
    printf 'caught  %-15s -> FAIL: %s\n' "$name" "$expect"
  else
    printf 'MISSED  %-15s expected: FAIL: %s\n' "$name" "$expect"
  fi
}

# Uses Python's default JSON reader: duplicate keys are silently merged.
mutant dupes_ok       "test_shared_cases" \
  's/^    dec = json.JSONDecoder(object_pairs_hook=_no_dupes, parse_constant=_no_constants)$/    dec = json.JSONDecoder()/'
# Forgets that True is an int in Python.
mutant bool_price     "test_shared_cases" \
  's/^        elif isinstance(price, bool) or not isinstance(price, int):$/        elif not isinstance(price, int):/'
# "Repairs" replies by cutting everything after the first JSON object.
mutant repairs        "test_shared_cases" \
  's/^    if end != len(text):$/    if False:/'
# Lets a truncated reply through.
mutant truncation_ok  "test_shared_cases" \
  's/^    if finish_reason == "length":$/    if False:/'
# Puts the scale into the seed, so paired games no longer share random numbers.
mutant seed_by_scale  "test_same_seeds_at_every_scale" \
  's/^        seed = call_seed(game\["fixture"\], game\["opener"\], msg_no, attempt)$/        seed = call_seed(game["fixture"] + game["scale"], game["opener"], msg_no, attempt)/'
# Leaves top_k to the server default (it becomes "off" with --generation-config vllm).
mutant no_top_k       "test_every_sampling_field_sent_with_the_registered_value" \
  '/^        "top_k": samp\["top_k"\],$/d'
# Turns thinking on.
mutant thinking_on    "test_every_sampling_field_sent_with_the_registered_value" \
  's/"enable_thinking": False/"enable_thinking": True/'
# Gives the seller the buyer's ceiling in its prompt.
mutant leaks_limit    "test_each_side_sees_only_its_own_limit" \
  's/^        limit = game\["F"\] if role == "seller" else game\["C"\]$/        limit = game["C"]/'
# Accept executes the accepting side's own last offer instead of the other side's.
mutant wrong_price    "test_agreement" \
  's/^            terminal, price, ended_by = "AGREEMENT", history\[-1\]\[1\], role$/            terminal, price, ended_by = "AGREEMENT", history[0][1], role/'
# Allows a 17th message.
mutant cap_17         "test_timeout_after_sixteen" \
  's/^    for msg_no in range(1, max_msgs + 1):$/    for msg_no in range(1, max_msgs + 2):/'
# Scores an infrastructure failure like a walk-away.
mutant void_scored    "test_infrastructure_failure_is_void_and_unscored" \
  's/^    if terminal != "INFRASTRUCTURE_VOID":$/    if True:/'
# Clamps the seller share into [0, 1].
mutant clamps         "test_accepting_below_your_floor_is_a_violation" \
  's/^        s = Fraction(price - F, width)$/        s = min(max(Fraction(price - F, width), Fraction(0)), Fraction(1))/'
# Ignores a non-empty reasoning field.
mutant reasoning_ok   "test_reasoning_is_flagged" \
  's/^        if reply.get("reasoning") not in (None, ""):$/        if False:/'

printf '\n%d of %d planted flaws caught\n' "$caught" "$total"
[ "$caught" -eq "$total" ]
