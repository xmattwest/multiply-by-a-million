# test_make_fixtures.sh: offline tests for scripts/make_fixtures.py.
#
# The key test re-derives every fixture and the whole run order in bash with
# shasum, from the written-down recipe alone (no Python), and compares with
# both a fresh run of the generator and the committed config/fixtures.json.
# If the two implementations agree, the recipe in the file is enough for
# anyone to check our fixtures were not picked after the fact.
#
# Usage: bash tests/test_make_fixtures.sh   (FX_SCRIPT overrides the script)

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${FX_SCRIPT:-$HERE/../scripts/make_fixtures.py}"
REAL="$HERE/../config/fixtures.json"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PHRASE="scale-invariance/fixtures/v1"

PASS=0
FAIL=0
RC=0
pass() { PASS=$((PASS + 1)); printf 'ok    %s\n' "$1"; }
fail() { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1"; }
assert_rc() { if [ "$RC" -eq "$1" ]; then pass "$2"; else fail "$2 (exit $RC, want $1)"; fi; }
run() { set +e; "$@" > "$TMP/out.txt" 2>&1; RC=$?; set -e; }

# --- Independent re-derivation (bash + shasum) -----------------------------------
SEED="$(printf '%s' "$PHRASE" | shasum -a 256 | cut -d' ' -f1)"
TWO48=$((1 << 48))
# derive LABEL N: prints a whole number in 0..N-1 (recipe in make_fixtures.py).
derive() {
  local label="$1" n="$2" k=0 h x limit
  limit=$(( (TWO48 / n) * n ))
  while :; do
    h="$(printf '%s|%s|%s' "$SEED" "$label" "$k" | shasum -a 256 | cut -c1-12)"
    x=$((16#$h))
    if [ "$x" -lt "$limit" ]; then echo $((x % n)); return; fi
    k=$((k + 1))
  done
}

: > "$TMP/fx_bash.txt"
FS=(); CS=()
i=1
while [ "$i" -le 30 ]; do
  f=$((75 + $(derive "fixture/$i/f" 21)))
  c=$((105 + $(derive "fixture/$i/c" 21)))
  FS[$i]=$f; CS[$i]=$c
  printf 'F%02d %d %d\n' "$i" "$f" "$c" >> "$TMP/fx_bash.txt"
  i=$((i + 1))
done
# Blocks: one per fixture x opener (canonical order), each listing its
# three scales. Shuffle the blocks, then the scales inside each block.
BK=()
i=1
while [ "$i" -le 30 ]; do
  for o in S B; do BK[${#BK[@]}]="$(printf 'F%02d %s' "$i" "$o")"; done
  i=$((i + 1))
done
j=$((${#BK[@]} - 1))
while [ "$j" -ge 1 ]; do
  r="$(derive "block/$j" $((j + 1)))"
  t="${BK[$j]}"; BK[$j]="${BK[$r]}"; BK[$r]="$t"
  j=$((j - 1))
done
: > "$TMP/games_bash.txt"
b=1
for blk in "${BK[@]}"; do
  fid="${blk% *}"; o="${blk#* }"; n=$((10#${fid#F}))
  if [ "$o" = S ]; then who=seller; else who=buyer; fi
  SC=(x1e2:100 x1e5:100000 x1e8:100000000)
  j=2
  while [ "$j" -ge 1 ]; do
    r="$(derive "within/$b/$j" $((j + 1)))"
    t="${SC[$j]}"; SC[$j]="${SC[$r]}"; SC[$r]="$t"
    j=$((j - 1))
  done
  for s in "${SC[@]}"; do
    tag="${s%%:*}"; v="${s#*:}"
    printf '%s-%s-%s %d %d %s %d\n' "$fid" "$tag" "$o" \
      $((FS[n] * v / 100)) $((CS[n] * v / 100)) "$who" "$b" >> "$TMP/games_bash.txt"
  done
  b=$((b + 1))
done

fixtures_of() { jq -r '.fixtures[] | "\(.id) \(.f_pct) \(.c_pct)"' "$1"; }
games_of() { jq -r '.games | sort_by(.run_order)[] | "\(.game_id) \(.F) \(.C) \(.opener) \(.block)"' "$1"; }

echo "== generator output vs. independent bash re-derivation =="
GEN="$TMP/gen.json"
run python3 "$SCRIPT" --out "$GEN"
assert_rc 0 "generator exits 0"
if [ "$(jq -r .seed_sha256 "$GEN")" = "$SEED" ]; then pass "seed is SHA-256 of the public phrase"; else fail "seed is SHA-256 of the public phrase"; fi
if [ "$(jq -r .seed_phrase "$GEN")" = "$PHRASE" ]; then pass "phrase recorded in the file"; else fail "phrase recorded in the file"; fi
if fixtures_of "$GEN" | cmp -s - "$TMP/fx_bash.txt"; then pass "all 30 fixtures match the bash re-derivation"; else fail "all 30 fixtures match the bash re-derivation"; fi
if games_of "$GEN" | cmp -s - "$TMP/games_bash.txt"; then pass "run order and every F, C, opener match the bash re-derivation"; else fail "run order and every F, C, opener match the bash re-derivation"; fi

echo "== the committed file =="
if [ -f "$REAL" ]; then
  if fixtures_of "$REAL" | cmp -s - "$TMP/fx_bash.txt" && games_of "$REAL" | cmp -s - "$TMP/games_bash.txt"; then
    pass "config/fixtures.json matches the bash re-derivation"; else fail "config/fixtures.json matches the bash re-derivation"; fi
else
  fail "config/fixtures.json exists"
fi

echo "== rejection sampling (exactly uniform draws) =="
# With n just above 2^47, about half of all 48-bit values get rejected, so a
# version without rejection disagrees with bash on these probes.
N=$(( (1 << 47) + 1 ))
probe_py="$(python3 - "$SCRIPT" "$SEED" "$N" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("mf", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(" ".join(str(m.derive(sys.argv[2], f"probe/{i}", int(sys.argv[3]))) for i in range(1, 9)))
PY
)"
probe_sh=""
for i in 1 2 3 4 5 6 7 8; do probe_sh="$probe_sh $(derive "probe/$i" "$N")"; done
if [ "$probe_py" = "${probe_sh# }" ]; then pass "derive() agrees with bash where rejection matters"; else fail "derive() agrees with bash where rejection matters"; fi

echo "== structure =="
check() {  # check "jq expression that must be true" "label"
  if jq -e "$1" "$GEN" >/dev/null 2>&1; then pass "$2"; else fail "$2"; fi
}
check '.fixtures | length == 30' "30 fixtures"
check '.games | length == 180' "180 games"
check '[.games[].game_id] | unique | length == 180' "game IDs unique"
check '[.games[] | [.fixture, .scale, .opener]] | unique | length == 180' "each fixture x scale x opener exactly once"
check '[.games[].run_order] | sort == [range(1;181)]' "run order is 1..180"
check 'all(.fixtures[]; .f_pct >= 75 and .f_pct <= 95 and .c_pct >= 105 and .c_pct <= 125)' "factors inside the ZOPA ranges"
check '.fixtures as $fx | all(.games[]; . as $g | ($fx[] | select(.id == $g.fixture)) as $f
       | $g.F * 100 == $f.f_pct * $g.V and $g.C * 100 == $f.c_pct * $g.V)' "F and C are the same percentages of V at every scale (the pairing)"
check 'all(.games[]; .F < .V and .V < .C)' "floor < V < ceiling in every game"
check 'all(.games[]; (.F|type) == "number" and (.F|floor) == .F and (.C|floor) == .C)' "prices are whole numbers"
check '[.games[] | select(.scale == "x1e8")] | all(.C <= 125000000)' "largest ceiling is 125,000,000"
check '[.games[].V] | unique == [100, 100000, 100000000]' "the three scales"
check '.games | group_by(.block) | length == 60 and all(.[]; length == 3 and ([.[].scale] | sort) == ["x1e2","x1e5","x1e8"] and ([.[] | [.fixture, .opener]] | unique | length) == 1)' "60 blocks, each one fixture and opener at all three scales"
check '.games | sort_by(.run_order) | [.[].block] == ([range(1;61)] | map(., ., .))' "each block's three games run back to back"
if grep -q '[^ -~]' "$GEN"; then fail "file is plain ASCII"; else pass "file is plain ASCII"; fi

echo "== deterministic, and the phrase matters =="
run python3 "$SCRIPT" --out "$TMP/gen2.json"
if cmp -s "$GEN" "$TMP/gen2.json"; then pass "two runs give byte-identical files"; else fail "two runs give byte-identical files"; fi
run python3 "$SCRIPT" --phrase "scale-invariance/fixtures/v2" --out "$TMP/other.json"
if fixtures_of "$TMP/other.json" | cmp -s - "$TMP/fx_bash.txt"; then fail "a different phrase gives different fixtures"; else pass "a different phrase gives different fixtures"; fi

echo "== --check =="
run python3 "$SCRIPT" --check --out "$GEN"
assert_rc 0 "--check passes on an untouched file"
sed 's/"f_pct": 81,/"f_pct": 82,/' "$GEN" > "$TMP/edited.json"
if cmp -s "$GEN" "$TMP/edited.json"; then fail "test setup: edit applied"; fi
run python3 "$SCRIPT" --check --out "$TMP/edited.json"
assert_rc 1 "--check catches a hand-edited fixture"
run python3 "$SCRIPT" --check --out "$TMP/missing.json"
assert_rc 2 "--check on a missing file -> exit 2"
run python3 "$SCRIPT" --phrase "" --out "$TMP/x.json"
assert_rc 2 "empty phrase -> exit 2"
if [ -e "$TMP/x.json" ]; then fail "empty phrase: nothing written"; else pass "empty phrase: nothing written"; fi

printf '\npassed: %d  failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
