#!/bin/bash
# test_launch_pod.sh: quick offline checks of the launcher's money safeguards.
#
# Runs a copy of scripts/ and config/ in a scratch folder (so the real
# logs/budget.tsv is never touched) against a fake RunPod API (fake curl)
# and a fake Keychain (fake security). Covers: refusing on budget, on an
# existing pod, on an unconfirmed cost; creating exactly once; and the
# teardown trap deleting the pod after every kind of failure.
# A quick check, not an exhaustive one (decided 2026-10-03: lighter testing).
#
# Usage: bash tests/test_launch_pod.sh

set -uo pipefail
export LC_ALL=C
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
CANARY="rpa_WRITEKEYCANARY0000000000000000000000000"
GPU="NVIDIA RTX PRO 6000 Blackwell Server Edition"
PASS=0; FAIL=0
pass() { PASS=$((PASS + 1)); printf 'ok    %s\n' "$1"; }
fail() { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1"; }
eq() { if [ "$1" = "$2" ]; then pass "$3"; else fail "$3 (got '$1', want '$2')"; fi; }
has() { if grep -qF -- "$2" "$1" 2>/dev/null; then pass "$3"; else fail "$3"; fi; }

mkdir -p "$TMP/bin" "$TMP/home/.ssh"
ssh-keygen -q -t ed25519 -N '' -C scale-invariance-pod -f "$TMP/home/.ssh/scale-invariance_ed25519"
cat > "$TMP/bin/security" <<EOF
#!/bin/bash
echo "$CANARY"
EOF
cat > "$TMP/bin/curl" <<'EOF'
#!/bin/bash
# Fake RunPod API. State lives in $FAKE/state: "none", "pod:<status>", "deleted".
F="${FAKE:?}"; { printf -- '---\n'; printf '%s\n' "$@"; } >> "$F/argv"
cat >> "$F/stdin"
m=GET; out=/dev/null; url=""; prev=""; body=""
for a in "$@"; do case "$prev" in -X) m="$a" ;; -o) out="$a" ;; --data-binary) body="${a#@}" ;; esac; case "$a" in https://*) url="$a" ;; esac; prev="$a"; done
p="${url#https://api.runpod.io/v2}"; st="$(cat "$F/state")"
reply() { printf '%s' "$2" > "$out"; printf '%s' "$1"; exit 0; }
pods() { if [ "$st" = deleted ] || [ "$st" = none ]; then echo "${FAKE_OTHER:-}"; else echo "{\"id\":\"pod1\",\"name\":\"$(cat "$F/name")\"}"; fi; }
case "$m $p" in
  "GET /catalog/gpus"*) reply 200 '[{"id":"NVIDIA RTX PRO 6000 Blackwell Server Edition","price":{"secure":2.09},"availability":"LOW"}]' ;;
  "GET /catalog/datacenters"*) reply 200 '{"dataCenters":[{"id":"US-NE-1","region":"NORTH_AMERICA","gpuAvailability":[{"id":"NVIDIA RTX PRO 6000 Blackwell Server Edition","availability":"LOW"}]},{"id":"EUR-IS-2","region":"EUROPE","gpuAvailability":[{"id":"NVIDIA RTX PRO 6000 Blackwell Server Edition","availability":"LOW"}]},{"id":"US-KS-2","region":"NORTH_AMERICA","gpuAvailability":[{"id":"NVIDIA H100 NVL","availability":"LOW"}]}]}' ;;
  "GET /pods") reply 200 "{\"pods\":[$(pods)],\"pagination\":{\"hasNextPage\":false}}" ;;
  "POST /pods")
    echo post >> "$F/posts"
    cp "$body" "$F/create_body.json" 2>/dev/null
    jq -r .name "$body" > "$F/name"
    case "${FAKE_CREATE:-ok}" in
      ok)        echo "pod:${FAKE_STATUS:-ERROR}" > "$F/state"; reply 201 '{"id":"pod1"}' ;;
      503made)   echo "pod:RUNNING" > "$F/state"; reply 503 '{"title":"upstream"}' ;;
      400)       reply 400 '{"detail":"no capacity"}' ;;
    esac ;;
  "GET /pods/pod1")
    case "$st" in deleted|none) reply 404 '{}' ;; pod:*) reply 200 "{\"id\":\"pod1\",\"status\":\"${st#pod:}\"}" ;; esac ;;
  "DELETE /pods/pod1")
    echo del >> "$F/deletes"
    [ "${FAKE_DELETE:-ok}" = ok ] && { echo deleted > "$F/state"; reply 204 ''; }
    reply 500 '{}' ;;
esac
printf '%s %s\n' "$m" "$p" >> "$F/unexpected"; printf '000'; exit 7
EOF
chmod +x "$TMP/bin/"*

# run NAME [VAR=value ...] -- [launcher args]
run() {
  local name="$1"; shift
  local envs=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  W="$TMP/w_$name"; rm -rf "$W"; mkdir -p "$W/logs" "$W/fake"
  cp -R "$HERE/../scripts" "$HERE/../config" "$HERE/../prereg" "$W/"
  printf '{"total_usd": 15.00}\n' > "$W/config/budget.json"   # these checks assume a $15 cap
  echo none > "$W/fake/state"
  [ -n "${LEDGER_SEED:-}" ] && printf '%b' "$LEDGER_SEED" > "$W/logs/budget.tsv"
  [ -n "${PRE:-}" ] && eval "$PRE"      # seed files in the scratch workspace (slot tests)
  OUT="$W/out"
  env -i PATH="$TMP/bin:/usr/bin:/bin:/opt/homebrew/bin" HOME="$TMP/home" USER=tester FAKE="$W/fake" \
    SI_TEARDOWN_SLEEP=0 SI_LOOKUP_SLEEP=0 SI_SLOT="${TEST_SLOT:-7}" ${envs[@]+"${envs[@]}"} \
    perl -e 'alarm shift; exec @ARGV' 60 /bin/bash "$W/scripts/launch_pod.sh" "$@" > "$OUT" 2>&1
  RC=$?
}
posts() { [ -f "$W/fake/posts" ] && grep -c . "$W/fake/posts" || echo 0; }
no_leak() {
  if grep -rlF -- "$CANARY" "$W" 2>/dev/null | grep -qv '/fake/stdin$'; then fail "$1: write key leaked"; else pass "$1: write key only on curl's stdin"; fi
}

echo "== dry run =="
run dry -- --mode probe --lifetime-min 15
eq "$RC" 0 "dry run exits 0"
has "$OUT" "worst case:     \$1.23" "worst case = (15 + 20 min) x \$2.09 + disk"
has "$OUT" "--confirm-cost 1.23" "tells you the number to confirm"
eq "$(posts)" 0 "dry run creates nothing"
run wrongconfirm -- --mode probe --lifetime-min 15 --confirm-cost 1.00
eq "$(posts)" 0 "a mismatched --confirm-cost creates nothing"

echo "== refusals before creating =="
run dup FAKE_OTHER='{"id":"x","name":"si-probe-old"}' -- --mode probe --lifetime-min 15 --confirm-cost 1.23
eq "$RC" 4 "an existing si- pod blocks the launch"
eq "$(posts)" 0 "dup: nothing created"
LEDGER_SEED='utc\tpod_name\tworst_usd\testimate_usd\tstatus\nx\tsi-a\t20.00\t13.90\tclosed\n'
run budget -- --mode probe --lifetime-min 15 --confirm-cost 1.23
eq "$RC" 4 "a launch that could pass \$15 is refused"
has "$OUT" "\$13.90 used or reserved" "closed launches count at their estimate"
eq "$(posts)" 0 "budget: nothing created"
LEDGER_SEED='utc\tpod_name\tworst_usd\testimate_usd\tstatus\nx\tsi-a\t14.00\t0.50\tUNCONFIRMED\n'
run unconfirmed -- --mode probe --lifetime-min 15 --confirm-cost 1.23
eq "$RC" 4 "an unconfirmed teardown counts at its worst case"
LEDGER_SEED=""

echo "== create once, always tear down =="
run boots_then_errors FAKE_STATUS=ERROR -- --mode probe --lifetime-min 15 --confirm-cost 1.23
eq "$RC" 6 "pod in ERROR: stops"
eq "$(posts)" 1 "created exactly once"
has "$OUT" "✓ pod deleted; pods of ours left on the account: 0" "teardown deletes and confirms"
has "$W/logs/budget.tsv" "closed" "ledger records the launch as closed"
no_leak boots_then_errors
run create503 FAKE_CREATE=503made -- --mode probe --lifetime-min 15 --confirm-cost 1.23
eq "$(posts)" 1 "a 503 on create is not retried"
eq "$(grep -c . "$W/fake/deletes" 2>/dev/null || echo 0)" 1 "the pod the 503 secretly made is found by name and deleted"
has "$OUT" "pods of ours left on the account: 0" "create503: none left"
run create400 FAKE_CREATE=400 -- --mode probe --lifetime-min 15 --confirm-cost 1.23
eq "$RC" 5 "a 400 on create stops"
eq "$(posts)" 1 "create400: not retried"
has "$OUT" "no capacity" "create400: shows RunPod's reason"
run stuck FAKE_STATUS=EXITED FAKE_DELETE=fail -- --mode probe --lifetime-min 5 --confirm-cost 0.88
has "$OUT" "COULD NOT CONFIRM THE POD IS GONE" "an undeletable pod is shouted about"
eq "$RC" 9 "and exits 9"
has "$W/logs/budget.tsv" "UNCONFIRMED" "ledger keeps its worst case"
eq "$(grep -c . "$W/fake/deletes")" 6 "delete was tried 6 times"

echo "== interp mode (additive) =="
run interp_noplugs -- --mode interp --model gemma-4-12b-it --lifetime-min 60
eq "$RC" 2 "interp without --interp-plugins is refused"
has "$OUT" "needs --interp-plugins" "says the plug-in list is required"
run interp_badplug -- --mode interp --model gemma-4-12b-it --lifetime-min 60 --interp-plugins nosuch
eq "$RC" 2 "an unknown plug-in is refused"
eq "$(posts)" 0 "unknown plug-in: nothing created"
run probe_plugs -- --mode probe --lifetime-min 15 --interp-plugins none
eq "$RC" 2 "--interp-plugins outside interp mode is refused"
run interp_dry -- --mode interp --model gemma-4-12b-it --lifetime-min 60 --interp-plugins concepts,e4,heads
eq "$RC" 0 "interp dry run exits 0"
has "$OUT" "interp code:    plug-ins concepts,e4,heads; create request" "dry run reports the plug-ins and the request size"
has "$OUT" "Launch plan (interp)" "interp dry run names the mode"
has "$OUT" "model:          gemma-4-12b-it" "interp dry run resolves the model (serve plumbing path)"
has "$OUT" "--confirm-cost" "interp dry run prints a worst-case to confirm"
eq "$(posts)" 0 "interp dry run creates nothing"
run interp_badmodel -- --mode interp --model no-such-model --lifetime-min 60
eq "$RC" 2 "interp rejects an unknown model"
eq "$(posts)" 0 "interp bad model: nothing created"
# A confirmed interp launch: the create body must carry the interp boot script
# and the interp code env. Reuse the worst case the dry run just printed.
IWORST="$(sed -n 's/.*--confirm-cost \([0-9.]*\).*/\1/p' "$TMP/w_interp_dry/out" | head -1)"
run interp_create FAKE_STATUS=ERROR -- --mode interp --model gemma-4-12b-it --lifetime-min 60 --interp-plugins concepts,e4,heads --confirm-cost "$IWORST"
eq "$(posts)" 1 "interp created exactly once"
if jq -e '.cmd[0] | contains("boot_interp.sh")' "$W/fake/create_body.json" >/dev/null 2>&1; then
  pass "interp create body runs boot_interp.sh"; else fail "interp create body runs boot_interp.sh"; fi
if jq -e '.env.SI_INTERP_B64 and .env.SI_INTERP_SHA256 and .env.SI_MODEL_NAME == "gemma-4-12b-it"' \
    "$W/fake/create_body.json" >/dev/null 2>&1; then
  pass "interp create body carries the interp code + model env"; else fail "interp create body carries the interp code + model env"; fi
LIST="$(jq -r .env.SI_INTERP_B64 "$W/fake/create_body.json" 2>/dev/null | base64 -D 2>/dev/null | tar -tzf - 2>/dev/null)"
if printf '%s\n' "$LIST" | grep -qx scripts/interp/tasks_concepts.py && printf '%s\n' "$LIST" | grep -qx scripts/interp/tasks_e4.py \
    && printf '%s\n' "$LIST" | grep -qx scripts/interp/tasks.py && ! printf '%s\n' "$LIST" | grep -q 'tasks_e3c.py\|tasks_buyers.py\|tasks_heads2.py'; then
  pass "interp code holds the core + the named plug-ins only"; else fail "interp code holds the core + the named plug-ins only"; fi
[ "$(wc -c < "$W/fake/create_body.json")" -lt 100000 ] && pass "create request under 100 KB" || fail "create request under 100 KB"
has "$OUT" "pods of ours left on the account: 0" "interp teardown confirms 0 pods"
no_leak interp_create
if jq -e '.dataCenterIds == ["US-NE-1"]' "$W/fake/create_body.json" >/dev/null 2>&1; then
  pass "--dc us (default): only North American data centers that list the GPU (not EUR-IS-2, not US-KS-2)"
else fail "--dc us (default): only North American data centers that list the GPU (not EUR-IS-2, not US-KS-2)"; fi

echo "== data centers (--dc) =="
run dc_any FAKE_STATUS=ERROR -- --mode interp --model gemma-4-12b-it --lifetime-min 60 --interp-plugins none --dc any --confirm-cost "$IWORST"
if jq -e 'has("dataCenterIds") | not' "$W/fake/create_body.json" >/dev/null 2>&1; then
  pass "--dc any sends no dataCenterIds"; else fail "--dc any sends no dataCenterIds"; fi
has "$OUT" "data centers:   any (scheduler chooses)" "--dc any is shown in the plan"
run dc_list -- --mode interp --model gemma-4-12b-it --lifetime-min 60 --interp-plugins none --dc EUR-IS-2,US-XX-9
has "$OUT" "data centers:   EUR-IS-2, US-XX-9 (--dc EUR-IS-2,US-XX-9)" "an explicit list is sent as given"
has "$OUT" "the catalog lists no NVIDIA RTX PRO 6000 Blackwell Server Edition in: US-XX-9" "an unstocked data center is flagged"
run dc_bad -- --mode interp --model gemma-4-12b-it --lifetime-min 60 --interp-plugins none --dc 'us;rm'
eq "$RC" 2 "a malformed --dc is refused"
eq "$(posts)" 0 "--dc dry runs and refusals create nothing"

echo "== serve mode still runs boot.sh (not touched) =="
run serve_dry -- --mode serve --model gemma-4-12b-it --lifetime-min 60
SWORST="$(sed -n 's/.*--confirm-cost \([0-9.]*\).*/\1/p' "$TMP/w_serve_dry/out" | head -1)"
run serve_create FAKE_STATUS=ERROR -- --mode serve --model gemma-4-12b-it --lifetime-min 60 --confirm-cost "$SWORST"
if jq -e '.cmd[0] | contains("boot.sh") and (contains("boot_interp.sh") | not)' \
    "$W/fake/create_body.json" >/dev/null 2>&1; then
  pass "serve create body still runs boot.sh"; else fail "serve create body still runs boot.sh"; fi
if jq -e '(.env | has("SI_INTERP_B64")) | not' "$W/fake/create_body.json" >/dev/null 2>&1; then
  pass "serve create body has no interp env"; else fail "serve create body has no interp env"; fi

echo "== slots (several pods at once) =="
OTHER='{"id":"x","name":"si-other-live"}'
run slot_dry -- --mode probe --lifetime-min 15 --slot 3
has "$OUT" "slot:           3 (tunnel port 18003; other live launches: 0)" "dry run names the slot and its port"
PRE='printf "%s si-other-live 2.50\n" "$$" > "$W/logs/pod_slot.1"'
run slot_live FAKE_OTHER="$OTHER" -- --mode probe --lifetime-min 15 --slot 2
has "$OUT" "our pods now:   0 not owned by a live launch" "a pod owned by a live launch is not counted against C13"
has "$OUT" "budget:         \$2.50 used or reserved" "the other live launch's worst case is reserved in the budget"
has "$OUT" "other live launches: 1" "the other live launch is reported"
PRE='printf "999999 si-other-live 2.50\n" > "$W/logs/pod_slot.1"'
run slot_stale FAKE_OTHER="$OTHER" -- --mode probe --lifetime-min 15 --slot 2
eq "$RC" 4 "a pod whose claim is stale (launcher gone) blocks the launch (C13)"
eq "$(posts)" 0 "nothing created with a stale claim"
PRE='printf "%s si-other-live 2.50\n" "$$" > "$W/logs/pod_slot.2"'
run slot_busy FAKE_OTHER="$OTHER" -- --mode probe --lifetime-min 15 --slot 2
eq "$RC" 4 "a slot in use by a live launch is refused"
has "$OUT" "slot 2 is in use" "says which slot is busy"
PRE=''
run slot_teardown FAKE_STATUS=ERROR -- --mode probe --lifetime-min 15 --slot 4 --confirm-cost 1.23
[ -e "$W/logs/pod_slot.4" ] && fail "claim file left after teardown" || pass "claim file removed at teardown"
has "$OUT" "pods of ours left on the account: 0" "teardown still confirms 0 pods"
unset PRE

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
