#!/bin/bash
# test_runpod_auth_check.sh: offline tests for scripts/runpod_auth_check.sh
# and the connection sequence built on it (scripts/connections/).
#
# No network by default. A fake `curl` placed first on PATH records its
# command-line arguments, its stdin, and its inherited environment, then
# returns a canned API reply for each GET (pods, network volumes, billing).
# All keys and secrets here are fake canaries; the tests assert that they
# never show up anywhere they shouldn't.
#
# Usage:
#   bash tests/test_runpod_auth_check.sh                  # offline only
#   bash tests/test_runpod_auth_check.sh --live-negative  # also send ONE request
#       with a fake key to the real API and expect 401 (free; nothing changes)

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/../scripts/runpod_auth_check.sh"
STEP01="$HERE/../scripts/connections/01_runpod.sh"
RUNNER="$HERE/../scripts/connections/run_all.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAKEBIN="$TMP/bin"
LOGS="$TMP/logs"
mkdir -p "$FAKEBIN" "$LOGS"

CANARY_KEY="rpa_CANARYKEY0000000000000000000000000000"   # fake API key
CANARY_ENV="hf_CANARYENVSECRET0000000000000000"           # fake pod env secret

PASS=0
FAIL=0
RC=0
pass() { PASS=$((PASS + 1)); printf 'ok    %s\n' "$1"; }
fail() { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1"; }
assert_rc() { if [ "$RC" -eq "$1" ]; then pass "$2"; else fail "$2 (exit $RC, want $1)"; fi; }
assert_has() { if grep -qF -- "$2" "$1"; then pass "$3"; else fail "$3"; fi; }
assert_lacks() { if [ -f "$1" ] && grep -qF -- "$2" "$1"; then fail "$3"; else pass "$3"; fi; }

# --- Fake curl -------------------------------------------------------------------
cat > "$FAKEBIN/curl" <<'FAKE'
#!/bin/bash
# Fake curl: records argv, stdin, and inherited RUNPOD_API_KEY; prints a canned
# body followed by "\n<http code> <seconds>", mimicking
# curl -w '\n%{http_code} %{time_total}'.
set -u
log="${FAKE_LOG_DIR:?}"
printf -- '---CALL---\n' >> "$log/argv"
printf '%s\n' "$@" >> "$log/argv"
stdin="$(cat)"
printf '%s\n' "$stdin" >> "$log/stdin"
printf '%s\n' "${RUNPOD_API_KEY:-}" >> "$log/child_env"
url=""
for a in "$@"; do case "$a" in https://*) url="$a" ;; esac; done
key="${stdin#Authorization: Bearer }"
reply() { printf '%s\n%s %s' "$2" "$1" "0.123456"; }
case "${FAKE_MODE:-ok}" in
  idle)
    case "$url" in
      */v2/pods)
        reply 200 '{"pods":[{"id":"podEXITED1","name":"old-pod","status":"EXITED","cost":1.59,"gpu":{"id":"NVIDIA RTX PRO 4500 Blackwell","count":1}}],"pagination":{"nextCursor":null,"hasNextPage":false}}' ;;
      *) reply 404 '{"title":"Not Found","status":404,"detail":"no such path"}' ;;
    esac ;;
  ok|billing403|fresh|nobilling)
    case "$url" in
      */v2/pods)
        # The stopped pod reports a nonzero cost, as the real API did on
        # 2026-10-02 (the docs say it should be 0).
        reply 200 "$(printf '{"pods":[{"id":"podEXITED1","name":"old-pod","status":"EXITED","cost":1.59,"gpu":{"id":"NVIDIA RTX PRO 4500 Blackwell","count":1},"env":{"HF_TOKEN":"%s"},"mounts":{"persistent":{"size":50,"path":"/workspace"}},"dataCenterId":"EU-RO-1","createdAt":"2026-07-12T00:00:00Z"},{"id":"podRUNNING2","name":"live-pod","status":"RUNNING","cost":0.69,"gpu":{"id":"NVIDIA L40S","count":1},"env":{"OTHER_SECRET":"%s"},"mounts":{"network":[{"volumeId":"vol123","path":"/workspace"}]},"dataCenterId":"EU-RO-1","createdAt":"2026-10-01T00:00:00Z"}],"pagination":{"nextCursor":null,"hasNextPage":false}}' "$FAKE_ENV_SECRET" "$FAKE_ENV_SECRET")" ;;
      */v2/network-volumes)
        reply 200 '{"networkVolumes":[{"id":"vol123","name":"weights","size":100,"dataCenter":"EU-RO-1","type":"STANDARD"}]}' ;;
      */v2/billing\?*)
        if [ "$FAKE_MODE" = "billing403" ]; then
          reply 403 '{"title":"Forbidden","status":403,"detail":"access denied"}'
        elif [ "$FAKE_MODE" = "nobilling" ]; then
          reply 200 '{"records":[],"metadata":{"recordCount":0}}'
        elif [ "$FAKE_MODE" = "fresh" ]; then
          today="$(date -u +%Y-%m-%d)"
          reply 200 "{\"records\":[{\"startTime\":\"${today}T00:00:00Z\",\"endTime\":\"${today}T23:59:59Z\",\"totalAmount\":0.1,\"storageStandardAmount\":0.1}],\"metadata\":{\"recordCount\":1}}"
        else
          reply 200 '{"records":[{"startTime":"2026-09-26T00:00:00Z","endTime":"2026-09-27T00:00:00Z","totalAmount":0.2333,"podGpuAmount":0,"podDiskAmount":0,"storageStandardAmount":0.2333},{"startTime":"2026-09-27T00:00:00Z","endTime":"2026-09-28T00:00:00Z","totalAmount":1.5,"podGpuAmount":1.27,"podDiskAmount":0,"storageStandardAmount":0.23},{"startTime":"2026-09-28T00:00:00Z","endTime":"2026-09-29T00:00:00Z","totalAmount":0.05,"podDiskAmount":0.05}],"metadata":{"recordCount":3,"totals":{"totalAmount":1.7833}}}'
        fi ;;
      *) reply 404 '{"title":"Not Found","status":404,"detail":"no such path"}' ;;
    esac ;;
  unauth)    reply 401 '{"title":"Unauthorized","status":401,"detail":"invalid api key"}' ;;
  forbidden) reply 403 '{"title":"Forbidden","status":403,"detail":"access denied"}' ;;
  ratelimit) reply 429 '{"title":"Too Many Requests","status":429,"detail":"slow down"}' ;;
  echo)      reply 401 "{\"title\":\"Unauthorized\",\"status\":401,\"detail\":\"bad key ${key}\"}" ;;
  nonjson)   reply 502 '<html><body>Bad Gateway</body></html>' ;;
  neterr)    echo "curl: (6) Could not resolve host: api.runpod.io" >&2; exit 6 ;;
esac
FAKE
chmod +x "$FAKEBIN/curl"

# run_with_fake MODE OUTFILE CMD...: run CMD with the fake curl first on PATH
# and a canary key in the environment; sets RC.
run_with_fake() {
  local mode="$1" out="$2"
  shift 2
  rm -f "$LOGS"/*
  set +e
  env PATH="$FAKEBIN:$PATH" FAKE_MODE="$mode" FAKE_LOG_DIR="$LOGS" \
      FAKE_ENV_SECRET="$CANARY_ENV" RUNPOD_API_KEY="$CANARY_KEY" \
      "$@" < /dev/null > "$out" 2>&1
  RC=$?
  set -e
}

# run_script MODE OUTFILE [BASH_FLAG]: run the full audit against the fake curl.
run_script() {
  local mode="$1" out="$2" flag="${3:-}"
  # $flag is intentionally unquoted so that an empty value adds no argument.
  run_with_fake "$mode" "$out" bash $flag "$SCRIPT"
}

count_calls() { if [ -f "$LOGS/argv" ]; then grep -c -- '---CALL---' "$LOGS/argv"; else echo 0; fi; }

echo "== static =="
if bash -n "$SCRIPT"; then pass "script parses under $(bash -c 'echo bash $BASH_VERSION')"; else fail "bash -n syntax check"; fi
if bash -n "$STEP01" && bash -n "$RUNNER"; then pass "connection scripts parse"; else fail "connection scripts syntax check"; fi
out="$TMP/badopt.txt"
run_with_fake ok "$out" bash "$SCRIPT" --bogus
assert_rc 2 "unknown option -> exit 2"
if [ "$(count_calls)" -eq 0 ]; then pass "unknown option: no HTTP request made"; else fail "unknown option still made a request"; fi

echo "== missing key =="
out="$TMP/nokey.txt"; rm -f "$LOGS"/*
set +e
env -u RUNPOD_API_KEY PATH="$FAKEBIN:$PATH" FAKE_LOG_DIR="$LOGS" \
    RUNPOD_KEYCHAIN_SERVICE="scale-invariance-test-missing-$$" \
    bash "$SCRIPT" > "$out" 2>&1
RC=$?
set -e
assert_rc 2 "no env var and no Keychain item -> exit 2"
assert_has "$out" "security add-generic-password" "prints how to store the key"
if [ ! -f "$LOGS/argv" ]; then pass "no HTTP request made without a key"; else fail "curl was called without a key"; fi

echo "== malformed keys =="
out="$TMP/badkey.txt"; rm -f "$LOGS"/*
set +e
env PATH="$FAKEBIN:$PATH" FAKE_LOG_DIR="$LOGS" RUNPOD_API_KEY="rpa_CANARYBAD000000000000 'x" \
    bash "$SCRIPT" > "$out" 2>&1
RC=$?
set -e
assert_rc 2 "key with space/quote -> exit 2"
assert_lacks "$out" "CANARYBAD" "malformed key is not echoed"
if [ ! -f "$LOGS/argv" ]; then pass "no HTTP request made with a malformed key"; else fail "curl was called with a malformed key"; fi
set +e
env PATH="$FAKEBIN:$PATH" FAKE_LOG_DIR="$LOGS" RUNPOD_API_KEY="rpa_short" \
    bash "$SCRIPT" > "$out" 2>&1
RC=$?
set -e
assert_rc 2 "too-short key -> exit 2"
assert_has "$out" "characters long" "too-short key reports its length only"

echo "== happy path (fake 200s) =="
out="$TMP/ok.txt"
run_script ok "$out"
assert_rc 0 "exit 0 when all three calls return 200"
assert_has "$out" "Key source: RUNPOD_API_KEY environment variable" "reports which key source was used"
assert_has "$out" "Pods: 2" "counts pods"
assert_has "$out" "status=EXITED" "shows pod status"
assert_has "$out" "volume_disk=50GB" "shows stopped pod's volume disk"
assert_has "$out" "network_volume=vol123" "shows network-volume attachment"
assert_has "$out" 'rate=$1.59/hr when running' "shows a stopped pod's running rate, labelled as such"
assert_has "$out" 'Pods that can bill compute now (not EXITED/TERMINATED): 1, combined rate $0.69/hr' "stopped pod's rate excluded from the live total"
assert_lacks "$out" '$2.28/hr' "stopped pod's rate not added to the live total"
assert_has "$out" "volume disks on 1 of them (50 GB) still bill until terminated" "names only the stopped pods that really hold a volume disk"
assert_has "$out" "GET /v2/billing (per day, last 7 days) -> 200 OK" "billing call reported"
assert_has "$out" '  2026-09-26  $0.23  (storageStandard=$0.23)' "billing day: rounds to cents and names the cost component"
assert_has "$out" '  2026-09-27  $1.50  (podGpu=$1.27, storageStandard=$0.23)' "billing day: pads cents and lists only nonzero components"
assert_has "$out" '  2026-09-28  $0.05  (podDisk=$0.05)' "billing day: pads single-digit cents"
assert_has "$out" '7-day total: $1.78' "billing total summed and rounded"
assert_has "$out" "the most recent billed day is 2026-09-28" "stale billing history is flagged, not read as 'nothing billing'"
assert_has "$out" "Network volumes: 1" "counts network volumes"
assert_has "$out" "Total: 100 GB" "totals network volume size"
assert_has "$out" "AUTH OK" "prints AUTH OK"
assert_lacks "$out" "$CANARY_KEY" "key absent from output"
assert_lacks "$out" "$CANARY_ENV" "pod env secret absent from output"
assert_lacks "$LOGS/argv" "$CANARY_KEY" "key absent from curl's command line"
assert_has "$LOGS/stdin" "Authorization: Bearer $CANARY_KEY" "key reached curl via stdin header"
if awk 'prev=="---CALL---" && $0!="-q" {bad=1} {prev=$0} END {exit bad}' "$LOGS/argv"; then
  pass "-q is curl's first argument on every call (~/.curlrc ignored)"
else
  fail "-q is not first on every curl call"
fi
assert_has "$LOGS/argv" "=https" "curl restricted to https"
assert_lacks "$LOGS/child_env" "$CANARY_KEY" "RUNPOD_API_KEY not inherited by curl"
calls="$(grep -c -- '---CALL---' "$LOGS/argv")"
if [ "$calls" -eq 3 ]; then pass "exactly 3 HTTP requests (pods, volumes, billing)"; else fail "expected 3 HTTP requests, saw $calls"; fi
if grep -q -- '-X' "$LOGS/argv" || grep -q -- '--request' "$LOGS/argv" || grep -q -- '--data' "$LOGS/argv"; then
  fail "a non-GET method or request body was used"
else
  pass "every request is a plain GET (no -X, no body)"
fi

echo "== key without billing access =="
out="$TMP/billing403.txt"
run_script billing403 "$out"
assert_rc 0 "billing 403 -> still exit 0 (auth already proven)"
assert_has "$out" "returned HTTP 403" "billing 403 reported as a warning"
assert_has "$out" "AUTH OK" "AUTH OK still printed"
assert_lacks "$out" "$CANARY_KEY" "key absent on billing 403"

echo "== billing history freshness =="
out="$TMP/fresh.txt"
run_script fresh "$out"
assert_rc 0 "fresh billing history -> exit 0"
assert_lacks "$out" "most recent billed day" "up-to-date history is not flagged as stale"
out="$TMP/nobilling.txt"
run_script nobilling "$out"
assert_rc 0 "empty billing history -> exit 0"
assert_has "$out" "No charges recorded in the last 7 days." "empty history reported"
assert_has "$out" "Billing history can lag" "empty history carries the lag caveat"

echo "== bash -x cannot leak the key =="
out="$TMP/xtrace.txt"
run_script ok "$out" -x
assert_rc 0 "exit 0 under bash -x"
assert_lacks "$out" "$CANARY_KEY" "key absent from xtrace output"

echo "== connection step 01 (brief, screen-safe) =="
out="$TMP/step01.txt"
run_with_fake ok "$out" bash "$STEP01"
assert_rc 0 "step 01 exits 0 when RunPod accepts the key"
assert_has "$out" "RunPod API" "step 01 prints its title"
assert_has "$out" "Key loaded from the RUNPOD_API_KEY environment variable (never displayed)" "step 01 names the key source"
assert_has "$out" "RunPod accepted the key: HTTP 200 in 0.12 s" "step 01 shows acceptance and response time"
assert_has "$out" 'Pods running right now: 1 (combined rate $0.69/hr)' "step 01 counts running pods only, with their rate"
for s in podEXITED1 podRUNNING2 old-pod live-pod vol123 EU-RO-1; do
  assert_lacks "$out" "$s" "step 01 is screen-safe: '$s' not shown"
done
assert_lacks "$out" "$CANARY_KEY" "step 01: key absent from output"
assert_lacks "$out" "$CANARY_ENV" "step 01: pod env secret absent from output"
assert_lacks "$LOGS/argv" "$CANARY_KEY" "step 01: key absent from curl's command line"
if [ "$(count_calls)" -eq 1 ]; then pass "step 01 makes exactly 1 request"; else fail "step 01 made $(count_calls) requests"; fi
if grep -q $'\033' "$out"; then fail "step 01 printed color codes into a file"; else pass "step 01: no color codes when output is not a terminal"; fi
run_with_fake idle "$out" bash "$STEP01"
assert_rc 0 "step 01 with only stopped pods -> exit 0"
assert_has "$out" "Pods running right now: 0" "stopped pods are not counted as running"
run_with_fake unauth "$out" bash "$STEP01"
assert_rc 1 "step 01 on 401 -> exit 1"
assert_has "$out" "returned 401" "step 01 reports the rejection"
assert_lacks "$out" "$CANARY_KEY" "step 01: key absent on 401"

echo "== connection runner (run_all.sh) =="
# This suite's fake curl only knows RunPod, so run the runner on a temporary
# copy that holds step 01 alone. Runner tests across all steps live in
# tests/test_connections.sh.
ONLY01="$TMP/only01"
mkdir -p "$ONLY01/scripts/connections"
cp "$SCRIPT" "$ONLY01/scripts/"
cp "$STEP01" "$RUNNER" "$ONLY01/scripts/connections/"
out="$TMP/runner.txt"
run_with_fake ok "$out" bash "$ONLY01/scripts/connections/run_all.sh"
assert_rc 0 "runner exits 0 when every check passes"
assert_has "$out" "] 01_runpod.sh" "runner announces each check by name"
assert_has "$out" "connection checks passed." "runner prints the final tally"
run_with_fake unauth "$out" bash "$RUNNER"
assert_rc 1 "runner returns the failing check's exit code"
assert_has "$out" "Stopped at 01_runpod.sh" "runner stops at the first failure and says where"
assert_lacks "$out" "connection checks passed." "runner does not claim success after a failure"

if [ "$(uname -s)" = "Darwin" ] && command -v script >/dev/null 2>&1; then
  echo "== colors on a real terminal (pseudo-terminal via script) =="
  out="$TMP/tty.txt"
  run_with_fake ok "$out" script -q /dev/null bash "$STEP01"
  if grep -q $'\033\[32m' "$out"; then pass "green check marks on a terminal"; else fail "no color on a terminal"; fi
  assert_lacks "$out" "$CANARY_KEY" "terminal output: key absent"
  run_with_fake ok "$out" env NO_COLOR=1 script -q /dev/null bash "$STEP01"
  if grep -q $'\033\[3' "$out"; then fail "NO_COLOR=1 still printed colors"; else pass "NO_COLOR=1 turns colors off"; fi
fi

echo "== error paths =="
out="$TMP/err.txt"
run_script unauth "$out"
assert_rc 1 "401 -> exit 1"
assert_has "$out" "returned 401" "401 is reported"
assert_lacks "$out" "$CANARY_KEY" "key absent on 401"
run_script forbidden "$out"
assert_rc 1 "403 -> exit 1"
assert_has "$out" "not allowed to read" "403 explains permissions"
run_script echo "$out"
assert_rc 1 "server echoing the key -> exit 1"
assert_lacks "$out" "$CANARY_KEY" "echoed key is redacted"
assert_has "$out" "[REDACTED]" "redaction marker shown"
run_script ratelimit "$out"
assert_rc 3 "429 -> exit 3"
run_script nonjson "$out"
assert_rc 3 "502 non-JSON -> exit 3"
assert_has "$out" "non-JSON" "non-JSON body reported without dumping it"
run_script neterr "$out"
assert_rc 3 "network error -> exit 3"
assert_lacks "$out" "$CANARY_KEY" "key absent on network error"

if [ "${1:-}" = "--live-negative" ]; then
  echo "== live negative (one real GET with a fake key; expect 401) =="
  out="$TMP/live.txt"
  set +e
  env RUNPOD_API_KEY="$CANARY_KEY" bash "$SCRIPT" > "$out" 2>&1
  RC=$?
  set -e
  assert_rc 1 "real API rejects a fake key -> exit 1"
  assert_has "$out" "returned 401" "real API answered 401"
  assert_lacks "$out" "$CANARY_KEY" "key absent from real 401 output"
  echo "  (real API said: $(grep -m1 'returned 401' "$out" | sed 's/^AUTH FAILED: //'))"
fi

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
