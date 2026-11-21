#!/bin/bash
# test_connections.sh: offline tests for connection step 02 (Hugging Face)
# and for the runner (scripts/connections/run_all.sh) across steps 01 and 02.
#
# No network. A fake `curl` placed first on PATH records its arguments,
# stdin, and inherited token variables, then returns a canned reply:
#   - Hugging Face model API: writes the body to the -o file and prints
#     "<http code> <seconds> <redirect url>", like curl -w does.
#   - RunPod /v2/pods (for step 01 inside the runner): prints body + code.
# FAKE_HF_MODE picks the Hugging Face scenario. All tokens are fake canaries.
#
# Usage: bash tests/test_connections.sh

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
STEP02="$HERE/../scripts/connections/02_huggingface.sh"
RUNNER="$HERE/../scripts/connections/run_all.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAKEBIN="$TMP/bin"
LOGS="$TMP/logs"
mkdir -p "$FAKEBIN" "$LOGS"

CANARY_HF="hf_CANARYHFTOKEN000000000000000000000"
CANARY_RP="rpa_CANARYKEY0000000000000000000000000000"
SHA_Q="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
SHA_G12="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
SHA_G31="842da3794eaa0b77d5f08bae87a17459d91ff475"
ESC=$'\033'

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
set -u
log="${FAKE_LOG_DIR:?}"
printf -- '---CALL---\n' >> "$log/argv"
printf '%s\n' "$@" >> "$log/argv"
stdin="$(cat)"
[ -n "$stdin" ] && printf '%s\n' "$stdin" >> "$log/stdin"
printf '%s|%s|%s|%s\n' "${HF_TOKEN:-}" "${HUGGING_FACE_HUB_TOKEN:-}" \
  "${HUGGINGFACE_HUB_TOKEN:-}" "${HF_API_TOKEN:-}" >> "$log/child_env"
url="" ; outfile="" ; follow=0 ; prev=""
for a in "$@"; do
  case "$prev" in -o) outfile="$a" ;; esac
  case "$a" in https://*) url="$a" ;; -L|--location) follow=1 ;; esac
  prev="$a"
done

case "$url" in
  https://api.runpod.io/v2/pods)
    printf '%s\n%s %s' '{"pods":[],"pagination":{"hasNextPage":false}}' 200 0.1
    exit 0 ;;
  https://huggingface.co/api/models/*) printf '%s\n' "$url" >> "$log/hf_urls" ;;
  *) echo "fake curl: unexpected URL $url" >&2; exit 99 ;;
esac

repo="${url#https://huggingface.co/api/models/}"
repo="${repo%%\?*}"
mode="${FAKE_HF_MODE:-ok}"

# Pinned-commit lookups: GET /api/models/<repo>/revision/<sha>. The real
# script asks curl for the status code only, so print just that.
case "$repo" in
  */revision/*)
    rrepo="${repo%%/revision/*}"; rsha="${repo#*/revision/}"
    printf '%s %s\n' "$rrepo" "$rsha" >> "$log/revision_calls"
    case "$mode" in
      moved)      code=200; body="{\"_id\":\"x\",\"id\":\"$rrepo\",\"sha\":\"$rsha\"}" ;;
      movedwrong) code=200; body="{\"_id\":\"x\",\"id\":\"$rrepo\",\"sha\":\"ffffffffffffffffffffffffffffffffffffffff\"}" ;;
      movedgone)  code=404; body='{"error":"Revision Not Found"}' ;;
      *)          code=500; body='{"error":"unexpected revision lookup in this mode"}' ;;
    esac
    [ -n "$outfile" ] && printf '%s' "$body" > "$outfile"
    printf '%s' "$code"
    exit 0 ;;
esac

code=200 ; redirect=""
case "$repo" in
  Qwen/Qwen3.8-27B-FP8)  sha=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a; day=2026-08-14 ;;
  google/gemma-4-12B-it) sha=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; day=2026-07-20 ;;
  google/gemma-4-31B-it) sha=842da3794eaa0b77d5f08bae87a17459d91ff475; day=2026-07-20 ;;
  mistralai/Mistral-Small-24B-Instruct-2501) sha=9527884be6e5616bdd54de542f9ae13384489724; day=2025-01-30 ;;
  microsoft/phi-4) sha=2db69c1c3e91a05d2c64a3185acfbaf36f744e25; day=2025-01-08 ;;
  *) sha=""; day="" ;;
esac
gated=false ; id="$repo" ; disabled=false
body=""
case "$mode:$repo" in
  ok:*) ;;
  gated:google/gemma-4-12B-it)    gated='"manual"' ;;
  gatedodd:google/gemma-4-12B-it) gated='"yes"' ;;
  missing:google/gemma-4-12B-it|missing2:google/gemma-4-12B-it)
    code=401; body='{"error":"Invalid username or password."}' ;;
  missing2:google/gemma-4-31B-it) code=429; body='{"error":"rate limited"}' ;;
  case:google/gemma-4-31B-it)
    code=307; redirect="https://huggingface.co/api/models/google/gemma-4-31b-it?expand%5B%5D=sha"
    body='Temporary Redirect.' ;;
  evilredirect:google/gemma-4-31B-it)
    code=307; redirect="https://evil.example/api/models/x/\$(id)${ESC_SEQ:-}/y"
    body='Temporary Redirect.' ;;
  disabled:Qwen/Qwen3.8-27B-FP8)  disabled=true ;;
  wrongid:Qwen/Qwen3.8-27B-FP8)   id="Qwen/Something-Else" ;;
  badsha:Qwen/Qwen3.8-27B-FP8)    sha="017b9c7" ;;
  nosha:Qwen/Qwen3.8-27B-FP8)     sha="" ;;
  moved:Qwen/Qwen3.8-27B-FP8|movedwrong:Qwen/Qwen3.8-27B-FP8|movedgone:Qwen/Qwen3.8-27B-FP8)
    sha="abcdef0123456789abcdef0123456789abcdef01"; day=2026-11-01 ;;
  nonjson:*)   body='<html>Bad Gateway</html>' ; code=200 ;;
  ratelimit:*) code=429; body='{"error":"rate limited"}' ;;
  server:*)    code=503; body='{"error":"unavailable"}' ;;
  neterr:*)    printf 'curl: (6) Could not resolve host: huggingface.co%s[31m\n' "$(printf '\033')" >&2; exit 6 ;;
  *) ;;
esac

# With -L, a redirect would be followed: the server then answers 200 for the
# name it redirected to.
if [ "$follow" -eq 1 ] && [ -n "$redirect" ]; then
  code=200
  id="google/gemma-4-31b-it"
  redirect=""
fi

if [ -z "$body" ]; then
  shaj="null"; [ -n "$sha" ] && shaj="\"$sha\""
  body="{\"_id\":\"x\",\"id\":\"$id\",\"sha\":$shaj,\"gated\":$gated,\"lastModified\":\"${day}T00:00:00.000Z\",\"disabled\":$disabled}"
fi
[ -n "$outfile" ] && printf '%s' "$body" > "$outfile"
printf '%s %s %s' "$code" "0.123456" "$redirect"
FAKE
chmod +x "$FAKEBIN/curl"

# run_fake MODE OUTFILE CMD...: run CMD with the fake curl first on PATH and
# canary tokens in the environment; sets RC.
run_fake() {
  local mode="$1" out="$2"
  shift 2
  rm -f "$LOGS"/*
  set +e
  env PATH="$FAKEBIN:$PATH" FAKE_HF_MODE="$mode" FAKE_LOG_DIR="$LOGS" ESC_SEQ="$ESC[31m" \
      PINS_FILE="${TEST_PINS:-$TMP/pins.json}" HF_TOKEN="$CANARY_HF" HUGGING_FACE_HUB_TOKEN="$CANARY_HF" \
      HUGGINGFACE_HUB_TOKEN="$CANARY_HF" HF_API_TOKEN="$CANARY_HF" \
      RUNPOD_API_KEY="$CANARY_RP" \
      "$@" < /dev/null > "$out" 2>&1
  RC=$?
  set -e
}
rev_calls() { if [ -f "$LOGS/revision_calls" ]; then wc -l < "$LOGS/revision_calls" | tr -d ' '; else echo 0; fi; }

# Test pins: the same repos and commits as config/pins.json, kept here so the
# tests do not depend on that file (one test below checks the real file).
cat > "$TMP/pins.json" <<PINS
{"models":[
 {"name":"qwen3.8-27b-fp8","repo":"Qwen/Qwen3.8-27B-FP8","revision":"$SHA_Q"},
 {"name":"gemma-4-12b-it","repo":"google/gemma-4-12B-it","revision":"$SHA_G12"},
 {"name":"gemma-4-31b-it","repo":"google/gemma-4-31B-it","revision":"$SHA_G31"}]}
PINS
hf_calls() { if [ -f "$LOGS/hf_urls" ]; then wc -l < "$LOGS/hf_urls" | tr -d ' '; else echo 0; fi; }

echo "== static =="
if bash -n "$STEP02"; then pass "02_huggingface.sh parses"; else fail "02_huggingface.sh syntax check"; fi
out="$TMP/badopt.txt"
run_fake ok "$out" bash "$STEP02" --bogus
assert_rc 2 "unknown option -> exit 2"
if [ "$(hf_calls)" -eq 0 ]; then pass "unknown option: no request made"; else fail "unknown option still made a request"; fi
run_fake ok "$out" bash "$STEP02" --help
assert_rc 0 "--help -> exit 0"

echo "== happy path: three public, ungated repos =="
out="$TMP/ok.txt"
run_fake ok "$out" bash "$STEP02"
assert_rc 0 "exit 0 when all three repos are found"
assert_has "$out" "Hugging Face model repos (anonymous: no token sent)" "prints its title"
assert_has "$out" "✓ Qwen/Qwen3.8-27B-FP8" "Qwen repo checked"
assert_has "$out" "✓ google/gemma-4-12B-it" "Gemma 12B repo checked"
assert_has "$out" "✓ google/gemma-4-31B-it" "Gemma 31B repo checked"
assert_has "$out" "found · not gated · pinned commit 017b9c7 is current" "Qwen line: gated status and pinned commit"
assert_has "$out" "pinned commit 707f0a3 is current" "Gemma 12B pinned commit"
assert_has "$out" "pinned commit 842da37 is current" "Gemma 31B pinned commit"
if [ "$(rev_calls)" -eq 0 ]; then pass "no pinned-commit lookup when every pin is current"; else fail "made $(rev_calls) pinned-commit lookups"; fi
assert_lacks "$out" "$SHA_Q" "full commit ID is shortened on screen"
if [ "$(hf_calls)" -eq 3 ]; then pass "exactly 3 requests, one per repo"; else fail "expected 3 requests, saw $(hf_calls)"; fi
assert_has "$LOGS/hf_urls" "https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8?expand%5B%5D=sha&expand%5B%5D=gated&expand%5B%5D=lastModified&expand%5B%5D=disabled" "requests only the needed fields, exact repo ID"
assert_has "$LOGS/hf_urls" "/api/models/google/gemma-4-12B-it?" "Gemma 12B requested with exact case"
assert_has "$LOGS/hf_urls" "/api/models/google/gemma-4-31B-it?" "Gemma 31B requested with exact case"
if grep -qi 'authorization' "$LOGS/argv"; then fail "an Authorization header was sent"; else pass "no Authorization header on the command line"; fi
if [ -s "$LOGS/stdin" ] 2>/dev/null; then fail "something was sent to curl on stdin"; else pass "nothing sent on stdin (no header file)"; fi
assert_lacks "$LOGS/argv" "$CANARY_HF" "HF token absent from curl's command line"
assert_lacks "$LOGS/child_env" "$CANARY_HF" "HF token variables not inherited by curl"
assert_lacks "$out" "$CANARY_HF" "HF token absent from output"
if awk 'prev=="---CALL---" && $0!="-q" {bad=1} {prev=$0} END {exit bad}' "$LOGS/argv"; then
  pass "-q is curl's first argument on every call (~/.curlrc ignored)"
else
  fail "-q is not first on every curl call"
fi
assert_has "$LOGS/argv" "=https" "curl restricted to https"
if grep -qxE -- '-L|--location|--location-trusted' "$LOGS/argv"; then fail "redirects are followed"; else pass "redirects are not followed (no -L)"; fi
if grep -qxE -- '-X|--request|--data|-d|--data-binary|-F|--form|-T|--upload-file' "$LOGS/argv"; then
  fail "a non-GET method or request body was used"
else
  pass "every request is a plain GET (no -X, no body)"
fi
if grep -q "$ESC" "$out"; then fail "printed color codes into a file"; else pass "no color codes when output is not a terminal"; fi

echo "== gated repo =="
out="$TMP/gated.txt"
run_fake gated "$out" bash "$STEP02"
assert_rc 0 "gated repo -> still exit 0 (a warning, not a failure)"
assert_has "$out" "! google/gemma-4-12B-it" "gated repo gets a warning mark"
assert_has "$out" "GATED (manual approval): downloads need a token · pinned commit 707f0a3" "gated repo flagged with its approval type"
assert_has "$out" "✓ google/gemma-4-31B-it" "other repos still reported"
run_fake gatedodd "$out" bash "$STEP02"
assert_rc 3 "unrecognized gated value -> exit 3"
assert_has "$out" "gated field was not recognized" "unrecognized gated value reported"

echo "== missing or misnamed repo =="
out="$TMP/missing.txt"
run_fake missing "$out" bash "$STEP02"
assert_rc 1 "missing repo (401) -> exit 1"
assert_has "$out" "✗ google/gemma-4-12B-it" "missing repo gets a failure mark"
assert_has "$out" "not found (HTTP 401)" "401 explained as not found"
assert_has "$out" "a private repo looks the same" "401 ambiguity explained"
if [ "$(hf_calls)" -eq 3 ]; then pass "remaining repos still checked after a failure"; else fail "stopped early: $(hf_calls) requests"; fi
assert_has "$out" "Hugging Face check failed (exit 1)" "failure summary printed"
run_fake missing2 "$out" bash "$STEP02"
assert_rc 3 "missing + rate-limited -> exit 3 (the worse outcome wins)"
out="$TMP/case.txt"
run_fake case "$out" bash "$STEP02"
assert_rc 1 "wrong-case name (307) -> exit 1"
assert_has "$out" "Hugging Face points to google/gemma-4-31b-it (letter case matters)" "redirect target named"
out="$TMP/evil.txt"
run_fake evilredirect "$out" bash "$STEP02"
assert_rc 1 "odd redirect -> exit 1"
assert_has "$out" "redirect target was not recognized" "odd redirect target not repeated"
if grep -q "$ESC" "$out"; then fail "escape code from a redirect reached the screen"; else pass "redirect escape code stripped"; fi
assert_lacks "$out" '$(id)' "odd redirect text not printed"

echo "== malformed answers =="
out="$TMP/bad.txt"
run_fake disabled "$out" bash "$STEP02"
assert_rc 1 "disabled repo -> exit 1"
assert_has "$out" "has disabled this repo" "disabled repo reported"
run_fake wrongid "$out" bash "$STEP02"
assert_rc 3 "answer for a different repo -> exit 3"
assert_has "$out" "different repo ID" "repo ID mismatch reported"
assert_lacks "$out" "Something-Else" "mismatched ID from the server is not echoed"
run_fake badsha "$out" bash "$STEP02"
assert_rc 3 "short commit ID from the server -> exit 3"
assert_has "$out" "commit ID is missing or malformed" "malformed commit ID reported"
run_fake nosha "$out" bash "$STEP02"
assert_rc 3 "missing commit ID -> exit 3"
run_fake nonjson "$out" bash "$STEP02"
assert_rc 3 "non-JSON 200 -> exit 3"
assert_has "$out" "not the expected JSON" "non-JSON reported without dumping it"
assert_lacks "$out" "<html>" "non-JSON body not printed"
run_fake ratelimit "$out" bash "$STEP02"
assert_rc 3 "429 -> exit 3"
assert_has "$out" "rate limited by Hugging Face (HTTP 429)" "429 explained"
run_fake server "$out" bash "$STEP02"
assert_rc 3 "503 -> exit 3"
assert_has "$out" "unexpected answer: HTTP 503" "503 reported"
run_fake neterr "$out" bash "$STEP02"
assert_rc 3 "network error -> exit 3"
assert_has "$out" "network error (curl: (6) Could not resolve host: huggingface.co" "curl's message shown"
if grep -q "$ESC" "$out"; then fail "escape code from curl's message reached the screen"; else pass "curl message escape code stripped"; fi

echo "== pinned commits =="
out="$TMP/pins.txt"
run_fake moved "$out" bash "$STEP02"
assert_rc 0 "repo moved past its pin, pin still there -> exit 0 (warning)"
assert_has "$out" "! Qwen/Qwen3.8-27B-FP8" "moved repo gets a warning mark"
assert_has "$out" "pinned commit 017b9c7 still available; repo moved on to abcdef0 (updated 2026-11-01)" "moved repo: pin still available, new commit named"
assert_has "$LOGS/revision_calls" "Qwen/Qwen3.8-27B-FP8 $SHA_Q" "pinned commit looked up by its full ID"
if [ "$(rev_calls)" -eq 1 ]; then pass "only the moved repo gets a pinned-commit lookup"; else fail "expected 1 pinned-commit lookup, saw $(rev_calls)"; fi
run_fake movedgone "$out" bash "$STEP02"
assert_rc 1 "pinned commit gone -> exit 1"
assert_has "$out" "✗ Qwen/Qwen3.8-27B-FP8" "gone pin gets a failure mark"
assert_has "$out" "pinned commit 017b9c7 is gone (HTTP 404); repo is now at abcdef0" "gone pin explained"
run_fake movedwrong "$out" bash "$STEP02"
assert_rc 3 "pin lookup answers for a different commit -> exit 3"
assert_has "$out" "not for pinned commit 017b9c7" "wrong pin answer reported"

echo "== pins file =="
mkpins() { printf '%s\n' "$1" > "$TMP/badpins.json"; }
mkpins '{"models":[{"repo":"Qwen/Qwen3.8-27B-FP8","revision":"017b9c7"}]}'
TEST_PINS="$TMP/badpins.json" run_fake ok "$out" bash "$STEP02"
assert_rc 2 "short pinned commit -> exit 2"
assert_has "$out" "must be 40 hex characters" "short pin explained"
if [ "$(hf_calls)" -eq 0 ]; then pass "bad pins file: no request made"; else fail "bad pins file still made requests"; fi
mkpins '{"models":[{"repo":"evil.example/x?y=1","revision":"'"$SHA_Q"'"}]}'
TEST_PINS="$TMP/badpins.json" run_fake ok "$out" bash "$STEP02"
assert_rc 2 "repo ID with URL characters -> exit 2"
assert_has "$out" "bad repo ID" "bad repo ID explained"
mkpins '{"models":[{"repo":"a/b/c","revision":"'"$SHA_Q"'"}]}'
TEST_PINS="$TMP/badpins.json" run_fake ok "$out" bash "$STEP02"
assert_rc 2 "repo ID with two slashes -> exit 2"
mkpins 'not json'
TEST_PINS="$TMP/badpins.json" run_fake ok "$out" bash "$STEP02"
assert_rc 2 "pins file that is not JSON -> exit 2"
mkpins '{"models":[]}'
TEST_PINS="$TMP/badpins.json" run_fake ok "$out" bash "$STEP02"
assert_rc 2 "pins file with no models -> exit 2"
TEST_PINS="$TMP/nope.json" run_fake ok "$out" bash "$STEP02"
assert_rc 2 "missing pins file -> exit 2"
assert_has "$out" "pins file not found" "missing pins file explained"
REAL_PINS="$HERE/../config/pins.json"
if [ -f "$REAL_PINS" ]; then
  TEST_PINS="$REAL_PINS" run_fake ok "$out" bash "$STEP02"
  assert_rc 0 "the real config/pins.json is valid and matches the expected commits"
else
  fail "config/pins.json not found"
fi

echo "== bash -x =="
out="$TMP/xtrace.txt"
run_fake ok "$out" bash -x "$STEP02"
assert_rc 0 "exit 0 under bash -x"
assert_lacks "$out" "$CANARY_HF" "HF token absent from xtrace output"

if [ "$(uname -s)" = "Darwin" ] && command -v script >/dev/null 2>&1; then
  echo "== colors on a real terminal (pseudo-terminal via script) =="
  out="$TMP/tty.txt"
  run_fake ok "$out" script -q /dev/null bash "$STEP02"
  if grep -q "$ESC\[32m" "$out"; then pass "green check marks on a terminal"; else fail "no color on a terminal"; fi
  run_fake ok "$out" env NO_COLOR=1 script -q /dev/null bash "$STEP02"
  if grep -q "$ESC\[3" "$out"; then fail "NO_COLOR=1 still printed colors"; else pass "NO_COLOR=1 turns colors off"; fi
fi

echo "== connection runner (run_all.sh): steps 01 and 02; 03 and 04 skip without a pod =="
out="$TMP/runner.txt"
run_fake ok "$out" bash "$RUNNER"
assert_rc 0 "runner exits 0 when every check passes"
assert_has "$out" "[1/4] 01_runpod.sh" "runner runs step 01 first"
assert_has "$out" "[2/4] 02_huggingface.sh" "runner picks up step 02"
assert_has "$out" "2 of 4 connection checks passed; 2 skipped (no pod running)." "runner prints the final tally"
assert_lacks "$out" "$CANARY_RP" "runner: RunPod key absent from output"
assert_lacks "$out" "$CANARY_HF" "runner: HF token absent from output"
run_fake missing "$out" bash "$RUNNER"
assert_rc 1 "runner returns the failing check's exit code"
assert_has "$out" "Stopped at 02_huggingface.sh (check 2 of 4" "runner says where it stopped"
assert_lacks "$out" "connection checks passed." "runner does not claim success after a failure"
out="$TMP/runner_nokey.txt"; rm -f "$LOGS"/*
set +e
env -u RUNPOD_API_KEY PATH="$FAKEBIN:$PATH" FAKE_HF_MODE=ok FAKE_LOG_DIR="$LOGS" \
    RUNPOD_KEYCHAIN_SERVICE="scale-invariance-test-missing-$$" \
    bash "$RUNNER" < /dev/null > "$out" 2>&1
RC=$?
set -e
assert_rc 2 "runner: step 01 fails without a key -> exit 2"
assert_has "$out" "Stopped at 01_runpod.sh" "runner stops at step 01"
if [ "$(hf_calls)" -eq 0 ]; then pass "step 02 is not run after step 01 fails"; else fail "step 02 ran after step 01 failed"; fi

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
