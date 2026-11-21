# mutation_check_pod_boot.sh: proves the boot-script tests can fail.
#
# For each mutant, writes a copy of scripts/pod/boot.sh with one deliberate
# flaw into a temporary folder, runs tests/test_pod_boot.sh against it
# (BOOT_UNDER_TEST), and checks that the matching test reports FAIL. An
# unmodified copy runs first and must pass. The real file is never
# modified. No network. Each test run has a hard 300-second limit, so a
# mutant that hangs counts as caught instead of hanging this script.
#
# Usage: bash tests/mutation_check_pod_boot.sh

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/scripts/pod/boot.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
caught=0
total=0

run_tests() {   # macOS has no `timeout`; perl's alarm gives a hard limit
  BOOT_UNDER_TEST="$1" perl -e 'alarm shift; exec @ARGV' 300 bash "$HERE/test_pod_boot.sh" 2>&1
}

cp "$SRC" "$TMP/baseline.sh"
if base="$(run_tests "$TMP/baseline.sh")"; then
  printf 'baseline: unmodified copy passes (%s)\n\n' "$(printf '%s\n' "$base" | grep -E '^passed:')"
else
  printf 'baseline FAILED; fix the tests before trusting this check:\n%s\n' \
    "$(printf '%s\n' "$base" | grep -E '^FAIL')"
  exit 1
fi

# mutant NAME "TEST THAT MUST FAIL (start of its name)" SED_PROGRAM
mutant() {
  local name="$1" expect="$2" prog="$3" f="$TMP/$1.sh" out rc
  total=$((total + 1))
  if ! sed -e "$prog" "$SRC" > "$f" || [ ! -s "$f" ]; then
    printf 'BROKEN  %-20s sed failed or produced an empty file; update this mutant\n' "$name"
    return
  fi
  if cmp -s "$SRC" "$f"; then
    printf 'BROKEN  %-20s the planted change did not apply; update this mutant\n' "$name"
    return
  fi
  out="$(run_tests "$f")"
  rc=$?
  if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qF "FAIL  $expect"; then
    caught=$((caught + 1))
    printf 'caught  %-20s -> %s\n' "$name" "$expect"
  elif [ "$rc" -ge 128 ]; then
    caught=$((caught + 1))
    printf 'caught  %-20s -> hit the 300 s hard limit (hung)\n' "$name"
  else
    printf 'MISSED  %-20s expected: %s\n' "$name" "$expect"
  fi
}

# The watchdog starts only after the download, so a hang there bills forever.
mutant watchdog_late      "watchdog starts before anything that can hang" \
  '/^watchdog &$/d; s/^WD_PID=\$!$/WD_PID=0/; s/^log "vLLM version: /watchdog \& WD_PID=$!; log "vLLM version: /'
# The lifetime cap is ignored above 6 h (or never enforced at all).
mutant no_lifetime_cap    "lifetime over 6 h refused" \
  's/-gt 21600 \]/-gt 9999999 ]/'
# The laptop's "delete now" request is ignored.
mutant ignore_delete_req  "the laptop can ask for deletion" \
  's/^    if \[ -e "\$RUN\/delete_now" \]; then reason=/    if false; then reason=/'
# A refused self-delete is given up on instead of retried.
mutant no_delete_retry    "then succeeds" \
  's/retrying in \$RETRY s" ;;/retrying in $RETRY s"; exit 1 ;;/'
# A 404 is taken as "already gone", though a key that can't see the pod gets 404 too.
mutant trusts_404         "a 404 is not trusted" \
  's/^      200|204) log "SELF_DELETE ok/      200|204|404) log "SELF_DELETE ok/'
# The pod key goes on curl's command line, where `ps` shows it.
mutant key_on_argv        "serve: pod key leaked" \
  's/-X DELETE -H @- "\$API/-X DELETE -H "Authorization: Bearer $POD_KEY" "$API/'
# The pod key stays in the environment, so vLLM inherits it.
mutant key_inherited      "vLLM does not inherit the pod key" \
  's/^unset RUNPOD_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN$/unset HF_TOKEN HUGGING_FACE_HUB_TOKEN/'
# sshd also reads /root/.ssh/authorized_keys, where RunPod may inject keys (C19).
mutant reads_root_keys    "sshd config: AuthorizedKeysFile" \
  's/^AuthorizedKeysFile \$AUTH$/AuthorizedKeysFile .ssh\/authorized_keys $AUTH/'
# Agent forwarding allowed: the pod could use the laptop's other keys.
mutant agent_forwarding   "sshd config: AllowAgentForwarding no" \
  's/^AllowAgentForwarding no$/AllowAgentForwarding yes/'
# Passwords allowed.
mutant passwords_ok       "sshd config: PasswordAuthentication no" \
  's/^PasswordAuthentication no$/PasswordAuthentication yes/'
# Forwarding to anywhere, not just vLLM.
mutant forward_anywhere   "sshd config: PermitOpen 127.0.0.1:8000" \
  's/^PermitOpen 127.0.0.1:8000$/PermitOpen any/'
# The config is never tested before starting sshd.
mutant sshd_untested      "sshd config is tested first" \
  's/^sshd -t -f "\$CONF" > "\$LOGDIR\/sshd_test.log" 2>&1 || fail "sshd rejected its config"$/:/'
# Any key is accepted, whatever its fingerprint.
mutant no_fpr_check       "wrongfpr: refuses" \
  's/^\[ "\$got" = "\$FPR" \] || fail/true || fail/'
# The key check is unanchored, so a key with a forced command slips in.
mutant loose_key_regex    "optkey: refuses" \
  "s/^re_key='\\^ssh-ed25519 /re_key='ssh-ed25519 /"
# The manifest itself isn't checked against its hash.
mutant no_manifest_hash   "manhash: refuses" \
  's/^\[ "\$(sha256_of "\$MAN")" = "\${SI_MANIFEST_SHA256:-}" \] || fail/true || fail/'
# File contents aren't hashed: a same-size swap goes through.
mutant no_file_sha        "samesize: refuses" \
  's/ && \[ "\$(sha256_of "\$dest.part")" = "\$sha" \]; then/; then/'
# File sizes aren't compared.
mutant no_file_size       "wrongsize: refuses" \
  's/if \[ "\$(bytes_of "\$dest.part")" = "\$size" \] && /if /'
# ".." in a manifest path is allowed.
mutant traversal_ok       "traversal: refuses" \
  '/^  case "\/\$path\/" in \*\/\.\/\*|\*\/\.\.\/\*) fail "manifest: bad path" ;; esac$/d'
# A failed weights check is only logged; vLLM starts anyway.
mutant bad_weights_serve  "tampered: vLLM started anyway" \
  's/^  \[ "\$ok" = 1 \] || fail "WEIGHTS_BAD/  [ "$ok" = 1 ] || log "WEIGHTS_BAD/'
# vLLM listens on every interface.
mutant vllm_all_ifaces    "vLLM gets exactly the infra-plan flags" \
  's/--host 127.0.0.1 --port 8000/--host 0.0.0.0 --port 8000/'
# vLLM may go online and fetch other files.
mutant vllm_online        "vLLM runs offline" \
  's/exec env HF_HUB_OFFLINE=1 vllm serve/exec env vllm serve/'
# curl reads ~/.curlrc for the downloads.
mutant curlrc_read        "curl's first argument is -q" \
  's/if curl -q -sS --fail -L/if curl -sS --fail -L/'
# Redirects may go to plain http.
mutant redirect_http      "redirects are limited to https" \
  "s/ --proto-redir '=https'//"
# Keys appearing in /root/.ssh are no longer reported (C19).
mutant c19_silent         "logs keys appearing in /root/.ssh" \
  's/^    if \[ "\$now" != "\$last" \]; then log "C19:/    if false; then log "C19:/'

printf '\n%d of %d planted flaws caught\n' "$caught" "$total"
[ "$caught" -eq "$total" ]
