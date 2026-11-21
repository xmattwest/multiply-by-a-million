#!/bin/bash
# test_pod_boot.sh: offline tests for scripts/pod/boot.sh.
#
# Runs the boot script on this Mac with every path under a scratch folder
# (SI_ROOT) and fakes first on PATH for the things only a pod has:
#   curl        Hugging Face downloads (served from a scratch "repo") and
#               RunPod's DELETE /v2/pods/{id} (status codes scripted per test);
#               records argv and stdin
#   apt-get, dpkg-query, sshd, nvidia-smi, vllm   record their arguments
# ssh-keygen, base64, shasum, awk and the rest are the real macOS tools.
# The pod key is a canary; the tests check it never leaves stdin.
# Every boot run has a hard time limit (perl alarm), so a hang fails a test
# instead of hanging the suite.
#
# Usage: bash tests/test_pod_boot.sh      (BOOT_UNDER_TEST=path overrides the script)

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
BOOT="${BOOT_UNDER_TEST:-$HERE/../scripts/pod/boot.sh}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAKEBIN="$TMP/bin"
mkdir -p "$FAKEBIN"

CANARY="rpa_PODKEYCANARY00000000000000000000000000"
POD="testpod123"
REPO="Org/Model-1B"
REV="0123456789abcdef0123456789abcdef01234567"
NAME="model-1b"

PASS=0
FAIL=0
pass() { PASS=$((PASS + 1)); printf 'ok    %s\n' "$1"; }
fail() { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1"; }
has()   { if grep -qF -- "$2" "$1" 2>/dev/null; then pass "$3"; else fail "$3"; fi; }
lacks() { if grep -qF -- "$2" "$1" 2>/dev/null; then fail "$3"; else pass "$3"; fi; }
count() { if [ -f "$1" ]; then grep -cF -- "$2" "$1"; else echo 0; fi; }
eq()    { if [ "$1" = "$2" ]; then pass "$3"; else fail "$3 (got '$1', want '$2')"; fi; }
before() {  # before FILE A B NAME: first line with A comes before first line with B
  local a b
  a="$(grep -nF -- "$2" "$1" | head -1 | cut -d: -f1)"
  b="$(grep -nF -- "$3" "$1" | head -1 | cut -d: -f1)"
  if [ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ]; then pass "$4"; else fail "$4"; fi
}

# --- Fakes ---------------------------------------------------------------------
cat > "$FAKEBIN/curl" <<'FAKE'
#!/bin/bash
log="${FAKE_LOG:?}"
{ printf -- '---CALL---\n'; printf '%s\n' "$@"; } >> "$log/curl_argv"
method=GET; out=""; url=""; prev=""; hstdin=0
for a in "$@"; do
  case "$prev" in -o) out="$a" ;; -X) method="$a" ;; -H) [ "$a" = "@-" ] && hstdin=1 ;; esac
  case "$a" in https://*) url="$a" ;; esac
  prev="$a"
done
[ "$hstdin" = 1 ] && cat >> "$log/curl_stdin"
case "$method $url" in
  "DELETE https://api.runpod.io/v2/pods/"*)
    printf '%s\n' "$url" >> "$log/deletes"
    n="$(grep -c . "$log/deletes")"
    set -- ${FAKE_DELETE_CODES:-204}
    [ "$n" -gt "$#" ] && n="$#"
    eval "code=\${$n}"
    printf '%s' "$code"
    exit 0 ;;
  "GET ${FAKE_HF_PREFIX:?}"*)
    path="${url#"$FAKE_HF_PREFIX"}"
    printf '%s\n' "$path" >> "$log/hf"
    [ "$path" = "${FAKE_HF_SLOW:-}" ] && sleep "${FAKE_HF_SLEEP:-0}"
    [ -f "$FAKE_HF_DIR/$path" ] || { echo "curl: (22) 404" >&2; exit 22; }
    cp "$FAKE_HF_DIR/$path" "$out"
    exit 0 ;;
esac
printf '%s %s\n' "$method" "$url" >> "$log/unexpected"
exit 99
FAKE
cat > "$FAKEBIN/sshd" <<'FAKE'
#!/bin/bash
{ printf -- '---CALL---\n'; printf '%s\n' "$@"; } >> "${FAKE_LOG:?}/sshd_args"
exit "${FAKE_SSHD_RC:-0}"
FAKE
cat > "$FAKEBIN/apt-get" <<'FAKE'
#!/bin/bash
printf '%s\n' "$*" >> "${FAKE_LOG:?}/apt"
exit 0
FAKE
cat > "$FAKEBIN/dpkg-query" <<'FAKE'
#!/bin/bash
printf '1:9.6p1-3ubuntu13.5'
FAKE
cat > "$FAKEBIN/nvidia-smi" <<'FAKE'
#!/bin/bash
case "$*" in
  *--query-gpu*) echo "NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.65.06, 97887 MiB" ;;
  *) echo "| NVIDIA-SMI 580.65.06   Driver Version: 580.65.06   CUDA Version: 13.0 |" ;;
esac
FAKE
cat > "$FAKEBIN/vllm" <<'FAKE'
#!/bin/bash
[ "${1:-}" = "--version" ] && { echo "0.30.0"; exit 0; }
printf '%s\n' "$@" > "${FAKE_LOG:?}/vllm_args"
{ echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset}"
  echo "pod_key_in_env=$(env | grep -c '^RUNPOD_API_KEY=')"
  echo "cwd=$(pwd)"; } > "$FAKE_LOG/vllm_env"
echo "INFO fake vllm serving"
exit 0
FAKE
chmod +x "$FAKEBIN"/*

# --- A tiny fake model, its manifest, two SSH keys -------------------------------
HF="$TMP/hf"
mkdir -p "$HF/sub"
printf '{"architectures":["Fake"]}\n' > "$HF/config.json"
head -c 200000 /dev/urandom > "$HF/model-00001.safetensors"
printf '{"tok":1}\n' > "$HF/sub/tokenizer.json"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
size() { wc -c < "$1" | tr -d ' '; }
mkman() {  # mkman OUT [extra line...]: manifest for the fake repo, like config/manifests
  local out="$1" p; shift
  { printf '# %s at %s\n' "$REPO" "$REV"
    printf '# path\tsize\tsha256\tsource\n'
    for p in config.json model-00001.safetensors sub/tokenizer.json; do
      printf '%s\t%s\t%s\thub\n' "$p" "$(size "$HF/$p")" "$(sha "$HF/$p")"
    done
    for p in "$@"; do printf '%s\n' "$p"; done; } > "$out"
}
MAN="$TMP/manifest.tsv"
mkman "$MAN"

ssh-keygen -q -t ed25519 -N '' -C scale-invariance-pod -f "$TMP/key" >/dev/null
ssh-keygen -q -t ed25519 -N '' -C other -f "$TMP/other" >/dev/null
ssh-keygen -q -t rsa -b 2048 -N '' -C rsa -f "$TMP/rsa" >/dev/null
KEY="$(cat "$TMP/key.pub")"
FPR="$(ssh-keygen -lf "$TMP/key.pub" | cut -d' ' -f2)"
OTHER_FPR="$(ssh-keygen -lf "$TMP/other.pub" | cut -d' ' -f2)"

# --- Running one boot ------------------------------------------------------------
# boot NAME [VAR=value ...]: fresh root and log folder, defaults, overrides last.
boot() {
  local name="$1" lim=30 a; shift
  for a in "$@"; do case "$a" in LIMIT=*) lim="${a#LIMIT=}" ;; esac; done
  S="$TMP/s_$name"; ROOT="$S/root"; L="$S/log"; OUT="$S/out"
  mkdir -p "$ROOT" "$L"
  env -i PATH="$FAKEBIN:/usr/bin:/bin" HOME="$S" TMPDIR="$S" \
    FAKE_LOG="$L" FAKE_HF_DIR="$HF" FAKE_HF_PREFIX="https://huggingface.co/$REPO/resolve/$REV/" \
    SI_ROOT="$ROOT" SI_TICK_S=0.1 SI_RETRY_S=0.1 \
    RUNPOD_POD_ID="$POD" RUNPOD_API_KEY="$CANARY" \
    SI_MODE=serve SI_LIFETIME_S=2 PUBLIC_KEY="$KEY" SI_SSH_KEY_FPR="$FPR" \
    SI_MODEL_NAME="$NAME" SI_MODEL_REPO="$REPO" SI_MODEL_REV="$REV" \
    SI_MANIFEST_B64="$(base64 < "${MANIFEST:-$MAN}" | tr -d '\n')" \
    SI_MANIFEST_SHA256="$(sha "${MANIFEST:-$MAN}")" \
    "$@" \
    perl -e 'alarm shift; exec @ARGV' "$lim" /bin/bash "$BOOT" > "$OUT" 2>&1
  RC=$?
}
no_key_leak() {   # the canary may appear only in the curl stdin log
  local hits
  hits="$(grep -rlF -- "$CANARY" "$S" 2>/dev/null | grep -v '/log/curl_stdin$' || true)"
  if [ -z "$hits" ]; then pass "$1: pod key appears nowhere but curl's stdin"; else fail "$1: pod key leaked to: $hits"; fi
}
failed_safely() {  # a refused boot: reason logged, nothing served, pod deleted
  has "$OUT" "BOOT_FAILED $2" "$1: refuses with '$2'"
  [ -f "$L/vllm_args" ] && fail "$1: vLLM started anyway" || pass "$1: vLLM never started"
  has "$OUT" "SELF_DELETE ok" "$1: pod deletes itself"
  no_key_leak "$1"
}

echo "== serve mode, everything right =="
boot serve
eq "$RC" 0 "exits 0 after the watchdog deletes the pod"
before "$OUT" "watchdog started" "installing openssh-server" "watchdog starts before anything that can hang"
has "$L/apt" "install -y -qq --no-install-recommends openssh-server" "installs openssh-server"
has "$OUT" "C19: /root/.ssh/authorized_keys at boot: absent" "logs the C19 check at boot"
has "$OUT" "SSH_READY (key $FPR only)" "SSH ready, with our key's fingerprint"
CONF="$ROOT/etc/ssh/si_sshd_config"
AUTH="$ROOT/etc/ssh/si_authorized_keys"
eq "$(sed -n 2p "$L/sshd_args")" "-t" "sshd config is tested first"
eq "$(sed -n 3,4p "$L/sshd_args" | tr '\n' ' ')" "-f $CONF " "sshd -t uses our config"
eq "$(sed -n 6,9p "$L/sshd_args" | tr '\n' ' ')" "-f $CONF -E $ROOT/var/log/si/sshd.log " "sshd starts with our config only"
for line in "AuthorizedKeysFile $AUTH" "AuthorizedKeysCommand none" "PasswordAuthentication no" \
    "KbdInteractiveAuthentication no" "AllowAgentForwarding no" "AllowTcpForwarding local" \
    "PermitOpen 127.0.0.1:8000" "PermitRootLogin prohibit-password" "UsePAM no" "X11Forwarding no" \
    "PermitTunnel no" "GatewayPorts no" "HostKey $ROOT/etc/ssh/si_host_ed25519_key"; do
  if grep -qxF -- "$line" "$CONF"; then pass "sshd config: $line"; else fail "sshd config: $line"; fi
done
eq "$(grep -c '^AuthorizedKeysFile' "$CONF")" 1 "exactly one AuthorizedKeysFile line"
eq "$(grep -c . "$AUTH")" 1 "authorized keys file holds exactly one key"
eq "$(cat "$AUTH")" "restrict,port-forwarding,permitopen=\"127.0.0.1:8000\" $KEY" "the key is ours, restricted to forwarding to vLLM"
eq "$(grep 'SSH_HOST_KEY ' "$OUT" | sed 's/.*SSH_HOST_KEY //')" "$(cut -d' ' -f1,2 "$ROOT/etc/ssh/si_host_ed25519_key.pub")" "prints the host key it actually uses"
has "$OUT" "SSH_HOST_KEY_FPR $(ssh-keygen -lf "$ROOT/etc/ssh/si_host_ed25519_key.pub" | cut -d' ' -f2)" "prints the host key fingerprint"
has "$OUT" "GPU: NVIDIA RTX PRO 6000" "logs the GPU"
has "$OUT" "host CUDA: CUDA Version: 13.0" "logs the host CUDA version"
eq "$(grep -c . "$L/hf")" 3 "downloads each manifest file once"
ok=1
for p in config.json model-00001.safetensors sub/tokenizer.json; do
  cmp -s "$HF/$p" "$ROOT/models/$NAME/$p" || ok=0
done
eq "$ok" 1 "downloaded files are byte-identical to the repo's"
eq "$(find "$ROOT/models/$NAME" -name '*.part' | grep -c .)" 0 "no partial files left"
first_hf="$(awk '/^---CALL---$/{n++} n==1' "$L/curl_argv" | sed -n 2p)"
eq "$first_hf" "-q" "curl's first argument is -q (ignore ~/.curlrc)"
has "$L/curl_argv" "--proto-redir" "redirects are limited to https"
has "$L/curl_argv" "https://huggingface.co/$REPO/resolve/$REV/sub/tokenizer.json" "downloads at the pinned commit"
has "$OUT" "WEIGHTS_OK files=3 bytes=" "weights check passes"
before "$OUT" "WEIGHTS_OK" "VLLM_STARTING" "vLLM starts only after the weights check"
want="serve
$ROOT/models/$NAME
--served-model-name
$NAME
--host
127.0.0.1
--port
8000
--max-model-len
8192
--generation-config
vllm
--language-model-only
--enable-log-requests
--seed
0
--gpu-memory-utilization
0.92"
eq "$(cat "$L/vllm_args")" "$want" "vLLM gets exactly the infra-plan flags, on 127.0.0.1"
has "$L/vllm_env" "HF_HUB_OFFLINE=1" "vLLM runs offline"
has "$L/vllm_env" "pod_key_in_env=0" "vLLM does not inherit the pod key"
has "$OUT" "[vllm] INFO fake vllm serving" "vLLM output reaches the pod log"
has "$OUT" "VLLM_EXITED" "vLLM exit is logged"
has "$OUT" "WATCHDOG lifetime of 2 s reached" "watchdog fires at the lifetime"
has "$OUT" "SELF_DELETE ok (HTTP 204)" "pod deletes itself"
eq "$(cat "$L/deletes")" "https://api.runpod.io/v2/pods/$POD" "DELETE goes to this pod only"
has "$L/curl_argv" "DELETE" "uses the DELETE method"
eq "$(cat "$L/curl_stdin")" "Authorization: Bearer $CANARY" "pod key reaches curl on stdin"
no_key_leak "serve"
[ -f "$L/unexpected" ] && fail "no unexpected requests ($(cat "$L/unexpected"))" || pass "no unexpected requests"

echo "== probe mode =="
boot probe SI_MODE=probe SI_LIFETIME_S=1
eq "$RC" 0 "probe exits 0"
has "$OUT" "SSH_READY" "probe brings SSH up"
[ -f "$L/hf" ] && fail "probe downloads nothing" || pass "probe downloads nothing"
[ -f "$L/vllm_args" ] && fail "probe starts no vLLM" || pass "probe starts no vLLM"
has "$OUT" "SELF_DELETE ok" "probe deletes itself"
boot podinfo SI_MODE=probe SI_LIFETIME_S=1 RUNPOD_DC_ID=EU-RO-1 RUNPOD_PUBLIC_IP='203.0.113.7;rm' RUNPOD_TCP_PORT_22=40122
has "$OUT" "pod: dc=EU-RO-1 public_ip=203.0.113.7 ssh_port=40122" "logs where the pod landed, sanitized"

echo "== a hanging download can't outlive the lifetime =="
boot hang SI_LIFETIME_S=1 FAKE_HF_SLOW=config.json FAKE_HF_SLEEP=5 LIMIT=20
before "$OUT" "SELF_DELETE ok" "WEIGHTS_OK" "self-delete happens while the download still hangs"

echo "== self-delete keeps trying until it works =="
boot retry SI_MODE=probe SI_LIFETIME_S=1 FAKE_DELETE_CODES="403 500 204"
has "$OUT" "SELF_DELETE FAILED (HTTP 403)" "a refused delete is logged (the 7a-1 question)"
has "$OUT" "SELF_DELETE FAILED (HTTP 500)" "a server error is logged"
has "$OUT" "SELF_DELETE ok (HTTP 204)" "then succeeds"
eq "$(grep -c . "$L/deletes")" 3 "three tries"
boot gone SI_MODE=probe SI_LIFETIME_S=1 FAKE_DELETE_CODES="404 204"
has "$OUT" "SELF_DELETE FAILED (HTTP 404)" "a 404 is not trusted as 'already gone'"
has "$OUT" "SELF_DELETE ok (HTTP 204)" "after a 404 it keeps trying"
boot badtick SI_MODE=probe SI_LIFETIME_S=1 SI_TICK_S='1;x' SI_RETRY_S=abc
has "$OUT" "SELF_DELETE ok" "malformed tick/retry settings fall back to defaults"

echo "== delete on request, and C19 watching =="
S="$TMP/s_req"
( sleep 1.5; mkdir -p "$S/root/root/.ssh"; printf 'ssh-ed25519 AAAAx a\nssh-ed25519 AAAAy b\n' > "$S/root/root/.ssh/authorized_keys"
  sleep 1; echo "laptop asked" > "$S/root/run/si/delete_now" ) &
boot req SI_MODE=probe SI_LIFETIME_S=600 LIMIT=20
wait
has "$OUT" "C19: /root/.ssh/authorized_keys changed: present (2 line(s)) (ignored)" "logs keys appearing in /root/.ssh"
has "$OUT" "WATCHDOG delete requested (laptop asked)" "the laptop can ask for deletion"
has "$OUT" "SELF_DELETE ok" "and the pod deletes itself"
lacks "$ROOT/etc/ssh/si_sshd_config" ".ssh/authorized_keys" "our sshd never reads /root/.ssh"

echo "== refusals: SSH key =="
boot twokeys PUBLIC_KEY="$KEY
$(cat "$TMP/other.pub")"
failed_safely twokeys "PUBLIC_KEY must hold exactly one key"
[ -f "$L/sshd_args" ] && fail "twokeys: sshd not started" || pass "twokeys: sshd not started"
boot wrongfpr SI_SSH_KEY_FPR="$OTHER_FPR"
failed_safely wrongfpr "PUBLIC_KEY fingerprint does not match"
[ -f "$L/sshd_args" ] && fail "wrongfpr: sshd not started" || pass "wrongfpr: sshd not started"
boot rsa PUBLIC_KEY="$(cat "$TMP/rsa.pub")" SI_SSH_KEY_FPR="$(ssh-keygen -lf "$TMP/rsa.pub" | cut -d' ' -f2)"
failed_safely rsa "PUBLIC_KEY is not a single ssh-ed25519 key"
boot optkey PUBLIC_KEY="command=\"sh\" $KEY"
failed_safely optkey "PUBLIC_KEY is not a single ssh-ed25519 key"
boot sshdbad FAKE_SSHD_RC=1
failed_safely sshdbad "sshd rejected its config"

echo "== refusals: weights =="
TAMPER="$TMP/hf_tampered"
cp -R "$HF" "$TAMPER"
printf 'x' >> "$TAMPER/sub/tokenizer.json"
boot tampered FAKE_HF_DIR="$TAMPER"
failed_safely tampered "WEIGHTS_BAD sub/tokenizer.json"
eq "$(count "$OUT" "WEIGHTS_MISMATCH sub/tokenizer.json")" 3 "tampered: three tries, then stop"
SAMESIZE="$TMP/hf_samesize"
cp -R "$HF" "$SAMESIZE"
printf '{"tok":2}\n' > "$SAMESIZE/sub/tokenizer.json"
boot samesize FAKE_HF_DIR="$SAMESIZE"
failed_safely samesize "WEIGHTS_BAD sub/tokenizer.json"
mkman "$TMP/m_size"
awk -F'\t' 'BEGIN{OFS="\t"} $1=="config.json"{$2=$2+1} {print}' "$TMP/m_size" > "$TMP/m_size2"
MANIFEST="$TMP/m_size2" boot wrongsize
failed_safely wrongsize "WEIGHTS_BAD config.json"
boot manhash SI_MANIFEST_SHA256="$(printf '%064d' 0)"
failed_safely manhash "manifest does not match SI_MANIFEST_SHA256"
[ -f "$L/hf" ] && fail "manhash: nothing downloaded" || pass "manhash: nothing downloaded"
mkman "$TMP/m_trav" "$(printf '../evil\t5\t%s\thub' "$(printf '%064d' 0)")"
MANIFEST="$TMP/m_trav" boot traversal
failed_safely traversal "manifest: bad path"
lacks "$L/hf" "evil" "traversal: never requested"
mkman "$TMP/m_abs" "$(printf '/etc/passwd\t5\t%s\thub' "$(printf '%064d' 0)")"
MANIFEST="$TMP/m_abs" boot absolute
failed_safely absolute "manifest: bad path"
mkman "$TMP/m_dup" "$(printf 'config.json\t%s\t%s\thub' "$(size "$HF/config.json")" "$(sha "$HF/config.json")")"
MANIFEST="$TMP/m_dup" boot duplicate
failed_safely duplicate "manifest: duplicate config.json"
boot missing404 FAKE_HF_DIR="$TMP/empty"
failed_safely missing404 "WEIGHTS_BAD config.json"
boot shortrev SI_MODEL_REV=0123456
failed_safely shortrev "SI_MODEL_REV must be a 40-character commit"
[ -f "$L/hf" ] && fail "shortrev: nothing downloaded" || pass "shortrev: nothing downloaded"
boot badrepo SI_MODEL_REPO="Org/../x"
failed_safely badrepo "SI_MODEL_REPO malformed"
mkdir -p "$TMP/s_exists/root/models/$NAME"
boot exists
failed_safely exists "model folder already exists"

echo "== refusals before the watchdog can run =="
boot nomode SI_MODE=
eq "$RC" 1 "no SI_MODE: exit 1"
has "$OUT" "BOOT_FAILED SI_MODE must be probe or serve" "no SI_MODE: says why"
[ -f "$L/sshd_args" ] && fail "no SI_MODE: nothing started" || pass "no SI_MODE: nothing started"
boot zerolife SI_LIFETIME_S=0
has "$OUT" "BOOT_FAILED SI_LIFETIME_S must be 1..21600" "lifetime 0 refused"
boot longlife SI_LIFETIME_S=21601
has "$OUT" "BOOT_FAILED SI_LIFETIME_S must be 1..21600" "lifetime over 6 h refused"
boot nokey RUNPOD_API_KEY=
has "$OUT" "BOOT_FAILED RUNPOD_API_KEY (pod-scoped) missing" "no pod key: refused, says the laptop must delete"
has "$OUT" "the laptop must delete this pod" "no pod key: tells the laptop"
boot badpod RUNPOD_POD_ID="x/../y"
has "$OUT" "BOOT_FAILED RUNPOD_POD_ID missing or malformed" "malformed pod ID refused"

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
