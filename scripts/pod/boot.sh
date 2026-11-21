#!/bin/bash
# boot.sh: the start-up script of a scale-invariance pod (Phase 7a).
#
# The launcher passes this whole file as the pod's command, running inside
# the official vllm/vllm-openai image (pinned by digest; infra plan D2).
# It runs as root on Ubuntu 24.04. Tested offline on macOS /bin/bash 3.2
# (tests/test_pod_boot.sh), so it sticks to bash 3.2 features.
#
# WHAT IT DOES, IN ORDER
#   1. Checks its inputs. Then, before anything else can hang, starts the
#      lifetime watchdog: when SI_LIFETIME_S runs out (or the file
#      $RUN/delete_now appears), the pod deletes itself with its own
#      pod-scoped RUNPOD_API_KEY, retrying until it works (infra plan 6.2).
#      Every later failure ends in that same self-delete.
#   2. SSH (D4, C19): installs openssh-server, makes a fresh host key and
#      prints it, so the laptop can pin it. sshd runs from our own config
#      file and accepts exactly one key: ours, from PUBLIC_KEY, checked
#      against SI_SSH_KEY_FPR, stored in a file of its own. Keys RunPod
#      might drop into /root/.ssh are ignored, and their appearance is
#      logged ("C19:" lines). Only local port forwarding to 127.0.0.1:8000
#      is allowed; no agent forwarding, no passwords.
#   3. serve mode only: downloads every file of the model at the pinned
#      commit, checks each file's size and SHA-256 against the manifest
#      (config/manifests/<model>.tsv, passed in SI_MANIFEST_B64 and itself
#      checked against SI_MANIFEST_SHA256). One mismatch: vLLM never starts.
#   4. serve mode only: runs vLLM on 127.0.0.1:8000 with the infra plan 2
#      flags and HF_HUB_OFFLINE=1. When vLLM exits, the pod stays up (SSH
#      for diagnosis) until the watchdog deletes it.
#   probe mode does 1 and 2 only (for testing self-delete cheaply).
#
# INPUTS (pod environment; none is secret, the API can read them all)
#   SI_MODE            probe | serve
#   SI_LIFETIME_S      seconds until the pod deletes itself (1..21600)
#   PUBLIC_KEY         our one ssh-ed25519 public key
#   SI_SSH_KEY_FPR     its SHA256:... fingerprint
#   serve mode: SI_MODEL_NAME, SI_MODEL_REPO, SI_MODEL_REV (40 hex),
#               SI_MANIFEST_B64, SI_MANIFEST_SHA256
#   from RunPod: RUNPOD_POD_ID, RUNPOD_API_KEY (pod-scoped; never printed)
#   tests only:  SI_ROOT (prefix for every path), SI_TICK_S, SI_RETRY_S
#
# LOG LINES the launcher reads (stdout = the pod log):
#   [si-boot] SSH_HOST_KEY <type> <key>     [si-boot] SSH_READY
#   [si-boot] WEIGHTS_OK files=<n> bytes=<b> manifest=<sha>
#   [si-boot] VLLM_STARTING                 [si-boot] BOOT_FAILED <reason>
#   [si-boot] SELF_DELETE ok|FAILED ...

set +x +v +a
set -uo pipefail
umask 022
export LC_ALL=C
PATH="$PATH:/usr/sbin:/sbin"

R="${SI_ROOT:-}"
RUN="$R/run/si"
LOGDIR="$R/var/log/si"
SSHDIR="$R/etc/ssh"
MODELS="$R/models"
TICK="${SI_TICK_S:-5}"
RETRY="${SI_RETRY_S:-60}"
re_secs='^[0-9]{1,4}(\.[0-9]{1,3})?$'
[[ "$TICK" =~ $re_secs ]] || TICK=5
[[ "$RETRY" =~ $re_secs ]] || RETRY=60
readonly API="https://api.runpod.io/v2"

log() {   # to the pod log (stdout) and, once it exists, to $LOGDIR/boot.log for SSH readers
  printf '[si-boot] %s\n' "$*"
  [ -d "$LOGDIR" ] && printf '[si-boot] %s\n' "$*" >> "$LOGDIR/boot.log"
  return 0
}

# The pod key goes into a shell variable that is not exported, and out of
# the environment, so vLLM and other children never inherit it.
POD_KEY="${RUNPOD_API_KEY:-}"
unset RUNPOD_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN

# --- 1. Inputs needed for the watchdog, then the watchdog --------------------
die_early() { log "BOOT_FAILED $*"; log "cannot self-delete; the laptop must delete this pod"; exit 1; }

MODE="${SI_MODE:-}"
case "$MODE" in probe|serve) ;; *) die_early "SI_MODE must be probe or serve" ;; esac
LIFETIME="${SI_LIFETIME_S:-}"
re_int='^[0-9]+$'
if ! [[ "$LIFETIME" =~ $re_int ]] || [ "$LIFETIME" -lt 1 ] || [ "$LIFETIME" -gt 21600 ]; then
  die_early "SI_LIFETIME_S must be 1..21600"
fi
POD_ID="${RUNPOD_POD_ID:-}"
re_id='^[A-Za-z0-9]{1,64}$'
[[ "$POD_ID" =~ $re_id ]] || die_early "RUNPOD_POD_ID missing or malformed"
[ -n "$POD_KEY" ] || die_early "RUNPOD_API_KEY (pod-scoped) missing"

mkdir -p "$RUN" "$LOGDIR" || die_early "cannot create $RUN"
rm -f "$RUN/delete_now"

self_delete() {   # prints the HTTP status; the key reaches curl on stdin only
  printf 'Authorization: Bearer %s\n' "$POD_KEY" |
    curl -q -sS --proto '=https' --tlsv1.2 --connect-timeout 10 --max-time 30 \
      -o /dev/null -w '%{http_code}' -X DELETE -H @- "$API/pods/$POD_ID" 2>/dev/null
}

c19_state() {     # what, if anything, sits where RunPod would inject keys
  local f="$R/root/.ssh/authorized_keys"
  if [ -f "$f" ]; then printf 'present (%s line(s))' "$(grep -c . "$f" 2>/dev/null || true)"; else printf 'absent'; fi
}

watchdog() {
  local end reason code last now
  log "watchdog started: lifetime $LIFETIME s"     # logged by the watchdog itself
  end=$(( $(date +%s) + LIFETIME ))
  last="$(c19_state)"
  log "C19: /root/.ssh/authorized_keys at boot: $last (our sshd never reads it)"
  while :; do
    now="$(date +%s)"
    if [ "$now" -ge "$end" ]; then reason="lifetime of $LIFETIME s reached"; break; fi
    if [ -e "$RUN/delete_now" ]; then reason="delete requested ($(head -c 200 "$RUN/delete_now" | tr -cd 'A-Za-z0-9 ._:-'))"; break; fi
    now="$(c19_state)"
    if [ "$now" != "$last" ]; then log "C19: /root/.ssh/authorized_keys changed: $now (ignored)"; last="$now"; fi
    sleep "$TICK"
  done
  log "WATCHDOG $reason; deleting this pod"
  while :; do
    code="$(self_delete)" || true
    case "$code" in
      200|204) log "SELF_DELETE ok (HTTP $code)"; exit 0 ;;
      # Not trusted as "gone": a key that can't see the pod may also get
      # 404. If the pod really is gone, this process is gone with it.
      *)       log "SELF_DELETE FAILED (HTTP ${code:-none}); retrying in $RETRY s" ;;
    esac
    sleep "$RETRY"
  done
}

watchdog &
WD_PID=$!
sleep 0.2     # let the watchdog's first lines reach the log before ours

fail() {          # every failure after this point: log, ask for deletion, wait
  log "BOOT_FAILED $*"
  printf 'boot failed\n' > "$RUN/delete_now"
  wait "$WD_PID"
  exit 1
}

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
bytes_of() { wc -c < "$1" | tr -d ' '; }

# --- 2. SSH: one key, our config, a fresh host key ---------------------------
FPR="${SI_SSH_KEY_FPR:-}"
KEY="${PUBLIC_KEY:-}"
re_key='^ssh-ed25519 [A-Za-z0-9+/]+={0,2}( [A-Za-z0-9._@-]{1,64})?$'
[ "$(printf '%s' "$KEY" | grep -c .)" = "1" ] || fail "PUBLIC_KEY must hold exactly one key"
[[ "$KEY" =~ $re_key ]] || fail "PUBLIC_KEY is not a single ssh-ed25519 key"
re_fpr='^SHA256:[A-Za-z0-9+/]{43}$'
[[ "$FPR" =~ $re_fpr ]] || fail "SI_SSH_KEY_FPR malformed"

mkdir -p "$SSHDIR" || fail "cannot create $SSHDIR"
# Always installed: the vLLM image ships without it (C9), and installing
# unconditionally means no other sshd on PATH is ever picked up by accident.
log "installing openssh-server"
DEBIAN_FRONTEND=noninteractive apt-get update -qq > "$LOGDIR/apt.log" 2>&1 &&
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends openssh-server \
    >> "$LOGDIR/apt.log" 2>&1 || fail "apt-get could not install openssh-server"
log "openssh-server version: $(dpkg-query -W -f='${Version}' openssh-server 2>/dev/null || echo unknown)"

AUTH="$SSHDIR/si_authorized_keys"
printf '%s\n' "$KEY" > "$RUN/key.pub"
got="$(ssh-keygen -lf "$RUN/key.pub" 2>/dev/null | cut -d' ' -f2)"
[ "$got" = "$FPR" ] || fail "PUBLIC_KEY fingerprint does not match SI_SSH_KEY_FPR"
printf 'restrict,port-forwarding,permitopen="127.0.0.1:8000" %s\n' "$KEY" > "$AUTH" || fail "cannot write $AUTH"
chmod 644 "$AUTH"

HOSTKEY="$SSHDIR/si_host_ed25519_key"
rm -f "$HOSTKEY" "$HOSTKEY.pub"
ssh-keygen -q -t ed25519 -N '' -C si-pod-host -f "$HOSTKEY" >/dev/null 2>&1 || fail "cannot make a host key"

CONF="$SSHDIR/si_sshd_config"
cat > "$CONF" <<EOF || fail "cannot write $CONF"
# scale-invariance pod sshd: one key, local forwarding to vLLM only
Port 22
ListenAddress 0.0.0.0
HostKey $HOSTKEY
PidFile $RUN/sshd.pid
AuthorizedKeysFile $AUTH
AuthorizedKeysCommand none
AuthorizedPrincipalsFile none
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitEmptyPasswords no
HostbasedAuthentication no
UsePAM no
PermitRootLogin prohibit-password
AllowUsers root
StrictModes yes
AllowAgentForwarding no
AllowTcpForwarding local
PermitOpen 127.0.0.1:8000
GatewayPorts no
X11Forwarding no
PermitTunnel no
PermitUserEnvironment no
MaxAuthTries 3
LoginGraceTime 20
ClientAliveInterval 30
ClientAliveCountMax 4
LogLevel VERBOSE
EOF
mkdir -p "$R/run/sshd"
# OpenSSH 9.x refuses to start unless run by absolute path ("sshd re-exec
# requires execution with an absolute path"; seen on the first probe, 2026-10-03).
SSHD_BIN="$(command -v sshd)"
case "$SSHD_BIN" in /*) ;; *) fail "sshd not found after install" ;; esac
"$SSHD_BIN" -t -f "$CONF" > "$LOGDIR/sshd_test.log" 2>&1 || fail "sshd rejected its config"
"$SSHD_BIN" -f "$CONF" -E "$LOGDIR/sshd.log" || fail "sshd did not start"
log "SSH_HOST_KEY $(cut -d' ' -f1,2 "$HOSTKEY.pub")"
log "SSH_HOST_KEY_FPR $(ssh-keygen -lf "$HOSTKEY.pub" | cut -d' ' -f2)"
log "SSH_READY (key $FPR only)"

log "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -4 | tr '\n' ';' || echo unknown)"
log "host CUDA: $(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1 || echo unknown)"
log "pod: dc=$(printf '%s' "${RUNPOD_DC_ID:-?}" | tr -cd 'A-Za-z0-9-?') public_ip=$(printf '%s' "${RUNPOD_PUBLIC_IP:-?}" | tr -cd '0-9.?') ssh_port=$(printf '%s' "${RUNPOD_TCP_PORT_22:-?}" | tr -cd '0-9?')"

if [ "$MODE" = "probe" ]; then
  log "probe mode: no model, no vLLM; waiting for the watchdog"
  wait "$WD_PID"
  exit 0
fi

# --- 3. Weights: download at the pinned commit, check every file -------------
NAME="${SI_MODEL_NAME:-}"; REPO="${SI_MODEL_REPO:-}"; REV="${SI_MODEL_REV:-}"
re_name='^[a-z0-9][a-z0-9._-]{0,63}$'
re_repo='^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$'
re_rev='^[0-9a-f]{40}$'
[[ "$NAME" =~ $re_name ]] || fail "SI_MODEL_NAME malformed"
[[ "$REPO" =~ $re_repo ]] || fail "SI_MODEL_REPO malformed"
[[ "$REV" =~ $re_rev ]] || fail "SI_MODEL_REV must be a 40-character commit"

MAN="$RUN/manifest.tsv"
printf '%s' "${SI_MANIFEST_B64:-}" | base64 -d > "$MAN" 2>/dev/null || fail "SI_MANIFEST_B64 is not base64"
[ "$(sha256_of "$MAN")" = "${SI_MANIFEST_SHA256:-}" ] || fail "manifest does not match SI_MANIFEST_SHA256"

DIR="$MODELS/$NAME"
[ ! -e "$DIR" ] || fail "model folder already exists: $DIR"
mkdir -p "$DIR" || fail "cannot create $DIR"

re_path='^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$'
re_sha='^[0-9a-f]{64}$'
n=0; total=0; seen=" "
TAB="$(printf '\t')"
while IFS="$TAB" read -r path size sha src extra; do
  case "$path" in ''|'#'*) continue ;; esac
  [[ "$path" =~ $re_path ]] || fail "manifest: bad path"
  case "/$path/" in */./*|*/../*) fail "manifest: bad path" ;; esac
  [[ "$size" =~ $re_int ]] || fail "manifest: bad size for $path"
  [[ "$sha" =~ $re_sha ]] || fail "manifest: bad sha256 for $path"
  [ -z "${extra:-}" ] || fail "manifest: extra column for $path"
  case "$seen" in *" $path "*) fail "manifest: duplicate $path" ;; esac
  seen="$seen$path "
  dest="$DIR/$path"
  mkdir -p "$(dirname "$dest")" || fail "cannot create folder for $path"
  ok=0
  for try in 1 2 3; do
    rm -f "$dest.part"
    if curl -q -sS --fail -L --proto '=https' --proto-redir '=https' --max-redirs 5 --tlsv1.2 \
         --connect-timeout 20 --max-time 3600 -o "$dest.part" \
         "https://huggingface.co/$REPO/resolve/$REV/$path" 2>> "$LOGDIR/download.log"; then
      if [ "$(bytes_of "$dest.part")" = "$size" ] && [ "$(sha256_of "$dest.part")" = "$sha" ]; then
        mv "$dest.part" "$dest" && ok=1 && break
      fi
      log "WEIGHTS_MISMATCH $path (try $try)"
    else
      log "download failed: $path (try $try)"
    fi
  done
  [ "$ok" = 1 ] || fail "WEIGHTS_BAD $path: wrong size or sha256 after 3 tries"
  n=$((n + 1)); total=$((total + size))
done < "$MAN"
[ "$n" -gt 0 ] || fail "manifest lists no files"
[ "$(find "$DIR" -type f | grep -c .)" = "$n" ] || fail "model folder has files the manifest doesn't list"
log "WEIGHTS_OK files=$n bytes=$total manifest=$(sha256_of "$MAN")"

# --- 4. vLLM on 127.0.0.1 only ------------------------------------------------
log "vLLM version: $(vllm --version 2>/dev/null | head -1 || echo unknown)"
log "VLLM_STARTING"
( cd / && exec env HF_HUB_OFFLINE=1 vllm serve "$DIR" --served-model-name "$NAME" \
    --host 127.0.0.1 --port 8000 --max-model-len 8192 \
    --generation-config vllm --language-model-only \
    --enable-log-requests --seed 0 --gpu-memory-utilization 0.92 ) 2>&1 |
  tee -a "$LOGDIR/vllm.log" | awk '{ print "[vllm] " $0; fflush() }' || true
log "VLLM_EXITED; pod stays up for diagnosis until the watchdog deletes it"
wait "$WD_PID"
