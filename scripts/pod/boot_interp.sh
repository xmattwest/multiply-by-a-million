#!/bin/bash
# boot_interp.sh: start-up script of a scale-invariance INTERP pod (plan_mechinterp §3 Step 1).
#
# Sibling of scripts/pod/boot.sh. Steps 1-3 (watchdog + self-delete, one-key
# SSH, weights download + per-file manifest check) are identical in intent to
# boot.sh; step 4 runs PyTorch hooks (scripts/interp/run_interp.py) instead of
# vLLM serve. The interp code is shipped IN this launch (base64 tar in
# SI_INTERP_B64, hash-checked against SI_INTERP_SHA256), so the pod never needs
# the repo or any network after the weights download (HF stays offline).
#
# The control API binds 127.0.0.1:8000 — the same address boot.sh's vLLM uses
# and the only one sshd PermitOpen's — so no sshd change and the launcher's
# existing tunnel (laptop 18000+slot -> pod 8000) reaches it unchanged.
#
# INPUTS add, over boot.sh serve mode:
#   SI_INTERP_B64      base64 of a tar.gz of scripts/interp/ + scripts/game_engine.py
#                      + prereg/settings.json (everything run_interp.py imports)
#   SI_INTERP_SHA256   sha256 of the decoded tar.gz
# Reuses: SI_MODE(=interp), SI_LIFETIME_S, PUBLIC_KEY, SI_SSH_KEY_FPR,
#         SI_MODEL_NAME, SI_MODEL_REPO, SI_MODEL_REV, SI_MANIFEST_B64,
#         SI_MANIFEST_SHA256, RUNPOD_POD_ID, RUNPOD_API_KEY.
#
# LOG LINES the launcher reads (stdout):
#   [si-boot] SSH_HOST_KEY ...   [si-boot] SSH_READY
#   [si-boot] WEIGHTS_OK files=<n> bytes=<b> manifest=<sha>
#   [si-boot] INTERP_CODE_OK sha=<sha>
#   [si-boot] INTERP_STARTING    [si-boot] BOOT_FAILED <reason>
#   [si-boot] SELF_DELETE ok|FAILED ...
# run_interp.py itself prints [interp] INTERP_READY ... when the model loads.

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
CODE="$R/workspace"
TICK="${SI_TICK_S:-5}"
RETRY="${SI_RETRY_S:-60}"
re_secs='^[0-9]{1,4}(\.[0-9]{1,3})?$'
[[ "$TICK" =~ $re_secs ]] || TICK=5
[[ "$RETRY" =~ $re_secs ]] || RETRY=60
readonly API="https://api.runpod.io/v2"

log() {
  printf '[si-boot] %s\n' "$*"
  [ -d "$LOGDIR" ] && printf '[si-boot] %s\n' "$*" >> "$LOGDIR/boot.log"
  return 0
}

POD_KEY="${RUNPOD_API_KEY:-}"
unset RUNPOD_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN

# --- 1. Inputs for the watchdog, then the watchdog ---------------------------
die_early() { log "BOOT_FAILED $*"; log "cannot self-delete; the laptop must delete this pod"; exit 1; }

MODE="${SI_MODE:-}"
case "$MODE" in interp) ;; *) die_early "SI_MODE must be interp for boot_interp.sh" ;; esac
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

self_delete() {
  printf 'Authorization: Bearer %s\n' "$POD_KEY" |
    curl -q -sS --proto '=https' --tlsv1.2 --connect-timeout 10 --max-time 30 \
      -o /dev/null -w '%{http_code}' -X DELETE -H @- "$API/pods/$POD_ID" 2>/dev/null
}

c19_state() {
  local f="$R/root/.ssh/authorized_keys"
  if [ -f "$f" ]; then printf 'present (%s line(s))' "$(grep -c . "$f" 2>/dev/null || true)"; else printf 'absent'; fi
}

watchdog() {
  local end reason code last now
  log "watchdog started: lifetime $LIFETIME s"
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
      *)       log "SELF_DELETE FAILED (HTTP ${code:-none}); retrying in $RETRY s" ;;
    esac
    sleep "$RETRY"
  done
}

watchdog &
WD_PID=$!
sleep 0.2

fail() {
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
# scale-invariance interp pod sshd: one key, local forwarding to the control API only
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

# --- 3b. Interp code: decode, check, unpack ----------------------------------
CODE_TGZ="$RUN/interp_code.tgz"
printf '%s' "${SI_INTERP_B64:-}" | base64 -d > "$CODE_TGZ" 2>/dev/null || fail "SI_INTERP_B64 is not base64"
[ "$(sha256_of "$CODE_TGZ")" = "${SI_INTERP_SHA256:-}" ] || fail "interp code does not match SI_INTERP_SHA256"
mkdir -p "$CODE" || fail "cannot create $CODE"
tar -xzf "$CODE_TGZ" -C "$CODE" 2>> "$LOGDIR/interp.log" || fail "cannot unpack the interp code"
[ -f "$CODE/scripts/interp/run_interp.py" ] || fail "interp code missing run_interp.py"
log "INTERP_CODE_OK sha=$(sha256_of "$CODE_TGZ")"

# --- 4. Run the interp control API (text-only load, hooks) -------------------
log "python: $(python3 --version 2>&1 | head -1 || echo unknown)"
log "INTERP_STARTING"
( cd "$CODE" && exec env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
    python3 "$CODE/scripts/interp/run_interp.py" \
      --model-dir "$DIR" --manifest "$MAN" --model-name "$NAME" --port 8000 --vllm-ref ) 2>&1 |
  tee -a "$LOGDIR/interp.log" | awk '{ print "[interp-out] " $0; fflush() }' || true
log "INTERP_EXITED; pod stays up for diagnosis until the watchdog deletes it"
wait "$WD_PID"
