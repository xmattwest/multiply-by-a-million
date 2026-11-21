#!/bin/bash
# launch_pod.sh: start one scale-invariance pod, use it, and always delete it.
#
# THIS SPENDS MONEY when run with --confirm-cost. Without it, it's a free
# dry run: it checks everything, prints the plan and the worst-case cost,
# and stops before creating anything.
#
# WHAT IT DOES
#   1. Free checks: write key in the Keychain; pins, manifest, project SSH
#      key; the GPU's live Secure Cloud price; no other "si-" pod already on
#      the account (C13); worst-case cost of this launch plus everything in
#      the budget ledger (logs/budget.tsv) stays within $15.
#      Worst case = (lifetime + image-download allowance) x $/hr + disk.
#   2. --confirm-cost must repeat that worst-case number exactly, so nothing
#      launches without the number having been seen.
#   3. Creates the pod ONCE (never retried). If the create call fails in a
#      way that might still have made a pod, it looks the pod up by its
#      unique name and deletes it.
#   4. From here on, a trap deletes the pod on every exit: normal end,
#      error, Ctrl-C. The launcher also enforces the lifetime itself (the
#      laptop-side watchdog), on top of the pod's own timer.
#   5. Waits for the pod, reads the SSH host key the pod prints in its log,
#      pins it, and runs the session:
#        probe: connection 03, then the self-delete test (asks the pod to
#               delete itself and watches whether it disappears).
#        serve: waits for the weights check, opens the SSH tunnel
#               (laptop 127.0.0.1:18000 -> pod 127.0.0.1:8000), waits for
#               vLLM, runs connections 03 and 04, then holds the session
#               until logs/pod_session.stop appears or the lifetime ends.
#   6. Teardown: closes the tunnel, deletes the pod, confirms it's gone,
#      counts our pods left (should be 0), records the cost estimate.
#
# SECRETS: the write key comes only from the Keychain item
# scale-invariance-runpod-write (no environment override), lives in an
# unexported variable, and reaches curl on stdin. Terminal output is
# screen-safe (no key, pod ID or IP); details go to logs/launch_<name>.details.
#
# USAGE
#   bash scripts/launch_pod.sh --mode probe --lifetime-min 15
#   bash scripts/launch_pod.sh --mode serve --model qwen3.8-27b-fp8 --lifetime-min 90
#   ... add --confirm-cost X.XX (the printed worst case) to actually launch.
#   Other options: --gpu-id "<catalog id>", --disk-gb N, --image-by-tag
#   --slot N (0-9; default $SI_SLOT or 0): run several pods at once, one per
#   slot. Each slot has its own session file, stop file and tunnel port
#   (18000+N). C13 then means: no si- pod that isn't owned by a live launch.
#   --interp-plugins all|none|a,b,... (required with --mode interp): which
#   scripts/interp/tasks_<name>.py plug-ins ship (scripts/lib/pack_interp.sh).
#   The code travels in the create request, which must stay under 100 KB;
#   the dry run builds that request and checks its size.
#   --dc us|any|ID,ID,... (default us): which data centers may host the pod.
#   "us" = every North American data center that lists the GPU in the live
#   catalog right now (Hugging Face downloads were ~3x faster there than in
#   EUR-IS-2 on 10-14); "any" = let the scheduler choose; or an explicit list.
#   Sent as dataCenterIds. Each serve/interp launch appends its download
#   speed and data center to logs/download_speed.tsv.
#
# Written for macOS /bin/bash 3.2.

set +x +v +a
set -uo pipefail
umask 077
export LC_ALL=C
unset SSLKEYLOGFILE

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
. "$HERE/lib/pod_ssh.sh"
. "$HERE/lib/pack_interp.sh"

readonly API="https://api.runpod.io/v2"
readonly PULL_ALLOWANCE_MIN=20
readonly KEYCHAIN_SERVICE="scale-invariance-runpod-write"
LEDGER="$ROOT/logs/budget.tsv"

MODE=""; SLOT="${SI_SLOT:-0}"; MODEL=""; LIFETIME_MIN=""; GPU_ID="NVIDIA RTX PRO 6000 Blackwell Server Edition"
DISK_GB=""; CONFIRM=""; BY_TAG=0; INTERP_PLUGINS=""; DC="us"
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift ;;
    --dc) DC="${2:-}"; shift ;;
    --model) MODEL="${2:-}"; shift ;;
    --lifetime-min) LIFETIME_MIN="${2:-}"; shift ;;
    --gpu-id) GPU_ID="${2:-}"; shift ;;
    --disk-gb) DISK_GB="${2:-}"; shift ;;
    --confirm-cost) CONFIRM="${2:-}"; shift ;;
    --image-by-tag) BY_TAG=1 ;;
    --slot) SLOT="${2:-}"; shift ;;
    --interp-plugins) INTERP_PLUGINS="${2:-}"; shift ;;
    -h|--help) sed -n '2,51p' "$0"; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '%s\n' "$*"; }
die() { local c="$1"; shift; printf 'STOP: %s\n' "$*" >&2; exit "$c"; }
for t in curl jq ssh ssh-keygen python3 security perl; do
  command -v "$t" >/dev/null 2>&1 || die 2 "$t not found"
done
case "$MODE" in probe|serve|interp) ;; *) die 2 "--mode probe|serve|interp" ;; esac
if [ "$MODE" = interp ]; then
  [ -n "$INTERP_PLUGINS" ] || die 2 "--mode interp needs --interp-plugins all|none|<name,name,...>"
else
  [ -z "$INTERP_PLUGINS" ] || die 2 "--interp-plugins is only for --mode interp"
fi
[[ "$LIFETIME_MIN" =~ ^[0-9]+$ ]] && [ "$LIFETIME_MIN" -ge 5 ] && [ "$LIFETIME_MIN" -le 360 ] ||
  die 2 "--lifetime-min must be 5..360"
si_set_slot "$SLOT" 2>/dev/null || die 2 "--slot must be 0..9"
[[ "$DC" =~ ^(us|any|[A-Z0-9-]+(,[A-Z0-9-]+)*)$ ]] || die 2 "--dc must be us, any, or a comma-separated list of data center IDs"
STOPFILE="$SI_STOPFILE"
mkdir -p "$ROOT/logs"
port_busy() { python3 -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 1)" "$1"; }
live_claims() {   # "<name> <worst>" for every other live launch (its launcher process still running)
  local f pid name worst
  for f in "$ROOT"/logs/pod_slot.*; do
    [ -f "$f" ] || continue
    [ "$f" = "$SI_SLOTFILE" ] && [ "${NAME:-}" != "" ] && continue
    read -r pid name worst < "$f" || continue
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && printf '%s %s\n' "$name" "$worst"
  done
  for f in "$ROOT"/logs/pod_session.env "$ROOT"/logs/pod_session.[0-9].env; do   # launches without a claim file
    [ -f "$f" ] || continue
    local s=0; case "$f" in *.[0-9].env) s="${f%.env}"; s="${s##*.}" ;; esac
    [ "$s" = "$SLOT" ] && continue
    [ -f "$ROOT/logs/pod_slot.$s" ] && continue
    port_busy $((18000 + s)) && name="$(sed -n 's/^POD_NAME=//p' "$f")" && [ -n "$name" ] && printf '%s 0\n' "$name"
  done
}
BUDGET_USD="$(jq -r '.total_usd' "$ROOT/config/budget.json" 2>/dev/null)"
[[ "$BUDGET_USD" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die 2 "config/budget.json: no valid total_usd"
BUDGET_USD="$(printf '%.2f' "$BUDGET_USD")"

# --- 1. Free checks ---------------------------------------------------------
RP_KEY="$(security find-generic-password -s "$KEYCHAIN_SERVICE" -a "${USER:-$(id -un)}" -w 2>/dev/null)" ||
  die 2 "no write key in the Keychain (service $KEYCHAIN_SERVICE)"
[ -n "$RP_KEY" ] || die 2 "write key is empty"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT     # replaced by the teardown trap once a pod may exist
rp() {   # rp METHOD PATH OUTFILE [BODYFILE] -> prints the HTTP status (000 = no reply)
  local m="$1" p="$2" out="$3" body="${4:-}" code
  if [ -n "$body" ]; then
    code="$(printf 'Authorization: Bearer %s\n' "$RP_KEY" | curl -q -sS --proto '=https' --tlsv1.2 \
      --connect-timeout 10 --max-time 60 -o "$out" -w '%{http_code}' -X "$m" -H @- \
      -H 'Content-Type: application/json' --data-binary @"$body" "$API$p" 2>/dev/null)"
  else
    code="$(printf 'Authorization: Bearer %s\n' "$RP_KEY" | curl -q -sS --proto '=https' --tlsv1.2 \
      --connect-timeout 10 --max-time 60 -o "$out" -w '%{http_code}' -X "$m" -H @- "$API$p" 2>/dev/null)"
  fi
  printf '%s' "${code:-000}"
}
rp_get() {   # GETs are safe to retry
  local code i
  for i in 1 2 3; do
    code="$(rp GET "$1" "$2")"
    case "$code" in 000|429|5??) sleep 3 ;; *) break ;; esac
  done
  printf '%s' "$code"
}
our_pods() {  # prints "<count>": si- pods not owned by another live launch (C13)
  local code known
  code="$(rp_get /pods "$T/pods.json")"
  [ "$code" = 200 ] || { printf 'ERR'; return; }
  jq -e '(.pagination.hasNextPage // false) | not' "$T/pods.json" >/dev/null || { printf 'ERR'; return; }
  known="$(live_claims | cut -d' ' -f1 | jq -R . | jq -s .)"
  jq --argjson k "$known" '[(.pods // .)[] | select((.name // "") | startswith("si-")) | select(.name as $n | $k | index($n) | not)] | length' "$T/pods.json"
}
other_live() { live_claims | grep -c . ; }

PINS="$ROOT/config/pins.json"
DIGEST="$(jq -r '.vllm_image.index_digest' "$PINS")"
[[ "$DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die 2 "pins.json: bad image digest"
IMAGE="vllm/vllm-openai@$DIGEST"
[ "$BY_TAG" = 1 ] && IMAGE="$(jq -r '.vllm_image.tag' "$PINS")"
PUB="$SI_SSH_KEY.pub"
[ -f "$SI_SSH_KEY" ] && [ -f "$PUB" ] || die 2 "project SSH key missing ($SI_SSH_KEY)"
PUBKEY="$(cat "$PUB")"
FPR="$(ssh-keygen -lf "$PUB" | cut -d' ' -f2)"
BOOT="$HERE/pod/boot.sh"
[ "$MODE" = interp ] && BOOT="$HERE/pod/boot_interp.sh"
bash -n "$BOOT" || die 2 "${BOOT##*/} has a syntax error"

if [ "$MODE" = serve ] || [ "$MODE" = interp ]; then
  REPO="$(jq -r --arg m "$MODEL" '.models[] | select(.name == $m) | .repo' "$PINS")"
  REV="$(jq -r --arg m "$MODEL" '.models[] | select(.name == $m) | .revision' "$PINS")"
  [ -n "$REPO" ] && [ -n "$REV" ] || die 2 "--model must be one of: $(jq -r '[.models[].name] | join(", ")' "$PINS")"
  MAN="$ROOT/config/manifests/$MODEL.tsv"
  [ -f "$MAN" ] || die 2 "no manifest for $MODEL"
  WEIGHT_BYTES="$(awk -F'\t' '!/^#/ && NF>=3 {s+=$2} END {printf "%.0f", s}' "$MAN")"
  [ -n "$DISK_GB" ] || DISK_GB="$(awk -v b="$WEIGHT_BYTES" 'BEGIN {printf "%d", b/1e9*1.15 + 40}')"
  TAG="$(printf '%s' "$MODEL" | tr -cd 'a-z0-9' | cut -c1-12)"
else
  MODEL="none"; [ -n "$DISK_GB" ] || DISK_GB=30; TAG="probe"
fi
[[ "$DISK_GB" =~ ^[0-9]+$ ]] && [ "$DISK_GB" -ge 10 ] && [ "$DISK_GB" -le 400 ] || die 2 "--disk-gb must be 10..400"

build_create() {   # build_create NAME -> $T/create.json (interp: packs the code once, into $T/interp_code.tgz)
  ENVJSON="$(jq -n --arg mode "$MODE" --arg life "$((LIFETIME_MIN * 60))" --arg pk "$PUBKEY" --arg fpr "$FPR" \
    '{SI_MODE: $mode, SI_LIFETIME_S: $life, PUBLIC_KEY: $pk, SI_SSH_KEY_FPR: $fpr}')"
  if [ "$MODE" = serve ] || [ "$MODE" = interp ]; then
    ENVJSON="$(printf '%s' "$ENVJSON" | jq --arg n "$MODEL" --arg r "$REPO" --arg v "$REV" \
      --arg b "$(base64 < "$MAN" | tr -d '\n')" --arg s "$(shasum -a 256 "$MAN" | cut -d' ' -f1)" \
      '. + {SI_MODEL_NAME: $n, SI_MODEL_REPO: $r, SI_MODEL_REV: $v, SI_MANIFEST_B64: $b, SI_MANIFEST_SHA256: $s}')"
  fi
  if [ "$MODE" = interp ]; then
    # Ship the pod-side interp code (and the files it imports) in the launch:
    # a reproducible tar.gz, hash-checked on the pod. No repo or network needed.
    CODE_TGZ="$T/interp_code.tgz"
    [ -s "$CODE_TGZ" ] || pack_interp_code "$ROOT" "$CODE_TGZ" "$INTERP_PLUGINS" || die 2 "could not pack the interp code"
    [ "$(wc -c < "$CODE_TGZ")" -lt 400000 ] || die 2 "interp code tar over 400 KB"
    ENVJSON="$(printf '%s' "$ENVJSON" | jq --arg b "$(base64 < "$CODE_TGZ" | tr -d '\n')" \
      --arg s "$(shasum -a 256 "$CODE_TGZ" | cut -d' ' -f1)" \
      '. + {SI_INTERP_B64: $b, SI_INTERP_SHA256: $s}')"
  fi
  jq -n --arg name "$1" --arg image "$IMAGE" --arg gpu "$GPU_ID" --argjson disk "$DISK_GB" \
    --rawfile boot "$BOOT" --argjson env "$ENVJSON" --argjson dcs "$DCJSON" \
    '{name: $name, image: $image, cloud: "SECURE",
      gpu: {id: $gpu, count: 1, minCudaVersion: "13.0"},
      ports: ["22/tcp"], disk: $disk, startSsh: false, startJupyter: false,
      entrypoint: ["/bin/bash", "-c"], cmd: [$boot], env: $env}
     + (if ($dcs | length) > 0 then {dataCenterIds: $dcs} else {} end)' > "$T/create.json"
}
# Data centers (--dc). Free GET; resolved now so the dry run shows the list and the request carries it.
DCJSON="[]"; DC_SHOW="any (scheduler chooses)"
if [ "$DC" != any ]; then
  code="$(rp_get "/catalog/datacenters?include=GPU_AVAILABILITY" "$T/dcs.json")"
  [ "$code" = 200 ] || die 3 "data center catalog: HTTP $code"
  STOCKED="$(jq -c --arg g "$GPU_ID" '[(.dataCenters // .)[] | select([(.gpuAvailability // [])[] | .id] | index($g)) | {id, region}]' "$T/dcs.json")"
  if [ "$DC" = us ]; then
    DCJSON="$(printf '%s' "$STOCKED" | jq -c '[.[] | select(.region == "NORTH_AMERICA") | .id] | sort')"
  else
    DCJSON="$(printf '%s' "$DC" | jq -R -c 'split(",")')"
    MISSING="$(jq -n -r --argjson want "$DCJSON" --argjson have "$STOCKED" '[$want[] | select(. as $w | [$have[].id] | index($w) | not)] | join(", ")')"
    [ -z "$MISSING" ] || say "  note: the catalog lists no $GPU_ID in: $MISSING"
  fi
  [ "$(printf '%s' "$DCJSON" | jq 'length')" -gt 0 ] || die 3 "no data center in --dc $DC lists $GPU_ID right now; try --dc any"
  DC_SHOW="$(printf '%s' "$DCJSON" | jq -r 'join(", ")') (--dc $DC)"
fi
# Build the request now (same-length stand-in name) so a dry run already knows it fits.
build_create "si-$TAG-$(date -u +%Y%m%dT%H%M%SZ)-0000"
CREATE_BYTES="$(wc -c < "$T/create.json" | tr -d ' ')"
[ "$CREATE_BYTES" -lt 100000 ] || die 2 "create request would be $CREATE_BYTES bytes (limit 100000); ship fewer plug-ins (--interp-plugins)"

code="$(rp_get "/catalog/gpus?include=AVAILABILITY&product=POD" "$T/gpus.json")"
[ "$code" = 200 ] || die 3 "GPU catalog: HTTP $code"
PRICE="$(jq -r --arg g "$GPU_ID" '[(if type=="array" then . else (.gpus // .data // .items) end)[] | select(.id == $g)][0].price.secure // empty' "$T/gpus.json")"
STOCK="$(jq -r --arg g "$GPU_ID" '[(if type=="array" then . else (.gpus // .data // .items) end)[] | select(.id == $g)][0].availability // "?"' "$T/gpus.json")"
[[ "$PRICE" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v p="$PRICE" 'BEGIN {exit !(p > 0)}' || die 3 "no Secure Cloud price for \"$GPU_ID\""

WORST="$(awk -v p="$PRICE" -v l="$LIFETIME_MIN" -v a="$PULL_ALLOWANCE_MIN" -v d="$DISK_GB" \
  'BEGIN { h = (l + a) / 60; c = p * h + d * 0.10 / 730 * h; printf "%.2f", int(c * 100 + 0.999999) / 100 }')"
COMMITTED="0.00"
if [ -f "$LEDGER" ]; then
  COMMITTED="$(awk -F'\t' 'NR > 1 { s += ($5 == "closed" ? $4 : $3) } END { printf "%.2f", s + 0 }' "$LEDGER")"
fi
RESERVED="$(live_claims | awk '{s += $2} END {printf "%.2f", s + 0}')"   # other live launches, not yet in the ledger
COMMITTED="$(awk -v a="$COMMITTED" -v b="$RESERVED" 'BEGIN {printf "%.2f", a + b}')"
AFTER="$(awk -v a="$COMMITTED" -v b="$WORST" 'BEGIN {printf "%.2f", a + b}')"

N="$(our_pods)"
[ "$N" = ERR ] && die 3 "could not list pods; not launching"

say "Launch plan ($MODE)"
say "  model:          $MODEL"
say "  GPU:            $GPU_ID, Secure Cloud, \$$PRICE/hr (stock: $STOCK)"
say "  data centers:   $DC_SHOW"
say "  image:          $IMAGE"
say "  lifetime:       $LIFETIME_MIN min (pod deletes itself; the launcher also enforces it)"
say "  container disk: $DISK_GB GB"
[ "$MODE" = interp ] && say "  interp code:    plug-ins $INTERP_PLUGINS; create request $CREATE_BYTES bytes (limit 100000)"
say "  worst case:     \$$WORST  = (lifetime + $PULL_ALLOWANCE_MIN min image download) x price + disk"
say "  budget:         \$$COMMITTED used or reserved + \$$WORST = \$$AFTER of \$$BUDGET_USD"
say "  slot:           $SLOT (tunnel port $SI_TUNNEL_PORT; other live launches: $(other_live))"
say "  our pods now:   $N not owned by a live launch"
[ "$N" = 0 ] || die 4 "an si- pod exists that no live launch owns; delete it before launching another (C13)"
if [ -f "$SI_SLOTFILE" ] && read -r _pid _rest < "$SI_SLOTFILE" && kill -0 "$_pid" 2>/dev/null; then
  die 4 "slot $SLOT is in use by another launch; pick another --slot"
fi
port_busy "$SI_TUNNEL_PORT" && die 4 "port $SI_TUNNEL_PORT is in use (slot $SLOT busy?); pick another --slot"
awk -v a="$AFTER" -v b="$BUDGET_USD" 'BEGIN {exit !(a <= b)}' ||
  die 4 "this launch could take the project past \$$BUDGET_USD; ask first"
if [ "$CONFIRM" != "$WORST" ]; then
  say ""
  say "Dry run: nothing created. To launch, add:  --confirm-cost $WORST"
  rm -rf "$T"
  exit 0
fi

# --- 3. Create once; from here on, always tear down -------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RAND="$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')"
NAME="si-$TAG-$STAMP-$RAND"
DETAILS="$ROOT/logs/launch_$NAME.details"
BOOTLOG="$ROOT/logs/launch_$NAME.bootlog"
: > "$DETAILS"
detail() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" >> "$DETAILS"; }

POD_ID=""; TUNNEL_PID=""; CREATE_SENT=0; TORN=0; CREATE_T=0; GONE=0
DEADLINE=0; SSH_HOST=""; SSH_PORT=""; KNOWN_HOSTS=""; COPIED=0; GONE_EARLY=0

copy_pod_logs() {   # the pod's /var/log/si (boot, sshd, vLLM logs) -> logs/pod_<name>/
  # RunPod's log API keeps only the last 5,000 lines; vLLM's request log
  # (the record of the settings it actually used) can be longer. Egress is free.
  [ "$COPIED" = 1 ] && return 0
  [ -n "$SSH_HOST" ] && [ -n "$KNOWN_HOSTS" ] && [ -f "$KNOWN_HOSTS" ] || return 0
  COPIED=1
  local dst="$ROOT/logs/pod_$NAME"
  mkdir -p "$dst"
  if si_ssh 'tar -C /var/log/si -czf - .' > "$dst/var_log_si.tgz" 2>/dev/null &&
     tar -xzf "$dst/var_log_si.tgz" -C "$dst" 2>/dev/null; then
    say "  pod logs copied to logs/pod_$NAME/ ($(ls "$dst" | grep -v '\.tgz$' | tr '\n' ' '))"
  else
    say "  could not copy the pod's logs over SSH (RunPod's log tail is in logs/launch_$NAME.bootlog)"
  fi
}

teardown() {
  [ "$TORN" = 1 ] && return
  TORN=1
  trap '' INT TERM HUP
  say ""
  say "Teardown"
  [ "$GONE_EARLY" = 1 ] || copy_pod_logs
  if [ -n "$TUNNEL_PID" ]; then kill "$TUNNEL_PID" 2>/dev/null; wait "$TUNNEL_PID" 2>/dev/null; say "  tunnel closed"; fi
  if [ -z "$POD_ID" ] && [ "$CREATE_SENT" = 1 ]; then
    # The create call failed or its reply was lost. A pod may still exist,
    # and may take a moment to show up in the list: look for ~1 minute.
    local j
    for j in 1 2 3 4 5 6; do
      if [ "$(rp_get /pods "$T/pods.json")" = 200 ]; then
        POD_ID="$(jq -r --arg n "$NAME" '[(.pods // .)[] | select(.name == $n)][0].id // empty' "$T/pods.json")"
      fi
      [ -n "$POD_ID" ] && { say "  found a pod by this launch's name; deleting it"; break; }
      sleep "${SI_LOOKUP_SLEEP:-10}"
    done
  fi
  if [ -n "$POD_ID" ]; then
    local i code
    for i in 1 2 3 4 5 6; do
      code="$(rp DELETE "/pods/$POD_ID" "$T/del.json")"
      detail "DELETE try $i: HTTP $code"
      code="$(rp_get "/pods/$POD_ID" "$T/pod.json")"
      if [ "$code" = 404 ] || [ "$(jq -r '.status // empty' "$T/pod.json" 2>/dev/null)" = TERMINATED ]; then GONE=1; break; fi
      sleep $((i * ${SI_TEARDOWN_SLEEP:-5}))
    done
  elif [ "$CREATE_SENT" = 1 ]; then
    GONE=1   # create sent but no pod by that name exists
  fi
  local left; left="$(our_pods)"
  if [ "$CREATE_SENT" = 1 ]; then
    local now charged status
    now="$(date +%s)"
    charged="$(awk -v p="$PRICE" -v s="$(( now - CREATE_T ))" 'BEGIN {printf "%.2f", int(p * s / 3600 * 100 + 0.999999) / 100}')"
    status=closed; [ "$GONE" = 1 ] || status=UNCONFIRMED
    [ -f "$LEDGER" ] || printf 'utc\tpod_name\tworst_usd\testimate_usd\tstatus\n' > "$LEDGER"
    printf '%s\t%s\t%s\t%s\t%s\n' "$STAMP" "$NAME" "$WORST" "$charged" "$status" >> "$LEDGER"
    say "  pod up for about $(( (now - CREATE_T) / 60 )) min; estimated cost \$$charged (billing confirms later)"
  fi
  rm -f "$SI_SESSION" "$STOPFILE" "$SI_SLOTFILE"
  local others; others="$(other_live)"
  rm -rf "$T"
  if [ "$GONE" = 1 ] && [ "$left" = 0 ]; then
    if [ "$others" = 0 ]; then say "  ✓ pod deleted; pods of ours left on the account: 0"
    else say "  ✓ pod deleted; pods of ours left on the account: 0 unaccounted ($others in other live launches)"; fi
  else
    say "  !!! COULD NOT CONFIRM THE POD IS GONE (our pods left: $left)."
    say "  !!! Delete \"$NAME\" in the RunPod console now. It may be billing."
    exit 9
  fi
}
trap teardown EXIT
trap 'say "(interrupted)"; exit 130' INT TERM HUP
printf '%s %s %s\n' "$$" "$NAME" "$WORST" > "$SI_SLOTFILE"     # claim the slot (removed at teardown)

build_create "$NAME"
[ "$(wc -c < "$T/create.json")" -lt 100000 ] || die 2 "create request over 100 KB"

say ""
say "Creating pod $NAME (once; never retried)"
CREATE_SENT=1
CREATE_T="$(date +%s)"
DEADLINE=$(( CREATE_T + LIFETIME_MIN * 60 ))
detail "data centers requested: $DCJSON"
code="$(rp POST /pods "$T/created.json" "$T/create.json")"
detail "POST /pods: HTTP $code"
case "$code" in
  201|200)
    POD_ID="$(jq -r '.id // empty' "$T/created.json")"
    [ -n "$POD_ID" ] || die 5 "created, but no pod ID in the reply" ;;
  *)
    detail "create reply: $(head -c 2000 "$T/created.json" 2>/dev/null | tr -cd '[:print:]')"
    say "  create failed: HTTP $code: $(jq -r '.detail // .title // empty' "$T/created.json" 2>/dev/null | tr -cd '[:print:]' | head -c 300)"
    die 5 "not retrying (C13); teardown checks for a pod by this name" ;;
esac
detail "pod id $POD_ID"
say "  created; billing has started"

left_min() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
past_deadline() { [ "$(date +%s)" -ge "$DEADLINE" ]; }

# --- 4. Wait for RUNNING and a public SSH port ------------------------------
say "Waiting for the pod (image download, then boot)"
last=""
while :; do
  past_deadline && die 6 "lifetime reached while waiting for the pod"
  code="$(rp_get "/pods/$POD_ID" "$T/pod.json")"
  if [ "$code" = 200 ]; then
    st="$(jq -r '.status // "?"' "$T/pod.json")"
    [ "$st" != "$last" ] && say "  status: $st ($(( ($(date +%s) - CREATE_T) / 60 )) min)" && last="$st"
    case "$st" in EXITED|ERROR|TERMINATED) die 6 "pod is $st" ;; esac
    hp="$(jq -r '(.ssh.direct // empty | "\(.host) \(.port)"),
                 ((.runtime.ports // [])[] | select(.private == 22) | "\(.ip) \(.public)")' "$T/pod.json" 2>/dev/null | head -1)"
    if [ "$st" = RUNNING ] && [ -n "$hp" ]; then SSH_HOST="${hp% *}"; SSH_PORT="${hp#* }"; break; fi
  fi
  sleep 10
done
[[ "$SSH_PORT" =~ ^[0-9]+$ ]] && [[ "$SSH_HOST" =~ ^[0-9A-Za-z.:-]+$ ]] || die 6 "odd SSH address from the API"
DC_ID="$(jq -r '.machine.dataCenterId // .dataCenterId // "?"' "$T/pod.json")"
detail "ssh $SSH_HOST:$SSH_PORT; dc $DC_ID"
if [ "$DCJSON" != "[]" ] && ! printf '%s' "$DCJSON" | jq -e --arg d "$DC_ID" 'index($d)' >/dev/null; then
  say "  note: placed in $DC_ID, which is not in the requested list"
fi

fetch_log() {   # the pod's container log so far -> $BOOTLOG (SSE stream, cut off after 10 s)
  printf 'Authorization: Bearer %s\n' "$RP_KEY" | curl -q -sS --proto '=https' --tlsv1.2 \
    --connect-timeout 10 --max-time 10 -o "$T/sse.txt" -H @- -H 'Accept: text/event-stream' \
    "$API/pods/$POD_ID/logs?source=container&tail=5000" 2>/dev/null
  grep '^data:' "$T/sse.txt" 2>/dev/null | sed 's/^data: *//' | jq -r '.line // empty' 2>/dev/null > "$T/log.txt"
  [ -s "$T/log.txt" ] && cp "$T/log.txt" "$BOOTLOG"
  return 0
}
marker() { grep -F "[si-boot] $1" "$BOOTLOG" 2>/dev/null | tail -1; }

say "Waiting for the pod's SSH host key in its log"
HK=""
while :; do
  past_deadline && die 6 "lifetime reached while waiting for SSH"
  fetch_log
  f="$(marker "BOOT_FAILED")"; [ -n "$f" ] && die 6 "boot failed: ${f#*BOOT_FAILED }"
  HK="$(marker "SSH_HOST_KEY ssh-ed25519 " | sed 's/.*SSH_HOST_KEY //')"
  [ -n "$HK" ] && [ -n "$(marker SSH_READY)" ] && break
  sleep 10
done
[[ "$HK" =~ ^ssh-ed25519\ [A-Za-z0-9+/]+=*$ ]] || die 6 "host key line malformed"
KNOWN_HOSTS="$ROOT/logs/known_hosts_$NAME"
printf '[%s]:%s %s\n' "$SSH_HOST" "$SSH_PORT" "$HK" > "$KNOWN_HOSTS"
{ printf 'POD_ID=%q\nPOD_NAME=%q\nMODE=%q\nMODEL=%q\n' "$POD_ID" "$NAME" "$MODE" "$MODEL"
  printf 'SSH_HOST=%q\nSSH_PORT=%q\nKNOWN_HOSTS=%q\n' "$SSH_HOST" "$SSH_PORT" "$KNOWN_HOSTS"; } > "$SI_SESSION"
say "  host key pinned ($(printf '%s\n' "$HK" > "$T/hk.pub"; ssh-keygen -lf "$T/hk.pub" | cut -d' ' -f2))"

# --- 5. Session ---------------------------------------------------------------
if [ "$MODE" = probe ]; then
  sleep 5   # sshd needs a moment after printing SSH_READY
  bash "$HERE/connections/03_pod.sh" || say "  (connection 03 failed; continuing to the self-delete test)"
  copy_pod_logs
  say ""
  say "Self-delete test: asking the pod to delete itself with its own key"
  si_ssh 'echo laptop-asked > /run/si/delete_now' || say "  could not reach the pod over SSH to ask"
  t0="$(date +%s)"; RESULT="failed"
  while [ $(( $(date +%s) - t0 )) -lt 180 ]; do
    fetch_log
    code="$(rp_get "/pods/$POD_ID" "$T/pod.json")"
    if [ "$code" = 404 ] || [ "$(jq -r '.status // empty' "$T/pod.json" 2>/dev/null)" = TERMINATED ]; then
      RESULT="passed"; GONE_EARLY=1; break
    fi
    sleep 5
  done
  grep -F '[si-boot] SELF_DELETE' "$BOOTLOG" 2>/dev/null | sed 's/^/  pod log: /' | tail -5
  if [ "$RESULT" = passed ]; then
    say "  ✓ SELF_DELETE_TEST passed: the pod deleted itself in $(( $(date +%s) - t0 )) s"
  else
    say "  ✗ SELF_DELETE_TEST failed: the pod did not delete itself within 3 min"
    say "    -> use the laptop-side watchdog (this launcher) plus short lifetimes"
  fi
  exit 0
fi

say "Waiting for the model files to download and pass their check"
DL_T0="$(date +%s)"
while :; do
  past_deadline && die 6 "lifetime reached while downloading"
  fetch_log
  f="$(marker "BOOT_FAILED")"; [ -n "$f" ] && die 6 "boot failed: ${f#*BOOT_FAILED }"
  w="$(marker "WEIGHTS_OK")"; [ -n "$w" ] && { say "  ${w#*\] }"; break; }
  sleep 20
done
DL_S=$(( $(date +%s) - DL_T0 )); [ "$DL_S" -gt 0 ] || DL_S=1
DL_MBS="$(awk -v b="$WEIGHT_BYTES" -v s="$DL_S" 'BEGIN {printf "%.0f", b / 1e6 / s}')"
say "  download: $(awk -v b="$WEIGHT_BYTES" 'BEGIN {printf "%.1f", b/1e9}') GB in $(( DL_S / 60 )) min $(( DL_S % 60 )) s (~$DL_MBS MB/s incl. the check; data center $DC_ID)"
SPEEDLOG="$ROOT/logs/download_speed.tsv"
[ -f "$SPEEDLOG" ] || printf 'utc\tpod_name\tmodel\tdata_center\tbytes\tseconds\tmb_per_s\n' > "$SPEEDLOG"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" "$NAME" "$MODEL" "$DC_ID" "$WEIGHT_BYTES" "$DL_S" "$DL_MBS" >> "$SPEEDLOG"
if [ "$MODE" = interp ]; then
  say "Waiting for the interp code to unpack"
  while :; do
    past_deadline && die 6 "lifetime reached while unpacking the interp code"
    fetch_log
    f="$(marker "BOOT_FAILED")"; [ -n "$f" ] && die 6 "boot failed: ${f#*BOOT_FAILED }"
    c="$(marker "INTERP_CODE_OK")"; [ -n "$c" ] && { say "  ${c#*\] }"; break; }
    sleep 10
  done
fi
si_ssh_opts
if [ "$MODE" = interp ] && port_busy "$SI_TUNNEL_PORT"; then
  die 6 "port $SI_TUNNEL_PORT was taken by another program on this laptop while the pod booted (a preview on slot $SLOT?); relaunch on a free --slot"
fi
perl -e 'alarm shift; exec @ARGV' $(( DEADLINE - $(date +%s) )) \
  ssh "${SI_SSH_OPTS[@]}" -N -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:$SI_TUNNEL_PORT:127.0.0.1:8000" "root@$SSH_HOST" &
TUNNEL_PID=$!
say "Tunnel open: laptop 127.0.0.1:$SI_TUNNEL_PORT -> pod 127.0.0.1:8000"
if [ "$MODE" = interp ]; then WHAT="the interp model (text-only) to load"; EXITMARK="INTERP_EXITED";
else WHAT="vLLM to load the model"; EXITMARK="VLLM_EXITED"; fi
say "Waiting for $WHAT"
while :; do
  past_deadline && die 6 "lifetime reached while loading"
  kill -0 "$TUNNEL_PID" 2>/dev/null || die 6 "the SSH tunnel closed"
  h="$(python3 -c "import sys; sys.path.insert(0, '$HERE'); import vllm_client as v; print(v.request($SI_TUNNEL_PORT, 'GET', '/health', total_timeout=5)[0])" 2>/dev/null)"
  [ "$h" = 200 ] && break
  fetch_log
  f="$(marker "$EXITMARK")"; [ -n "$f" ] && die 6 "the pod process exited during start-up (see the boot log)"
  sleep 15
done
bash "$HERE/connections/03_pod.sh" || say "  (connection 03 reported a problem)"
if [ "$MODE" = interp ]; then
  fetch_log
  r="$(grep -F '[interp] INTERP_READY' "$BOOTLOG" 2>/dev/null | tail -1)"
  [ -n "$r" ] && say "  ${r#*] }"
  say ""
  say "SESSION_READY: interp control API reachable at http://127.0.0.1:$SI_TUNNEL_PORT through the tunnel."
  say "  Drive it with:  python3 scripts/launch_interp.py --attach --slot $SLOT ..."
else
  bash "$HERE/connections/04_vllm.sh" || say "  (connection 04 reported a problem)"
  fetch_log
  say ""
  say "SESSION_READY: vLLM reachable at http://127.0.0.1:$SI_TUNNEL_PORT through the tunnel."
fi
say "  End it any time:  touch logs/${STOPFILE##*/}   (auto-teardown in $(left_min) min)"
while :; do
  [ -e "$STOPFILE" ] && { say "  stop requested"; break; }
  past_deadline && { say "  lifetime reached"; break; }
  kill -0 "$TUNNEL_PID" 2>/dev/null || { say "  the SSH tunnel closed"; break; }
  sleep 5
done
fetch_log
exit 0
