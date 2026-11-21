#!/bin/bash
# runpod_auth_check.sh: read-only RunPod API auth check (Phase 1).
#
# WHAT IT DOES
#   Sends three read-only GET requests to the RunPod REST API v2:
#     GET /v2/pods             lists your pods (anything that may be billing)
#     GET /v2/network-volumes  lists your network volumes (they bill even when unattached)
#     GET /v2/billing          what you were actually charged, per day, last 7 days
#   Then prints a short summary with no secrets in it. It never creates,
#   starts, stops, or deletes anything, and these calls cost nothing.
#
# ABOUT A POD'S "cost" FIELD, AND BILLING LAG
#   The API reference says `cost` is the current USD/hour and 0 when a pod is
#   stopped (EXITED). On 2026-10-02 the live API returned $1.59 and $0.99/hr
#   for two stopped pods, while billing history showed no compute charges
#   for them. So this script labels `cost` "rate when running", adds it up
#   only for pods that are not stopped, and reports actual daily charges
#   from the billing endpoint. That history can lag (on 2026-10-02 it had
#   no records after 2026-09-27), so a stale history is flagged rather than
#   read as "nothing is billing".
#
# WHERE THE KEY COMES FROM (first match wins)
#   1. $RUNPOD_API_KEY, if set (explicit override; useful off macOS)
#   2. macOS Keychain item: service "scale-invariance-runpod", account $USER
#      Store it once (you will be prompted; nothing echoes or reaches history):
#        security add-generic-password -U -s scale-invariance-runpod \
#          -a "$USER" -l "RunPod API key (scale-invariance)" -w
#
# HOW THE KEY IS PROTECTED
#   - Never printed, never written to a file, never logged.
#   - Never placed on a command line, where any local process could read it
#     with `ps`. It reaches curl on stdin as a header line (curl -H @-).
#   - Tracing (bash -x) is switched off before the key is read.
#   - ~/.curlrc is ignored (curl -q), so a stray "verbose" or "trace" setting
#     cannot dump request headers.
#   - The API host is hard-coded, so an environment variable cannot redirect
#     the key to a different server.
#   - Pod environment variables are never printed. GET /v2/pods returns their
#     values, and those can hold other secrets (for example HF_TOKEN).
#
# EXIT CODES
#   0 auth OK   1 key rejected (401/403)   2 local setup problem
#   3 network error, rate limit, or unexpected response
#
# USAGE
#   bash scripts/runpod_auth_check.sh           full audit: pods, volumes, billing
#   bash scripts/runpod_auth_check.sh --brief   one call, three screen-safe lines
#       (no key, IDs, or names). Used by scripts/connections/01_runpod.sh.
#
# Written for macOS /bin/bash 3.2: no associative arrays, mapfile, or ${x,,}.

# Before anything else: no tracing, no auto-export. A trace (bash -x, or an
# exported SHELLOPTS) would print the key when it is assigned.
set +x +v +a
set -euo pipefail
umask 077
export LC_ALL=C
unset SSLKEYLOGFILE   # if set, curl would write TLS session secrets to disk

readonly API_BASE="https://api.runpod.io/v2"
readonly KEYCHAIN_SERVICE="${RUNPOD_KEYCHAIN_SERVICE:-scale-invariance-runpod}"
readonly KEYCHAIN_ACCOUNT="${RUNPOD_KEYCHAIN_ACCOUNT:-${USER:-$(id -un)}}"

die() {
  local code="$1"
  shift
  printf 'ERROR: %s\n' "$*" >&2
  exit "$code"
}

MODE="details"
while [ $# -gt 0 ]; do
  case "$1" in
    --brief) MODE="brief" ;;
    -h|--help)
      printf 'Usage: bash %s [--brief]\n' "$0"
      printf '  (no option)  full read-only audit: pods, network volumes, billing\n'
      printf '  --brief      one read-only call; three screen-safe lines\n'
      exit 0 ;;
    *) die 2 "unknown option: $1 (try --help)" ;;
  esac
  shift
done

# Colors only on a real terminal, and never when NO_COLOR is set.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
  C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
  C_OK=""; C_WARN=""; C_BOLD=""; C_OFF=""
fi

# jq helper: format a number as dollars and cents, e.g. 1.5 -> "$1.50".
readonly JQ_USD='def usd: ((. // 0) * 100 | round) as $c
  | "$\($c / 100 | floor).\(if ($c % 100) < 10 then "0" else "" end)\($c % 100)";'

command -v curl >/dev/null 2>&1 || die 2 "curl not found."
command -v jq >/dev/null 2>&1 || die 2 "jq not found (install with: brew install jq)."

# --- 1. Load the key into a shell variable (not exported) -------------------
RUNPOD_KEY=""
KEY_SOURCE=""
if [ -n "${RUNPOD_API_KEY:-}" ]; then
  RUNPOD_KEY="$RUNPOD_API_KEY"
  KEY_SOURCE="RUNPOD_API_KEY environment variable"
elif command -v security >/dev/null 2>&1; then
  if RUNPOD_KEY="$(security find-generic-password -s "$KEYCHAIN_SERVICE" \
      -a "$KEYCHAIN_ACCOUNT" -w 2>/dev/null)"; then
    KEY_SOURCE="macOS Keychain (service=$KEYCHAIN_SERVICE, account=$KEYCHAIN_ACCOUNT)"
  else
    RUNPOD_KEY=""
  fi
fi
# Child processes (curl, jq) do not need the env copy, so drop it.
unset RUNPOD_API_KEY

if [ -z "$RUNPOD_KEY" ]; then
  cat >&2 <<EOF
ERROR: no RunPod API key found.
Looked for \$RUNPOD_API_KEY, then the Keychain item
  service="$KEYCHAIN_SERVICE" account="$KEYCHAIN_ACCOUNT"
To store a key, run this and paste the key when prompted (twice):
  security add-generic-password -U -s "$KEYCHAIN_SERVICE" -a "$KEYCHAIN_ACCOUNT" -l "RunPod API key (scale-invariance)" -w
EOF
  exit 2
fi

# A RunPod key is letters, digits, and _ . - only. Anything else (a space, a
# quote, a newline from a bad paste) is rejected here. That also prevents
# header injection. The key itself is never shown.
case "$RUNPOD_KEY" in
  *[!A-Za-z0-9_.-]*)
    die 2 "the key from $KEY_SOURCE contains unexpected characters (space, quote, or newline?). Re-store it." ;;
esac
if [ "${#RUNPOD_KEY}" -lt 20 ] || [ "${#RUNPOD_KEY}" -gt 200 ]; then
  die 2 "the key from $KEY_SOURCE is ${#RUNPOD_KEY} characters long, which looks wrong (partial paste?). Re-store it."
fi

# --- 2. Read-only API calls ----------------------------------------------------
HTTP_CODE=""
HTTP_TIME=""
HTTP_BODY=""

# api_get PATH: GET $API_BASE$PATH; sets HTTP_CODE, HTTP_TIME (seconds), and
# HTTP_BODY.
# printf is a shell builtin, so the key never appears in any process's argv.
api_get() {
  local path="$1" out status
  if ! out="$(printf 'Authorization: Bearer %s\n' "$RUNPOD_KEY" \
      | curl -q -sS --proto '=https' --tlsv1.2 \
          --connect-timeout 10 --max-time 30 \
          -H @- -H 'Accept: application/json' \
          -w '\n%{http_code} %{time_total}' \
          "${API_BASE}${path}")"; then
    die 3 "network error on GET ${path} (curl's message is above). Nothing was changed."
  fi
  status="${out##*$'\n'}"
  HTTP_CODE="${status%% *}"
  HTTP_TIME="${status#* }"
  HTTP_BODY="${out%$'\n'*}"
}

# Print the API's error title/detail, truncated, with the key scrubbed in case
# a server ever echoes it back.
api_error_detail() {
  local msg
  msg="$(printf '%s' "$HTTP_BODY" \
    | jq -r '"\(.title // "error"): \(.detail // "(no detail)")"' 2>/dev/null)" \
    || msg="(non-JSON response body, ${#HTTP_BODY} bytes)"
  msg="${msg//"$RUNPOD_KEY"/[REDACTED]}"
  printf '%.200s' "$msg"
}

require_ok() {
  local path="$1"
  case "$HTTP_CODE" in
    200) return 0 ;;
    401)
      printf 'AUTH FAILED: GET %s returned 401 (%s).\n' "$path" "$(api_error_detail)" >&2
      printf 'The key from %s was rejected: wrong, disabled, revoked, or partly pasted.\n' "$KEY_SOURCE" >&2
      exit 1 ;;
    403)
      printf 'AUTH FAILED: GET %s returned 403 (%s).\n' "$path" "$(api_error_detail)" >&2
      printf 'The key is valid but not allowed to read this resource. Check its permissions in the console.\n' >&2
      exit 1 ;;
    429)
      die 3 "GET $path was rate limited (429). Wait a minute and retry." ;;
    *)
      die 3 "GET $path returned HTTP ${HTTP_CODE} ($(api_error_detail))." ;;
  esac
}

# --- Brief mode: one call, three lines, nothing account-specific ---------------
# Screen-safe for recordings: no key, no pod or volume IDs, no names.
if [ "$MODE" = "brief" ]; then
  api_get "/pods"
  require_ok "/pods"
  if ! counts="$(printf '%s' "$HTTP_BODY" | jq -r "${JQ_USD}"'
      [(.pods // [])[] | select(.status != "EXITED" and .status != "TERMINATED")] as $live
      | "\($live | length) \([$live[] | .cost // 0] | add // 0 | usd)"' 2>/dev/null)"; then
    die 3 "GET /pods answered 200 but not with the expected JSON."
  fi
  running="${counts%% *}"
  rate="${counts#* }"
  case "$HTTP_TIME" in
    ''|*[!0-9.]*) secs="?" ;;
    *) secs="$(printf '%.2f' "$HTTP_TIME")" ;;
  esac
  case "$KEY_SOURCE" in
    macOS*) where="the macOS Keychain" ;;
    *) where="the RUNPOD_API_KEY environment variable" ;;
  esac
  printf '%sRunPod API%s\n' "$C_BOLD" "$C_OFF"
  printf '  %s✓%s Key loaded from %s (never displayed)\n' "$C_OK" "$C_OFF" "$where"
  printf '  %s✓%s RunPod accepted the key: HTTP 200 in %s s\n' "$C_OK" "$C_OFF" "$secs"
  if [ "$running" = "0" ]; then
    printf '  %s✓%s Pods running right now: 0\n' "$C_OK" "$C_OFF"
  else
    printf '  %s!%s Pods running right now: %s (combined rate %s/hr)\n' \
      "$C_WARN" "$C_OFF" "$running" "$rate"
  fi
  exit 0
fi

printf 'RunPod read-only auth check (REST API v2)\n'
printf 'Key source: %s\n\n' "$KEY_SOURCE"

# Pods. Print only non-secret fields: never env, never SSH details or IPs.
api_get "/pods"
require_ok "/pods"
printf '%s' "$HTTP_BODY" | jq -r '
  (.pods // []) as $p
  | "GET /v2/pods -> 200 OK",
    "Pods: \($p | length)",
    ( $p[]
      | "  - \(.id)  \"\(.name // "-")\"  status=\(.status // "?")"
        + "  gpu=\(.gpu.count // 0)x \(.gpu.id // "none")"
        + "  rate=$\(.cost // 0)/hr when running  dc=\(.dataCenterId // "-")"
        + ( if .mounts.persistent then "  volume_disk=\(.mounts.persistent.size)GB"
            elif ((.mounts.network // []) | length) > 0 then "  network_volume=\(.mounts.network[0].volumeId)"
            else "" end )
        + "  created=\(.createdAt // "-")" ),
    ( [$p[] | select(.status != "EXITED" and .status != "TERMINATED")] as $live
      | "  Pods that can bill compute now (not EXITED/TERMINATED): \($live | length), combined rate $\([$live[] | .cost // 0] | add // 0)/hr" ),
    ( [$p[] | select(.status == "EXITED")] as $stopped
      | [$stopped[] | select(.mounts.persistent)] as $vd
      | if ($stopped | length) == 0 then empty
        elif ($vd | length) > 0
        then "  Stopped (EXITED) pods bill no compute, but volume disks on \($vd | length) of them (\([$vd[] | .mounts.persistent.size] | add) GB) still bill until terminated."
        else "  Stopped (EXITED) pods bill no compute, and none of them holds its own volume disk."
        end ),
    ( if (.pagination.hasNextPage // false)
      then "  WARNING: more pods exist on another page; check the console."
      else empty end )
'
printf '\n'

# Network volumes.
api_get "/network-volumes"
require_ok "/network-volumes"
printf '%s' "$HTTP_BODY" | jq -r '
  (.networkVolumes // []) as $v
  | "GET /v2/network-volumes -> 200 OK",
    "Network volumes: \($v | length)",
    ( $v[] | "  - \(.id)  \"\(.name // "-")\"  \(.size // 0) GB  dc=\(.dataCenter // "-")  type=\(.type // "-")" ),
    ( if ($v | length) > 0
      then "  Total: \([$v[] | .size // 0] | add) GB. Network volumes keep billing whether or not a pod is attached."
      else empty end )
'
printf '\n'

# Actual charges per day: the ground truth for "is anything costing money?".
# A key without billing access gets a warning here, not a failure; auth was
# already proven by the two calls above.
api_get "/billing?bucketSize=day&lastN=7"
if [ "$HTTP_CODE" = "200" ]; then
  printf '%s' "$HTTP_BODY" | jq -r "${JQ_USD}"'
    (.records // []) as $r
    | "GET /v2/billing (per day, last 7 days) -> 200 OK",
      ( if ($r | length) == 0 then "  No charges recorded in the last 7 days."
        else ( $r[]
          | "  \(.startTime[0:10])  \(.totalAmount | usd)"
            + ( [ to_entries[]
                  | select((.key | endswith("Amount")) and .key != "totalAmount" and ((.value // 0) > 0))
                  | "\(.key | sub("Amount$"; ""))=\(.value | usd)" ]
                | if length > 0 then "  (" + join(", ") + ")" else "" end ) )
        end ),
      "  7-day total: \([$r[] | .totalAmount // 0] | add // 0 | usd)",
      ( if ($r | length) == 0
        then "  (Billing history can lag; the console Billing page is authoritative.)"
        else ([$r[] | .startTime[0:10]] | max) as $lastday
          | if (now - ($lastday + "T00:00:00Z" | fromdateiso8601)) > 172800
            then "  Note: the most recent billed day is \($lastday). Either nothing was charged after that, or the history lags; the console Billing page is authoritative."
            else empty end
        end )
  '
else
  printf 'Billing: GET /v2/billing returned HTTP %s (%s), so it was skipped. Auth itself is fine.\n' \
    "$HTTP_CODE" "$(api_error_detail)"
fi

printf '\nAUTH OK. Read-only checks finished; nothing was created or changed.\n'
printf '(Pod environment variables are deliberately not shown; they can hold secrets.)\n'
