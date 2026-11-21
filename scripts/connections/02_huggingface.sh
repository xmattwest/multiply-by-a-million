#!/bin/bash
# 02_huggingface.sh: connection check 2. Are our model repos on Hugging
# Face, are they gated, and is the exact version we pinned still there?
#
# WHAT IT DOES
#   Reads the repos and pinned commits from config/pins.json, then makes one
#   anonymous, read-only GET per repo to the public Hugging Face Hub API:
#     GET https://huggingface.co/api/models/<repo>?expand[]=sha&expand[]=gated&...
#   The expand[] parameters ask for just four fields (about 200 bytes) instead
#   of the full model page. Costs nothing, changes nothing, needs no account.
#   Output is screen-safe: repo names and commit IDs are public anyway.
#
#   If a repo's current commit is not the pinned one (the maker published
#   an update), a second GET asks whether the pinned commit still exists:
#     GET https://huggingface.co/api/models/<repo>/revision/<pinned commit>
#   Still there: a warning (we keep using the pin). Gone: a failure.
#
# ANONYMOUS ON PURPOSE
#   - No token is read, stored, or sent. Any Hugging Face token variables in
#     the environment are unset before curl runs.
#   - ~/.curlrc is ignored (curl -q), so a config file cannot add a header.
#   - Redirects are not followed. Hugging Face answers a wrong-case repo name
#     with a 307 redirect; we report the name it points to instead of quietly
#     using it, because the exact repo ID is part of the experiment record.
#
# READING THE ANSWERS (see docs/research_notes.md, M4)
#   200  repo is public. "gated" is false, "auto", or "manual".
#   401  repo does not exist, OR it is private. To an anonymous request
#        Hugging Face makes the two look the same.
#   307  the name is right except for letter case (or the repo was renamed).
#   The server's text is untrusted: only validated fields are printed, and
#   non-printable characters (such as terminal escape codes) are stripped.
#
# EXIT CODES
#   0 all repos found and every pinned commit exists (a gated repo, or a
#     repo that has moved past its pin, is a warning, not a failure)
#   1 a repo is missing, misnamed, or disabled, or a pinned commit is gone
#   2 local setup problem (including a missing or malformed pins file)
#   3 network error, rate limit, or unexpected response
#
# USAGE
#   bash scripts/connections/02_huggingface.sh
#
# Written for macOS /bin/bash 3.2.

set +x +v +a
set -euo pipefail
umask 077
export LC_ALL=C
unset SSLKEYLOGFILE
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_API_TOKEN

readonly HF_API="https://huggingface.co/api/models"
readonly FIELDS='expand%5B%5D=sha&expand%5B%5D=gated&expand%5B%5D=lastModified&expand%5B%5D=disabled'
PINS="${PINS_FILE:-$(cd "$(dirname "$0")" && pwd)/../../config/pins.json}"

die() {
  local code="$1"
  shift
  printf 'ERROR: %s\n' "$*" >&2
  exit "$code"
}

case "${1:-}" in
  "") ;;
  -h|--help)
    printf 'Usage: bash %s\n' "$0"
    printf '  Anonymous read-only check of the three model repos on Hugging Face.\n'
    exit 0 ;;
  *) die 2 "unknown option: $1 (try --help)" ;;
esac

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
  C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_BAD=$'\033[31m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
  C_OK=""; C_WARN=""; C_BAD=""; C_BOLD=""; C_OFF=""
fi

command -v curl >/dev/null 2>&1 || die 2 "curl not found."
command -v jq >/dev/null 2>&1 || die 2 "jq not found (install with: brew install jq)."

# Exact repo IDs (letter case matters) and pinned commits, from the pins
# file. Every value is validated before it goes near a URL.
[ -f "$PINS" ] || die 2 "pins file not found: $PINS"
if ! pinlist="$(jq -r '.models[] | "\(.repo)\t\(.revision)"' "$PINS" 2>/dev/null)" || [ -z "$pinlist" ]; then
  die 2 "could not read models from $PINS"
fi
REPOS=()
PINNED=()
while IFS=$'\t' read -r repo rev; do
  shown="$(printf '%s' "$repo" | tr -cd '[:print:]' | cut -c1-60)"
  case "$repo" in
    */*/*|*[!A-Za-z0-9._/-]*|/*|*/) die 2 "pins file: bad repo ID ($shown)" ;;
    */*) ;;
    *) die 2 "pins file: bad repo ID ($shown)" ;;
  esac
  case "$rev" in *[!0-9a-f]*|'') die 2 "pins file: bad commit for $shown" ;; esac
  [ "${#rev}" -eq 40 ] || die 2 "pins file: commit for $shown must be 40 hex characters"
  REPOS+=("$repo")
  PINNED+=("$rev")
done <<PINS_EOF
$pinlist
PINS_EOF

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# clean TEXT: printable ASCII only (drops escape codes and newlines), max 120 chars.
clean() {
  printf '%s' "$1" | tr -cd '[:print:]' | cut -c1-120
}

WORST=0
worse() { if [ "$1" -gt "$WORST" ]; then WORST="$1"; fi; }

ok_line()   { printf '  %s✓%s %-22s %s\n' "$C_OK" "$C_OFF" "$1" "$2"; }
warn_line() { printf '  %s!%s %-22s %s\n' "$C_WARN" "$C_OFF" "$1" "$2"; }
bad_line()  { printf '  %s✗%s %-22s %s\n' "$C_BAD" "$C_OFF" "$1" "$2"; }

check_repo() {
  local repo="$1" pin="$2" body="$TMP/body" meta code rest redirect fields
  local id sha gated day disabled suggestion
  rm -f "$body" "$TMP/err"
  if ! meta="$(curl -q -sS -g --proto '=https' --tlsv1.2 \
        --connect-timeout 10 --max-time 20 --max-filesize 1048576 \
        -H 'Accept: application/json' -o "$body" \
        -w '%{http_code} %{time_total} %{redirect_url}' \
        "${HF_API}/${repo}?${FIELDS}" < /dev/null 2>"$TMP/err")"; then
    bad_line "$repo" "network error ($(clean "$(cat "$TMP/err" 2>/dev/null)"))"
    worse 3
    return
  fi
  code="${meta%% *}"
  rest="${meta#* }"
  case "$rest" in
    *" "*) redirect="${rest#* }" ;;
    *) redirect="" ;;
  esac

  case "$code" in
    200) ;;
    401|404)
      bad_line "$repo" "not found (HTTP $code). Check the spelling; a private repo looks the same."
      worse 1
      return ;;
    301|302|307|308)
      suggestion="${redirect#*/api/models/}"
      suggestion="${suggestion%%\?*}"
      case "$suggestion" in
        */*/*|/*|*/|'') suggestion="" ;;
        */*) case "$suggestion" in *[!A-Za-z0-9._/-]*) suggestion="" ;; esac ;;
        *) suggestion="" ;;
      esac
      if [ -n "$suggestion" ]; then
        bad_line "$repo" "not this exact name (HTTP $code). Hugging Face points to $suggestion (letter case matters)."
      else
        bad_line "$repo" "not this exact name (HTTP $code); the redirect target was not recognized."
      fi
      worse 1
      return ;;
    429)
      bad_line "$repo" "rate limited by Hugging Face (HTTP 429). Wait a minute and retry."
      worse 3
      return ;;
    *)
      bad_line "$repo" "unexpected answer: HTTP $(clean "$code")"
      worse 3
      return ;;
  esac

  # Every field is non-empty ("-" if absent): bash treats tab as whitespace,
  # so `read` would merge an empty field into its neighbour.
  if ! fields="$(jq -r 'def f: if . == null or . == "" then "-" else tostring end;
                        [ (.id | f),
                          (.sha | f),
                          (if .gated == false then "false" elif (.gated | type) == "string" then .gated else "?" end | f),
                          ((.lastModified // "") | tostring | .[0:10] | f),
                          (if .disabled == true then "true" else "false" end)
                        ] | @tsv' "$body" 2>/dev/null)"; then
    bad_line "$repo" "HTTP 200, but the answer was not the expected JSON."
    worse 3
    return
  fi
  IFS=$'\t' read -r id sha gated day disabled <<EOF
$fields
EOF

  if [ "$id" != "$repo" ]; then
    bad_line "$repo" "HTTP 200, but the answer is for a different repo ID."
    worse 3
    return
  fi
  case "$sha" in
    *[!0-9a-f]*|'') sha="" ;;
  esac
  if [ "${#sha}" -ne 40 ]; then
    bad_line "$repo" "HTTP 200, but the commit ID is missing or malformed."
    worse 3
    return
  fi
  case "$day" in
    [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
    *) day="unknown date" ;;
  esac
  if [ "$disabled" = "true" ]; then
    bad_line "$repo" "found, but Hugging Face has disabled this repo."
    worse 1
    return
  fi
  local access warn=0 pinmsg
  case "$gated" in
    false) access="not gated" ;;
    auto|manual) access="GATED ($gated approval): downloads need a token"; warn=1 ;;
    *)
      bad_line "$repo" "HTTP 200, but the gated field was not recognized."
      worse 3
      return ;;
  esac

  if [ "$sha" = "$pin" ]; then
    pinmsg="pinned commit ${pin:0:7} is current"
  else
    # The repo moved on. Is the pinned commit still downloadable?
    if ! code="$(curl -q -sS -g --proto '=https' --tlsv1.2 \
          --connect-timeout 10 --max-time 20 --max-filesize 1048576 \
          -H 'Accept: application/json' -o "$body" -w '%{http_code}' \
          "${HF_API}/${repo}/revision/${pin}?expand%5B%5D=sha" < /dev/null 2>"$TMP/err")"; then
      bad_line "$repo" "network error checking pinned commit ${pin:0:7}"
      worse 3
      return
    fi
    case "$code" in
      200)
        if [ "$(jq -r '.sha // ""' "$body" 2>/dev/null)" != "$pin" ]; then
          bad_line "$repo" "HTTP 200, but the answer is not for pinned commit ${pin:0:7}."
          worse 3
          return
        fi
        pinmsg="pinned commit ${pin:0:7} still available; repo moved on to ${sha:0:7} (updated $day)"
        warn=1 ;;
      401|404)
        bad_line "$repo" "found, but pinned commit ${pin:0:7} is gone (HTTP $code); repo is now at ${sha:0:7}"
        worse 1
        return ;;
      429)
        bad_line "$repo" "rate limited by Hugging Face (HTTP 429). Wait a minute and retry."
        worse 3
        return ;;
      *)
        bad_line "$repo" "checking pinned commit: unexpected answer HTTP $(clean "$code")"
        worse 3
        return ;;
    esac
  fi

  if [ "$warn" -eq 1 ]; then
    warn_line "$repo" "found · $access · $pinmsg"
  else
    ok_line "$repo" "found · $access · $pinmsg"
  fi
}

printf '%sHugging Face model repos%s (anonymous: no token sent)\n' "$C_BOLD" "$C_OFF"
i=0
while [ "$i" -lt "${#REPOS[@]}" ]; do
  check_repo "${REPOS[$i]}" "${PINNED[$i]}"
  i=$((i + 1))
done

if [ "$WORST" -ne 0 ]; then
  printf 'Hugging Face check failed (exit %d). Nothing was downloaded or changed.\n' "$WORST" >&2
fi
exit "$WORST"
