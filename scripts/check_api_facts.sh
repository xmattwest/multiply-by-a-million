#!/bin/bash
# check_api_facts.sh: are the facts our design rests on still true?
#
# WHAT IT DOES
#   Reads docs/api_facts.txt. Each line names a public page (RunPod spec or
#   docs, vLLM source at tag v0.30.0, a Hugging Face model) and an exact
#   piece of text that page must contain ("has") or must not contain
#   ("lacks"). Fetches each page once, anonymously, and prints one line per
#   fact: ✓ still true, ✗ the page changed, ? the page could not be read.
#   Free and read-only: plain GETs, no account, no key, nothing changed.
#
#   Run it before finalizing Phase 4 and before every paid Phase 7 step.
#   A ✗ means: re-read that source and update docs/api_map.md or
#   docs/research_notes.md before relying on the fact.
#
# SAFETY
#   - Only four hosts are allowed (below). The facts file is validated in
#     full before any request, so a bad line cannot send curl elsewhere.
#   - No token or key is read or sent; Hugging Face token variables and
#     RUNPOD_API_KEY are unset before curl runs. ~/.curlrc is ignored
#     (curl -q). Redirects are not followed.
#   - Page text is never printed; only the labels from our own file, with
#     non-printable characters stripped.
#
# EXIT CODES
#   0 every fact holds   1 at least one fact changed
#   2 facts file missing or malformed   3 a page could not be read
#
# USAGE
#   bash scripts/check_api_facts.sh
#   API_FACTS_FILE=path bash scripts/check_api_facts.sh   (another facts file)
#
# Written for macOS /bin/bash 3.2.

set +x +v +a
set -euo pipefail
umask 077
export LC_ALL=C
unset SSLKEYLOGFILE
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_API_TOKEN RUNPOD_API_KEY

HERE="$(cd "$(dirname "$0")" && pwd)"
FACTS="${API_FACTS_FILE:-$HERE/../docs/api_facts.txt}"
readonly MAX_BYTES=5242880

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
    printf '  Checks that the facts in docs/api_facts.txt still hold (anonymous, read-only).\n'
    exit 0 ;;
  *) die 2 "unknown option: $1 (try --help)" ;;
esac

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
  C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_BAD=$'\033[31m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
  C_OK=""; C_WARN=""; C_BAD=""; C_BOLD=""; C_OFF=""
fi

command -v curl >/dev/null 2>&1 || die 2 "curl not found."
[ -f "$FACTS" ] || die 2 "facts file not found: $FACTS"

clean() { printf '%s' "$1" | tr -cd '[:print:]' | cut -c1-90; }

allowed_url() {
  case "$1" in
    *[!A-Za-z0-9._~/%?=\&+:-]*) return 1 ;;
  esac
  case "$1" in
    https://api.runpod.io/*|https://docs.runpod.io/*|https://raw.githubusercontent.com/*|https://huggingface.co/*) return 0 ;;
  esac
  return 1
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
: > "$TMP/facts"
: > "$TMP/urls"

# --- 1. Validate every line before any request -----------------------------
TAB=$'\t'
lineno=0
while IFS= read -r line || [ -n "$line" ]; do
  lineno=$((lineno + 1))
  case "$line" in ''|'#'*) continue ;; esac
  IFS="$TAB" read -r id kind url label text extra <<EOF
$line
EOF
  if [ -z "${text:-}" ] || [ -n "${extra:-}" ]; then
    die 2 "$FACTS line $lineno: need exactly 5 TAB-separated fields."
  fi
  case "$id" in *[!A-Za-z0-9._-]*) die 2 "$FACTS line $lineno: bad ID." ;; esac
  case "$kind" in has|lacks) ;; *) die 2 "$FACTS line $lineno: second field must be has or lacks." ;; esac
  allowed_url "$url" || die 2 "$FACTS line $lineno: URL not allowed (only https on api.runpod.io, docs.runpod.io, raw.githubusercontent.com, huggingface.co)."
  printf '%s\n' "$line" >> "$TMP/facts"
  grep -qxF -- "$url" "$TMP/urls" || printf '%s\n' "$url" >> "$TMP/urls"
done < "$FACTS"

nfacts="$(wc -l < "$TMP/facts" | tr -d ' ')"
nurls="$(wc -l < "$TMP/urls" | tr -d ' ')"
[ "$nfacts" -gt 0 ] || die 2 "$FACTS lists no facts."

# --- 2. Fetch each page once ------------------------------------------------------
i=0
while IFS= read -r url; do
  i=$((i + 1))
  code="000"
  if code="$(curl -q -sS -g --proto '=https' --tlsv1.2 \
        --connect-timeout 10 --max-time 40 --max-filesize "$MAX_BYTES" \
        -o "$TMP/body.$i" -w '%{http_code}' "$url" < /dev/null 2>"$TMP/err.$i")"; then
    :
  else
    code="net"
  fi
  printf '%s\n' "$code" > "$TMP/code.$i"
done < "$TMP/urls"

# --- 3. Check each fact -------------------------------------------------------
printf '%sAPI facts%s (anonymous, read-only: %d facts from %d pages)\n' \
  "$C_BOLD" "$C_OFF" "$nfacts" "$nurls"
held=0
changed=0
unreadable=0
while IFS="$TAB" read -r id kind url label text; do
  n="$(grep -n -x -F -- "$url" "$TMP/urls" | head -1 | cut -d: -f1)"
  code="$(cat "$TMP/code.$n")"
  shown="$(printf '%-7s %s' "$(clean "$id")" "$(clean "$label")")"
  if [ "$code" != "200" ] || [ ! -s "$TMP/body.$n" ]; then
    unreadable=$((unreadable + 1))
    if [ "$code" = "net" ]; then why="network error"
    elif [ "$code" = "200" ]; then why="empty page"
    else why="HTTP $(clean "$code")"; fi
    printf '  %s?%s %s: page unavailable (%s)\n' "$C_WARN" "$C_OFF" "$shown" "$why"
    continue
  fi
  if grep -qF -- "$text" "$TMP/body.$n"; then found=1; else found=0; fi
  if { [ "$kind" = "has" ] && [ "$found" -eq 1 ]; } || { [ "$kind" = "lacks" ] && [ "$found" -eq 0 ]; }; then
    held=$((held + 1))
    printf '  %s✓%s %s\n' "$C_OK" "$C_OFF" "$shown"
  else
    changed=$((changed + 1))
    if [ "$kind" = "has" ]; then why="expected text is gone"; else why="unwanted text appeared"; fi
    printf '  %s✗%s %s: CHANGED (%s)\n' "$C_BAD" "$C_OFF" "$shown" "$why"
  fi
done < "$TMP/facts"

printf '%d of %d facts still hold.' "$held" "$nfacts"
[ "$changed" -gt 0 ] && printf ' %d changed: re-read those sources before relying on them.' "$changed"
[ "$unreadable" -gt 0 ] && printf ' %d could not be checked.' "$unreadable"
printf '\n'

if [ "$unreadable" -gt 0 ]; then exit 3; fi
if [ "$changed" -gt 0 ]; then exit 1; fi
exit 0
