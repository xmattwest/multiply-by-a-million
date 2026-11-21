#!/bin/bash
# pin_wp2d.sh: resolve the two WP2d models' current commit on Hugging Face and
# write the exact 40-hex revision into config/pins.json, in place of the
# PENDING_HUB_PIN placeholders. Read-only against the hub (no token, no
# weights). After this, run `bash scripts/make_manifests.sh` to fingerprint
# every pinned model (now including the two new ones).
#
# WHAT IT DOES
#   For each of mistral-small-24b-it and phi-4-14b in config/pins.json whose
#   revision is still PENDING_HUB_PIN, asks the hub for the repo's current
#   commit SHA, prints it for you to see, and (unless DRY_RUN=1) rewrites
#   pins.json with that SHA. Models already pinned to a real SHA are left
#   untouched. All-or-nothing: resolves both before writing either.
#
#   Free and read-only. Total download is a few KB of JSON.
#
# EXIT CODES
#   0 wrote (or, with DRY_RUN=1, previewed) both pins
#   2 local problem (missing tool, bad pins file, nothing pending)
#   3 network error or unexpected answer (nothing written)
#
# USAGE
#   DRY_RUN=1 bash scripts/pin_wp2d.sh    # show the SHAs, write nothing
#   bash scripts/pin_wp2d.sh              # resolve and write pins.json
#   then: bash scripts/make_manifests.sh  # fingerprint all pinned models
#
# Written for macOS /bin/bash 3.2.

set -euo pipefail
umask 022
export LC_ALL=C
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_API_TOKEN 2>/dev/null || true

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
PINS="${PINS_FILE:-$ROOT/config/pins.json}"
DRY_RUN="${DRY_RUN:-0}"

die() { local c="$1"; shift; printf 'ERROR: %s\n' "$*" >&2; exit "$c"; }

command -v curl >/dev/null 2>&1 || die 2 "curl not found."
command -v jq   >/dev/null 2>&1 || die 2 "jq not found."
command -v python3 >/dev/null 2>&1 || die 2 "python3 not found."
[ -f "$PINS" ] || die 2 "pins file not found: $PINS"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# Which models still need a real SHA?
pending="$(jq -r '.models[] | select(.revision == "PENDING_HUB_PIN") | [.name, .repo] | @tsv' "$PINS")"
[ -n "$pending" ] || die 2 "no models with PENDING_HUB_PIN in $PINS (nothing to do)."

get() {
  curl -q -sS -g --proto '=https' --tlsv1.2 --connect-timeout 10 --max-time 60 \
    -o "$2" -w '%{http_code}' "$1" < /dev/null
}

# Resolve every pending model to a SHA first; only write once all succeed.
mapfile_names=""; mapfile_shas=""
while IFS=$'\t' read -r name repo; do
  [ -n "$name" ] || continue
  # The model info endpoint reports the current commit SHA at .sha
  code="$(get "https://huggingface.co/api/models/$repo" "$TMP/info.json")" \
    || die 3 "network error resolving $repo"
  [ "$code" = "200" ] || die 3 "resolving $repo: HTTP $code"
  sha="$(jq -r '.sha // empty' "$TMP/info.json")"
  case "$sha" in ''|*[!0-9a-f]*) die 3 "$repo: hub did not return a hex commit sha" ;; esac
  [ "${#sha}" -eq 40 ] || die 3 "$repo: commit sha is not 40 hex characters ($sha)"
  printf '%-24s %s -> %s\n' "$name" "$repo" "$sha"
  mapfile_names="$mapfile_names$name "
  mapfile_shas="$mapfile_shas$sha "
done <<EOF
$pending
EOF

if [ "$DRY_RUN" = "1" ]; then
  printf 'DRY_RUN=1: resolved above; pins.json NOT modified.\n'
  exit 0
fi

# Write each resolved SHA into pins.json via a small, deterministic python edit.
set -- $mapfile_names
names=("$@")
set -- $mapfile_shas
shas=("$@")
i=0
while [ "$i" -lt "${#names[@]}" ]; do
  nm="${names[$i]}"; sh="${shas[$i]}"
  python3 - "$PINS" "$nm" "$sh" <<'PY'
import json, sys
pins_path, name, sha = sys.argv[1], sys.argv[2], sys.argv[3]
with open(pins_path) as f:
    d = json.load(f)
for m in d["models"]:
    if m["name"] == name:
        if m["revision"] != "PENDING_HUB_PIN":
            print(f"skip {name}: already pinned to {m['revision']}")
            break
        m["revision"] = sha
        with open(pins_path, "w") as f:
            json.dump(d, f, indent=2)
            f.write("\n")
        print(f"pinned {name} -> {sha}")
        break
else:
    sys.exit(f"model {name} not found in pins")
PY
  i=$((i + 1))
done

printf '\nDone. Next: bash scripts/make_manifests.sh   (fingerprints all pinned models)\n'
