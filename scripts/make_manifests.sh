#!/bin/bash
# make_manifests.sh: write a file-by-file fingerprint list for each pinned model.
#
# WHAT IT DOES
#   For each model in config/pins.json, asks Hugging Face (anonymously) for
#   the list of files at the pinned commit, and writes
#   config/manifests/<name>.tsv with one line per file:
#       path <TAB> size in bytes <TAB> sha256 <TAB> where the hash came from
#   - Big files (weights, stored with Git LFS/Xet): Hugging Face publishes
#     each file's SHA-256 in the file list ("lfs.oid"). Recorded as "hub".
#   - Small files (configs, chat template, README): the file list gives only
#     a git hash, so the file is downloaded at the pinned commit and hashed
#     here. Recorded as "computed". Total download is about 10 MB.
#   In Phase 7 the pod hashes what it downloaded and compares against these
#   files, so we know the weights we served are exactly the pinned ones.
#
#   Optional per model in pins.json: "exclude": ["<path>", ...] leaves those
#   repo files out of the manifest, so the pod never downloads them (e.g. a
#   second copy of the weights in another format). Each excluded path must
#   exist in the repo, and is named in the manifest header.
#   Free and read-only. No token. Weights are never downloaded here.
#   Output files are deterministic: run it twice, get identical files.
#   All or nothing: every pin is checked before any request, and the real
#   manifest files are replaced only after every model succeeded.
#   Refuses (exit 3) anything it can't vouch for: a paged file list, a
#   strange path, a malformed fingerprint, sizes that disagree, a duplicate.
#
# EXIT CODES
#   0 manifests written   2 local problem (incl. bad pins)
#   3 network error or unexpected answer (nothing written)
#
# USAGE
#   bash scripts/make_manifests.sh
#   Tests: bash tests/test_make_manifests.sh (offline)
#
# Written for macOS /bin/bash 3.2.

set +x +v +a
set -euo pipefail
umask 022
export LC_ALL=C
unset SSLKEYLOGFILE
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_API_TOKEN

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
PINS="${PINS_FILE:-$ROOT/config/pins.json}"
OUTDIR="${MANIFEST_DIR:-$ROOT/config/manifests}"

die() {
  local code="$1"
  shift
  printf 'ERROR: %s\n' "$*" >&2
  exit "$code"
}

command -v curl >/dev/null 2>&1 || die 2 "curl not found."
command -v jq >/dev/null 2>&1 || die 2 "jq not found."
command -v shasum >/dev/null 2>&1 || die 2 "shasum not found."
[ -f "$PINS" ] || die 2 "pins file not found: $PINS"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# get URL OUTFILE: anonymous GET, no redirects; saves response headers to
# $TMP/hdr and prints the HTTP code.
get() {
  curl -q -sS -g --proto '=https' --tlsv1.2 --connect-timeout 10 --max-time 120 \
    --max-filesize 104857600 -D "$TMP/hdr" -o "$2" -w '%{http_code}' "$1" < /dev/null
}

# A file path is safe if it is relative, has no "..", and uses only plain
# characters (so it can't break the TSV or escape the model folder on the pod).
safe_path() {
  case "$1" in ''|/*|*..*|*/|*//*|*[!A-Za-z0-9._/-]*) return 1 ;; esac
  return 0
}
is_uint() { case "$1" in ''|*[!0-9]*) return 1 ;; esac; return 0; }

if ! models="$(jq -r '.models[] | [.name, .repo, .revision, ((.exclude // []) | join(" "))] | @tsv' "$PINS" 2>/dev/null)" || [ -z "$models" ]; then
  die 2 "could not read models from $PINS"
fi

# Check every pin before making any request.
seen=" "
while IFS=$'\t' read -r name repo rev excl; do
  case "$name" in ''|*[!a-z0-9._-]*) die 2 "bad model name in pins" ;; esac
  case "$seen" in *" $name "*) die 2 "model name listed twice in pins: $name" ;; esac
  seen="$seen$name "
  case "$repo" in */*/*|*[!A-Za-z0-9._/-]*|/*|*/) die 2 "bad repo in pins for $name" ;; */*) ;; *) die 2 "bad repo in pins for $name" ;; esac
  case "$rev" in ''|*[!0-9a-f]*) die 2 "bad revision for $name" ;; esac
  [ "${#rev}" -eq 40 ] || die 2 "revision for $name must be 40 hex characters"
  for x in $excl; do safe_path "$x" || die 2 "bad exclude path for $name"; done
done <<EOF
$models
EOF

# Build every manifest in TMP first; only replace the real files once all
# of them succeeded, so a failure never leaves a mix of old and new.
mkdir -p "$TMP/out"
while IFS=$'\t' read -r name repo rev excl; do
  code="$(get "https://huggingface.co/api/models/$repo/tree/$rev?recursive=true" "$TMP/tree.json")" \
    || die 3 "network error listing $repo"
  [ "$code" = "200" ] || die 3 "listing $repo at ${rev:0:7}: HTTP $code"
  # Long file lists come in pages, with a "Link: <…>; rel=\"next\"" header.
  # We don't follow pages, so refuse rather than write a partial list.
  if grep -qi '^link:.*rel="\{0,1\}next' "$TMP/hdr"; then
    die 3 "file list for $repo is split into pages; this script reads only one page"
  fi
  jq -e 'type == "array" and length > 0' "$TMP/tree.json" >/dev/null 2>&1 \
    || die 3 "file list for $repo is not a non-empty JSON array"

  out="$TMP/$name.tsv"
  : > "$out"
  for x in $excl; do
    jq -e --arg p "$x" 'any(.[]; .type == "file" and .path == $p)' "$TMP/tree.json" >/dev/null \
      || die 3 "$repo: excluded file $x is not in the repo"
  done
  is_excluded() { case " $excl " in *" $1 "*) return 0 ;; esac; return 1; }
  # Big files (LFS/Xet): SHA-256 straight from the hub's file list.
  jq -r '.[] | select(.type == "file" and .lfs != null)
         | [.path, (.size|tostring), (.lfs.size|tostring), (.lfs.oid|tostring)] | @tsv' \
    "$TMP/tree.json" > "$TMP/big.tsv"
  while IFS=$'\t' read -r path size lsize oid; do
    [ -n "$path" ] || continue
    is_excluded "$path" && continue
    safe_path "$path" || die 3 "unexpected file name in $repo"
    is_uint "$size" || die 3 "$repo/$path: size is not a whole number"
    [ "$size" = "$lsize" ] || die 3 "$repo/$path: file list gives two different sizes"
    case "$oid" in *[!0-9a-f]*) die 3 "$repo/$path: fingerprint is not lowercase hex" ;; esac
    [ "${#oid}" -eq 64 ] || die 3 "$repo/$path: fingerprint is not 64 hex characters"
    printf '%s\t%s\t%s\thub\n' "$path" "$size" "$oid" >> "$out"
  done < "$TMP/big.tsv"
  # Small files: download at the pinned commit and hash.
  jq -r '.[] | select(.type == "file" and .lfs == null) | [.path, (.size|tostring)] | @tsv' \
    "$TMP/tree.json" > "$TMP/small.tsv"
  while IFS=$'\t' read -r path size; do
    [ -n "$path" ] || continue
    is_excluded "$path" && continue
    safe_path "$path" || die 3 "unexpected file name in $repo"
    is_uint "$size" || die 3 "$repo/$path: size is not a whole number"
    code="$(get "https://huggingface.co/$repo/raw/$rev/$path" "$TMP/f")" || die 3 "network error fetching $path"
    [ "$code" = "200" ] || die 3 "fetching $repo/$path at ${rev:0:7}: HTTP $code"
    got="$(wc -c < "$TMP/f" | tr -d ' ')"
    [ "$got" = "$size" ] || die 3 "$repo/$path: downloaded $got bytes, file list says $size"
    printf '%s\t%s\t%s\tcomputed\n' "$path" "$size" "$(shasum -a 256 "$TMP/f" | cut -d' ' -f1)" >> "$out"
  done < "$TMP/small.tsv"

  [ -s "$out" ] || die 3 "no files listed for $repo"
  sort -o "$out" "$out"
  dups="$(cut -f1 "$out" | uniq -d | head -1)"
  [ -z "$dups" ] || die 3 "$repo: a file is listed twice"
  {
    printf '# %s at %s\n' "$repo" "$rev"
    [ -z "$excl" ] || printf '# excluded (pins.json): %s\n' "$excl"
    printf '# path\tsize\tsha256\tsource (hub = Hugging Face file list, computed = hashed here)\n'
    cat "$out"
  } > "$TMP/out/$name.tsv"
done <<EOF
$models
EOF

mkdir -p "$OUTDIR"
while IFS=$'\t' read -r name repo rev excl; do
  mv -f "$TMP/out/$name.tsv" "$OUTDIR/$name.tsv"
  body="$(grep -v '^#' "$OUTDIR/$name.tsv")"
  nfiles="$(printf '%s\n' "$body" | wc -l | tr -d ' ')"
  bytes="$(printf '%s\n' "$body" | awk -F'\t' '{s += $2} END {printf "%.0f", s}')"
  printf '%-18s %3d files, %s bytes -> %s\n' "$name" "$nfiles" "$bytes" "config/manifests/$name.tsv"
done <<EOF
$models
EOF
