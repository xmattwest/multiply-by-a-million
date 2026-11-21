# test_make_manifests.sh: offline tests for scripts/make_manifests.sh.
#
# No network. A fake `curl` placed first on PATH serves a tiny made-up model
# repo: a file list (like Hugging Face's /tree endpoint) and the raw bytes of
# its small files. FAKE_MM_MODE picks a scenario (good repo, broken answers).
# The expected fingerprints are computed here, independently, with shasum.
#
# Usage: bash tests/test_make_manifests.sh

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${MM_SCRIPT:-$HERE/../scripts/make_manifests.sh}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAKEBIN="$TMP/bin"
LOGS="$TMP/logs"
mkdir -p "$FAKEBIN" "$LOGS"

REV_A="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
REV_B="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
BIG_OID="1111111111111111111111111111111111111111111111111111111111111111"
CANARY_HF="hf_CANARYHFTOKEN000000000000000000000"

PASS=0
FAIL=0
RC=0
pass() { PASS=$((PASS + 1)); printf 'ok    %s\n' "$1"; }
fail() { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1"; }
assert_rc() { if [ "$RC" -eq "$1" ]; then pass "$2"; else fail "$2 (exit $RC, want $1)"; fi; }
assert_has() { if [ -f "$1" ] && grep -qF -- "$2" "$1"; then pass "$3"; else fail "$3"; fi; }
assert_lacks() { if [ -f "$1" ] && grep -qF -- "$2" "$1"; then fail "$3"; else pass "$3"; fi; }

# The made-up repo's small files. Sizes and hashes are derived from these
# bytes here, so the test does not trust the script's own arithmetic.
SMALL="$TMP/small"
mkdir -p "$SMALL/sub"
printf '{"model_type":"toy"}\n' > "$SMALL/config.json"
printf 'chat template {{ x }}\n' > "$SMALL/sub/chat_template.jinja"
size_of() { wc -c < "$1" | tr -d ' '; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
SZ_CFG="$(size_of "$SMALL/config.json")"; SHA_CFG="$(sha_of "$SMALL/config.json")"
SZ_TPL="$(size_of "$SMALL/sub/chat_template.jinja")"; SHA_TPL="$(sha_of "$SMALL/sub/chat_template.jinja")"
export SMALL SZ_CFG SZ_TPL BIG_OID

cat > "$FAKEBIN/curl" <<'FAKE'
#!/bin/bash
set -u
log="${FAKE_LOG_DIR:?}"
mode="${FAKE_MM_MODE:-ok}"
printf -- '---CALL---\n' >> "$log/argv"
printf '%s\n' "$@" >> "$log/argv"
printf '%s\n' "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}${HUGGINGFACE_HUB_TOKEN:-}${HF_API_TOKEN:-}" >> "$log/child_env"
url="" ; outfile="" ; hdrfile="" ; prev=""
for a in "$@"; do
  case "$prev" in -o) outfile="$a" ;; -D) hdrfile="$a" ;; esac
  case "$a" in https://*) url="$a" ;; esac
  prev="$a"
done
printf '%s\n' "$url" >> "$log/urls"
hdr="HTTP/2 200"
case "$url" in
  https://huggingface.co/api/models/*/tree/*)
    [ "$mode" = neterr ] && { echo "curl: (6) Could not resolve host" >&2; exit 6; }
    [ "$mode" = tree404 ] && { printf '{"error":"Revision Not Found"}' > "$outfile"; : > "$hdrfile"; printf 404; exit 0; }
    bigpath="model-00001.safetensors"; bigsize=5000000000; lsize=$bigsize; oid="$BIG_OID"
    tplpath="sub/chat_template.jinja"
    case "$mode" in
      paged)     hdr="$hdr"$'\n''link: <https://huggingface.co/api/models/x/tree/y?cursor=abc>; rel="next"' ;;
      evilpath)  bigpath="../../etc/x.safetensors" ;;
      tabpath)   tplpath='sub/a\tb.jinja' ;;   # JSON escape for a tab
      shortoid)  oid="1111" ;;
      upperoid)  oid="$(printf '%s' "$BIG_OID" | tr 1 A)" ;;
      sizemix)   lsize=4999999999 ;;
      notarray)  printf '{"error":"x"}' > "$outfile"; printf '%s\n' "$hdr" > "$hdrfile"; printf 200; exit 0 ;;
      empty)     printf '[]' > "$outfile"; printf '%s\n' "$hdr" > "$hdrfile"; printf 200; exit 0 ;;
    esac
    dupe=""
    [ "$mode" = dupe ] && dupe=",{\"type\":\"file\",\"oid\":\"g\",\"size\":$SZ_CFG,\"path\":\"config.json\"}"
    cat > "$outfile" <<JSON
[{"type":"file","oid":"g1","size":$SZ_CFG,"path":"config.json"},
 {"type":"directory","oid":"d1","size":0,"path":"sub"},
 {"type":"file","oid":"g2","size":$SZ_TPL,"path":"$tplpath"},
 {"type":"file","oid":"g3","size":$bigsize,"lfs":{"oid":"$oid","size":$lsize,"pointerSize":134},"path":"$bigpath"}$dupe]
JSON
    printf '%s\n' "$hdr" > "$hdrfile"
    printf 200 ;;
  https://huggingface.co/*/raw/*)
    rel="${url#https://huggingface.co/}"; rel="${rel#*/*/raw/*/}"
    # Any repo named Org/Missing has no raw files (used for "all or nothing").
    case "$url" in https://huggingface.co/Org/Missing/*) printf 'Entry not found' > "$outfile"; : > "$hdrfile"; printf 404; exit 0 ;; esac
    if [ "$mode" = rawfail ]; then printf 'nope' > "$outfile"; : > "$hdrfile"; printf 500; exit 0; fi
    if [ -f "$SMALL/$rel" ]; then
      cp "$SMALL/$rel" "$outfile"
      [ "$mode" = truncated ] && [ "$rel" = config.json ] && printf '{"mo' > "$outfile"
      printf '%s\n' "$hdr" > "$hdrfile"; printf 200
    else
      printf 'Entry not found' > "$outfile"; : > "$hdrfile"; printf 404
    fi ;;
  *) echo "fake curl: unexpected URL $url" >&2; exit 99 ;;
esac
FAKE
chmod +x "$FAKEBIN/curl"

# Two toy models, so "all or nothing" can be tested.
write_pins() {  # write_pins FILE REPO2
  cat > "$1" <<PINS
{"models":[
 {"name":"toy-a","repo":"Org/Toy-A","revision":"$REV_A"},
 {"name":"toy-b","repo":"${2:-Org/Toy-B}","revision":"$REV_B"}]}
PINS
}
write_pins "$TMP/pins.json"

# run_mm MODE PINS OUTDIR: run the script against the fake curl; sets RC.
run_mm() {
  local mode="$1" pins="$2" outdir="$3"
  rm -f "$LOGS"/*
  set +e
  env PATH="$FAKEBIN:$PATH" FAKE_MM_MODE="$mode" FAKE_LOG_DIR="$LOGS" \
      PINS_FILE="$pins" MANIFEST_DIR="$outdir" \
      SMALL="$SMALL" SZ_CFG="$SZ_CFG" SZ_TPL="$SZ_TPL" BIG_OID="$BIG_OID" \
      HF_TOKEN="$CANARY_HF" HUGGING_FACE_HUB_TOKEN="$CANARY_HF" \
      HUGGINGFACE_HUB_TOKEN="$CANARY_HF" HF_API_TOKEN="$CANARY_HF" \
      bash "$SCRIPT" < /dev/null > "$TMP/out.txt" 2>&1
  RC=$?
  set -e
}
calls() { if [ -f "$LOGS/urls" ]; then wc -l < "$LOGS/urls" | tr -d ' '; else echo 0; fi; }

echo "== static =="
if bash -n "$SCRIPT"; then pass "make_manifests.sh parses"; else fail "make_manifests.sh syntax check"; fi

echo "== happy path =="
OUT="$TMP/m1"
run_mm ok "$TMP/pins.json" "$OUT"
assert_rc 0 "exit 0 for a good repo"
M="$OUT/toy-a.tsv"
assert_has "$M" "# Org/Toy-A at $REV_A" "header names the repo and full pinned commit"
assert_has "$M" "$(printf 'model-00001.safetensors\t5000000000\t%s\thub' "$BIG_OID")" "big file: hub fingerprint and size"
assert_has "$M" "$(printf 'config.json\t%s\t%s\tcomputed' "$SZ_CFG" "$SHA_CFG")" "small file: hashed locally, matches an independent shasum"
assert_has "$M" "$(printf 'sub/chat_template.jinja\t%s\t%s\tcomputed' "$SZ_TPL" "$SHA_TPL")" "file in a subfolder: hashed"
assert_lacks "$M" "$(printf 'sub\t0')" "folders are not listed as files"
if [ "$(grep -vc '^#' "$M")" -eq 3 ]; then pass "exactly 3 file lines"; else fail "exactly 3 file lines (saw $(grep -vc '^#' "$M"))"; fi
if grep -v '^#' "$M" | cut -f1 | sort -c 2>/dev/null; then pass "lines sorted by path"; else fail "lines sorted by path"; fi
if [ -f "$OUT/toy-b.tsv" ]; then pass "a manifest per model"; else fail "a manifest per model"; fi
assert_has "$TMP/out.txt" "toy-a                3 files, $((5000000000 + SZ_CFG + SZ_TPL)) bytes" "summary line: file count and byte total"
assert_has "$LOGS/urls" "https://huggingface.co/api/models/Org/Toy-A/tree/$REV_A?recursive=true" "file list asked for at the full pinned commit"
assert_has "$LOGS/urls" "https://huggingface.co/Org/Toy-A/raw/$REV_A/config.json" "small file fetched at the pinned commit"
assert_lacks "$LOGS/urls" "safetensors" "weights are never downloaded"
if [ "$(calls)" -eq 6 ]; then pass "6 requests for 2 models (1 list + 2 small files each)"; else fail "expected 6 requests, saw $(calls)"; fi
if grep -qx -- '-q' "$LOGS/argv" && [ "$(grep -c -- '---CALL---' "$LOGS/argv")" -eq "$(grep -A1 -- '---CALL---' "$LOGS/argv" | grep -cx -- '-q')" ]; then
  pass "-q is first on every curl call"; else fail "-q is first on every curl call"; fi
if grep -qE -- '^(-L|--location)$' "$LOGS/argv"; then fail "redirects are not followed"; else pass "redirects are not followed"; fi
if [ "$(grep -c -- "--proto" "$LOGS/argv")" -eq 6 ] && grep -qx -- '=https' "$LOGS/argv"; then pass "https only on every call"; else fail "https only on every call"; fi
if grep -q 'CANARY' "$LOGS/child_env"; then fail "HF token variables not inherited by curl"; else pass "HF token variables not inherited by curl"; fi
if grep -qi 'authorization' "$LOGS/argv"; then fail "no Authorization header sent"; else pass "no Authorization header sent"; fi

echo "== deterministic =="
OUT2="$TMP/m2"
run_mm ok "$TMP/pins.json" "$OUT2"
if cmp -s "$OUT/toy-a.tsv" "$OUT2/toy-a.tsv" && cmp -s "$OUT/toy-b.tsv" "$OUT2/toy-b.tsv"; then
  pass "two runs give byte-identical manifests"; else fail "two runs give byte-identical manifests"; fi

echo "== refuses bad answers (exit 3) =="
check_bad() {  # check_bad MODE "message fragment" "label"
  local d="$TMP/bad_$1"
  run_mm "$1" "$TMP/pins.json" "$d"
  assert_rc 3 "$3 -> exit 3"
  assert_has "$TMP/out.txt" "$2" "$3: explained"
  if [ -e "$d/toy-a.tsv" ]; then fail "$3: no manifest written"; else pass "$3: no manifest written"; fi
}
check_bad paged     "split into pages"                "file list split into pages"
check_bad evilpath  "unexpected file name"            "big file with a ../ path"
check_bad tabpath   "unexpected file name"            "small file with a tab in its name"
check_bad shortoid  "not 64 hex characters"           "short fingerprint"
check_bad upperoid  "not lowercase hex"               "fingerprint with non-hex letters"
check_bad sizemix   "two different sizes"             "size and LFS size disagree"
check_bad truncated "file list says $SZ_CFG"          "small file downloaded short"
check_bad dupe      "listed twice"                    "same path listed twice"
check_bad notarray  "not a non-empty JSON array"      "file list not an array"
check_bad empty     "not a non-empty JSON array"      "empty file list"
check_bad tree404   "HTTP 404"                        "pinned commit not found"
check_bad rawfail   "HTTP 500"                        "small file fetch fails"
check_bad neterr    "network error"                   "network error"

echo "== all or nothing =="
KEEP="$TMP/keep"
mkdir -p "$KEEP"
printf 'old a\n' > "$KEEP/toy-a.tsv"
printf 'old b\n' > "$KEEP/toy-b.tsv"
write_pins "$TMP/pins_missing.json" "Org/Missing"
# toy-a lists fine; Org/Missing's raw files don't exist -> 404 on the second model.
run_mm ok "$TMP/pins_missing.json" "$KEEP"
assert_rc 3 "second model fails -> exit 3"
if [ "$(cat "$KEEP/toy-a.tsv")" = "old a" ] && [ "$(cat "$KEEP/toy-b.tsv")" = "old b" ]; then
  pass "a failure leaves every old manifest untouched"; else fail "a failure leaves every old manifest untouched"; fi

echo "== bad pins (exit 2, no requests) =="
check_pins() {  # check_pins 'JSON' "label"
  printf '%s\n' "$1" > "$TMP/badpins.json"
  run_mm ok "$TMP/badpins.json" "$TMP/bp"
  assert_rc 2 "$2 -> exit 2"
  if [ "$(calls)" -eq 0 ]; then pass "$2: no request made"; else fail "$2: no request made"; fi
}
check_pins '{"models":[{"name":"x","repo":"Org/X","revision":"abc1234"}]}' "short revision"
check_pins '{"models":[{"name":"x","repo":"Org/X","revision":"'"$REV_A"'"},{"name":"y","repo":"a/b?c","revision":"'"$REV_A"'"}]}' "bad repo in a later model"
check_pins '{"models":[{"name":"../x","repo":"Org/X","revision":"'"$REV_A"'"}]}' "model name with a path in it"
check_pins '{"models":[{"name":"x","repo":"Org/X","revision":"'"$REV_A"'"},{"name":"x","repo":"Org/Y","revision":"'"$REV_B"'"}]}' "same model name twice"
check_pins '{"models":[]}' "no models"
check_pins 'not json' "pins not JSON"
run_mm ok "$TMP/nope.json" "$TMP/bp"
assert_rc 2 "missing pins file -> exit 2"

echo "== exclude (pins.json) =="
cat > "$TMP/pins_ex.json" <<PINS
{"models":[{"name":"toy-a","repo":"Org/Toy-A","revision":"$REV_A","exclude":["sub/chat_template.jinja"]}]}
PINS
run_mm ok "$TMP/pins_ex.json" "$TMP/ex"
if [ "$RC" -eq 0 ]; then pass "exclude: exit 0"; else fail "exclude: exit $RC"; fi
if ! grep -q 'chat_template.jinja' <(grep -v '^#' "$TMP/ex/toy-a.tsv"); then pass "excluded file left out of the manifest"; else fail "excluded file still listed"; fi
if grep -q '^config.json' "$TMP/ex/toy-a.tsv"; then pass "other files still listed"; else fail "other files missing"; fi
if grep -qx '# excluded (pins.json): sub/chat_template.jinja' "$TMP/ex/toy-a.tsv"; then pass "exclusion named in the header"; else fail "exclusion not named in the header"; fi
if ! grep -q 'raw/.*chat_template.jinja' "$LOGS/urls"; then pass "excluded file never fetched"; else fail "excluded file was fetched"; fi
cat > "$TMP/pins_ex2.json" <<PINS
{"models":[{"name":"toy-a","repo":"Org/Toy-A","revision":"$REV_A","exclude":["nope.bin"]}]}
PINS
run_mm ok "$TMP/pins_ex2.json" "$TMP/ex2"
if [ "$RC" -eq 3 ]; then pass "excluding a file not in the repo -> exit 3"; else fail "missing excluded file: exit $RC, want 3"; fi
cat > "$TMP/pins_ex3.json" <<PINS
{"models":[{"name":"toy-a","repo":"Org/Toy-A","revision":"$REV_A","exclude":["../etc/x"]}]}
PINS
run_mm ok "$TMP/pins_ex3.json" "$TMP/ex3"
if [ "$RC" -eq 2 ] && [ "$(calls)" -eq 0 ]; then pass "bad exclude path -> exit 2, no requests"; else fail "bad exclude path: exit $RC"; fi

echo "== the real files =="
REAL="$HERE/../config/manifests"
for f in "$REAL"/*.tsv; do
  n="$(basename "$f" .tsv)"
  bad="$(grep -v '^#' "$f" | awk -F'\t' 'NF != 4 || $2 !~ /^[0-9]+$/ || $3 !~ /^[0-9a-f]{64}$/ || ($4 != "hub" && $4 != "computed")' | head -1)"
  if [ -z "$bad" ]; then pass "$n.tsv: every line is path, size, 64-hex sha256, source"; else fail "$n.tsv: malformed line"; fi
  want="$(jq -r --arg n "$n" '.models[] | select(.name == $n) | "# \(.repo) at \(.revision)"' "$HERE/../config/pins.json")"
  if [ -n "$want" ] && [ "$(head -1 "$f")" = "$want" ]; then pass "$n.tsv: header matches config/pins.json"; else fail "$n.tsv: header matches config/pins.json"; fi
done

printf '\npassed: %d  failed: %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
