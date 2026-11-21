#!/bin/bash
# run_all.sh: run every connection check in this folder, in number order
# (01_*.sh, 02_*.sh, ...). Stops at the first check that fails, because
# later connections build on earlier ones. Each check is read-only and free.
#
# Usage: bash scripts/connections/run_all.sh

set -uo pipefail
export LC_ALL=C

here="$(cd "$(dirname "$0")" && pwd)"
total=0
for f in "$here"/[0-9][0-9]_*.sh; do
  [ -e "$f" ] && total=$((total + 1))
done
if [ "$total" -eq 0 ]; then
  printf 'No connection checks found in %s\n' "$here" >&2
  exit 2
fi

n=0
skipped=0
for f in "$here"/[0-9][0-9]_*.sh; do
  n=$((n + 1))
  name="$(basename "$f")"
  printf '\n[%d/%d] %s\n' "$n" "$total" "$name"
  bash "$f"
  rc=$?
  if [ "$rc" -eq 10 ]; then          # 10 = skipped: needs a running pod
    skipped=$((skipped + 1))
    continue
  fi
  if [ "$rc" -ne 0 ]; then
    printf '\nStopped at %s (check %d of %d, exit %d). Later checks were not run.\n' \
      "$name" "$n" "$total" "$rc" >&2
    exit "$rc"
  fi
done
if [ "$skipped" -eq 0 ]; then
  printf '\n%d of %d connection checks passed.\n' "$total" "$total"
else
  printf '\n%d of %d connection checks passed; %d skipped (no pod running).\n' \
    "$((total - skipped))" "$total" "$skipped"
fi
