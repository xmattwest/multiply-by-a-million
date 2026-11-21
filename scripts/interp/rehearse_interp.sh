#!/bin/bash
# rehearse_interp.sh: free local dress rehearsal of an interp session (no pod, no money).
#
# Packs scripts/interp + its imports EXACTLY as launch_pod.sh --mode interp does (the same
# scripts/lib/pack_interp.sh, with the plug-in list launch_interp.py would pass), unpacks it
# into a scratch "pod" folder, starts
# run_interp.py there on the CPU with a tiny random-weight Gemma 4 model of the 31B's class,
# then drives a small E0 through the real laptop driver (launch_interp.py --attach).
# Catches packaging, import, API and protocol bugs; says nothing about the real model's
# numbers (random weights; the check against vLLM is skipped and labelled REHEARSAL).
#
# Two chats can rehearse at the same time with different REHEARSE_SLOT values (default 9; never a slot
# a paid session is using). Create the throwaway venv and tiny model with
#   bash scripts/interp/make_rehearsal_model.sh
# Needs (made once, outside the repo):
#   /tmp/si-interp-venv   python venv with torch 2.13.0, transformers 5.17.0, xgrammar
#   /tmp/si-tiny-31b      tiny Gemma4ForConditionalGeneration + the pinned 31B tokenizer files
#   /tmp/si-tiny-31b.tsv  its manifest
# Usage: bash scripts/interp/rehearse_interp.sh [e0|e1e2|e3|e3b]      (default e0; macOS bash 3.2)
set -uo pipefail
export LC_ALL=C
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
EXP="${1:-e0}"; case "$EXP" in e0|e1e2|e3|e3b) ;; *) [ -f "$HERE/exp_$EXP.py" ] || { echo "usage: $0 [e0|e1e2|e3|e3b|<plug-in>]"; exit 2; };; esac
EXTRA="${REHEARSE_ARGS:-}"          # extra launch_interp flags for a plug-in, e.g. REHEARSE_ARGS="--sweep-step 2"
PY=/tmp/si-interp-venv/bin/python; MODEL=/tmp/si-tiny-31b; MAN=/tmp/si-tiny-31b.tsv
SLOT="${REHEARSE_SLOT:-9}"; case "$SLOT" in [1-9]) ;; *) echo "REHEARSE_SLOT must be 1-9"; exit 2;; esac
POD="/tmp/si-rehearsal-pod-$SLOT"; PORT=$((18000 + SLOT))      # one rehearsal per slot can run at once
for f in "$PY" "$MODEL/config.json" "$MAN"; do [ -e "$f" ] || { echo "missing $f (see the header)"; exit 2; }; done
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT busy"; exit 2; }

rm -rf "$POD"; mkdir -p "$POD/workspace"
# The launcher's own packer and plug-in list.
. "$ROOT/scripts/lib/pack_interp.sh"
PLUGS="$(python3 "$ROOT/scripts/launch_interp.py" --experiment "$EXP" --print-pod-plugins)" || { echo "no plug-in list for $EXP"; exit 1; }
pack_interp_code "$ROOT" "$POD/code.tgz" "$PLUGS" || { echo "pack failed (plug-ins $PLUGS)"; exit 1; }
echo "plug-ins: $PLUGS"
tar -xzf "$POD/code.tgz" -C "$POD/workspace" || exit 1
echo "packed: $(tar -tzf "$POD/code.tgz" | grep -v '/$' | tr '\n' ' ')"

( cd "$POD/workspace" && SI_INTERP_DEVICE=cpu SI_INTERP_REHEARSAL=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 exec "$PY" scripts/interp/run_interp.py --model-dir "$MODEL" --manifest "$MAN" \
    --model-name gemma-4-31b-it --port $PORT ) > "$POD/interp.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
for i in $(seq 1 90); do grep -q CONTROL_API "$POD/interp.log" && break; kill -0 $PID 2>/dev/null || break; sleep 2; done
grep '^\[interp\]' "$POD/interp.log" | cut -c1-220
grep -q 'INTERP_READY' "$POD/interp.log" || { echo "REHEARSAL FAILED: pod process not ready"; tail -20 "$POD/interp.log"; exit 1; }

perl -e 'alarm 900; exec @ARGV' python3 "$ROOT/scripts/launch_interp.py" --attach --slot $SLOT --model gemma-4-31b-it \
  --roles seller --fixtures 3 --samples 1 --experiment "$EXP" --steer-layer 3 --samples-e3 1 $EXTRA > "$POD/driver.out" 2>&1
RC=$?
tail -25 "$POD/driver.out"
echo "--- pod log tail ---"; grep -E '^\[interp\] task|failed|Error' "$POD/interp.log" | tail -8 | cut -c1-220
RUN="$(sed -n 's/^Data: \(data\/interp\/[^ ]*\).*/\1/p' "$POD/driver.out" | head -1)"
if [ -n "$RUN" ]; then
  python3 - "$ROOT/$RUN" <<'EOF'
import json, sys, collections
d = sys.argv[1]
rows = [json.loads(l) for l in open(d + "/results.jsonl")]
c = collections.Counter(r["step"] for r in rows)
gen = [r for r in rows if r["step"] in ("sampled", "greedy")]
print("results per step:", dict(c))
if gen:
    print("constrained by grammar:", sum(1 for r in gen if r.get("constrained")), "of", len(gen),
          "| parsed as valid JSON moves:", sum(1 for r in gen if not r.get("errors")), "of", len(gen))
e2 = [r for r in rows if r["step"] == "e2 patch"]
if e2:
    print("e2 items:", len(e2), "| rows per item:", sorted({len(r["rows"]) for r in e2}),
          "| prefix cache:", sorted({r.get("prefix_cache") for r in e2}),
          "| clean rows consistent:", sum(1 for r in e2 if r.get("clean_consistent")), "of", len(e2))
fit = [r for r in rows if r["step"] == "e1 fit"]
if fit:
    f = fit[0]
    print("e1 fit:", f["n_negotiation"], "negotiation +", f["n_plain"], "plain; layers", f["layers"],
          "| positions", sorted(f["within"]), "| transfer", sorted(f["transfer"]))
e3 = [r for r in rows if r["step"] == "e3 steer"]
if e3:
    allrows = [x for r in e3 for x in r["rows"]]
    print("e3 items:", len(e3), "| rows per item:", sorted({len(r["rows"]) for r in e3}), "| loo:", sorted({r["loo"] for r in e3}),
          "| prefill fired:", sorted({str(r["prefill_fired"]) for r in e3}), "| n_post:", sorted({r["n_post"] for r in e3}),
          "| grammar:", sorted({r["grammar"] for r in e3}), "| vectors:", sorted({x["vector"] for x in allrows}))
for step in ("e3b bridge", "e3b steer"):
    rs = [r for r in rows if r["step"] == step]
    if rs:
        allrows = [x for r in rs for x in r["rows"]]
        print(step + ":", len(rs), "items | rows per item:", sorted({len(r["rows"]) for r in rs}), "| loo:", sorted({r["loo"] for r in rs}),
              "| hook calls:", sorted({str(r.get("hook_calls", [])[:1]) for r in rs}),
              "| aligned:", sum(1 for x in allrows if x.get("aligned")), "of", len(allrows) if step == "e3b steer" else "-")
vecinfo = [r for r in rows if r["step"] in ("e3 vectors", "e3b vectors")]
if vecinfo:
    print("e3 vectors:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in vecinfo[0].items() if k != "fixtures"})
s = json.load(open(d + "/summary.json"))
print("summary keys:", sorted(s), "| analysis error:", s.get("analysis_error"))
EOF
  rm -rf "$ROOT/$RUN"
fi
[ $RC = 0 ] && echo "REHEARSAL PASSED (protocol and code paths; numbers meaningless)" || echo "REHEARSAL FAILED (driver exit $RC)"
exit $RC
