#!/bin/bash
# 04_vllm.sh: connection check 4. Is the model server answering, privately?
#
# Needs a serve-mode pod and its SSH tunnel (scripts/launch_pod.sh opens
# it: laptop 127.0.0.1:18000 -> pod 127.0.0.1:8000). Without one it says so
# and exits 10 ("skipped"). Free. Uses scripts/vllm_client.py, so it can
# only reach 127.0.0.1. Screen-safe output.
#
# Checks: /health is 200; /version is the pinned 0.30.0; /v1/models serves
# exactly our model name.

set -uo pipefail
export LC_ALL=C
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../lib/pod_ssh.sh"

printf 'vLLM (model server, through the tunnel)\n'
if ! si_load_session || [ "${MODE:-}" != serve ]; then
  printf '  – no model server running right now; skipped\n'
  exit 10
fi
perl -e 'alarm shift; exec @ARGV' 60 python3 - "$HERE/.." "$SI_TUNNEL_PORT" "$MODEL" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1])
import vllm_client as v
port, model = int(sys.argv[2]), sys.argv[3]

def get(path):
    status, raw, err = v.request(port, "GET", path, total_timeout=10)
    return status, raw, err

s, raw, err = get("/health")
if s != 200:
    print(f"  ✗ health: {err or 'HTTP %s' % s}")
    sys.exit(1)
print("  ✓ health: OK, reached through the private tunnel")
s, raw, err = get("/version")
ver = json.loads(raw).get("version") if s == 200 else None
if ver != "0.30.0":
    print(f"  ✗ version: {ver!r}, expected '0.30.0'")
    sys.exit(1)
print("  ✓ version: vLLM 0.30.0 (the pinned release)")
s, raw, err = get("/v1/models")
ids = [m.get("id") for m in json.loads(raw).get("data", [])] if s == 200 else []
if ids != [model]:
    print(f"  ✗ models: {ids!r}, expected [{model!r}]")
    sys.exit(1)
info = json.loads(raw)["data"][0]
print(f"  ✓ serving exactly one model: {model} (context window {info.get('max_model_len')} tokens)")
EOF
