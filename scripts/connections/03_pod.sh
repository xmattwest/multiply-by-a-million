#!/bin/bash
# 03_pod.sh: connection check 3. Can we get into the GPU machine, safely?
#
# Needs a running pod (scripts/launch_pod.sh writes logs/pod_session.env).
# Without one it says so and exits 10 ("skipped"). Free: it only talks to
# the pod over SSH. Screen-safe: no IP, port, pod ID or key.
#
# Checks: SSH works with only the project key and the host key the pod
# printed at boot (pinned; a different machine would be refused); the GPU
# and host CUDA; whether anything was injected into /root/.ssh (C19); in
# serve mode, that the model files passed their fingerprint check.

set -uo pipefail
export LC_ALL=C
. "$(cd "$(dirname "$0")" && pwd)/../lib/pod_ssh.sh"

printf 'Pod (GPU machine)\n'
if ! si_load_session; then
  printf '  – no pod running right now; skipped (start one with scripts/launch_pod.sh)\n'
  exit 10
fi
clean() { tr -cd 'A-Za-z0-9 ,._:()/=-' | head -c 160; }

if [ "$(si_ssh 'echo ok' 2>/dev/null)" = ok ]; then
  printf '  ✓ SSH: in with the project key; the host key matches the one the pod printed at boot\n'
else
  printf '  ✗ SSH: could not log in (wrong host key, refused key, or unreachable)\n'
  exit 1
fi
gpu="$(si_ssh 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' 2>/dev/null | head -1 | clean)"
cuda="$(si_ssh "nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1" 2>/dev/null | clean)"
if [ -n "$gpu" ]; then printf '  ✓ GPU: %s; host %s\n' "$gpu" "${cuda:-CUDA version unknown}"; else printf '  ✗ GPU: nvidia-smi gave nothing\n'; exit 1; fi
c19="$(si_ssh 'if [ -f /root/.ssh/authorized_keys ]; then grep -c . /root/.ssh/authorized_keys; else echo absent; fi' 2>/dev/null | clean)"
case "$c19" in
  absent) printf '  ✓ only our key is accepted (nothing was injected into /root/.ssh)\n' ;;
  *)      printf '  ! RunPod put %s key line(s) into /root/.ssh; our SSH server ignores that file (C19)\n' "$c19" ;;
esac
if [ "${MODE:-}" = serve ]; then
  w="$(si_ssh 'grep -o "WEIGHTS_OK files=[0-9]* bytes=[0-9]*" /var/log/si/boot.log | tail -1' 2>/dev/null | clean)"
  if [ -n "$w" ]; then
    n="$(printf '%s' "$w" | sed 's/.*files=\([0-9]*\).*/\1/')"
    printf '  ✓ model files: all %s match their fingerprints\n' "$n"
  else
    printf '  ✗ model files: not verified (yet)\n'
    exit 1
  fi
fi
