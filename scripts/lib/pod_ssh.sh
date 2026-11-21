# pod_ssh.sh: shared SSH settings for talking to a scale-invariance pod.
# Sourced (not run) by scripts/launch_pod.sh and scripts/connections/03_*, 04_*.
#
# Reads the session file the launcher writes (logs/pod_session.env):
#   POD_ID POD_NAME MODE MODEL SSH_HOST SSH_PORT KNOWN_HOSTS
# The host and port are not secret, but they're not screen-safe either:
# callers never print them.
#
# Every ssh call ignores ~/.ssh/config and the agent, uses only the project
# key, pins the host key printed by the pod at boot, and never forwards the
# agent (infra plan D4).

SI_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SI_SSH_KEY="$HOME/.ssh/scale-invariance_ed25519"

# Slots let several pods run at once (one per terminal). Slot 0, the
# default, keeps the original file names and port. Slot N (1-9) uses
# logs/pod_session.N.env, logs/pod_session.N.stop and port 18000+N.
# The slot comes from SI_SLOT (exported, so child scripts follow it).
si_set_slot() {
  local s="${1:-0}"
  [[ "$s" =~ ^[0-9]$ ]] || { printf 'slot must be 0..9\n' >&2; return 2; }
  SI_SLOT="$s"; export SI_SLOT
  if [ "$s" = 0 ]; then
    SI_SESSION="$SI_ROOT_DIR/logs/pod_session.env"; SI_STOPFILE="$SI_ROOT_DIR/logs/pod_session.stop"
  else
    SI_SESSION="$SI_ROOT_DIR/logs/pod_session.$s.env"; SI_STOPFILE="$SI_ROOT_DIR/logs/pod_session.$s.stop"
  fi
  SI_SLOTFILE="$SI_ROOT_DIR/logs/pod_slot.$s"     # "<launcher pid> <pod name> <worst usd>" while a launch is live
  SI_TUNNEL_PORT=$((18000 + s))
}
si_set_slot "${SI_SLOT:-0}"

si_load_session() {   # returns 1 if there's no live session file
  [ -f "$SI_SESSION" ] || return 1
  # shellcheck disable=SC1090
  . "$SI_SESSION"
  [ -n "${SSH_HOST:-}" ] && [ -n "${SSH_PORT:-}" ] && [ -f "${KNOWN_HOSTS:-/nonexistent}" ]
}

si_ssh_opts() {       # prints nothing; fills the SI_SSH_OPTS array
  SI_SSH_OPTS=(-F /dev/null -i "$SI_SSH_KEY" -o IdentitiesOnly=yes -o IdentityAgent=none
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$KNOWN_HOSTS"
    -o GlobalKnownHostsFile=/dev/null -o ForwardAgent=no -o ForwardX11=no
    -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4
    -o LogLevel=ERROR -p "$SSH_PORT")
}

si_ssh() {            # si_ssh 'remote command'   (hard limit: 120 s)
  si_ssh_opts
  perl -e 'alarm shift; exec @ARGV' 120 ssh "${SI_SSH_OPTS[@]}" "root@$SSH_HOST" "$@"
}
