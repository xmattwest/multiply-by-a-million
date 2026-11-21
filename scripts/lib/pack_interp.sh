# pack_interp.sh: the pod-side interp code as one tar.gz. Sourced by launch_pod.sh (--mode interp) and
# scripts/interp/rehearse_interp.sh, so a rehearsal packs exactly what a launch ships. macOS bash 3.2.
#
#   pack_interp_code ROOT OUT PLUGINS
#     PLUGINS: "all", "none", or comma-separated plug-in names (scripts/interp/tasks_<name>.py) to ship.
#     The other tasks_*.py plug-ins are left out; the core (tasks.py, hooks.py, model_io.py, run_interp.py, ...)
#     always ships. Why: the code travels inside RunPod's create request, which has a size cap (100 KB here).
#     The laptop side picks the list (scripts/launch_interp.py, pod_plugins: the session's plug-in + the
#     tasks_* modules it imports).

pack_interp_code() {
  local root="$1" out="$2" keep="${3:-}" f n ex=()
  case "$keep" in
    all|none) ;;
    *) [[ "$keep" =~ ^[a-z0-9_]+(,[a-z0-9_]+)*$ ]] || { printf 'bad plug-in list: "%s"\n' "$keep" >&2; return 2; } ;;
  esac
  if [ "$keep" != all ]; then
    for n in ${keep//,/ }; do
      [ "$n" = none ] || [ -f "$root/scripts/interp/tasks_$n.py" ] || { printf 'no plug-in scripts/interp/tasks_%s.py\n' "$n" >&2; return 2; }
    done
    for f in "$root"/scripts/interp/tasks_*.py; do
      [ -f "$f" ] || continue
      n="${f##*/tasks_}"; n="${n%.py}"
      case ",$keep," in *",$n,"*) ;; *) ex+=("--exclude=tasks_$n.py") ;; esac
    done
  fi
  ( cd "$root" && COPYFILE_DISABLE=1 tar --numeric-owner --no-mac-metadata --no-xattrs --no-fflags \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' --exclude='*.sh' --exclude='mi_figures*' --exclude='exp_*' \
      ${ex[@]+"${ex[@]}"} -czf "$out" scripts/interp scripts/game_engine.py scripts/render_prompt.py prereg/settings.json )
}
