#!/bin/bash
# 01_runpod.sh: connection check 1. Does RunPod's API accept our key?
#
# Sends one read-only request (list pods), so it costs nothing and changes
# nothing. Prints three screen-safe lines (no key, no IDs, no names).
# For the full account audit (pods, volumes, billing), run:
#   bash scripts/runpod_auth_check.sh
exec bash "$(cd "$(dirname "$0")" && pwd)/../runpod_auth_check.sh" --brief
