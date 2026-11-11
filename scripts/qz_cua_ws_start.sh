#!/usr/bin/env bash
set -euo pipefail
task_root=/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq
[[ $(uname -s) == Linux && $(hostname) == cua-test-qi--91bb3a6ab7a7-oxj4j3ju6p ]] || exit 2
task_state="$task_root/.runtime/cua-ssh/$(hostname)"
task_source=$(cd "$(dirname "$0")" && pwd)/qz_cua_ws_bridge.py
[[ -d "$task_state" && -f "$task_source" ]] || exit 2
umask 077
exec 9>"$task_state/ws-setup.lock"
flock -n 9 || exit 3
if [[ -f "$task_state/ws.pid" ]] && kill -0 "$(cat "$task_state/ws.pid")" 2>/dev/null; then
  echo 'Existing bridge process; inspect before starting another.'
  exit 0
fi
if ! python3 -m pip --version >/dev/null 2>&1; then
  apt-get update >"$task_state/ws-install.log" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip >>"$task_state/ws-install.log" 2>&1
fi
PIP_CACHE_DIR="$task_state/pip-cache" python3 -m pip install --target "$task_state/ws-deps" websockets==15.0.1 >"$task_state/ws-pip.log" 2>&1
nohup env PYTHONPATH="$task_state/ws-deps" python3 -B "$task_source" >"$task_state/ws.log" 2>&1 9>&- &
echo "$!" >"$task_state/ws.pid"
sleep 1
kill -0 "$(cat "$task_state/ws.pid")"
cat "$task_state/ws.log"
