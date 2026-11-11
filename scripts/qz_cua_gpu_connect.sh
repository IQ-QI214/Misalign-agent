#!/usr/bin/env bash
# Run once in the GPU terminal after allocating an instance.
set -euo pipefail
umask 077
task_base=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi
task_project="$task_base/codex-workspace/Misalign-agent"
task_relay="$task_project/.runtime/gpu-relay-6f159838"
task_worker="$task_base/gpu-relay/gpu-worker.sh"
task_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "$task_count" -ge 1 ]] || { echo 'No GPU is visible' >&2; exit 1; }
[[ -x "$task_worker" ]] || { echo 'Existing GPU relay worker is missing' >&2; exit 1; }
mkdir -p "$task_relay/log" "$task_relay/state"
if [[ -f "$task_relay/state/ready" ]]; then
  task_pid=$(cat "$task_relay/state/ready")
  if [[ "$(cat "$task_relay/state/hostname")" == "$(hostname)" ]] &&
     [[ "$task_pid" =~ ^[0-9]+$ ]] && kill -0 "$task_pid" 2>/dev/null &&
     tr '\0' ' ' <"/proc/$task_pid/cmdline" | grep -Fq "$task_worker"; then
    echo "MISALIGN_RELAY_READY host=$(hostname) pid=$task_pid"; exit 0
  fi
fi
GPU_RELAY_DIR="$task_relay" nohup "$task_worker" >>"$task_relay/log/worker.log" 2>&1 </dev/null &
task_pid=$!
for task_try in {1..30}; do
  if [[ "$(cat "$task_relay/state/ready" 2>/dev/null || true)" == "$task_pid" ]] && kill -0 "$task_pid" 2>/dev/null; then
    echo "MISALIGN_RELAY_READY host=$(hostname) pid=$task_pid"; exit 0
  fi
  sleep 0.2
done
tail -n 15 "$task_relay/log/worker.log"
exit 1
