#!/usr/bin/env bash
# CPU terminal entry point. GPU work starts only with explicit --run.
set -euo pipefail
cua_root=/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq
cua_project="$cua_root/Misalign-agent"
cd "$cua_project"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$cua_project:$cua_project/references/cua_lite"
exec "$cua_root/.runtime/cua-host/venv/bin/python" -B scripts/qz_cua_batch.py "$@"
