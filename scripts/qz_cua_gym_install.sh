#!/usr/bin/env bash
# CPU host dependencies, using the Python and uv already prepared on QZ.
set -euo pipefail
[[ $(uname -s) == Linux ]] || exit 2
task_root=/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq
task_runtime="$task_root/.runtime/cua-host"
task_source="$task_root/Misalign-agent/references/cua_lite"
test -f "$task_source/pyproject.toml"
exec 9>"$task_runtime/gym-install.lock"
flock -n 9 || exit 3
mkdir -p "$task_runtime/tmp"
export TMPDIR="$task_runtime/tmp" PYTHONDONTWRITEBYTECODE=1 UV_HTTP_TIMEOUT=15 UV_HTTP_RETRIES=1
"$task_runtime/tools/bin/uv" pip install --no-cache \
  --default-index https://mirrors.aliyun.com/pypi/simple \
  --python "$task_runtime/venv/bin/python" -e "$task_source[gym]"
"$task_runtime/venv/bin/python" -B -c 'import yaml, docker, fastapi, lite.gym; print("CPU_GYM_IMPORT_PASSED")'
