#!/usr/bin/env bash
# Run after Python 3.12 and uv have been prepared on the QZ CPU host.
set -euo pipefail
[[ $(uname -s) == Linux ]] || exit 2
task_root=/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq
task_runtime="$task_root/.runtime/cua-host"
task_project="$task_root/Misalign-agent"
export PATH="$task_runtime/venv/bin:$task_runtime/tools/bin:$PATH"
export VIRTUAL_ENV="$task_runtime/venv" PYTHONDONTWRITEBYTECODE=1
export UV_NO_CONFIG=1 UV_NO_SOURCES=1
export UV_CONSTRAINT="$task_project/configs/cua/cpu_constraints.txt"
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
export UV_HTTP_TIMEOUT=20 UV_HTTP_RETRIES=1 UV_CACHE_DIR="$task_runtime/uv-cache"
export TMPDIR="$task_runtime/tmp" HF_HOME="$task_runtime/hf"
export HF_ENDPOINT=https://huggingface.co XDG_CACHE_HOME="$task_runtime/cache"
export UV_PYTHON_BIN_DIR="$task_runtime/bin"
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY
uv pip install --python "$task_runtime/venv/bin/python" \
  --index-url https://mirror.nju.edu.cn/pytorch/whl/cpu/ \
  -r "$UV_CONSTRAINT"
cd "$task_project/references/cua_lite"
bash lite/gym/envs/lite/osworld/scripts/install.sh provision
