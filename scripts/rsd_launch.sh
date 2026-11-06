#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Entry point only: invoking it explicitly starts one two-GPU job.
# Use `python -m rsd.train ... --dry-run` first to inspect its configuration.
exec torchrun --standalone --nproc-per-node=2 -m rsd.train "$@"
