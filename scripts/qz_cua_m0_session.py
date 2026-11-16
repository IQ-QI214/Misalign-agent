#!/usr/bin/env python3
"""Supervise one CPU M0 attempt and record its actual process exit status."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from scripts.qz_cua_dind import ROOT, checked_path, json_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--cached-runtime', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-id', action='append')
    parser.add_argument('--reference-run', type=Path)
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({'status': 'PLAN_ONLY', 'gpu_used': False, 'outer_timeout_seconds': 1500}))
        return 0
    if platform.system() != 'Linux' or '--91bb3a6ab7a7-' not in platform.node():
        raise SystemExit('THIS_QZ_CPU_NOTEBOOK_REQUIRED')
    output = checked_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exit_path = output.with_suffix('.exit.json')
    launch_path = output.with_suffix('.launch.json')
    if output.exists() or exit_path.exists() or launch_path.exists():
        raise SystemExit('NEW_ATTEMPT_PATH_REQUIRED')
    project = Path(__file__).resolve().parents[1]
    runtime = ROOT / '.runtime/cua-host/venv'
    if args.cached_runtime:
        cache = ROOT / '.runtime/cua-dind/cua-test-qi--91bb3a6ab7a7-oxj4j3ju6p/disk/cua-python-cache-v1'
        if json.loads((cache / 'receipt.json').read_text())['status'] != 'CACHED_RUNTIME_VERIFIED':
            raise SystemExit('VERIFIED_RUNTIME_CACHE_REQUIRED')
        from scripts.qz_cua_runtime_cache import bind_cached_runtime
        runtime = bind_cached_runtime(cache) / 'venv'
    env = dict(os.environ, PYTHONPATH=str(project) + ':' + str(project / 'references/cua_lite'),
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        DOCKER_HOST='unix:///var/run/docker.sock')
    command = [str(runtime / 'bin/python'), '-u', '-B', '-X', 'faulthandler',
               '-m', 'scripts.qz_cua_m0', '--run', '--output', str(output)]
    for case_id in args.case_id or []:
        command += ['--case-id', case_id]
    if args.reference_run:
        command += ['--reference-run', str(checked_path(args.reference_run))]
    started = time.monotonic()
    report = {'status': 'STARTED', 'output': str(output), 'runtime': str(runtime),
              'gpu_used': False, 'supervisor_pid': os.getpid(),
              'started_at': datetime.now(timezone.utc).isoformat(), 'outer_timeout_seconds': 1500}
    with output.with_suffix('.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        report['child_pid'] = child.pid
        json_write(launch_path, report)
        try:
            report['returncode'] = child.wait(timeout=1500)
        except subprocess.TimeoutExpired:
            report['outer_timeout'] = True
            child.terminate()
            try:
                report['returncode'] = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                report['returncode'] = child.wait()
    report.update(status='PROCESS_EXITED', seconds=round(time.monotonic() - started, 3),
                  ended_at=datetime.now(timezone.utc).isoformat())
    json_write(exit_path, report)
    print(json.dumps(report), flush=True)
    return 0 if report['returncode'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
