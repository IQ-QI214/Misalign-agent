#!/usr/bin/env python3
"""Restore the saved CUA CPU runtime before entering the frozen M1 session.

Supports an absolute script path from any working directory. Default is a plan;
--prepare-cpu restores only CPU storage/runtime, and --run requests the bounded
GPU measurement through the existing qualified session.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import qz_cua_dind as dind
from scripts.qz_cua_runtime_cache import bind_cached_runtime
from cua_defense.assets import source_manifest

STATE = dind.ROOT / '.runtime/cua-dind/cua-test-qi--91bb3a6ab7a7-oxj4j3ju6p'


def restore_cpu(project, state, qualification, report):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(project) + ':' + str(project / 'references/cua_lite'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    report['stage'] = 'resume_existing_storage'
    subprocess.run([sys.executable, '-B', '-m', 'scripts.qz_cua_dind_resume', '--run'],
                   cwd=project, env=env, check=True)
    report['stage'] = 'bind_existing_cache'
    cache = state / 'disk/cua-python-cache-v1'
    if not (cache / 'receipt.json').is_file():
        raise RuntimeError('CACHE_MISSING_AFTER_STORAGE_RESUME: inspect saved storage; no cache was recreated')
    cached = json.loads((cache / 'receipt.json').read_text())
    if cached.get('status') != 'CACHED_RUNTIME_VERIFIED':
        raise RuntimeError('CACHED_RUNTIME_NOT_VERIFIED: preserved for inspection')
    runtime = bind_cached_runtime(cache) / 'venv/bin/python'
    report['stage'] = 'cpu_runtime_probe'
    probe = ('import json,sys; import torch,transformers; from PIL import Image; '
             'assert sys.version_info[:2] == (3,12); assert torch.version.cuda is None; '
             'print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,'
             'transformers=transformers.__version__,cuda=torch.version.cuda)))')
    result = subprocess.run([str(runtime), '-B', '-c', probe], cwd=project, env=env,
                            capture_output=True, text=True, check=True, timeout=120)
    report['runtime'] = json.loads(result.stdout)
    report['stage'] = 'check_existing_qualification'
    gate = json.loads(qualification.read_text())
    if (gate.get('status') != 'READY_FOR_GPU_MEASUREMENT'
            or gate.get('source_sha256') != source_manifest(project)):
        raise RuntimeError('CPU_QUALIFICATION_MISMATCH: no qualification was rewritten')
    report['qualification_sha256'] = hashlib.sha256(qualification.read_bytes()).hexdigest()
    report['status'] = 'CPU_ENVIRONMENT_READY'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-cpu', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--cpu-scripted', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--qualification', type=Path,
                        default=PROJECT / 'validation/cua_m1_cpu_ready.json')
    args = parser.parse_args(argv)
    if not (args.prepare_cpu or args.run or args.cpu_scripted):
        print(json.dumps({'status': 'PLAN_ONLY', 'gpu_dispatched': False,
                          'cpu_notebook': 'cua-test-qi (04ab0f8a)',
                          'next': '--prepare-cpu restores existing storage/runtime; --run then enters M1'}))
        return 0
    if args.prepare_cpu and (args.output is not None or args.resume):
        parser.error('--output and --resume apply only to an M1 session')
    if (platform.system() != 'Linux' or os.geteuid() != 0
            or '--91bb3a6ab7a7-' not in platform.node()):
        raise SystemExit('THIS_QZ_CPU_NOTEBOOK_REQUIRED: use cua-test-qi, notebook 04ab0f8a')
    qualification = args.qualification.resolve()
    output = args.output.resolve() if args.output is not None else None
    root = dind.checked_path(dind.ROOT / '.runtime/cua-host/prepared')
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    receipt = root / f'cpu-start-{stamp}-{uuid.uuid4().hex[:8]}.json'
    report = {'status': 'STARTED', 'created_at': datetime.now(timezone.utc).isoformat(),
              'gpu_dispatched_by_bootstrap': False, 'formatted': False, 'downloaded': False,
              'bootstrap_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (root / 'cpu-start.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            restore_cpu(PROJECT, STATE, qualification, report)
        except Exception as error:
            report.update(status='FAILED_RESOURCES_PRESERVED', error=str(error))
            if isinstance(error, subprocess.CalledProcessError):
                report['child_returncode'] = error.returncode
                report['child_stderr_tail'] = (error.stderr or '')[-3000:]
            dind.json_write(receipt, report)
            print(json.dumps(report), flush=True)
            print('RECEIPT=' + str(receipt), flush=True)
            return 1
        dind.json_write(receipt, report)
        print(json.dumps(report), flush=True)
        print('RECEIPT=' + str(receipt), flush=True)
        if args.prepare_cpu:
            return 0
        command = [sys.executable, '-B', '-m', 'scripts.qz_cua_m1_session',
                   '--run' if args.run else '--cpu-scripted', '--qualification', str(qualification)]
        if output is not None:
            command += ['--output', str(output)]
        if args.resume:
            command += ['--resume']
        # The existing session retains qualification, GPU relay and budget checks.
        return subprocess.run(command, cwd=PROJECT, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
