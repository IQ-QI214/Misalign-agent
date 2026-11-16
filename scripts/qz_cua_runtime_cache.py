#!/usr/bin/env python3
"""Cache the existing CPU Python runtime on the verified project ext4 mount.

Copies installed code, not models; leaves the original environment untouched.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

from scripts import qz_cua_dind as dind


def bind_cached_runtime(target):
    """Keep runtime lookups out of the slow FUSE parent directory chain."""
    alias = Path('/run/cua-dind/python-cache')
    found = dind.run(['findmnt', '-rn', '-M', alias], check=False)
    if found.returncode:
        if alias.exists() and any(alias.iterdir()):
            raise RuntimeError('Runtime alias directory is not empty')
        alias.mkdir(parents=True, exist_ok=True, mode=0o700)
        dind.run(['mount', '--bind', target, alias])
    actual, expected = alias.stat(), target.stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError('Runtime alias does not refer to this cache')
    return alias


def repair_record_files(source, target):
    """Verify local copies against wheel RECORD; copy only missing/bad files."""
    source_site = source / 'lib/python3.12/site-packages'
    cache_venv = target / 'venv'
    jobs = {}
    source_verified = []
    verified = 0
    for record in source_site.glob('*.dist-info/RECORD'):
        with record.open() as stream:
            for name, expected, size in csv.reader(stream):
                if not expected:
                    continue
                relative = os.path.normpath('lib/python3.12/site-packages/' + name)
                if relative.startswith('../') or os.path.isabs(relative):
                    raise RuntimeError('RECORD path escapes the venv')
                jobs[relative] = expected
    def matches(path, expected):
        try:
            algorithm, encoded = expected.split('=', 1)
            h = hashlib.new(algorithm)
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            return base64.urlsafe_b64encode(h.digest()).decode().rstrip('=') == encoded
        except FileNotFoundError:
            return False
    previous_cwd = Path.cwd()
    # Relative local paths avoid repeated permission checks through FUSE parents.
    os.chdir(cache_venv)
    try:
        missing = [(name, expected) for name, expected in jobs.items() if not matches(Path(name), expected)]
        verified = len(jobs) - len(missing)
        print(json.dumps({'record_verified': verified, 'record_files_to_repair': len(missing)}), flush=True)
        def copy(job):
            name, expected = job
            dst = Path(name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, dst)
            if not matches(dst, expected):
                # Overlapping distributions can leave stale RECORD entries.
                # The installed source is the runtime we must preserve.
                actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
                if hashlib.sha256(dst.read_bytes()).hexdigest() != actual:
                    raise RuntimeError('Copied installed-source hash differs: ' + name)
                source_verified.append({'path': name, 'installed_source_sha256': actual})
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(copy, missing))
    finally:
        os.chdir(previous_cwd)
    return {'hashed_files_verified': verified + len(missing), 'files_repaired': len(missing),
            'record_disagreements_verified_against_installed_source': source_verified}


def copy_runtime(base, source, target):
    """Parallelize disjoint package directories on the metadata-limited source."""
    jobs = []
    def collect(src, dst, split):
        dst.mkdir(parents=True, exist_ok=True)
        for entry in src.iterdir():
            if entry.name in split and entry.is_dir() and not entry.is_symlink():
                collect(entry, dst / entry.name, split[entry.name])
            else:
                jobs.append((entry, dst))
    collect(base, target / 'python', {'lib': {'python3.12': {}}})
    collect(source, target / 'venv', {'lib': {'python3.12': {'site-packages': {}}}})
    def copy(job):
        src, dst = job
        subprocess.run(['cp', '-a', str(src), str(dst)], check=True, timeout=600)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy, jobs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume-incomplete', action='store_true')
    parser.add_argument('--repair-records', action='store_true',
                        help='Complete a partial venv copy using installed wheel checksums')
    args = parser.parse_args()
    if not args.run:
        print('PLAN_ONLY: copy existing CPU Python and venv to the project ext4 mount')
        return 0
    if (platform.system() != 'Linux' or os.geteuid() != 0
            or '--91bb3a6ab7a7-' not in platform.node()):
        raise SystemExit('THIS_QZ_CPU_NOTEBOOK_REQUIRED')
    state = dind.ROOT / '.runtime/cua-dind/cua-test-qi--91bb3a6ab7a7-oxj4j3ju6p'
    if not dind.owned_mount(state, state / 'docker-data.ext4'):
        raise SystemExit('VERIFIED_EXT4_MOUNT_REQUIRED')
    source = dind.ROOT / '.runtime/cua-host/venv'
    base = (source / 'bin/python').resolve().parents[1]
    base.relative_to(dind.ROOT / '.runtime/cua-host/python')
    target = dind.checked_path(state / 'disk/cua-python-cache-v1')
    with (state / 'runtime-cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            previous = json.loads((target / 'receipt.json').read_text())
            if (not args.resume_incomplete or previous.get('status') != 'FAILED_CACHE_PRESERVED'
                    or previous.get('source') != str(source)):
                raise SystemExit('CACHE_ALREADY_EXISTS: inspect the existing receipt before reuse')
            dind.json_write(target / ('previous-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json'), previous)
        if shutil.disk_usage(state / 'disk').free < 5 * 2**30:
            raise SystemExit('NEED_5_GIB_FREE_FOR_RUNTIME_CACHE')
        target.mkdir(mode=0o700, exist_ok=True)
        report = {'status': 'COPYING', 'source': str(source), 'target': str(target),
                  'gpu_used': False, 'model_weights_copied': False, 'copy_workers': 8,
                  'started_at': datetime.now(timezone.utc).isoformat()}
        dind.json_write(target / 'receipt.json', report)
        started = time.monotonic()
        try:
            if args.repair_records:
                if not args.resume_incomplete or not (target / 'python/bin/python3.12').exists():
                    raise RuntimeError('Record repair requires an existing partial cache and Python base')
                report['record_check'] = repair_record_files(source, target)
            else:
                print('COPYING_EXISTING_RUNTIME workers=8', flush=True)
                copy_runtime(base, source, target)
            for name in ('python', 'python3', 'python3.12'):
                link = target / 'venv/bin' / name
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to('../../python/bin/python3.12')
            alias = bind_cached_runtime(target)
            report['runtime_alias'] = str(alias)
            cfg = target / 'venv/pyvenv.cfg'
            cfg.write_text('\n'.join('home = ' + str(alias / 'python/bin')
                if line.startswith('home = ') else line for line in cfg.read_text().splitlines()) + '\n')
            checked = ['torch/__init__.py', 'transformers/__init__.py', 'PIL/Image.py']
            report['source_sha256'] = {}
            for relative in checked:
                src = source / 'lib/python3.12/site-packages' / relative
                dst = target / 'venv/lib/python3.12/site-packages' / relative
                digest = hashlib.sha256(src.read_bytes()).hexdigest()
                if hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Copied source hash differs: ' + relative)
                report['source_sha256'][relative] = digest
            report['copy_seconds'] = round(time.monotonic() - started, 3)
            print('CHECKING_CACHED_IMPORTS', flush=True)
            probe = ('import time,json,sys; t=time.monotonic(); import torch,transformers; '
                     'from transformers import AutoProcessor; '
                     'print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,'
                     'transformers=transformers.__version__,prefix=sys.prefix,seconds=time.monotonic()-t)))')
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
            result = subprocess.run([str(alias / 'venv/bin/python'), '-B', '-c', probe],
                env=env, capture_output=True, text=True, timeout=120, check=True)
            report['import_check'] = json.loads(result.stdout)
            if report['import_check']['cuda'] is not None:
                raise RuntimeError('Expected the existing CPU-only Torch runtime')
            report['allocated_bytes'] = int(subprocess.check_output(['du', '-sx', '-B1', str(target)], text=True).split()[0])
            report['status'] = 'CACHED_RUNTIME_VERIFIED'
        except Exception as error:
            report.update(status='FAILED_CACHE_PRESERVED', error=repr(error))
        dind.json_write(target / 'receipt.json', report)
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'CACHED_RUNTIME_VERIFIED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
