#!/usr/bin/env python3
"""Resume the existing CUA notebook's ext4 Docker storage after a restart."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import uuid

from scripts import qz_cua_dind as dind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    state = dind.ROOT / '.runtime/cua-dind/cua-test-qi--91bb3a6ab7a7-oxj4j3ju6p'
    if not args.run:
        print(json.dumps({'status': 'PLAN_ONLY', 'reuse': str(state), 'format': False, 'download': False}))
        return 0
    if (platform.system() != 'Linux' or os.geteuid() != 0
            or '--91bb3a6ab7a7-' not in platform.node()):
        raise SystemExit('THIS_QZ_CPU_NOTEBOOK_REQUIRED')
    dind.checked_path(state)
    image = dind.checked_path(state / 'docker-data.ext4')
    config = dind.checked_path(state / 'daemon.json')
    owner = dind.checked_path(state / 'owner.json')
    if not all(p.is_file() for p in (image, config, owner)):
        raise SystemExit('EXISTING_STORAGE_REQUIRED')
    os.umask(0o077)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    receipt = state / f'resume-{attempt}.json'
    report = {'status': 'STARTED', 'stage': 'ownership', 'gpu_used': False,
              'formatted': False, 'downloaded': False, 'previous_owner': json.loads(owner.read_text()),
              'current_owner': {'hostname': platform.node(),
                  'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}}
    with (state / 'setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if '--91bb3a6ab7a7-' not in report['previous_owner'].get('hostname', ''):
                raise RuntimeError('Saved storage owner is a different notebook')
            expected_root = str(state / 'disk/docker')
            saved_config = json.loads(config.read_text())
            if saved_config.get('data-root') != expected_root or saved_config.get('hosts') != [dind.SOCKET]:
                raise RuntimeError('Unexpected saved Docker storage/socket configuration')
            info = dind.docker_info()
            if not info:
                if dind.run(['pgrep', '-x', 'dockerd'], check=False).returncode != 1 or dind.SOCKET_FILE.exists():
                    raise RuntimeError('Unhealthy existing daemon/socket requires inspection')
                report['stage'] = 'cgroups'
                dind.prepare_cgroups()
                report['stage'] = 'mount_existing_image'
                if dind.run(['blkid', '-p', '-s', 'TYPE', '-o', 'value', image]).stdout.strip() != 'ext4':
                    raise RuntimeError('Existing image must be ext4; never auto-format')
                if not dind.owned_mount(state, image):
                    dind.loop_nodes()
                    device = dind.run(['losetup', '--find', '--show', '--nooverlap', '--direct-io=on', image]).stdout.strip()
                    report['loop_device'] = device
                    (state / 'disk').mkdir(exist_ok=True)
                    dind.run(['mount', '-t', 'ext4', device, state / 'disk'])
                if not dind.owned_mount(state, image):
                    raise RuntimeError('Restored mount does not match saved storage')
                report['stage'] = 'start_daemon'
                dind.run(['dockerd', '--validate', '--config-file', config])
                Path('/run/cua-dind').mkdir(mode=0o700, exist_ok=True)
                env = dict(os.environ)
                for key in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH', 'DOCKER_DRIVER'):
                    env.pop(key, None)
                temporary = state / 'disk/tmp'
                temporary.mkdir(exist_ok=True)
                env['DOCKER_TMPDIR'] = str(temporary)
                with (state / f'dockerd-resume-{attempt}.log').open('ab') as log:
                    daemon = subprocess.Popen(['dockerd', '--config-file', str(config)],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, env=env)
                report['daemon_pid'] = daemon.pid
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if daemon.poll() is not None:
                        raise RuntimeError('Docker exited; inspect the saved resume log')
                    try:
                        info = dind.docker_info()
                    except subprocess.TimeoutExpired:
                        report['startup_probe_timeouts'] = report.get('startup_probe_timeouts', 0) + 1
                        info = None
                    if info:
                        break
                    time.sleep(1)
            if not info or info.get('DockerRootDir') != expected_root or info.get('Driver') != 'overlay2':
                raise RuntimeError('Docker does not match saved project storage')
            if not dind.owned_mount(state, image):
                raise RuntimeError('Running Docker mount ownership is unverified')
            report['daemon'] = {k: info.get(k) for k in ('ServerVersion', 'Driver', 'DockerRootDir', 'Images')}
            dind.json_write(owner, report['current_owner'])
            report['stage'] = 'offline_container_check'
            smoke = dind.run(['docker', '--host', dind.SOCKET, 'run', '--rm', '--pull', 'never',
                '--label', 'project=misalign-cua', '--network', 'none', '--cpus', '1',
                '--memory', '64m', 'hello-world:latest'], check=False, timeout=30)
            report['smoke'] = {'returncode': smoke.returncode, 'stdout': smoke.stdout, 'stderr': smoke.stderr}
            if smoke.returncode:
                raise RuntimeError('Offline container check failed; no image was downloaded')
            report['status'] = 'DOCKER_RESUMED_OFFLINE_CHECK_PASSED'
        except Exception:
            report.update(status='FAILED_RESOURCES_PRESERVED', error=traceback.format_exc())
        dind.json_write(receipt, report)
        print(json.dumps(report, indent=2))
        print('RECEIPT=' + str(receipt))
    return 0 if report['status'] == 'DOCKER_RESUMED_OFFLINE_CHECK_PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
