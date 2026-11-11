#!/usr/bin/env python3
"""Project-owned DinD for the user-confirmed QZ Kata CPU instance; stdlib only.

Default: print a plan without writes. --apply --kata-confirmed starts Docker.
Never formats a block device or an existing image. Does not install software.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import time
import uuid

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')
SOCKET = 'unix:///var/run/docker.sock'
SOCKET_FILE = Path('/var/run/docker.sock')
GIB = 1024 ** 3


def run(argv, *, check=True, timeout=20):
    env = os.environ.copy()
    if argv[0] == 'docker':
        for key in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH'):
            env.pop(key, None)
    result = subprocess.run([str(a) for a in argv], text=True, capture_output=True, timeout=timeout, env=env)
    if check and result.returncode:
        raise RuntimeError(f'{argv[0]} exited {result.returncode}: {result.stderr[-3000:]}')
    return result


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def checked_path(path):
    # Reject symlink escapes, including an existing symlink at a final component.
    if path.resolve() != path or (path != ROOT and ROOT not in path.parents):
        raise RuntimeError(f'Path must remain physically inside the authorized root: {path}')
    return path


def config_for(state):
    return {
        'hosts': [SOCKET], 'data-root': str(state / 'disk/docker'),
        'exec-root': '/run/cua-dind/exec', 'pidfile': '/run/cua-dind/dockerd.pid',
        'storage-driver': 'overlay2', 'features': {'containerd-snapshotter': False},
        'exec-opts': ['native.cgroupdriver=cgroupfs'],
        'ip': '127.0.0.1', 'iptables': True,
        'log-driver': 'json-file', 'log-opts': {'max-size': '10m', 'max-file': '3'},
    }


def create_image(image, size_gib):
    # O_EXCL + format the new regular FILE, never /dev/loopN. Interrupted creation
    # is deliberately not auto-reformatted on retry; preserve it for inspection.
    with image.open('xb') as stream:
        stream.truncate(size_gib * GIB)
    run(['mkfs.ext4', '-F', '-m', '0', image], timeout=120)


def prepare_cgroups():
    cg = Path('/sys/fs/cgroup')
    required = {'cpu', 'memory', 'io', 'pids', 'cpuset'}
    if not required <= set((cg / 'cgroup.controllers').read_text().split()):
        raise RuntimeError('Required cgroup v2 controllers are not available')
    if required <= set((cg / 'cgroup.subtree_control').read_text().split()):
        return
    # QZ manual: move root-cgroup processes to a child before enabling controllers.
    child = cg / 'cua-dind-init'
    child.mkdir(exist_ok=True)
    for pid in (cg / 'cgroup.procs').read_text().split():
        try:
            (child / 'cgroup.procs').write_text(pid)
        except OSError:
            if Path('/proc', pid).exists():
                raise
    (cg / 'cgroup.subtree_control').write_text(' '.join('+' + x for x in sorted(required)))
    if not required <= set((cg / 'cgroup.subtree_control').read_text().split()):
        raise RuntimeError('cgroup controller activation did not complete')


def loop_nodes():
    for path, kind, major, minor in [(Path('/dev/loop-control'), stat.S_IFCHR, 10, 237)] + [
        (Path(f'/dev/loop{i}'), stat.S_IFBLK, 7, i) for i in range(8)
    ]:
        if not path.exists():
            os.mknod(path, kind | 0o600, os.makedev(major, minor))


def docker_info():
    result = run(['docker', '--host', SOCKET, 'info', '--format', '{{json .}}'], check=False, timeout=10)
    return json.loads(result.stdout) if result.returncode == 0 else None


def owned_mount(state, image):
    result = run(['findmnt', '-rn', '-M', state / 'disk', '-o', 'SOURCE,FSTYPE'], check=False)
    if result.returncode:
        return False
    fields = result.stdout.split()
    if len(fields) != 2 or fields[1] != 'ext4' or not fields[0].startswith('/dev/loop'):
        raise RuntimeError('Unexpected existing mount; refusing to replace it')
    backing = run(['losetup', '-n', '-O', 'BACK-FILE', fields[0]]).stdout.strip()
    if Path(backing).resolve() != image:
        raise RuntimeError('Mounted loop does not belong to this project image')
    return True


def apply_setup(state, args, report):
    owner = state / 'owner.json'
    identity = {'hostname': platform.node(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    image = state / 'docker-data.ext4'
    for name in ('owner.json', 'docker-data.ext4', 'disk', 'daemon.json', 'dockerd.log'):
        checked_path(state / name)
    if owner.exists():
        if json.loads(owner.read_text()) != identity:
            raise RuntimeError('Instance identity changed; inspect the previous mount before recovery')
    elif image.exists() or (state / 'disk').exists():
        raise RuntimeError('Unowned existing storage; refusing to adopt or format it')

    info = docker_info()
    expected_root = str(state / 'disk/docker')
    if info:
        if info.get('DockerRootDir') != expected_root or info.get('Driver') != 'overlay2':
            raise RuntimeError('An unrelated Docker daemon is running; it was left unchanged')
        if not owner.exists() or not owned_mount(state, image):
            raise RuntimeError('Daemon storage ownership/mount could not be verified')
        report['daemon_action'] = 'REUSED_VERIFIED_PROJECT_DAEMON'
    else:
        active = run(['pgrep', '-x', 'dockerd'], check=False)
        if active.returncode != 1 or SOCKET_FILE.exists():
            raise RuntimeError('A daemon process/socket already exists but is not healthy; inspect first')
        free = shutil.disk_usage(ROOT).free
        report['workspace_free_bytes_before'] = free
        minimum = 16 if image.exists() else args.size_gib + 16
        if free < minimum * GIB:
            raise RuntimeError(f'Need {minimum} GiB free on the project mount before starting Docker')
        if not owner.exists():
            json_write(owner, identity)
        config = state / 'daemon.json'
        json_write(config, config_for(state))
        run(['dockerd', '--validate', '--config-file', config])
        prepare_cgroups()
        if not image.exists():
            create_image(image, args.size_gib)
            report['image_action'] = 'CREATED_NEW_SPARSE_FILE'
        else:
            report['image_action'] = 'REUSED_NO_FORMAT'
        fs = run(['blkid', '-p', '-s', 'TYPE', '-o', 'value', image]).stdout.strip()
        if fs != 'ext4':
            raise RuntimeError('Existing image is not valid ext4; never auto-reformat')
        if not owned_mount(state, image):
            loop_nodes()
            device = run(['losetup', '--find', '--show', '--nooverlap', '--direct-io=on', image]).stdout.strip()
            report['loop_device'] = device
            (state / 'disk').mkdir(exist_ok=True)
            run(['mount', '-t', 'ext4', device, state / 'disk'])
        if not owned_mount(state, image):
            raise RuntimeError('Project ext4 mount verification failed')
        Path('/run/cua-dind').mkdir(mode=0o700, exist_ok=True)
        env = os.environ.copy()
        for key in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH', 'DOCKER_DRIVER'):
            env.pop(key, None)
        temp = state / 'disk/tmp'
        temp.mkdir(exist_ok=True)
        env['DOCKER_TMPDIR'] = str(temp)
        with (state / 'dockerd.log').open('ab') as log:
            daemon = subprocess.Popen(['dockerd', '--config-file', str(config)], stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        report['daemon_pid'] = daemon.pid
        for _ in range(30):
            if daemon.poll() is not None:
                raise RuntimeError(f'dockerd exited; see {state / "dockerd.log"}')
            info = docker_info()
            if info:
                break
            time.sleep(1)
        if not info:
            raise RuntimeError('Daemon readiness deadline reached; preserve logs and inspect before retry')
        report['daemon_action'] = 'STARTED'
    if info.get('DockerRootDir') != expected_root or info.get('Driver') != 'overlay2':
        raise RuntimeError('Started daemon does not use expected storage')
    report['daemon'] = {k: info.get(k) for k in ('ServerVersion', 'Driver', 'DockerRootDir', 'CgroupVersion', 'NCPU', 'MemTotal')}
    report['status'] = 'DAEMON_READY_CONTAINER_NOT_YET_VERIFIED'
    # A tiny official image verifies create/start/exit; no task data or host mounts.
    name = 'cua-dind-smoke-' + uuid.uuid4().hex[:12]
    report['smoke_container'] = name
    result = run(['docker', '--host', SOCKET, 'run', '--name', name,
                  '--label', 'project=misalign-cua', '--network', 'none', '--cpus', '1',
                  '--memory', '64m', 'hello-world:latest'], check=False, timeout=180)
    report['smoke'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    if result.returncode:
        raise RuntimeError('Daemon is ready, but container pull/run failed; see smoke receipt')
    inspected = run(['docker', '--host', SOCKET, 'inspect', name])
    container = json.loads(inspected.stdout)[0]
    report['smoke']['image_id'] = container['Image']
    report['smoke']['state'] = container['State']
    report['status'] = 'DIND_SMOKE_PASSED_GUI_NOT_RUN'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--kata-confirmed', action='store_true', help='Only in the new QZ development-mode instance')
    parser.add_argument('--size-gib', type=int, choices=range(30, 129), default=64, metavar='30..128')
    args = parser.parse_args(argv)
    if platform.system() != 'Linux':
        print('NOT_QZ_LINUX_NO_CHANGES')
        return 2
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', platform.node()):
        raise RuntimeError('Invalid instance hostname')
    state = ROOT / '.runtime/cua-dind' / platform.node()
    print(json.dumps({'root': str(ROOT), 'state': str(state), 'image_capacity_gib': args.size_gib,
                      'note': 'Sparse capacity, not reserved physical space; GUI/VM/model not installed'}, ensure_ascii=False))
    if not args.apply:
        return 0
    if not args.kata_confirmed or os.geteuid() != 0:
        raise RuntimeError('--apply requires root and --kata-confirmed in the QZ development-mode instance')
    checked_path(state)
    if not ROOT.is_dir() or run(['stat', '-fc', '%T', '/sys/fs/cgroup']).stdout.strip() != 'cgroup2fs':
        raise RuntimeError('Expected project mount or cgroup v2 is missing')
    for executable in ('docker', 'dockerd', 'losetup', 'mkfs.ext4', 'mount', 'findmnt', 'blkid', 'pgrep'):
        if not shutil.which(executable):
            raise RuntimeError(f'Missing {executable}; no software was installed')
    os.umask(0o077)
    state.mkdir(parents=True, exist_ok=True)
    checked_path(state / 'setup.lock')
    with (state / 'setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = {'schema_version': 1, 'time_utc': datetime.now(timezone.utc).isoformat(),
                  'status': 'SETUP_STARTED', 'gui_runtime': 'NOT_RUN',
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        rc = 0
        try:
            apply_setup(state, args, report)
        except Exception as exc:
            report['failed_stage_status'] = report['status']
            report['status'] = 'FAILED_RESOURCES_PRESERVED'
            report['error'] = str(exc)
            rc = 1
        report['workspace_free_bytes_after'] = shutil.disk_usage(ROOT).free
        image = state / 'docker-data.ext4'
        if image.exists():
            image_stat = image.stat()
            report['image_logical_bytes'] = image_stat.st_size
            report['image_allocated_bytes'] = image_stat.st_blocks * 512
        receipt = state / ('attempt-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8] + '.json')
        json_write(receipt, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f'RECEIPT={receipt}')
        return rc


if __name__ == '__main__':
    raise SystemExit(main())
