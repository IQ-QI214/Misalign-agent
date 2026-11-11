#!/usr/bin/env python3
"""Read-only QZ/Kata/DinD checks. No install, daemon start, mount, or format."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


WRITE_BOUNDARY = Path('/inspire/hdd/global_user/wenming-253108090054/zhujiaqi')
DEFAULT_PROJECT = WRITE_BOUNDARY / 'codex-workspace/Misalign-agent'


def command(argv):
    """Retain missing commands and timeouts as observations, never success."""
    if not shutil.which(argv[0]):
        return {'status': 'MISSING_COMMAND', 'argv': argv, 'returncode': None}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=8)
    except subprocess.TimeoutExpired:
        return {'status': 'TIMEOUT', 'argv': argv, 'returncode': None}
    except OSError as exc:
        return {'status': 'ERROR', 'argv': argv, 'returncode': None, 'error': str(exc)}
    return {
        'status': 'OK' if result.returncode == 0 else 'ERROR',
        'argv': argv, 'returncode': result.returncode,
        'stdout': result.stdout.strip()[:12000],
        'stderr': result.stderr.strip()[:3000],
    }


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {'status': 'UNAVAILABLE', 'reason': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=DEFAULT_PROJECT)
    args = parser.parse_args()
    report = {
        'schema_version': 1, 'probe': 'qz_cua_dind_preflight_v1',
        'scope': 'READ_ONLY_NO_INSTALL_NO_GPU',
        'platform': platform.system(), 'architecture': platform.machine(),
        'kata_mode': 'REQUIRES_PLATFORM_CONFIRMATION',
    }
    if platform.system() != 'Linux':
        report['status'] = 'NOT_QZ_LINUX_NO_CHECKS_EXECUTED'
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    project = args.project_root.resolve()
    boundary = WRITE_BOUNDARY.resolve()
    if project != boundary and boundary not in project.parents:
        parser.error('--project-root must be under the authorized QZ workspace')
    report['uid'] = os.geteuid()
    report['kernel'] = platform.release()
    report['project_root'] = str(project)
    report['project_exists'] = project.is_dir()
    report['cgroup'] = {
        'mount': command(['findmnt', '-T', '/sys/fs/cgroup', '-o', 'TARGET,FSTYPE,OPTIONS']),
        'controllers': read('/sys/fs/cgroup/cgroup.controllers'),
        'subtree_control': read('/sys/fs/cgroup/cgroup.subtree_control'),
        'memory_max': read('/sys/fs/cgroup/memory.max'),
        'cpu_max': read('/sys/fs/cgroup/cpu.max'),
    }
    report['commands'] = {name: shutil.which(name) for name in (
        'docker', 'dockerd', 'dockerd-entrypoint.sh', 'losetup', 'mkfs.ext4',
        'mount', 'findmnt', 'truncate', 'mknod', 'git', 'python3', 'uv',
    )}
    report['storage'] = {
        'system': command(['df', '-h', '/']),
        'workspace': command(['df', '-h', str(boundary)]),
        'workspace_fs': command(['findmnt', '-T', str(boundary), '-o', 'TARGET,FSTYPE,OPTIONS']),
        'docker_mount': command(['findmnt', '-T', '/var/lib/docker', '-o', 'TARGET,SOURCE,FSTYPE']),
        'loop_devices': command(['losetup', '--list', '--output', 'NAME,BACK-FILE']),
    }
    # Probe only the local socket; ignore an unrelated context or remote endpoint.
    docker = ['docker', '--host', 'unix:///var/run/docker.sock']
    report['docker_version'] = command(docker + ['version', '--format', '{{json .}}'])
    report['docker_daemon'] = command(docker + [
        'info', '--format',
        '{"server_version":{{json .ServerVersion}},"driver":{{json .Driver}},'
        '"data_root":{{json .DockerRootDir}},"cgroup_version":{{json .CgroupVersion}},'
        '"cpus":{{json .NCPU}},"memory_bytes":{{json .MemTotal}}}',
    ])
    report['status'] = 'COLLECTED_NOT_A_RUNTIME_PASS'
    report['next_step'] = (
        'Inspect this report before starting a project-owned daemon. '
        'No filesystem, loop device or daemon was changed by this script.'
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
