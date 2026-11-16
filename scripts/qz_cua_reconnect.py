#!/usr/bin/env python3
"""Restart this notebook's saved SSH/rtunnel bridge after a QZ CPU restart.

Reuses existing keys and the verified binary. Optional SSH package repair on QZ.
No Docker changes or GPU work.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')
ORIGINAL = ROOT / '.runtime/cua-ssh/cua-test-qi--91bb3a6ab7a7-oxj4j3ju6p'
BINARY_SHA256 = '854e98eefee7f91d2716a9c7e06ce3181c53e4c2fbfd8b22731a0d1233d4695a'
EXPECTED_HOST_KEY = 'SHA256:Eb0g5hAGBL5/nu/DiLpYQYKyic9LBznilSpPTRK/qTQ'


def find_executable(name):
    found = shutil.which(name)
    if found:
        return found
    for directory in ('/usr/bin', '/usr/sbin', '/bin', '/sbin'):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def ensure_openssh(state, install_missing):
    """Run only after the QZ notebook guard and reconnect lock are acquired."""
    executables = {name: find_executable(name) for name in ('ssh-keygen', 'sshd')}
    missing = [name for name, path in executables.items() if path is None]
    if not missing:
        return executables, False
    if not install_missing:
        raise SystemExit('SSH_TOOLS_MISSING: ' + ', '.join(missing)
                         + '; rerun with --run --install-missing-ssh on this QZ CPU')
    apt = find_executable('apt-get')
    if apt is None:
        raise SystemExit('APT_GET_MISSING: SSH packages were not installed')
    package_for = {'ssh-keygen': 'openssh-client', 'sshd': 'openssh-server'}
    packages = [package_for[name] for name in missing]
    cache = state / 'apt-cache'
    (cache / 'archives/partial').mkdir(parents=True, exist_ok=True)
    log_path = state / 'ssh-install.log'
    command = [apt, '-o', f'Dir::Cache={cache}',
               '-o', 'Acquire::Retries=1', '-o', 'Acquire::http::Timeout=20',
               '-o', 'Acquire::https::Timeout=20', '-o', 'DPkg::Lock::Timeout=30']
    env = {**os.environ, 'DEBIAN_FRONTEND': 'noninteractive'}
    print(json.dumps({'status': 'INSTALLING_MISSING_SSH', 'packages': packages,
                      'log': str(log_path)}), flush=True)
    with log_path.open('ab') as log:
        for args in (['update'], ['install', '-y', '--no-install-recommends', *packages]):
            try:
                subprocess.run(command + args, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=log, check=True, timeout=240)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise SystemExit(f'SSH_INSTALL_FAILED: {type(exc).__name__}; log={log_path}') from exc
    executables = {name: find_executable(name) for name in executables}
    if not all(executables.values()):
        raise SystemExit(f'SSH_TOOLS_STILL_MISSING: inspect {log_path}')
    # Only discard package downloads created in this script's private cache.
    for archive in (cache / 'archives').glob('*.deb'):
        archive.unlink()
    return executables, True


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def owned_process(pidfile, required):
    if not pidfile.exists():
        return None
    pid = int(pidfile.read_text().strip())
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        return None
    command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if not all(part in command for part in required):
        raise RuntimeError('Saved PID is now an unrelated process; nothing was killed')
    return pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--install-missing-ssh', action='store_true',
                        help='Install missing OpenSSH tools using apt, only on the guarded QZ CPU')
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({'status': 'PLAN_ONLY', 'ports': [22224, 30004],
                          'gpu_used': False, 'downloads': False, 'reuse_existing_keys': True,
                          'ssh_install_if_missing': args.install_missing_ssh}))
        return 0
    host = platform.node()
    if (platform.system() != 'Linux' or os.geteuid() != 0 or not ROOT.is_dir()
            or '--91bb3a6ab7a7-' not in host):
        raise SystemExit('THIS_QZ_CPU_NOTEBOOK_REQUIRED')
    binary = ORIGINAL / 'fkqz'
    key = ORIGINAL / 'host_key'
    authorized = ORIGINAL / 'authorized_keys'
    if not all(p.is_file() for p in (binary, key, authorized)):
        raise SystemExit('SAVED_BRIDGE_FILES_MISSING: no replacement keys or binaries created')
    if hashlib.sha256(binary.read_bytes()).hexdigest() != BINARY_SHA256:
        raise SystemExit('SAVED_BRIDGE_BINARY_HASH_MISMATCH')
    state = ROOT / '.runtime/cua-ssh' / host
    if state.resolve() != state:
        raise SystemExit('STATE_PATH_ESCAPE')
    os.umask(0o077)
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'reconnect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        executables, ssh_installed = ensure_openssh(state, args.install_missing_ssh)
        pub = subprocess.check_output([executables['ssh-keygen'], '-y', '-f', str(key)])
        fingerprint = subprocess.check_output(
            [executables['ssh-keygen'], '-lf', '-'], input=pub).decode().split()[1]
        if fingerprint != EXPECTED_HOST_KEY:
            raise SystemExit('SAVED_HOST_KEY_MISMATCH')
        sshd = executables['sshd']
        config = state / 'sshd_config'
        ssh_pidfile = state / 'sshd.pid'
        bridge_pidfile = state / 'fkqz.pid'
        ssh_pid = owned_process(ssh_pidfile, ['sshd', str(config)])
        bridge_pid = owned_process(bridge_pidfile, [str(binary), '127.0.0.1:22224', '0.0.0.0:30004'])
        if listening(22224) and ssh_pid is None:
            raise RuntimeError('SSH port is owned by an unverified process; left unchanged')
        if listening(30004) and bridge_pid is None:
            raise RuntimeError('Bridge port is owned by an unverified process; left unchanged')
        if ssh_pid is None:
            runtime = Path('/run/cua-ssh')
            runtime.mkdir(mode=0o700, exist_ok=True)
            auth = runtime / 'authorized_keys'
            if auth.exists() and auth.read_bytes() != authorized.read_bytes():
                raise RuntimeError('Existing runtime authorized keys differ; left unchanged')
            auth.write_bytes(authorized.read_bytes())
            auth.chmod(0o600)
            Path('/run/sshd').mkdir(mode=0o755, exist_ok=True)
            config.write_text('\n'.join([
                'Port 22224', 'ListenAddress 0.0.0.0', f'HostKey {key}',
                f'PidFile {ssh_pidfile}', f'AuthorizedKeysFile {auth}',
                'PermitRootLogin prohibit-password', 'PasswordAuthentication no',
                'KbdInteractiveAuthentication no', 'AuthenticationMethods publickey',
                'PubkeyAuthentication yes', 'UsePAM no', 'AllowUsers root',
                'AllowAgentForwarding no', 'AllowTcpForwarding local', 'X11Forwarding no',
                'Subsystem sftp internal-sftp', '']))
            subprocess.run([sshd, '-t', '-f', str(config)], check=True)
            subprocess.run([sshd, '-f', str(config), '-E', str(state / 'sshd.log')], check=True)
        if bridge_pid is None:
            with (state / 'fkqz.log').open('ab') as log:
                child = subprocess.Popen([str(binary), '127.0.0.1:22224', '0.0.0.0:30004'],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            bridge_pidfile.write_text(str(child.pid) + '\n')
        for _ in range(20):
            if listening(22224) and listening(30004):
                break
            time.sleep(0.25)
        if not (listening(22224) and listening(30004)):
            raise RuntimeError('Bridge readiness failed; saved logs remain in ' + str(state))
        report = {'status': 'CUA_RELAY_READY', 'hostname': host,
                  'host_key_fingerprint': fingerprint, 'ssh_port': 22224, 'forward_port': 30004,
                  'gpu_used': False, 'docker_touched': False,
                  'ssh_packages_installed': ssh_installed, 'downloads': ssh_installed}
        (state / 'reconnect.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
