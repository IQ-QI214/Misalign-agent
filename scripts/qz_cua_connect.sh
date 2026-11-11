#!/usr/bin/env bash
# Run once in the new QZ CPU instance, passing the operator's existing PUBLIC key.
set -euo pipefail
task_root=/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq
[[ $(uname -s) == Linux && $(id -u) == 0 && -d "$task_root" ]] || exit 2
[[ $(hostname) == cua-test-qi--91bb3a6ab7a7-oxj4j3ju6p ]] || { echo 'Unexpected instance'; exit 2; }
task_key=${1:?Pass the existing SSH public key}
[[ "$task_key" == ssh-ed25519\ * && "$task_key" != *$'\n'* ]] || exit 2
task_state="$task_root/.runtime/cua-ssh/$(hostname)"
umask 077
mkdir -p "$task_state"
[[ $(realpath "$task_state") == "$task_state" ]] || exit 2
exec 9>"$task_state/setup.lock"
flock -n 9 || exit 3
if ! command -v sshd >/dev/null && [[ ! -x /usr/sbin/sshd ]]; then
  mkdir -p "$task_state/apt-cache/archives/partial"
  apt-get -o "Dir::Cache=$task_state/apt-cache" update >"$task_state/install.log" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get -o "Dir::Cache=$task_state/apt-cache" \
    install -y --no-install-recommends openssh-server >>"$task_state/install.log" 2>&1
fi
task_sshd=$(command -v sshd || echo /usr/sbin/sshd)
printf '%s\n' "$task_key" >"$task_state/authorized_keys"
ssh-keygen -lf "$task_state/authorized_keys" >/dev/null
mkdir -p /run/cua-ssh
chmod 700 /run/cua-ssh
install -m 600 "$task_state/authorized_keys" /run/cua-ssh/authorized_keys
if [[ ! -e "$task_state/host_key" ]]; then
  ssh-keygen -q -t ed25519 -N '' -f "$task_state/host_key"
fi
mkdir -p /run/sshd
cat >"$task_state/sshd_config" <<EOF
Port 22224
ListenAddress 0.0.0.0
HostKey $task_state/host_key
PidFile $task_state/sshd.pid
AuthorizedKeysFile /run/cua-ssh/authorized_keys
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
AuthenticationMethods publickey
PubkeyAuthentication yes
UsePAM no
AllowUsers root
AllowAgentForwarding no
AllowTcpForwarding local
X11Forwarding no
Subsystem sftp internal-sftp
EOF
"$task_sshd" -t -f "$task_state/sshd_config"
if [[ -f "$task_state/sshd.pid" ]] && kill -0 "$(cat "$task_state/sshd.pid")" 2>/dev/null; then
  task_pid=$(cat "$task_state/sshd.pid")
  tr '\0' ' ' <"/proc/$task_pid/cmdline" | grep -Fq "$task_state/sshd_config" || exit 4
else
  "$task_sshd" -f "$task_state/sshd_config" -E "$task_state/sshd.log"
fi
echo 'CUA_SSH_READY port=22224'
hostname -I
ssh-keygen -lf "$task_state/host_key.pub"
echo "SSH_LOG=$task_state/sshd.log"
