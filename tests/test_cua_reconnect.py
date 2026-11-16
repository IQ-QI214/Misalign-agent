"""Regression coverage for missing SSH tools; never installs local packages."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import qz_cua_reconnect as reconnect


class ReconnectTests(unittest.TestCase):
    def test_existing_tools_do_not_install(self):
        with patch.object(reconnect, 'find_executable', side_effect=lambda name: '/usr/bin/' + name), \
                patch.object(reconnect.subprocess, 'run') as run:
            tools, installed = reconnect.ensure_openssh(Path('/unused'), True)
        self.assertEqual(tools['ssh-keygen'], '/usr/bin/ssh-keygen')
        self.assertFalse(installed)
        run.assert_not_called()

    def test_missing_tools_install_then_recheck(self):
        available = {'apt-get': '/usr/bin/apt-get'}
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if 'install' in command:
                self.assertIn('openssh-client', command)
                self.assertIn('openssh-server', command)
                available.update({'ssh-keygen': '/usr/bin/ssh-keygen', 'sshd': '/usr/sbin/sshd'})

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(reconnect, 'find_executable', side_effect=available.get), \
                patch.object(reconnect.subprocess, 'run', side_effect=run):
            tools, installed = reconnect.ensure_openssh(Path(tmp), True)
        self.assertTrue(installed)
        self.assertEqual(tools['sshd'], '/usr/sbin/sshd')
        self.assertEqual(len(calls), 2)

    def test_install_failure_is_explicit_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(reconnect, 'find_executable', side_effect=lambda n: '/usr/bin/apt-get' if n == 'apt-get' else None), \
                patch.object(reconnect.subprocess, 'run', side_effect=subprocess.CalledProcessError(100, ['apt-get'])):
            with self.assertRaisesRegex(SystemExit, 'SSH_INSTALL_FAILED'):
                reconnect.ensure_openssh(Path(tmp), True)
            self.assertTrue((Path(tmp) / 'ssh-install.log').exists())

    def test_main_checks_missing_tools_before_key_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            original = root / 'saved'
            original.mkdir()
            for name in ('fkqz', 'host_key', 'authorized_keys'):
                (original / name).write_bytes(b'fixture')
            with patch.object(reconnect, 'ROOT', root), \
                    patch.object(reconnect, 'ORIGINAL', original), \
                    patch.object(reconnect, 'BINARY_SHA256', hashlib.sha256(b'fixture').hexdigest()), \
                    patch.object(reconnect.platform, 'system', return_value='Linux'), \
                    patch.object(reconnect.platform, 'node', return_value='test--91bb3a6ab7a7-restarted'), \
                    patch.object(reconnect.os, 'geteuid', return_value=0), \
                    patch.object(reconnect.os, 'umask'), \
                    patch.object(reconnect, 'find_executable', return_value=None), \
                    patch.object(reconnect.subprocess, 'check_output') as inspect_key, \
                    patch('sys.argv', ['qz_cua_reconnect.py', '--run']):
                with self.assertRaisesRegex(SystemExit, 'SSH_TOOLS_MISSING: ssh-keygen, sshd'):
                    reconnect.main()
                inspect_key.assert_not_called()
            self.assertEqual((original / 'host_key').read_bytes(), b'fixture')


if __name__ == '__main__':
    unittest.main()
