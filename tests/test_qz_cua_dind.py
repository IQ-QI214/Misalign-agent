"""Storage guard tests. Standard library; never invokes Docker, mount or mkfs."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('qz_dind', Path(__file__).parents[1] / 'scripts/qz_cua_dind.py')
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class DindGuards(unittest.TestCase):
    def test_mac_apply_cannot_write_or_run_commands(self):
        with patch.object(D.platform, 'system', return_value='Darwin'), patch.object(D, 'run') as run, patch.object(Path, 'mkdir') as mkdir:
            self.assertEqual(D.main(['--apply', '--kata-confirmed']), 2)
        run.assert_not_called()
        mkdir.assert_not_called()

    def test_plan_cannot_write_or_run_commands(self):
        with patch.object(D.platform, 'system', return_value='Linux'), patch.object(D, 'run') as run, patch.object(Path, 'mkdir') as mkdir:
            self.assertEqual(D.main([]), 0)
        run.assert_not_called()
        mkdir.assert_not_called()

    def test_existing_image_never_formatted(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'existing.ext4'
            image.write_bytes(b'preserve-this')
            with patch.object(D, 'run') as run:
                with self.assertRaises(FileExistsError):
                    D.create_image(image, 64)
            run.assert_not_called()
            self.assertEqual(image.read_bytes(), b'preserve-this')

    def test_new_image_formats_only_the_exclusively_created_file(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'new.ext4'
            with patch.object(D, 'GIB', 1024), patch.object(D, 'run') as run:
                D.create_image(image, 1)
            self.assertTrue(image.is_file())
            self.assertEqual(image.stat().st_size, 1024)
            run.assert_called_once_with(['mkfs.ext4', '-F', '-m', '0', image], timeout=120)

    def test_foreign_loop_mount_is_not_reused(self):
        outputs = [subprocess.CompletedProcess([], 0, stdout='/dev/loop3 ext4\n'),
                   subprocess.CompletedProcess([], 0, stdout='/foreign/data.img\n')]
        with patch.object(D, 'run', side_effect=outputs):
            with self.assertRaisesRegex(RuntimeError, 'does not belong'):
                D.owned_mount(D.ROOT, D.ROOT / 'docker-data.ext4')

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / 'escape').symlink_to('/tmp')
            with patch.object(D, 'ROOT', root):
                with self.assertRaises(RuntimeError):
                    D.checked_path(root / 'escape/file')

    def test_unrelated_daemon_is_not_modified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            info = {'DockerRootDir': '/someone/else', 'Driver': 'overlay2'}
            with patch.object(D, 'ROOT', root), patch.object(D, 'docker_info', return_value=info), patch.object(Path, 'read_text', return_value='mock-boot'), patch.object(D, 'run') as run, patch.object(D, 'prepare_cgroups') as cg, patch.object(D, 'create_image') as create:
                with self.assertRaisesRegex(RuntimeError, 'unrelated Docker'):
                    D.apply_setup(root, SimpleNamespace(size_gib=64), {})
            run.assert_not_called()
            cg.assert_not_called()
            create.assert_not_called()

    def test_low_space_does_not_create_storage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with patch.object(D, 'ROOT', root), patch.object(D, 'SOCKET_FILE', root / 'absent.sock'), patch.object(D, 'docker_info', return_value=None), patch.object(Path, 'read_text', return_value='mock-boot'), patch.object(D, 'run', return_value=subprocess.CompletedProcess([], 1)), patch.object(D.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * D.GIB)), patch.object(D, 'prepare_cgroups') as cg, patch.object(D, 'create_image') as create:
                with self.assertRaisesRegex(RuntimeError, 'Need 80 GiB'):
                    D.apply_setup(root, SimpleNamespace(size_gib=64), {})
            self.assertFalse((root / 'owner.json').exists())
            cg.assert_not_called()
            create.assert_not_called()

    def test_docker_config_keeps_persistent_storage_inside_root(self):
        config = D.config_for(D.ROOT / '.runtime/cua-dind/instance')
        self.assertEqual(config['hosts'], ['unix:///var/run/docker.sock'])
        self.assertTrue(config['data-root'].startswith(str(D.ROOT) + '/'))
        self.assertFalse(config['features']['containerd-snapshotter'])
        self.assertEqual(config['storage-driver'], 'overlay2')
        self.assertEqual(config['ip'], '127.0.0.1')


if __name__ == '__main__':
    unittest.main()
