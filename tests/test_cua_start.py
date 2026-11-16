"""Startup ordering and fail-closed checks; never mount storage or dispatch GPUs."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import qz_cua_start as start


class StartupTests(unittest.TestCase):
    def test_plan_and_wrong_host_do_not_restore(self):
        with patch.object(start, 'restore_cpu') as restore, patch.object(Path, 'mkdir') as mkdir:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(start.main([]), 0)
            with patch.object(start.platform, 'system', return_value='Darwin'):
                with self.assertRaisesRegex(SystemExit, 'THIS_QZ_CPU_NOTEBOOK_REQUIRED'):
                    start.main(['--run'])
            restore.assert_not_called()
            mkdir.assert_not_called()

    def test_failed_disk_restore_cannot_bind_or_probe(self):
        with patch.object(start.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['resume'])), \
                patch.object(start, 'bind_cached_runtime') as bind:
            with self.assertRaises(subprocess.CalledProcessError):
                start.restore_cpu(Path('/project'), Path('/state'), Path('/gate'), {})
            bind.assert_not_called()

    def test_disk_is_restored_before_cache_is_resolved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / 'disk/cua-python-cache-v1'
            qualification = root / 'gate.json'
            qualification.write_text(json.dumps({'status': 'READY_FOR_GPU_MEASUREMENT', 'source_sha256': {}}))
            events = []

            def run(command, **kwargs):
                if 'scripts.qz_cua_dind_resume' in command:
                    events.append('restore')
                    cache.mkdir(parents=True)
                    (cache / 'receipt.json').write_text('{"status":"CACHED_RUNTIME_VERIFIED"}')
                    return subprocess.CompletedProcess(command, 0)
                events.append('probe')
                return subprocess.CompletedProcess(command, 0, '{"cuda": null}')

            def bind(target):
                self.assertTrue((target / 'receipt.json').is_file())
                events.append('bind')
                return target

            with patch.object(start.subprocess, 'run', side_effect=run), \
                    patch.object(start, 'bind_cached_runtime', side_effect=bind), \
                    patch.object(start, 'source_manifest', return_value={}):
                report = {}
                start.restore_cpu(root, root, qualification, report)
            self.assertEqual(events, ['restore', 'bind', 'probe'])
            self.assertEqual(report['status'], 'CPU_ENVIRONMENT_READY')

    def test_missing_cache_after_restore_stops_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(start.subprocess, 'run'), patch.object(start, 'bind_cached_runtime') as bind:
                with self.assertRaisesRegex(RuntimeError, 'CACHE_MISSING_AFTER_STORAGE_RESUME'):
                    start.restore_cpu(Path(folder), Path(folder), Path(folder) / 'gate', {})
                bind.assert_not_called()

    def test_prepare_cpu_returns_without_starting_m1(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(start.platform, 'system', return_value='Linux'), \
                    patch.object(start.platform, 'node', return_value='cua-test-qi--91bb3a6ab7a7-test'), \
                    patch.object(start.os, 'geteuid', return_value=0), \
                    patch.object(start.dind, 'ROOT', Path(folder).resolve()), \
                    patch.object(start, 'restore_cpu') as restore, \
                    patch.object(start.subprocess, 'run') as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(start.main(['--prepare-cpu']), 0)
                restore.assert_called_once()
                run.assert_not_called()

    def test_changed_qualification_is_rejected_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / 'disk/cua-python-cache-v1'
            cache.mkdir(parents=True)
            (cache / 'receipt.json').write_text('{"status":"CACHED_RUNTIME_VERIFIED"}')
            qualification = root / 'gate.json'
            original = '{"status":"READY_FOR_GPU_MEASUREMENT","source_sha256":{"source":"old"}}'
            qualification.write_text(original)
            result = subprocess.CompletedProcess([], 0, '{"cuda": null}')
            with patch.object(start.subprocess, 'run', return_value=result), \
                    patch.object(start, 'bind_cached_runtime', return_value=cache), \
                    patch.object(start, 'source_manifest', return_value={'source': 'changed'}):
                with self.assertRaisesRegex(RuntimeError, 'CPU_QUALIFICATION_MISMATCH'):
                    start.restore_cpu(root, root, qualification, {})
            self.assertEqual(qualification.read_text(), original)


if __name__ == '__main__':
    unittest.main()
