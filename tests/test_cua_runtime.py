"""CPU tests for the QZ-only native runtime and trainer handoff."""
import asyncio
from pathlib import Path
import tempfile
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(sys.platform == 'linux', 'QZ Linux runtime required')
class RuntimeContracts(unittest.TestCase):
    def test_final_file_is_read_before_native_desktop_close(self):
        from cua_defense.runtime import VerifiedTaskEnv
        class Fake:
            def __init__(self):
                self._computer=SimpleNamespace(interface=self)
                self.reads=0
            async def read_bytes(self,path):
                assert self._computer is not None
                self.reads+=1
                return b'finished'
            async def close(self): self._computer=None
        fake=Fake();wrapped=VerifiedTaskEnv(fake,'/synthetic',b'finished')
        asyncio.run(wrapped.close());asyncio.run(wrapped.close())
        self.assertEqual(wrapped.final_task_score,1)
        self.assertEqual(fake.reads,1)

    def test_resource_setup_retries_only_after_closing_owned_environment(self):
        from cua_defense.runtime import TaskResourceEnv
        class Fake:
            def __init__(self):
                self.resets = self.closes = 0
                self._computer = SimpleNamespace(interface=self)
            async def reset(self):
                self.resets += 1
                return 'observation'
            async def close(self):
                self.closes += 1
            async def run_command(self, command):
                return SimpleNamespace(returncode=0 if self.resets == 2 else 1)
        fake = Fake()
        wrapped = TaskResourceEnv(fake, ['/expected'])
        self.assertEqual(asyncio.run(wrapped.reset()), 'observation')
        self.assertEqual((fake.resets, fake.closes, wrapped.setup_attempts), (2, 1, 2))

    def test_milestone_requires_result_and_restores_training_mode(self):
        from cua_defense.milestones import MilestoneEvaluation
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            modes = []
            model = SimpleNamespace(eval=lambda: modes.append('eval'), train=lambda: modes.append('train'))
            callback = MilestoneEvaluation(root, None, [0], 1)
            control = SimpleNamespace(should_training_stop=False)
            def server(model, processor, directory, **kwargs):
                (directory / 'evaluation_result.json').write_text('{}')
            with patch('cua_defense.milestones.serve_requests', side_effect=server):
                callback.evaluate(SimpleNamespace(global_step=0), control, model)
                callback.evaluate(SimpleNamespace(global_step=0), control, model)
            self.assertEqual(callback.completed, {0})
            self.assertEqual(modes, ['eval', 'train'])
            self.assertFalse(control.should_training_stop)
            self.assertTrue((root / 'step-000000/stopped.json').exists())

    def test_absolute_inference_deadline(self):
        from cua_defense.inference import serve_requests
        import time
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'request-0.json').write_text('{}')
            with patch('cua_defense.inference.answer_request') as answer:
                serve_requests(None, None, root, deadline=time.monotonic()-1)
                answer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
