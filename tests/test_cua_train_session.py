"""CPU launch-plan checks using explicit synthetic fixtures and no GPU calls."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(sys.platform == 'linux', 'QZ Linux runtime required')
class TrainingSessionContracts(unittest.TestCase):
    def check_plan(self, metadata):
        import torch
        from scripts import qz_cua_train_session as session
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'fixture.parquet'
            data.write_bytes(b'explicit CPU test fixture')
            table = SimpleNamespace(to_pylist=lambda: [{'metadata': json.dumps(metadata)}])
            example = {'input_ids': torch.zeros((1, 8), dtype=torch.long),
                       'labels': torch.tensor([[-100]*5+[1,2,3]])}
            args = ['session', '--data', str(data), '--eval-config', str(session.PROJECT / 'configs/cua/milestone_gpu.json'),
                    '--output', str(root / 'output')]
            with patch.object(sys, 'argv', args), patch('pyarrow.parquet.read_table', return_value=table), \
                 patch('transformers.AutoProcessor.from_pretrained', return_value=object()), \
                 patch('cua_defense.training.StepDataset', return_value=[example]), \
                 patch('subprocess.run') as launch, contextlib.redirect_stdout(io.StringIO()) as captured:
                session.main()
                launch.assert_not_called()
                self.assertFalse((root / 'output').exists())
                return json.loads(captured.getvalue())

    def test_native_metadata_id_is_accepted_in_plan_only(self):
        report = self.check_plan({'others': {'research_training_eligible': True, 'task_id': 'fixture_training_task'}})
        self.assertEqual(report['supervised_examples'], 1)
        self.assertEqual(report['unique_target_tokens'], 3)
        self.assertFalse(report['gpu_dispatched'])

    def test_evaluation_overlap_is_rejected(self):
        from scripts import qz_cua_train_session as session
        manifest = json.loads((session.PROJECT / 'configs/cua/milestone_tasks_v1.json').read_text())
        task_id = manifest['tasks'][0]['task_id']
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.check_plan({'others': {'research_training_eligible': True, 'task_id': task_id}})

    def test_diagnostic_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'diagnostic'):
            self.check_plan({'others': {'research_training_eligible': False, 'task_id': 'fixture'}})


if __name__ == '__main__':
    unittest.main()
