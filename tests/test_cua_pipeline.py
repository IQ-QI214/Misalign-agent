"""CPU regression checks for data eligibility, mailbox ownership, and replay."""
import asyncio
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

from cua_defense.events import EventMonitor
from cua_defense.inference import serve_requests
from cua_defense.mailbox import MailboxClient, teacher_prompt
from cua_defense.replay import file_fingerprint
from cua_defense.tasks import ATTACKS, variant
from scripts.qz_cua_export_sft import eligible


class PipelineContracts(unittest.TestCase):
    def test_owned_worker_launch_has_independent_wall_limit(self):
        from scripts import qz_cua_batch as batch
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            heartbeat = project / '.runtime/gpu-relay-6f159838/state/heartbeat'
            heartbeat.parent.mkdir(parents=True)
            heartbeat.write_text(str(time.time()))
            with patch.object(batch, 'SHARED_PROJECT', project), patch.object(batch, 'SHARED', project / 'shared'), \
                 patch.object(batch.subprocess, 'run') as execute:
                batch.launch_worker({'gpu_model_path': '/fixture/weights'}, project / 'mailbox', budget_seconds=60)
                command = execute.call_args.args[0][-1]
                self.assertIn('timeout --signal=TERM --kill-after=30s 60s', command)
                self.assertIn('CUDA_VISIBLE_DEVICES=0', command)

    def test_risk_variants_share_authorized_goal_and_evaluator(self):
        row = {'task_id': 'fixture', 'instruction': 'Edit the title.',
               'metadata': {'config': [], 'evaluator': {'func': 'fixture'}, 'others': {}}}
        variants = [variant(row, family) for family in ATTACKS]
        self.assertEqual(len({v['instruction'] for v in variants}), 1)
        self.assertTrue(all(v['metadata']['evaluator'] == row['metadata']['evaluator'] for v in variants))
        self.assertEqual(row['metadata']['config'], [])
        self.assertNotEqual(variants[0]['task_id'], variants[1]['task_id'])

    def test_teacher_feedback_keeps_student_prompt_suffix(self):
        prompt = '<|im_start|>system\nRules<|im_end|>\n<|im_start|>user\n<|image_pad|><|im_end|>\n<|im_start|>assistant\n'
        changed = teacher_prompt(prompt, 'Past execution failed.')
        self.assertEqual(changed[changed.index('<|im_end|>'):], prompt[prompt.index('<|im_end|>'):])
        self.assertEqual(teacher_prompt(prompt, ''), prompt)
        with self.assertRaises(ValueError):
            teacher_prompt('unknown template', 'feedback')

    def test_unsafe_or_unknown_episodes_cannot_be_positives(self):
        row = {'data_role': 'development', 'model_run': True, 'execution_valid': True, 'task_score': 1,
               'condition': 'maintenance', 'safe_completion': True, 'risk': {'coverage_complete': True}}
        self.assertTrue(eligible(row))
        for update in [{'data_role': 'evaluation'}, {'data_role': None}, {'model_run': False}, {'execution_valid': False}, {'task_score': 0},
                       {'safe_completion': None}, {'risk': {'coverage_complete': False}}]:
            self.assertFalse(eligible({**row, **update}))

    def test_monitor_uncertainty_and_known_violation(self):
        monitor = EventMonitor('owned', '/owned', 'sha')
        self.assertIsNone(monitor.summary()['violation'])
        monitor.events = [{'type': 'violation'}, {'type': 'monitor_error'}]
        self.assertTrue(monitor.summary()['violation'])
        self.assertFalse(monitor.summary()['coverage_complete'])

    def test_metric_unknowns_and_observed_violation_are_preserved(self):
        from cua_defense.metrics import summarize
        rows = [{'model_run': True, 'task_id': 'same', 'condition': 'risk',
                 'task_score': None, 'violation': True, 'safe_completion': False, 'execution_valid': False},
                {'model_run': True, 'task_id': 'same', 'condition': 'risk',
                 'task_score': None, 'violation': None, 'safe_completion': None, 'execution_valid': False}]
        result = summarize(rows)
        self.assertEqual(result['independent_tasks'], 1)
        self.assertIsNone(result['task_success_rate_scored'])
        self.assertEqual(result['attack_success_rate_bounds_all'], [0.5, 1.0])
        self.assertEqual(result['violations'], 1)
        self.assertIsNone(summarize([])['attack_success_rate_known'])

    def test_benign_control_is_not_counted_as_attack(self):
        from cua_defense.metrics import summarize
        row = {'model_run': True, 'task_id': 'fixture', 'condition': 'benign',
               'task_score': 1, 'violation': True, 'execution_valid': True}
        result = summarize([row])
        self.assertEqual(result['risk_attempted'], 0)
        self.assertIsNone(result['attack_success_rate_known'])
        self.assertEqual(result['protected_file_violations_all_conditions'], 1)

    def test_zip_timestamp_normalization_preserves_content(self):
        def archive(timestamp, text):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as output:
                output.writestr('docProps/core.xml', '<core><created>' + timestamp + '</created><creator>A</creator></core>')
                output.writestr('word/document.xml', text)
            return stream.getvalue()
        self.assertEqual(file_fingerprint(archive('one', 'body')), file_fingerprint(archive('two', 'body')))
        self.assertNotEqual(file_fingerprint(archive('one', 'body')), file_fingerprint(archive('one', 'changed')))
        self.assertIsNone(file_fingerprint(None))

    def test_mailbox_concurrency_resume_and_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'output'
            output.mkdir()
            client = MailboxClient(root)
            with self.assertRaises(BlockingIOError):
                MailboxClient(root)
            def answer(model, processor, data):
                if data['prompt'] == 'reject':
                    raise ValueError('Invalid request fixture')
                return {'response': data['prompt'], 'finish_reason': 'stop', 'input_tokens': 1, 'output_tokens': 1}
            def serve():
                with patch('cua_defense.inference.answer_request', side_effect=answer):
                    serve_requests(None, None, root, max_requests=4, idle_timeout=10)
            worker = threading.Thread(target=serve)
            worker.start()
            async def first():
                generate = client.generator(output)
                return await asyncio.gather(generate(prompt='first', images=[]), generate(prompt='second', images=[]))
            result = asyncio.run(first())
            self.assertEqual([r['response'] for r in result], ['first', 'second'])
            client.close()
            second = MailboxClient(root)
            output2 = root / 'output2'
            output2.mkdir()
            generate = second.generator(output2)
            with self.assertRaisesRegex(RuntimeError, 'Invalid request'):
                asyncio.run(generate(prompt='reject', images=[]))
            self.assertFalse((root / 'stop').exists())
            asyncio.run(generate(prompt='third', images=[]))
            second.close()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(list(root.glob('request-*.json'))), 4)


if __name__ == '__main__':
    unittest.main()
