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
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
import zipfile

from cua_defense.events import EventMonitor
from cua_defense.inference import serve_requests
from cua_defense.mailbox import MailboxClient, teacher_prompt
from cua_defense.replay import file_fingerprint
from cua_defense.tasks import ATTACKS, variant
from scripts.qz_cua_export_sft import eligible


class PipelineContracts(unittest.TestCase):
    def test_shift_click_validation_and_abort_cleanup(self):
        from cua_defense.desktop_actions import click_uses_shift, release_shift_on_failure
        for arguments in [{'keys': ['shift']}, {'keys': ['Shift']}, {'shift_key': 'True'}, {'shift_key': True}]:
            self.assertTrue(click_uses_shift(arguments))
        for arguments in [{}, {'keys': []}, {'shift_key': False}]:
            self.assertFalse(click_uses_shift(arguments))
        for arguments in [{'keys': 'shift'}, {'keys': ['ctrl']}, {'shift_key': 1},
                          {'keys': ['shift'], 'shift_key': False}]:
            with self.assertRaises(ValueError):
                click_uses_shift(arguments)
        interface = SimpleNamespace(key_up=AsyncMock())
        cleanup = []
        async def abort():
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(release_shift_on_failure(interface, abort(), cleanup))
        interface.key_up.assert_awaited_once_with('Shift_L')
        self.assertEqual(cleanup, [{'modifier': 'Shift_L', 'released': True}])

    @unittest.skipUnless(Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq').is_dir(),
                         'Native CUA action space is verified on QZ')
    def test_native_shift_click_roundtrip_preserves_plain_actions(self):
        from cua_defense.desktop_actions import make_shift_click_action_space
        from lite.agents.models.qwen3_5.action_space import Qwen3_5DesktopActionSpace
        from lite.core.tools.calls import tool_call_arguments
        extended, original = make_shift_click_action_space(), Qwen3_5DesktopActionSpace()
        self.assertEqual(type(extended).get_tool_names(), type(original).get_tool_names())
        from lite.agents.models.qwen3_5.adapter import Qwen3_5DesktopUseAdapter
        adapter = Qwen3_5DesktopUseAdapter(action_space=extended)
        self.assertIn('Shift-click', adapter._build_tools_section())
        plain = {'name': 'computer_use', 'arguments': {'action': 'left_click', 'coordinate': [197, 624]}}
        self.assertEqual(extended.convert_tool_calls_from_agent([plain]), original.convert_tool_calls_from_agent([plain]))
        for extra in [{'keys': ['shift']}, {'shift_key': 'True'}]:
            call = {**plain, 'arguments': {**plain['arguments'], **extra}}
            converted = extended.convert_tool_calls_from_agent([plain, call, plain])
            actions = tool_call_arguments(converted[0])['actions']
            self.assertEqual([item['action'] for item in actions], ['click', 'key_down', 'click', 'key_up', 'click'])
            rendered = extended.convert_tool_calls_to_agent(converted)
            self.assertEqual(rendered[1]['arguments']['keys'], ['shift'])
            self.assertEqual(extended.convert_tool_calls_from_agent(rendered), converted)

    def test_initial_selection_feedback_requests_model_chosen_click(self):
        from cua_defense.action_contract import feedback_generator
        schema = {'name': 'computer_use', 'parameters': {'properties': {'action': {}, 'keys': {}, 'coordinate': {}}}}
        initial = {'tool_calls': [{'name': 'computer_use', 'arguments': {'action': 'key', 'keys': ['ctrl', 'a']}}]}
        corrected = {'tool_calls': [{'name': 'computer_use', 'arguments': {'action': 'left_click', 'coordinate': [500, 400]}}]}
        adapter = SimpleNamespace(parse_raw_assistant_response=json.loads,
            action_space=SimpleNamespace(get_tool_schemas=lambda: [schema]))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prompts = []
            def generator(directory, **options):
                index = 0
                async def generate(**request):
                    nonlocal index
                    prompts.append(request['prompt'])
                    result = {'response': json.dumps([initial, corrected][index]), 'finish_reason': 'stop'}
                    (directory / f'response-{index}.json').write_text(json.dumps(result))
                    index += 1
                    return result
                return generate
            generate = feedback_generator(SimpleNamespace(generator=generator), adapter, output, require_initial_click=True)
            prompt = '<|im_start|>system\nRules<|im_end|>\n<|im_start|>user\nTask<|im_end|>\n<|im_start|>assistant\n'
            result = asyncio.run(generate(prompt=prompt, images=[]))
            self.assertEqual(json.loads(result['response']), corrected)
            self.assertIn('initial_content_activation', prompts[1])
            self.assertNotIn('[500, 400]', prompts[1])
            events = json.loads((output / 'action_feedback.json').read_text())
            self.assertEqual([event['accepted'] for event in events], [False, True])

    def test_modifier_hook_accepts_native_positional_coordinates(self):
        from scripts.qz_cua_focus_diagnostic import install_shift_click_control
        original = AsyncMock()
        interface = SimpleNamespace(left_click=original, key_down=AsyncMock(), key_up=AsyncMock())
        state, events = {'shift': False, 'turn': 0}, []
        install_shift_click_control(interface, state, events)
        asyncio.run(interface.left_click(10, 20))
        original.assert_awaited_with(x=10, y=20)
        interface.key_down.assert_not_awaited()
        state.update(shift=True, turn=6)
        asyncio.run(interface.left_click(378, 674))
        original.assert_awaited_with(x=378, y=674)
        interface.key_down.assert_awaited_once_with('Shift_L')
        interface.key_up.assert_awaited_once_with('Shift_L')
        self.assertTrue(events[0]['completed'])

    def test_modifier_click_releases_shift_when_click_fails(self):
        from scripts.qz_cua_focus_diagnostic import shifted_click
        interface = SimpleNamespace(key_down=AsyncMock(), key_up=AsyncMock())
        click = AsyncMock(side_effect=RuntimeError('click failed'))
        with self.assertRaisesRegex(RuntimeError, 'click failed'):
            asyncio.run(shifted_click(interface, click, x=378, y=674))
        interface.key_down.assert_awaited_once_with('Shift_L')
        interface.key_up.assert_awaited_once_with('Shift_L')

    def test_calc_control_replaces_modifier_click_with_supported_chords(self):
        from scripts.qz_cua_focus_diagnostic import calc_selection_control
        actions = calc_selection_control(23)
        self.assertEqual(len(actions), 29)
        self.assertEqual(sum('["shift", "down"]' in response for response in actions), 20)
        self.assertFalse(any('shift_key' in response for response in actions))
        self.assertIn('terminate', actions[-1])
        with self.assertRaises(ValueError):
            calc_selection_control(2)

    def test_keys_on_click_are_rejected_without_blocking_keyboard_chords(self):
        from cua_defense.action_contract import argument_issues
        schema = {'name': 'computer_use', 'parameters': {'properties': {'action': {}, 'coordinate': {}, 'keys': {}}}}
        message = {'tool_calls': [{'name': 'computer_use', 'arguments': {
            'action': 'left_click', 'coordinate': [197, 621], 'keys': ['shift']}}]}
        self.assertEqual(argument_issues(message, [schema]), [])
        issue = argument_issues(message, [schema], check_key_usage=True)[0]
        self.assertEqual(issue['inactive_arguments'], ['keys'])
        self.assertNotIn('keys', issue['declared_arguments'])
        message['tool_calls'][0]['arguments'] = {'action': 'key', 'keys': ['shift', 'down']}
        self.assertEqual(argument_issues(message, [schema], check_key_usage=True), [])
        message['tool_calls'][0]['name'] = 'another_tool'
        self.assertEqual(argument_issues(message, [schema], check_key_usage=True), [])

    def test_feedback_retry_is_bounded_preserves_prefix_and_records_all_proposals(self):
        from cua_defense.action_contract import ActionContractError, feedback_generator
        schema = {'name': 'computer_use', 'parameters': {'properties': {'action': {}, 'coordinate': {}}}}
        def response(extra):
            arguments = {'action': 'left_click', 'coordinate': [500, 400]}
            arguments.update(extra)
            return json.dumps({'tool_calls': [{'name': 'computer_use', 'arguments': arguments}]})
        adapter = SimpleNamespace(parse_raw_assistant_response=json.loads,
            action_space=SimpleNamespace(get_tool_schemas=lambda: [schema]))
        prompt = '<|im_start|>system\nRules<|im_end|>\n<|im_start|>user\nTask<|im_end|>\n<|im_start|>assistant\n'
        for recover in [True, False]:
            for prefix in [[], [{'response': response({}), 'finish_reason': 'stop'}]]:
                with self.subTest(recover=recover, prefix=bool(prefix)), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    output = root / 'episode'
                    output.mkdir()
                    client = MailboxClient(root)
                    proposals = [response({'shift_key': 'True'}), response({} if recover else {'shift_key': 'True'})]
                    def answer(model, processor, request):
                        return {'response': proposals.pop(0), 'finish_reason': 'stop', 'input_tokens': 1, 'output_tokens': 1}
                    with patch('cua_defense.inference.answer_request', side_effect=answer):
                        worker = threading.Thread(target=serve_requests, args=(None, None, root),
                                                  kwargs={'max_requests': 2, 'idle_timeout': 5})
                        worker.start()
                        generate = feedback_generator(client, adapter, output, replay=prefix)
                        try:
                            if prefix:
                                asyncio.run(generate(prompt=prompt, images=[]))
                                self.assertEqual(json.loads((output / 'response-0.json').read_text())['source'], 'prefix_replay')
                            if recover:
                                asyncio.run(generate(prompt=prompt, images=[]))
                                saved = json.loads((output / f'response-{len(prefix)}.json').read_text())
                                self.assertEqual(saved['correction_attempts'], 1)
                            else:
                                with self.assertRaises(ActionContractError):
                                    asyncio.run(generate(prompt=prompt, images=[]))
                                self.assertFalse((output / f'response-{len(prefix)}.json').exists())
                            retry = json.loads((root / 'request-1.json').read_text())
                            self.assertIn('Action argument feedback', retry['prompt'])
                            self.assertIn('shift_key', retry['prompt'])
                            self.assertEqual(retry['prompt'][retry['prompt'].index('<|im_end|>'):], prompt[prompt.index('<|im_end|>'):])
                            events = json.loads((output / 'action_feedback.json').read_text())
                            self.assertEqual(len(events), 2)
                            self.assertEqual(events[-1]['accepted'], recover)
                        finally:
                            (root / 'stop').touch()
                            client.close()
                            worker.join(timeout=5)
                        self.assertFalse(worker.is_alive())
                        self.assertEqual(len(list(root.glob('request-*.json'))), 2)

    def test_protocol_names_order_and_matched_panel(self):
        from scripts.qz_cua_batch import protocol_arms, select_rows, summarize_protocols
        arms = [{'name': 'baseline'}, {'name': 'feedback', 'argument_feedback': True}]
        self.assertEqual(protocol_arms({}), [{'name': 'baseline'}])
        for invalid in [[], [{'name': 'same'}, {'name': 'same'}], [{'name': 'bad', 'argument_feedback': 'yes'}],
                        [{'name': 'bad', 'check_key_usage': 'yes'}], [{'name': 'bad', 'shift_click': 'yes'}],
                        [{'name': 'bad', 'require_initial_click': True}]]:
            with self.assertRaises(ValueError):
                protocol_arms({'protocols': invalid})
        rows = [{'task_id': name} for name in ['calc', 'writer']]
        ordered = select_rows(rows, {'task_ids': ['writer', 'calc'], 'task_order': 'configured'})
        self.assertEqual([row['task_id'] for row in ordered], ['writer', 'calc'])
        results = [{'task_id': task, 'condition': 'clean', 'repeat': 0, 'model_run': True,
                    'execution_valid': True, 'task_score': 0, 'protocol': arm}
                   for arm in arms for task in ['writer', 'calc']]
        results[-1].update(execution_valid=False, task_score=None)
        report = summarize_protocols(results, arms, 2)
        self.assertEqual(report['matched_completed_cases'], [{'task_id': 'writer', 'condition': 'clean', 'repeat': 0}])
        self.assertEqual(report['planned_episodes_per_protocol'], 2)
        self.assertEqual(report['matched_protocol_metrics']['baseline']['task_scored'], 1)
        self.assertEqual(report['matched_protocol_metrics']['feedback']['task_scored'], 1)

    def test_checkpoint_preserves_files_and_focus_probe_failure_is_diagnostic(self):
        from cua_defense.replay import checkpoint
        interface = SimpleNamespace(
            run_command=AsyncMock(side_effect=[SimpleNamespace(returncode=0), TimeoutError('focus probe')]),
            read_bytes=AsyncMock(return_value=b'actual output'))
        env = SimpleNamespace(_computer=SimpleNamespace(interface=interface))
        row = {'metadata': {'evaluator': {'result': {'type': 'vm_file', 'path': '/output.docx'}}}}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = asyncio.run(checkpoint(env, row, Mock(size=(1920, 1088)), directory, 3))
            self.assertEqual((directory / record['file_snapshots']['/output.docx']).read_bytes(), b'actual output')
            self.assertEqual(record['files']['/output.docx'], file_fingerprint(b'actual output'))
            self.assertEqual(record['focus']['returncode'], None)
            self.assertIn('TimeoutError', record['focus']['error'])

    def test_argument_audit_reports_dropped_modifiers_without_rewriting_actions(self):
        from cua_defense.action_contract import audit_arguments
        schema = {'type': 'function', 'function': {'name': 'computer_use',
                  'parameters': {'properties': {'action': {}, 'coordinate': {}}}}}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            turn = directory / 'turn_0006'
            turn.mkdir()
            path = turn / '03_actions.json'
            raw = json.dumps({'agent_message': {'tool_calls': [{'name': 'computer_use',
                'arguments': {'action': 'left_click', 'coordinate': [197, 624], 'shift_key': 'True'}}]}})
            path.write_text(raw)
            report = audit_arguments(directory, [schema])
            self.assertEqual(report['affected_turns'], 1)
            self.assertEqual(report['issues'][0]['undeclared_arguments'], ['shift_key'])
            self.assertEqual(path.read_text(), raw)
            self.assertEqual(audit_arguments(directory, [schema['function']]), report)
            schema['function']['parameters']['properties']['shift_key'] = {}
            self.assertEqual(audit_arguments(directory, [schema])['affected_turns'], 0)

    def test_chat_eos_is_included_when_model_defaults_to_endoftext(self):
        from types import SimpleNamespace
        from cua_defense.inference import eos_token_ids
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044))
        processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=248046))
        self.assertEqual(eos_token_ids(model, processor), [248044, 248046])
        model.generation_config.eos_token_id = [248044, 248046]
        self.assertEqual(eos_token_ids(model, processor), [248044, 248046])
        model.generation_config.eos_token_id = None
        self.assertEqual(eos_token_ids(model, processor), [248046])
        processor.tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, 'EOS token'):
            eos_token_ids(model, processor)

    def test_calibration_interleaves_domains_before_applying_limit(self):
        from scripts.qz_cua_batch import select_rows
        rows = [{'task_id': name, 'metadata': {'others': {'domain': domain}}}
                for name, domain in [('c1', 'calc'), ('c2', 'calc'), ('c3', 'calc'), ('w1', 'writer')]]
        config = {'task_order': 'round_robin_domain'}
        self.assertEqual([row['task_id'] for row in select_rows(rows, config)], ['c1', 'w1', 'c2', 'c3'])
        self.assertEqual([row['task_id'] for row in select_rows(rows, config, 2)], ['c1', 'w1'])
        self.assertEqual(select_rows(rows, {}), rows)
        self.assertEqual(select_rows(rows, {**config, 'task_ids': ['c2', 'w1']}), [rows[1], rows[3]])
        with self.assertRaisesRegex(ValueError, 'Unknown task ordering'):
            select_rows(rows, {'task_order': 'unknown'})

    def test_export_rejects_invalid_supervision_boundaries_before_conversion(self):
        from scripts.qz_cua_export_sft import convert_trajectory
        for prefix in [-1, True, 1.5]:
            with self.assertRaisesRegex(ValueError, 'nonnegative step index'):
                convert_trajectory(Path('unused'), {'supervise_from_step': prefix}, {}, Path('.'), diagnostic=True)
        with self.assertRaisesRegex(ValueError, 'verified positive'):
            convert_trajectory(Path('unused'), {'data_role': 'diagnostic'}, {}, Path('.'))

    def test_monitor_can_use_a_separate_inotify_uid(self):
        ready = json.dumps({'type': 'ready', 'initial_sha256': 'known', 'monitor_uid': 65532})
        closed = json.dumps({'type': 'closed', 'final_sha256': 'known'})
        with patch('cua_defense.events.subprocess.Popen') as launch:
            process = launch.return_value
            process.stdout = io.StringIO(ready + '\n' + closed + '\n')
            process.stderr = io.StringIO('')
            process.stdin = io.StringIO()
            process.returncode = 0
            monitor = EventMonitor('owned', '/public-fixture', 'known', monitor_user='65532:65532')
            monitor.start()
            monitor.close()
            self.assertEqual(launch.call_args.args[0][4], '65532:65532')
            self.assertTrue(monitor.summary()['coverage_complete'])
            self.assertFalse(monitor.summary()['violation'])
            self.assertEqual(EventMonitor('owned', '/private', 'known').monitor_user, '0')

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
