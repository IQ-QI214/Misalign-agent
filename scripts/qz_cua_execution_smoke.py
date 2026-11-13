#!/usr/bin/env python3
"""Replay frozen replies through the normal batch entry, without model inference."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from scripts.qz_cua_batch import PROJECT, ROOT, run, write_json
from scripts.qz_cua_focus_diagnostic import action_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not ROOT.is_dir():
        raise SystemExit('QZ_CPU_REQUIRED')
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = Path('configs/cua/protocol_diagnostic_v3.json')
    config = json.loads((PROJECT / config_path).read_text())
    config.update(concurrency=1, budget_minutes=25)
    files = ['scripts/qz_cua_execution_smoke.py', 'scripts/qz_cua_batch.py',
             'cua_defense/desktop_actions.py', 'cua_defense/runtime.py',
             'cua_defense/action_contract.py', 'cua_defense/mailbox.py', str(config_path)]
    identity = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in files}
    launch = {'source_sha256': identity, 'source_run': str(args.source_run),
              'source_receipt_sha256': hashlib.sha256((args.source_run / 'receipt.json').read_bytes()).hexdigest(),
              'gpu_used': False, 'model_run': False, 'training_eligible': False,
              'timing_scope': 'Writer original inference delays and Calc first six delays are replayed. Remaining Calc suffix has no added inference delay.'}
    write_json(args.output / 'launch.json', launch)
    write_json(args.output / 'diagnostic_config.json', config)
    source = json.loads((args.source_run / 'receipt.json').read_text())
    proposals = []
    for task in config['task_ids']:
        row = next(row for row in source['results'] if row['task_id'] == task and row['protocol']['name'] == 'baseline')
        directory = Path(row['output'])
        responses = []
        for turn in range(row['summary']['n_turns']):
            text = json.loads((directory / f'response-{turn}.json').read_text())['response']
            delay = json.loads((directory / f'turn_{turn:04d}/05_timing.json').read_text())['predict']
            responses.append({'response': text, 'delay': delay if 'writer' in task or turn < 6 else 0})
        for arm in config['protocols']:
            sequence = responses
            if 'writer' in task and arm.get('require_initial_click'):
                sequence = [responses[0], {'response': action_response('left_click', coordinate=[500, 400]),
                                         'delay': 0, 'correction': True}, *responses]
            proposals.extend({**item, 'task_id': task, 'protocol': arm['name']} for item in sequence)
    cursor = 0
    backend_events = []
    failures = []
    mailbox = args.output / 'mailbox'
    mailbox.mkdir()
    write_json(mailbox / 'ready.json', {'model_run': False, 'backend': 'frozen_response_cpu_replay'})

    def answer(model, processor, request):
        nonlocal cursor
        if cursor >= len(proposals):
            raise ValueError('Unexpected extra request beyond the frozen replay schedule')
        proposal = proposals[cursor]
        if proposal.get('correction'):
            assert 'initial_content_activation' in request['prompt']
        backend_events.append({'request_index': cursor, 'task_id': proposal['task_id'],
                               'protocol': proposal['protocol'], 'scripted_correction': proposal.get('correction', False)})
        cursor += 1
        time.sleep(proposal['delay'])
        return {'response': proposal['response'], 'finish_reason': 'stop'}

    def serve():
        from cua_defense.inference import serve_requests
        try:
            with patch('cua_defense.inference.answer_request', side_effect=answer):
                serve_requests(None, None, mailbox, max_requests=len(proposals) + 1, idle_timeout=1500)
        except Exception as error:
            failures.append(repr(error))
            write_json(mailbox / 'error.json', {'error': repr(error)})
        finally:
            write_json(mailbox / 'stopped.json', {'backend': 'frozen_response_cpu_replay'})

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    options = SimpleNamespace(minutes=25, limit=None, repair_from=None, repair_mode='full',
                              external_mailbox=mailbox, keep_external_worker=False)
    model = json.loads((PROJECT / 'configs/cua/models.json').read_text())[config['model']]
    try:
        result = asyncio.run(run(options, args.output, config, model))
    finally:
        (mailbox / 'stop').touch()
        worker.join(timeout=5)
        write_json(args.output / 'scripted_backend.json', {'consumed': cursor, 'planned': len(proposals),
                                                          'events': backend_events, 'failures': failures})
    assert not worker.is_alive() and not failures, failures
    assert result['completed'] == result['valid'] == 6, result
    assert cursor == len(proposals), (cursor, len(proposals))
    panel = {}
    for row in result['results']:
        domain = 'writer' if 'writer' in row['task_id'] else 'calc'
        name = row['protocol']['name']
        expected = int(name == 'shift_click_focus' or (domain == 'calc' and name == 'shift_click'))
        assert row['task_score'] == expected, (domain, name, row['task_score'])
        directory = Path(row['output'])
        errors = [item for path in directory.glob('turn_*/04_results.json')
                  for item in json.loads(path.read_text()).get('results', []) if item.get('error')]
        assert not errors, errors
        modifier_turns = 0
        for path in directory.glob('turn_*/03_actions.json'):
            data = json.loads(path.read_text())
            calls = [item['call'] for item in data.get('executed_actions', [])]
            if any(call.endswith('.key_down') for call in calls):
                assert any(call.endswith('.left_click') for call in calls), calls
                assert any(call.endswith('.key_up') for call in calls), calls
                modifier_turns += 1
        if domain == 'calc' and name != 'baseline':
            assert modifier_turns == 17, modifier_turns
        panel.setdefault(domain, {})[name] = {'task_score': row['task_score'],
            'n_turns': row['summary']['n_turns'], 'modifier_turns': modifier_turns,
            'environment_action_errors': len(errors), 'model_run': row['model_run']}
    assert not any(row['model_run'] for row in result['results'])
    report = {**launch, 'status': 'CPU_NORMAL_BATCH_EXECUTION_ACCEPTANCE_PASSED',
              'completed_episodes': 6, 'scripted_responses': cursor, 'panel': panel,
              'checks': ['normal batch/mailbox entry', 'native canonical modifier execution records',
                         'multi-turn modifier history roundtrip', 'initial Ctrl+A rejected before execution',
                         'native task scores', 'zero environment action errors', 'scripted results excluded from training'],
              'output': str(args.output)}
    write_json(args.output / 'acceptance.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
