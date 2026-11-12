#!/usr/bin/env python3
"""CPU-only native GUI/batch/mailbox acceptance for the three protocol arms."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import patch

from scripts.qz_cua_batch import PROJECT, ROOT, STATE, run, write_json


def scripted_answer(model, processor, request):
    prompt = request['prompt']
    if 'CPU_DIAGNOSTIC_CLICK' in prompt:
        response = '<tool_call>\n<function=computer_use>\n<parameter=action>terminate</parameter>\n<parameter=status>failure</parameter>\n</function>\n</tool_call>'
    else:
        extra = '' if 'Action argument feedback' in prompt or 'GUI operation guidance' in prompt else '<parameter=shift_key>True</parameter>\n'
        response = ('Action: CPU_DIAGNOSTIC_CLICK\n<tool_call>\n<function=computer_use>\n'
                    '<parameter=action>left_click</parameter>\n<parameter=coordinate>[500, 400]</parameter>\n' +
                    extra + '</function>\n</tool_call>')
    return {'response': response, 'finish_reason': 'stop'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not ROOT.is_dir():
        raise SystemExit('QZ_CPU_REQUIRED')
    args.output.mkdir(parents=True, exist_ok=False)
    files = ['scripts/qz_cua_protocol_smoke.py', 'scripts/qz_cua_batch.py',
             'cua_defense/action_contract.py', 'cua_defense/mailbox.py', 'configs/cua/protocol_diagnostic.json']
    identity = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in files}
    write_json(args.output / 'launch.json', {'source_sha256': identity, 'gpu_used': False})
    config = json.loads((PROJECT / 'configs/cua/protocol_diagnostic.json').read_text())
    config.update(task_ids=['synth_writer_f_writer_93__doc_font_0001'], max_steps=4, concurrency=1)
    model = json.loads((PROJECT / 'configs/cua/models.json').read_text())[config['model']]
    mailbox = args.output / 'mailbox'
    mailbox.mkdir()
    write_json(mailbox / 'ready.json', {'model_run': False, 'backend': 'scripted_cpu_diagnostic'})
    write_json(args.output / 'diagnostic_config.json', config)
    failures = []

    def serve():
        from cua_defense.inference import serve_requests
        try:
            with patch('cua_defense.inference.answer_request', side_effect=scripted_answer):
                serve_requests(None, None, mailbox, max_requests=20, idle_timeout=600)
        except Exception as error:
            failures.append(repr(error))
            write_json(mailbox / 'error.json', {'error': repr(error)})
        finally:
            write_json(mailbox / 'stopped.json', {'backend': 'scripted_cpu_diagnostic'})

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    options = SimpleNamespace(minutes=10, limit=None, repair_from=None, repair_mode='full',
                              external_mailbox=mailbox, keep_external_worker=False)
    try:
        result = asyncio.run(run(options, args.output, config, model))
    finally:
        (mailbox / 'stop').touch()
        worker.join(timeout=5)
    assert not worker.is_alive() and not failures, failures
    assert result['completed'] == result['valid'] == 3, result
    records = {row['protocol']['name']: row for row in result['results']}
    baseline, hints, feedback = [records[name] for name in ['baseline', 'operation_guidance', 'argument_feedback']]
    assert baseline['action_contract']['affected_turns'] == 1
    assert hints['action_contract']['affected_turns'] == feedback['action_contract']['affected_turns'] == 0
    assert [baseline['proposal_accounting']['responses'], hints['proposal_accounting']['responses'],
            feedback['proposal_accounting']['responses']] == [2, 2, 3]
    assert feedback['action_feedback'][0]['issues'][0]['undeclared_arguments'] == ['shift_key']
    assert feedback['action_feedback'][1]['accepted']
    directory = Path(feedback['output'])
    candidate = json.loads((directory / 'action_feedback/turn-0000/response-1.json').read_text())
    retry = json.loads((mailbox / f"request-{candidate['request_id']}.json").read_text())
    assert 'Action argument feedback' in retry['prompt'] and 'shift_key' in retry['prompt']
    first = json.loads((directory / 'action_feedback/turn-0000/response-0.json').read_text())
    initial = json.loads((mailbox / f"request-{first['request_id']}.json").read_text())
    assert [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in initial['images']] == [
        hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in retry['images']]
    actions = json.loads((directory / 'turn_0000/03_actions.json').read_text())
    assert len(actions['executed_actions']) == 1
    assert 'shift_key' not in actions['agent_message']['tool_calls'][0]['arguments']
    assert len(result['matched_completed_cases']) == 1
    report = {'status': 'CPU_NATIVE_PROTOCOL_ACCEPTANCE_PASSED', 'gpu_used': False,
              'model_run': False, 'training_eligible': False, 'source_sha256': identity,
              'completed_protocol_episodes': 3, 'scripted_model_responses': 7,
              'checks': ['native Qwen XML parsing', 'undeclared arguments detected',
                         'corrective feedback present in next request', 'retry image content unchanged',
                         'only accepted action reaches GUI', 'one-variable guidance arm',
                         'all correction requests accounted', 'three-arm matched task reporting'],
              'output': str(args.output)}
    write_json(args.output / 'acceptance.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
