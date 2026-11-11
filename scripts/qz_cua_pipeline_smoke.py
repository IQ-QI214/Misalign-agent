#!/usr/bin/env python3
"""Exercise real GUI batch plumbing with deterministic fake model responses on QZ CPU."""
import argparse
import asyncio
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import uuid
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen3vl_4b', 'qwen35_9b'], default='qwen35_9b')
    parser.add_argument('--os-harm', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--minutes', type=float, default=30)
    args = parser.parse_args()
    from scripts import qz_cua_batch as batch
    from cua_defense.inference import serve_requests
    if not batch.ROOT.is_dir():
        raise SystemExit('QZ_REQUIRED')
    output = batch.STATE / 'prepared' / ('batch-cpu-smoke-' + args.model + '-' + uuid.uuid4().hex[:8])
    output.mkdir()
    config = json.loads((batch.PROJECT / 'configs/cua/next_gpu.json').read_text())
    config.update(conditions=['clean', 'maintenance'], max_steps=2, concurrency=1)
    if args.os_harm:
        config.update(json.loads((batch.PROJECT / 'configs/cua/os_harm_gpu.json').read_text()))
        config.update(max_steps=2, concurrency=1)
        from cua_defense.os_harm import load_rows
        choices = load_rows(config['cohort'])
        config['task_ids'] = [next(row['task_id'] for row in choices if row['metadata']['others']['domain'] == 'libreoffice_writer')]
    if args.config:
        config.update(json.loads(args.config.read_text()))
        config.update(max_steps=2, conditions=['clean'], concurrency=2)
    model = json.loads((batch.PROJECT / 'configs/cua/models.json').read_text())[args.model]
    options = SimpleNamespace(limit=None if args.config else 1, repair_from=None, repair_mode='full', external_mailbox=None,
                              keep_external_worker=False, minutes=args.minutes)
    response = ('<tool_call>\n{"name":"terminate","arguments":{"status":"success"}}\n</tool_call>'
                if args.model == 'qwen3vl_4b' else
                '<tool_call>\n<function=terminate>\n<parameter=status>success</parameter>\n</function>\n</tool_call>')
    threads = []
    def launch(model, mailbox, budget_seconds=None):
        def worker():
            (mailbox / 'ready.json').write_text(json.dumps({'model_run': False, 'device': 'cpu', 'fake_responses': True}))
            try:
                with patch('cua_defense.inference.answer_request', return_value={'response': response, 'finish_reason': 'stop'}):
                    serve_requests(None, None, mailbox, idle_timeout=600)
            finally:
                (mailbox / 'stopped.json').write_text('{}')
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
    with patch.object(batch, 'launch_worker', side_effect=launch) as launched:
        report = asyncio.run(batch.run(options, output, config, model))
        assert launched.call_count == 1
        if args.config:
            entries = json.loads((batch.PROJECT / config['cohort']).read_text())['tasks']
            expected = sum(row['task_id'] in config.get('task_ids', [entry['task_id'] for entry in entries]) for row in entries)
        else:
            expected = 1 if args.os_harm else 2
        assert report['valid'] == expected, report
        assert all(not r['model_run'] for r in report['results'])
        if args.os_harm:
            assert report['results'][0]['task_score'] is None
            assert report['results'][0]['official_safety'] is None
        elif not args.config:
            attacked = next(r for r in report['results'] if r['condition'] != 'clean')
            assert attacked['risk']['coverage_complete'] and attacked['violation'] is False, attacked
        resumed = asyncio.run(batch.run(options, output, config, model))
        assert resumed['status'] == 'NO_PENDING_JOBS' and launched.call_count == 1
        options.external_mailbox = output / 'resume-mailbox'
        options.external_mailbox.mkdir()
        asyncio.run(batch.run(options, output, config, model))
        assert (options.external_mailbox / 'evaluation_result.json').exists()
        assert (options.external_mailbox / 'stop').exists()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    result = {'status': 'CPU_NATIVE_BATCH_SMOKE_PASSED', 'model': args.model, 'model_run': False,
              'gpu_used': False, 'valid_episodes': report['valid'], 'benchmark': 'os_harm' if args.os_harm else config['phase'],
              'cohort': config['cohort'],
              'risk_coverage_complete': not args.os_harm and not args.config,
              'completed_resume_dispatched_worker': False, 'output': str(output)}
    result['completed_external_evaluation_notified'] = True
    (output / 'smoke_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
