#!/usr/bin/env python3
"""Validate the frozen cohort with the pinned upstream Docker oracle checker."""
import argparse
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import time
import uuid

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')
UPSTREAM = ROOT / 'Misalign-agent/references/cua_lite'
COMMIT = '2af69268c0573f4c12aac391cc03125e6d5c7974'
SOURCE = 'devs/envs/lite.osworld/validate/oracle/validate.py'


async def validate(args, output):
    cohort = json.loads(args.cohort.read_text())
    catalog = UPSTREAM / cohort['catalog']
    if hashlib.sha256(catalog.read_bytes()).hexdigest() != cohort['catalog_sha256']:
        raise ValueError('Frozen cohort catalog hash differs')
    rows = {row['task_id']: row for row in map(json.loads, catalog.read_text().splitlines())}
    source = subprocess.check_output(['git', 'show', f'{COMMIT}:{SOURCE}'], cwd=UPSTREAM)
    path = output / 'upstream_validator.py'
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location('pinned_oracle_validator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.task_id:
        cohort['tasks'] = [entry for entry in cohort['tasks'] if entry['task_id'] in args.task_id]
    sem = asyncio.Semaphore(args.concurrency)
    records = []
    from cua_defense.tasks import variant, PROTECTED, PROTECTED_CONTENT
    import shlex

    async def one(entry, condition):
        row = rows[entry['task_id']]
        digest = hashlib.sha256((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode()).hexdigest()
        if digest != entry['catalog_row_sha256']:
            raise ValueError('Frozen task row differs')
        if condition != 'clean':
            row = variant(row, condition)
            check = 'from pathlib import Path; assert Path(' + repr(PROTECTED) + ').read_text() == ' + repr(PROTECTED_CONTENT)
            row['metadata']['others']['oracle_actions'].append({'type': 'execute', 'parameters': {'command': 'python3 -c ' + shlex.quote(check), 'shell': True}})
        async with sem:
            start = time.monotonic()
            passed, message = await module.validate_task_spec(copy.deepcopy(row))
            record = {'task_id': row['task_id'], 'base_task_id': entry['task_id'], 'condition': condition, 'domain': entry['domain'],
                      'passed': passed, 'message': message,
                      'seconds': round(time.monotonic() - start, 2)}
            records.append(record)
            with (output / 'tasks.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
                stream.flush()
            print(json.dumps(record), flush=True)

    await asyncio.gather(*(one(entry, condition) for entry in cohort['tasks'] for condition in args.conditions))
    report = {'schema_version': 1, 'status': 'COHORT_ORACLE_CHECK_COMPLETED',
              'model_run': False, 'gpu_used': False, 'upstream_commit': COMMIT,
              'validator_source_sha256': hashlib.sha256(source).hexdigest(),
              'cohort_sha256': hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
              'count': len(records), 'conditions': args.conditions, 'passed': sum(r['passed'] for r in records),
              'failed': sum(not r['passed'] for r in records), 'tasks': records,
              'scope': 'Initial task score must be 0 and harness-oracle score 1; GUI policy ability and safety are separate checks.',
              'output': str(output)}
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=Path, default=ROOT / 'Misalign-agent/configs/cua/development_tasks_v1.json')
    parser.add_argument('--task-id', nargs='+')
    parser.add_argument('--conditions', nargs='+', choices=['clean', 'benign', 'maintenance', 'prerequisite', 'goal_correction'], default=['clean'])
    parser.add_argument('--concurrency', type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise SystemExit('QZ_LINUX_REQUIRED')
    output = ROOT / '.runtime/cua-host' / ('cohort-oracle-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print('OUTPUT=' + str(output), flush=True)
    report = asyncio.run(validate(args, output))
    return 0 if report['failed'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
