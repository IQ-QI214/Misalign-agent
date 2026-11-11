#!/usr/bin/env python3
"""CPU-only setup coverage for pinned OS-Harm tasks and injection vectors."""
import asyncio
import argparse
import hashlib
import json
from pathlib import Path
import platform
import traceback
import uuid

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')


async def check(output, all_cases=False, concurrency=2, limit=None, case_ids=None):
    import lite.gym as gym
    from lite.gym.sandbox.base import _take_screenshot
    from cua_defense.tasks import register_row
    from cua_defense.os_harm import IMAGE, install_injection, task_row
    source = ROOT / '.runtime/cua-host/references/os-harm'
    cases = json.loads((ROOT / '.runtime/cua-host/prepared/os_harm_cases.json').read_text())
    if case_ids is not None:
        cases = [case for case in cases if case['case_id'] in case_ids]
    selected = {}
    bases = set()
    vectors = set()
    for case in cases:
        base = (case['domain'], case['task_id'])
        kind = case['injection']['type']
        if all_cases or base not in bases or kind not in vectors:
            selected[case['case_id']] = case
            bases.add(base)
            vectors.add(kind)
    records = []
    sem = asyncio.Semaphore(concurrency)

    async def one(case):
        async with sem:
            row = task_row(case, source)
            env = gym.make(register_row(row), image=IMAGE, max_steps=30)
            record = {'case_id': case['case_id'], 'domain': case['domain'], 'task_id': case['task_id'],
                      'vector': case['injection']['type'], 'passed': False}
            try:
                await asyncio.wait_for(env.reset(), timeout=600)
                await asyncio.wait_for(install_injection(env, source, case['injection']), timeout=300)
                await asyncio.sleep(3)
                image = await _take_screenshot(env._computer)
                filename = hashlib.sha256(case['case_id'].encode()).hexdigest()[:16] + '.png'
                (output / filename).write_bytes(image)
                record.update(passed=True, screenshot=filename, screenshot_sha256=hashlib.sha256(image).hexdigest())
            except Exception:
                record['error'] = traceback.format_exc()
            finally:
                await env.close()
            records.append(record)
            (output / 'tasks.jsonl').open('a').write(json.dumps(record) + '\n')
            print(json.dumps({k:v for k,v in record.items() if k!='error'}), flush=True)
    chosen = list(selected.values())[:limit]
    await asyncio.gather(*(one(case) for case in chosen))
    report = {'schema_version': 1, 'status': 'OS_HARM_SETUP_CHECK_COMPLETED', 'model_run': False,
              'gpu_used': False, 'source_cases': len(cases), 'setup_cases_checked': len(records),
              'passed': sum(r['passed'] for r in records), 'failed': sum(not r['passed'] for r in records),
              'official_scores': 'UNKNOWN_PENDING_QUALIFIED_JUDGE', 'image': IMAGE,
              'tasks': records, 'output': str(output)}
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise SystemExit('QZ_LINUX_REQUIRED')
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--case-ids', type=Path)
    args = parser.parse_args()
    output = ROOT / '.runtime/cua-host/prepared' / ('os-harm-smoke-' + uuid.uuid4().hex[:8])
    output.mkdir()
    print('OUTPUT=' + str(output), flush=True)
    report = asyncio.run(check(output, args.all, args.concurrency, args.limit, json.loads(args.case_ids.read_text()) if args.case_ids else None))
    print(json.dumps({k:v for k,v in report.items() if k!='tasks'}, indent=2))
