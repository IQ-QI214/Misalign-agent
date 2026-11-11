#!/usr/bin/env python3
"""QZ-only task reinitialization and evaluator probe, using harness actions."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import traceback
import uuid

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')
TASK_ID = 'synth_os_f_os_01__edit_server_name_0001'


async def probe(output, report):
    import lite.gym as gym
    from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_actions
    from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn

    # Private runtime fields are used only by this pinned-version harness probe.
    env = gym.make(f'lite.osworld@{TASK_ID}', max_steps=5)
    marker = f'/tmp/cua-state-probe-{uuid.uuid4().hex}'

    async def inspect(stage):
        result = await env._computer.interface.run_command(
            f"sha256sum /home/user/Desktop/nginx-site.conf; test ! -e {marker}")
        score = await evaluate_final_fn(env._task, env._computer)
        record = {'file_sha256': result.stdout.split()[0],
                  'marker_absent': result.returncode == 0, 'task_score': score}
        report[stage] = record
        return record

    try:
        frame = await env.reset()
        (output / 'initial.png').write_bytes(frame.image)
        initial = await inspect('initial')
        if initial['task_score'] != 0:
            raise RuntimeError('Initial task state must fail the completion evaluator')
        await env._computer.interface.run_command(f'touch {marker}')
        # Harness oracle validates the evaluator. It is never model training data.
        rows = Path('lite/gym/envs/lite/osworld/data/train.synth.jsonl')
        row = next(json.loads(line) for line in rows.open()
                   if json.loads(line)['task_id'] == TASK_ID)
        await dispatch_actions(env._computer, row['metadata']['others']['oracle_actions'])
        solved = await inspect('harness_solved')
        if solved['task_score'] != 1 or solved['marker_absent']:
            raise RuntimeError('Oracle completion or marker mutation failed')
        frame = await env.reset()
        (output / 'warm_reset.png').write_bytes(frame.image)
        warm = await inspect('warm_reset')
        if warm['file_sha256'] != initial['file_sha256'] or warm['task_score'] != 0:
            raise RuntimeError('Warm reset did not reconstruct the task file')
        await env.close()
        env = gym.make(f'lite.osworld@{TASK_ID}', max_steps=5)
        frame = await env.reset()
        (output / 'fresh_container.png').write_bytes(frame.image)
        fresh = await inspect('fresh_container')
        if fresh != initial:
            raise RuntimeError('Fresh container did not reconstruct checked initial state')
        report['status'] = 'TASK_FILE_REINITIALIZATION_PASSED'
    finally:
        await env.close()


def main():
    if platform.system() != 'Linux' or not ROOT.is_dir():
        print('QZ_LINUX_REQUIRED')
        return 2
    output = ROOT / '.runtime/cua-host' / (
        'state-probe-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        + '-' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    report = {'schema_version': 1, 'task_id': TASK_ID, 'model_run': False,
              'actions_source': 'harness_oracle_and_state_mutation',
              'arbitrary_checkpoint_restore': 'NOT_TESTED',
              'status': 'STARTED', 'output': str(output),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        asyncio.run(probe(output, report))
    except Exception:
        report['status'] = 'FAILED'
        report['error'] = traceback.format_exc()
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'TASK_FILE_REINITIALIZATION_PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
