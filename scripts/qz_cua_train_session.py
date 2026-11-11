#!/usr/bin/env python3
"""CPU orchestration of one-GPU SFT with native GUI milestone evaluation."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from types import SimpleNamespace
import uuid
from scripts.qz_cua_batch import PROJECT, STATE, SHARED, SHARED_PROJECT, GPU_EXEC, run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--eval-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--interval', type=int, default=25)
    parser.add_argument('--minutes', type=float, default=60)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.steps <= 0 or args.interval <= 0 or args.minutes <= 0:
        raise ValueError('Steps, interval, and budget must be positive')
    config = json.loads(args.eval_config.read_text())
    if config['data_role'] != 'evaluation':
        raise ValueError('Milestone evaluation config must have evaluation data role')
    model = json.loads((PROJECT / 'configs/cua/models.json').read_text())[config['model']]
    from cua_defense.tasks import load_rows
    eval_ids = {r['task_id'] for r in load_rows(PROJECT / 'references/cua_lite', PROJECT / config['cohort'])}
    import pyarrow.parquet as pq
    rows = pq.read_table(args.data).to_pylist()
    for row in rows:
        metadata = json.loads(row['metadata'])
        if metadata.get('others', {}).get('research_training_eligible') is not True:
            raise ValueError('Unverified or diagnostic training row')
        if metadata['others']['task_id'].split('__risk_')[0] in eval_ids:
            raise ValueError('Training task overlaps milestone evaluation')
    if not rows:
        raise ValueError('Training dataset is empty')
    from transformers import AutoProcessor
    from cua_defense.training import StepDataset
    processor = AutoProcessor.from_pretrained(model['processor_path'], local_files_only=True)
    dataset = StepDataset(args.data, processor)
    lengths = []
    targets = 0
    for index in range(len(dataset)):
        example = dataset[index]
        lengths.append(example['input_ids'].shape[-1])
        targets += int((example['labels'] != -100).sum())
    milestones = sorted({0, args.steps, *range(args.interval, args.steps, args.interval)})
    plan = {'status': 'CPU_TRAIN_SESSION_PLAN_PASSED', 'gpu_dispatched': False,
            'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
            'training_rows': len(rows), 'milestones': milestones, 'evaluation_tasks': len(eval_ids),
            'supervised_examples': len(lengths), 'unique_target_tokens': targets, 'max_sequence_tokens': max(lengths),
            'model': model['model_id'], 'gpu_count': 1, 'budget_minutes': args.minutes}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    heartbeat = SHARED_PROJECT / '.runtime/gpu-relay-6f159838/state/heartbeat'
    if not heartbeat.exists() or time.time() - float(heartbeat.read_text()) > 30:
        raise RuntimeError('GPU relay offline')
    args.output.mkdir(parents=True, exist_ok=False)
    session = SHARED / ('train-' + uuid.uuid4().hex[:12])
    session.mkdir()
    shutil.copytree(PROJECT / 'cua_defense', session / 'code/cua_defense')
    shutil.copyfile(PROJECT / 'scripts/qz_cua_train.py', session / 'train.py')
    shutil.copyfile(args.data, session / 'train.parquet')
    plan['shared_session'] = str(session)
    write_json(args.output / 'plan.json', plan)
    command = shlex.join([str(SHARED_PROJECT / '.venv-rsd/bin/python'), str(session / 'train.py'),
        '--data', str(session / 'train.parquet'), '--model', model['gpu_model_path'],
        '--output', str(session / 'output'), '--steps', str(args.steps), '--eval-root', str(session / 'eval'),
        '--milestones', *map(str, milestones), '--budget-minutes', str(args.minutes)])
    supervised = 'timeout --signal=TERM --kill-after=30s ' + str(int(args.minutes*60)) + 's ' + command
    supervised += '; training_rc=$?; printf \'%s\\n\' "$training_rc" >' + shlex.quote(str(session / 'exit_code'))
    script = '\n'.join(['source .runtime/qz/runtime_env.sh', 'export CUDA_VISIBLE_DEVICES=0',
        'export PYTHONPATH=' + shlex.quote(str(session / 'code')),
        'nohup bash -lc ' + shlex.quote(supervised) +
        ' >' + shlex.quote(str(session / 'training.log')) + ' 2>&1 </dev/null &',
        'echo $! >' + shlex.quote(str(session / 'worker.pid'))])
    subprocess.run(['env', 'GPU_RELAY_DIR=' + str(SHARED_PROJECT / '.runtime/gpu-relay-6f159838'),
        str(GPU_EXEC), '--timeout', '45', '--cwd', str(SHARED_PROJECT), '--', 'bash', '-lc', script], check=True)
    deadline = time.monotonic() + args.minutes * 60
    evaluated = set()
    while time.monotonic() < deadline:
        for directory in sorted((session / 'eval').glob('step-*')):
            if directory.name in evaluated or not (directory / 'ready.json').exists():
                continue
            options = SimpleNamespace(limit=None, repair_from=None, repair_mode='full',
                external_mailbox=directory, keep_external_worker=False, minutes=max(0.01, (deadline-time.monotonic())/60))
            target = args.output / directory.name
            target.mkdir()
            asyncio.run(run(options, target, config, model))
            evaluated.add(directory.name)
        if (session / 'exit_code').exists():
            break
        time.sleep(2)
    shutdown_deadline = time.monotonic() + 45
    while not (session / 'exit_code').exists() and time.monotonic() < shutdown_deadline:
        time.sleep(1)
    if (session / 'output').exists():
        shutil.copytree(session / 'output', args.output / 'training', dirs_exist_ok=True)
    shutil.copyfile(session / 'training.log', args.output / 'training.log')
    write_json(args.output / 'receipt.json', {'status': 'TRAIN_SESSION_RETURNED',
        'exit_code': int((session / 'exit_code').read_text()) if (session / 'exit_code').exists() else None,
        'completed_milestones': sorted(evaluated), 'requested_milestones': milestones,
        'training_result_present': (session / 'output/training_result.json').exists(),
        'temporary_session_retained': str(session),
        'retirement_policy': 'Normal completion reloads, hashes, and retires adapter weights. Inspect interrupted output before cleanup.'})
    print('OUTPUT=' + str(args.output))


if __name__ == '__main__':
    main()
