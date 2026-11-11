#!/usr/bin/env python3
"""One native CUA-Lite agent episode using a serialized shared-mailbox VLM."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')
MODEL = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/models/Qwen__Qwen3-VL-4B-Instruct')


async def run(args, output, report):
    import importlib
    importlib.import_module("jsonschema")
    importlib.import_module("pyarrow")
    from transformers import AutoProcessor
    import lite.gym as gym
    from lite.agents.factory import make
    from lite.agents.core.agent.logger import TrajectoryLogger

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    calls = 0

    async def generate(*, prompt, images, **kwargs):
        nonlocal calls
        number = calls
        calls += 1
        paths = []
        for index, image in enumerate(images or []):
            path = args.mailbox / f'image-{number}-{index}.png'
            image.save(path)
            paths.append(str(path))
        temporary = args.mailbox / f'request-{number}.tmp'
        temporary.write_text(json.dumps({'prompt': prompt, 'images': paths}) + '\n')
        temporary.replace(args.mailbox / f'request-{number}.json')
        deadline = time.monotonic() + 600
        response = args.mailbox / f'response-{number}.json'
        while not response.exists():
            error = args.mailbox / 'error.json'
            if error.exists():
                raise RuntimeError(error.read_text())
            if (args.mailbox / 'stopped.json').exists():
                raise RuntimeError('GPU worker stopped before responding')
            if time.monotonic() > deadline:
                raise TimeoutError('GPU response timed out')
            await asyncio.sleep(0.5)
        data = json.loads(response.read_text())
        (output / response.name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        print(f'MODEL_TURN={calls} INPUT={data["input_tokens"]} OUTPUT={data["output_tokens"]}', flush=True)
        return {'response': data['response'], 'finish_reason': data['finish_reason']}

    env = gym.make(f'lite.osworld@{args.task_id}', max_steps=20,
                   loop_detect=0, extra_tools=['response', 'terminate'])
    try:
        logger = TrajectoryLogger(output, save_data=True, save_video=False,
                                  debug_artifacts=True, env_id='lite.osworld', task_id=args.task_id)
        agent = make('Qwen/Qwen3-VL-4B-Instruct', env=env,
                     processor=processor, generate_fn=generate)
        await agent.sample(env, hooks=[logger])
        report['status'] = 'EPISODE_RETURNED'
    finally:
        await env.close()
        report['model_calls'] = calls
        (args.mailbox / 'stop').touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mailbox', type=Path, required=True)
    parser.add_argument('--task-id', default='synth_os_f_os_01__edit_server_name_0001')
    args = parser.parse_args()
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise SystemExit('QZ_LINUX_REQUIRED')
    output = ROOT / '.runtime/cua-host' / ('model-pilot-' + args.mailbox.name)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'task_id': args.task_id,
              'model': 'Qwen/Qwen3-VL-4B-Instruct', 'model_run': True,
              'training_updates': 0, 'max_steps': 20, 'loop_detect': 0,
              'status': 'STARTED', 'output': str(output),
              'time_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        asyncio.run(run(args, output, report))
    except Exception:
        report['status'] = 'FAILED'
        report['error'] = traceback.format_exc()
        (args.mailbox / 'stop').touch()
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'EPISODE_RETURNED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
