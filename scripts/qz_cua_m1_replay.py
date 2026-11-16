#!/usr/bin/env python3
"""Replay one saved M1 response sequence on CPU, keeping the original result intact."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

from scripts.qz_cua_m0 import ROOT, PROJECT, save
from cua_defense.m1 import pilot_task
from cua_defense.measurement import digest_json


async def replay(args, report):
    import lite.gym as gym
    from lite.agents.factory import make
    from lite.agents.core.agent.logger import TrajectoryLogger
    from transformers import AutoProcessor
    from cua_defense.observation import model_image, pixel_hash
    from cua_defense.runtime import VerifiedTaskEnv
    from cua_defense.tasks import register_row

    original = json.loads((args.source / 'receipt.json').read_text())
    task, row = pilot_task(original['family'], original['condition'])
    if digest_json(row) != original['contract_sha256']:
        raise ValueError('Task changed since the saved trajectory')
    responses = [args.source / f'response-{i}.json' for i in range(original['turns'])]
    texts = [json.loads(p.read_text()) for p in responses]
    report.update(source_case=original['case_id'], source_valid=original['execution_valid'],
                  response_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in responses])
    model = json.loads((PROJECT / 'configs/cua/models.json').read_text())['qwen35_9b']
    processor = AutoProcessor.from_pretrained(model['processor_path'], local_files_only=True)
    timings = report['timings'] = []

    class TimedEnv(VerifiedTaskEnv):
        async def timed(self, phase, operation):
            started = time.monotonic()
            event = {'phase': phase, 'status': 'STARTED'}
            timings.append(event)
            save(args.output / 'receipt.json', report)
            try:
                result = await operation
                event['status'] = 'RETURNED'
                return result
            except BaseException as error:
                event.update(status='ERROR', error=type(error).__name__)
                raise
            finally:
                event['seconds'] = round(time.monotonic() - started, 3)
                save(args.output / 'receipt.json', report)
                print(json.dumps(event), flush=True)

        async def reset(self):
            return await self.timed('reset', super().reset())

        async def step(self, actions):
            return await self.timed('step-' + str(self.step_index + 1), super().step(actions))

        async def close(self):
            if self.env._computer is not None and not (args.output / 'final-target.txt').exists():
                try:
                    data = await self.env._computer.interface.read_bytes(self.target)
                    (args.output / 'final-target.txt').write_bytes(data)
                except Exception as error:
                    report['diagnostic_read_error'] = str(error)
            return await self.timed('close', super().close())

    env = TimedEnv(gym.make(register_row(row), max_steps=len(texts), seed=original['seed'],
                           loop_detect=0, extra_tools=['response', 'terminate']),
                   '/home/user/Desktop/' + task['filename'], task['after'].encode())
    turn = 0
    report['observations'] = []

    async def generate(**data):
        nonlocal turn
        if turn >= len(texts):
            raise ValueError('Replay requested an unsaved model response')
        images = [model_image(image) for image in data['images']]
        hashes = [pixel_hash(image) for image in images]
        prior = json.loads((args.source / f'observation-{turn}.json').read_text())
        report['observations'].append({'turn': turn, 'input_sha256': hashes,
            'matches_source_images': hashes == [r['input_pixel_sha256'] for r in prior['images']],
            'matches_source_prompt': data['prompt'] == (args.source / f'prompt-{turn}.txt').read_text()})
        images[-1].save(args.output / f'input-{turn}.png')
        result = texts[turn]
        turn += 1
        return {'response': result['response'], 'finish_reason': result['finish_reason']}

    started = time.monotonic()
    try:
        agent = make(model['model_id'], env=env, processor=processor, generate_fn=generate)
        await asyncio.wait_for(agent.sample(env, hooks=[TrajectoryLogger(args.output,
            save_data=True, save_video=False)]), args.timeout)
        summary = json.loads((args.output / 'summary.json').read_text())
        report['native_score'] = summary['episode_return']
        report['status'] = 'REPLAY_COMPLETED'
    except Exception:
        report.update(status='REPLAY_FAILED', error=traceback.format_exc())
    finally:
        await env.close()
        report.update(replayed_responses=turn, independent_score=env.final_task_score,
                      final_target_sha256=env.final_target_sha256, risk=env.risk,
                      seconds=round(time.monotonic() - started, 3))
    if report.get('native_score') != report['independent_score']:
        report['evaluator_agreement'] = False
    elif report['independent_score'] is not None:
        report['evaluator_agreement'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if not args.run:
        print('PLAN_ONLY: saved responses, fresh CPU desktop, no model inference')
        return 0
    if platform.system() != 'Linux' or '--91bb3a6ab7a7-' not in platform.node():
        raise SystemExit('THIS_QZ_CPU_NOTEBOOK_REQUIRED')
    for path in (args.source, args.output):
        path.resolve().relative_to(ROOT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'STARTED', 'gpu_used': False, 'model_run': False,
              'training_eligible': False, 'original_result_overwritten': False,
              'timeout_seconds': args.timeout, 'source': str(args.source),
              'scope': 'Saved action-text replay with generation latency removed; not a new policy rollout.'}
    try:
        asyncio.run(replay(args, report))
    except Exception:
        report.update(status='REPLAY_FAILED', error=traceback.format_exc())
    save(args.output / 'receipt.json', report)
    print(json.dumps(report), flush=True)
    return 0 if report['status'] == 'REPLAY_COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
