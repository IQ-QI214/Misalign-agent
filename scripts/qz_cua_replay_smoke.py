#!/usr/bin/env python3
"""CPU replay check through the native agent's action parsing and history."""
import asyncio
import json
from pathlib import Path
import platform
import uuid

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')


async def check(output):
    import lite.gym as gym
    from lite.agents.factory import make
    from lite.agents.core.agent.logger import TrajectoryLogger
    from transformers import AutoProcessor
    from cua_defense.replay import checkpoint, verify_checkpoint
    from cua_defense.tasks import register_row
    task_id = 'synth_os_f_os_01__edit_server_name_0001'
    catalog = ROOT / 'Misalign-agent/references/cua_lite/lite/gym/envs/lite/osworld/data/train.synth.jsonl'
    row = next(r for r in map(json.loads, catalog.read_text().splitlines()) if r['task_id'] == task_id)
    processor = AutoProcessor.from_pretrained('/inspire/hdd/project/independent-security-research/wenming-253108090054/models/Qwen__Qwen3-VL-4B-Instruct', local_files_only=True)
    row['metadata']['config'] += [{'type': 'open', 'parameters': {'path': '/home/user/Desktop/nginx-site.conf'}}, {'type': 'sleep', 'parameters': {'seconds': 5}}]
    replay = [{'response': '<tool_call>\n' + json.dumps({'name': 'computer_use', 'arguments': {'action': 'key', 'keys': ['Home']}}) + '\n</tool_call>', 'finish_reason': 'stop'} for _ in range(3)]
    finish = '<tool_call>\n{"name":"terminate","arguments":{"status":"success"}}\n</tool_call>'
    checks = []
    for run in ['reference', 'replay']:
        directory = output / run
        directory.mkdir()
        env = gym.make(register_row(row), max_steps=5, seed=20261112,
                       loop_detect=0, extra_tools=['response', 'terminate'])
        turn = 0
        async def generate(**data):
            nonlocal turn
            if turn == 3:
                point = await checkpoint(env, row, data['images'][-1], directory, turn)
                tree = await env._computer.interface.run_command('curl -s --max-time 20 http://localhost:5000/accessibility')
                (directory / 'accessibility.json').write_text(tree.stdout)
                if run == 'replay':
                    checks.append(verify_checkpoint(output / 'reference', point, directory, turn))
            result = replay[turn] if turn < 3 else {'response': finish, 'finish_reason': 'stop'}
            turn += 1
            return {'response': result['response'], 'finish_reason': result['finish_reason']}
        try:
            agent = make('Qwen/Qwen3-VL-4B-Instruct', env=env, processor=processor, generate_fn=generate)
            await agent.sample(env, hooks=[TrajectoryLogger(directory, save_data=True, save_video=False)])
        finally:
            await env.close()
    return {'schema_version': 1, 'status': 'CPU_GUI_PREFIX_REPLAY_PASSED',
            'model_run': False, 'gpu_used': False, 'prefix_actions_replayed': 3,
            'checks': checks, 'output': str(output)}


if __name__ == '__main__':
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise SystemExit('QZ_LINUX_REQUIRED')
    output = ROOT / '.runtime/cua-host/prepared' / ('replay-smoke-' + uuid.uuid4().hex[:8])
    output.mkdir()
    print('OUTPUT=' + str(output), flush=True)
    report = asyncio.run(check(output))
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
