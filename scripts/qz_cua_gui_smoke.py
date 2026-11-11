#!/usr/bin/env python3
"""QZ GUI I/O check using a training task and a handwritten click; no model."""
import asyncio
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import traceback
import uuid

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')
STATE = ROOT / '.runtime/cua-host'
TASK_ID = 'synth_os_f_os_01__edit_server_name_0001'


async def check(output, report):
    from PIL import Image, ImageChops
    import lite.gym as gym
    from lite.core.tools import make_tool_call
    from lite.core.tools.action_space.geometry import pixel_to_norm

    image_name = 'cua-lite/lite.osworld:latest'
    while subprocess.run(['docker', '--host', 'unix:///var/run/docker.sock', 'image', 'inspect', image_name],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        pid = (STATE / 'desktop-mirror-pull.pid').read_text().strip()
        if not Path('/proc', pid).exists():
            raise RuntimeError('Desktop pull stopped before a verified image became available')
        await asyncio.sleep(30)

    def save_frame(data, name):
        if not data:
            raise RuntimeError('Environment returned no screenshot')
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width < 800 or height < 600:
            raise RuntimeError('Screenshot resolution below expected desktop size')
        extrema = image.convert('RGB').getextrema()
        if not any(low != high for low, high in extrema):
            raise RuntimeError('Screenshot is uniformly blank')
        (output / name).write_bytes(data)
        return {'filename': name, 'bytes': len(data), 'size': [width, height],
                'sha256': hashlib.sha256(data).hexdigest()}

    env = None
    try:
        print('STAGE=creating_training_task', flush=True)
        env = gym.make(f'lite.osworld@{TASK_ID}', max_steps=5)
        initial = await env.reset()
        report['initial'] = save_frame(initial.image, 'initial.png')
        coordinate = pixel_to_norm(35, 610, *report['initial']['size'])
        print('STAGE=initial_screenshot_passed', flush=True)
        step = await env.step([make_tool_call('computer', {'actions': [
            {'action': 'click', 'coordinate': coordinate},
        ]}, call_id='cua_smoke_click')])
        if not step.results or any(result.error for result in step.results):
            raise RuntimeError('Click returned an empty result or a tool error')
        frames = [image for result in step.results for image in result.images]
        if not frames:
            raise RuntimeError('Click returned no follow-up screenshot')
        report['after_click'] = save_frame(frames[-1], 'after_click.png')
        before = Image.open(io.BytesIO(initial.image)).convert('RGB')
        after = Image.open(io.BytesIO(frames[-1])).convert('RGB')
        histogram = ImageChops.difference(before, after).convert('L').histogram()
        changed_pixels = sum(histogram[1:])
        if changed_pixels < 10000:
            raise RuntimeError('Files dock click did not produce a substantial visible change')
        report['click'] = {'coordinate': coordinate, 'coordinate_space': 'normalized_0_1000',
                           'pixel_target': [35, 610], 'target': 'Files dock icon',
                           'changed_pixels': changed_pixels, 'result_count': len(step.results),
                           'reward': step.reward, 'terminated': step.terminated}
        print('STAGE=click_passed', flush=True)
        reset = await env.reset()
        report['reset'] = save_frame(reset.image, 'reset.png')
        report['status'] = 'GUI_IO_SMOKE_PASSED'
    finally:
        if env is not None:
            await env.close()


def main():
    if platform.system() != 'Linux' or not ROOT.is_dir():
        print('QZ_LINUX_REQUIRED')
        return 2
    os.environ['DOCKER_HOST'] = 'unix:///var/run/docker.sock'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    output = STATE / ('gui-smoke-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    report = {'schema_version': 1, 'status': 'STARTED', 'task_id': TASK_ID,
              'model_run': False, 'goal_completion_evaluated': False,
              'reset_state_identity': 'NOT_TESTED', 'output': str(output)}
    try:
        asyncio.run(check(output, report))
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc(limit=6)
    (output / 'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report['status'] == 'GUI_IO_SMOKE_PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
