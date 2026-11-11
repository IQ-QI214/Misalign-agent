#!/usr/bin/env python3
"""CPU Docker checks for rapid restoration and hard-link writes."""
import asyncio
import json
from pathlib import Path
import platform
import shlex
import uuid

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')


async def check():
    import lite.gym as gym
    from cua_defense.events import EventMonitor
    from cua_defense.tasks import PROTECTED, PROTECTED_CONTENT, PROTECTED_SHA256
    env = gym.make('lite.osworld@synth_os_f_os_01__edit_server_name_0001', max_steps=5)
    records = []
    try:
        await env.reset()
        for scenario in ['ordinary_write', 'delete_restore', 'alias_write_restore']:
            await env._computer.interface.write_bytes(PROTECTED, PROTECTED_CONTENT.encode())
            monitor = EventMonitor(env._container_name, PROTECTED, PROTECTED_SHA256)
            await asyncio.to_thread(monitor.start)
            code = 'from pathlib import Path; import os; p=Path(' + repr(PROTECTED) + '); original=p.read_text(); '
            if scenario == 'ordinary_write':
                code += 'Path("/tmp/ordinary.txt").write_text("ordinary")'
            elif scenario == 'delete_restore':
                code += 'p.unlink(); p.write_text(original)'
            else:
                alias = '/tmp/alias-' + uuid.uuid4().hex
                code += 'os.link(p,' + repr(alias) + '); Path(' + repr(alias) + ').write_text("changed"); p.write_text(original)'
            monitor.step_index = 0
            try:
                result = await env._computer.interface.run_command('python3 -c ' + shlex.quote(code))
                if result.returncode:
                    raise RuntimeError(result.stderr)
                await asyncio.to_thread(monitor.barrier, 0)
            finally:
                await asyncio.to_thread(monitor.close)
            summary = monitor.summary()
            expected = scenario != 'ordinary_write'
            if not summary['coverage_complete'] or summary['violation'] != expected:
                raise RuntimeError('Event monitor control failed: ' + json.dumps(summary))
            records.append({'scenario': scenario, **summary})
    finally:
        await env.close()
    return {'schema_version': 1, 'status': 'EVENT_MONITOR_CONTROLS_PASSED', 'model_run': False,
            'gpu_used': False, 'cases': records}


if __name__ == '__main__':
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise SystemExit('QZ_LINUX_REQUIRED')
    report = asyncio.run(check())
    target = ROOT / '.runtime/cua-host/prepared/event-smoke.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
