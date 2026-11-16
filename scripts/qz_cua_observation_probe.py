#!/usr/bin/env python3
"""Six fresh-desktop controls for a new, explicitly transformed input protocol."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time
import traceback

from cua_defense.measurement import compare_prefix, digest_json, outcome
from cua_defense.observation import PROTOCOL, geometry, pixel_hash, save_inputs, stable_task
from scripts.qz_cua_m0 import ROOT, PROJECT, control_actions, save, task_row


async def run(config, output, report):
    import lite.gym as gym
    from lite.agents.factory import make
    from lite.agents.core.agent.logger import TrajectoryLogger
    from transformers import AutoProcessor
    from cua_defense.runtime import MonitoredEnv
    from cua_defense.tasks import register_row
    from cua_defense.replay import checkpoint
    from scripts.qz_cua_judge_controls import TASKS
    model = json.loads((PROJECT / 'configs/cua/models.json').read_text())['qwen35_9b']
    processor = AutoProcessor.from_pretrained(model['processor_path'], local_files_only=True)
    deadline = time.monotonic() + config['budget_minutes'] * 60
    references = {}
    for family, repeat in itertools.product(config['families'], range(config['repetitions'])):
        if time.monotonic() >= deadline:
            report['stop_reason'] = 'BUDGET_EXHAUSTED'
            break
        task = next(t for t in TASKS if t['name'] == family)
        row = stable_task(task_row(task))
        case_id = f'{family}__r{repeat}'
        directory = output / case_id
        directory.mkdir()
        record = {'case_id': case_id, 'execution_valid': False, 'training_eligible': False}
        sequence = control_actions(task, 'safe_completion')
        responses = []
        env, turn, task_score = None, 0, None
        started = time.monotonic()
        try:
            env = MonitoredEnv(gym.make(register_row(row), max_steps=len(sequence)+1,
                seed=config['seed'], loop_detect=0, extra_tools=['response', 'terminate']))
            async def generate(**data):
                nonlocal turn, task_score
                images = save_inputs(data['images'], directory, turn)
                if turn <= 1:
                    point = await checkpoint(env, row, data['images'][-1], directory, turn)
                    window = await geometry(env._computer.interface)
                    captured = {'contract_sha256': digest_json(row),
                        'prompt_sha256': hashlib.sha256(data['prompt'].encode()).hexdigest(),
                        'images_sha256': [pixel_hash(image) for image in images],
                        'raw_images_sha256': [pixel_hash(image) for image in data['images']],
                        'prior_responses_sha256': digest_json(responses), 'files': point['files'],
                        'focus_stdout': point['focus']['stdout'].strip(),
                        'prior_violation': any(e['type']=='violation' for e in env.monitor.events),
                        'window': window}
                    save(directory / f'prefix-{turn}.json', captured)
                    if turn == 1:
                        record['prefix'] = captured
                        if repeat == 0:
                            references[family] = captured
                            record['comparison'] = {'status': 'REFERENCE'}
                        else:
                            reference = references[family]
                            record['comparison'] = compare_prefix(reference, captured)
                            record['comparison']['window_match'] = reference['window'] == window
                            record['comparison']['raw_image_match'] = reference['raw_images_sha256'] == captured['raw_images_sha256']
                if turn == len(sequence)-1:
                    target = '/home/user/Desktop/' + task['filename']
                    task_score = int((await env._computer.interface.read_bytes(target)).decode() == task['after'])
                text = sequence[turn]
                responses.append(text)
                save(directory / f'response-{turn}.json', {'response': text, 'source': 'scripted_control'})
                turn += 1
                return {'response': text, 'finish_reason': 'stop'}
            agent = make(model['model_id'], env=env, processor=processor, generate_fn=generate)
            await asyncio.wait_for(agent.sample(env, hooks=[TrajectoryLogger(directory, save_data=True,
                save_video=False)]), min(config['episode_timeout_seconds'], max(1,deadline-time.monotonic())))
            summary = json.loads((directory / 'summary.json').read_text())
            record['native_score'] = summary['episode_return']
            if turn != len(sequence) or summary['episode_return'] != task_score:
                raise RuntimeError('Independent utility or control schedule mismatch')
            record['execution_valid'] = True
        except Exception:
            record['error'] = traceback.format_exc()
        finally:
            risk = None
            if env is not None:
                try:
                    await env.close()
                    risk = env.risk
                except Exception:
                    record['close_error'] = traceback.format_exc()
                    record['execution_valid'] = False
            record['risk'] = risk
            record['labels'] = outcome(task_score, risk, record['execution_valid'])
            record['seconds'] = round(time.monotonic()-started,3)
            record['control_match'] = record['execution_valid'] and record['labels']['safe_completion'] is True
            report['results'].append(record)
            save(directory / 'receipt.json', record)
            save(output / 'receipt.json', report)
            print(json.dumps({'case_id':case_id,'control_match':record['control_match'],
                              'comparison':record.get('comparison'),'error':record.get('error')}),flush=True)
        if not record['control_match']:
            report['stop_reason'] = 'FIRST_INVALID_CONTROL'
            break
    rows = report['results']
    pairs = [r['comparison'] for r in rows if r.get('comparison',{}).get('status')!='REFERENCE'
             and 'comparison' in r]
    matched = sum(p['status']=='OBSERVED_PREFIX_MATCH' and p.get('window_match') is True for p in pairs)
    report.update(status='PROBE_COMPLETE' if len(rows)==report['registered'] and all(r['control_match'] for r in rows)
                  else 'PROBE_INCOMPLETE', attempted=len(rows), valid=sum(r['control_match'] for r in rows),
                  prefix_pairs=len(pairs), matched_pairs=matched,
                  observation_qualified=len(pairs)==report['registered']-len(config['families']) and matched==len(pairs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    config = json.loads((PROJECT/'configs/cua/observation_probe.json').read_text())
    report = {'phase':'M0_observation_v2','status':'PLAN_ONLY','registered':6,'protocol':PROTOCOL,
              'model_run':False,'gpu_used':False,'training_eligible':False,'training_updates':0,
              'full_environment_equivalence':'NOT_ESTABLISHED','causal_branch_eligible':False,
              'created_at':datetime.now(timezone.utc).isoformat(),'results':[],
              'source_sha256':{p:hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() for p in
                 ['scripts/qz_cua_observation_probe.py','cua_defense/observation.py','scripts/qz_cua_m0.py',
                  'cua_defense/measurement.py','configs/cua/observation_probe.json']}}
    if not args.run:
        print(json.dumps(report,indent=2)); return 0
    if platform.system()!='Linux' or not ROOT.is_dir():
        raise SystemExit('QZ_CPU_REQUIRED')
    output=args.output.resolve();output.relative_to(ROOT.resolve());output.mkdir(exist_ok=False)
    report['status']='STARTED';save(output/'receipt.json',report)
    try:
        asyncio.run(run(config,output,report))
    except Exception:
        report.update(status='PROBE_FAILED',error=traceback.format_exc())
    save(output/'receipt.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)
    return 0 if report['status']=='PROBE_COMPLETE' else 1


if __name__=='__main__':
    raise SystemExit(main())
