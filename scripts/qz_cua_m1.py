#!/usr/bin/env python3
"""CPU acceptance or explicit, bounded GPU frozen-policy pilot. No training."""
import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import time
import traceback
import uuid

from cua_defense.assets import model_manifest, source_manifest
from cua_defense.measurement import digest_json, outcome
from cua_defense.m1 import analysis, jobs, pilot_task, task_manifest
from cua_defense.observation import PROTOCOL, geometry, save_inputs
from scripts.qz_cua_m0 import ROOT, PROJECT, control_actions, save
from scripts.qz_cua_batch import SHARED, SHARED_PROJECT, GPU_EXEC


def launch(model,mailbox,seconds):
    heartbeat=SHARED_PROJECT/'.runtime/gpu-relay-6f159838/state/heartbeat'
    if not heartbeat.exists() or time.time()-float(heartbeat.read_text())>30:
        raise RuntimeError('GPU relay offline; restore it before requesting a GPU run')
    snapshot=mailbox/'source';snapshot.mkdir()
    shutil.copytree(PROJECT/'cua_defense',snapshot/'cua_defense',ignore=shutil.ignore_patterns('__pycache__'))
    worker=snapshot/'qz_cua_hf_worker.py'
    supervisor=snapshot/'qz_cua_worker_session.py'
    shutil.copyfile(PROJECT/'scripts/qz_cua_hf_worker.py',worker)
    shutil.copyfile(PROJECT/'scripts/qz_cua_worker_session.py',supervisor)
    command=['python3',str(supervisor),'--python',str(SHARED_PROJECT/'.venv-rsd/bin/python'),
             '--worker',str(worker),'--model',model['gpu_model_path'],'--mailbox',str(mailbox),
             '--seconds',str(max(1,min(1800,seconds)))]
    shell='\n'.join(['source .runtime/qz/runtime_env.sh','export CUDA_VISIBLE_DEVICES=0',
        'export PYTHONPATH='+shlex.quote(str(snapshot)),
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1',
        'nohup '+shlex.join(command)+' >'+shlex.quote(str(mailbox/'supervisor.log'))+' 2>&1 </dev/null &'])
    subprocess.run(['env','GPU_RELAY_DIR='+str(SHARED_PROJECT/'.runtime/gpu-relay-6f159838'),str(GPU_EXEC),
        '--timeout','45','--cwd',str(SHARED_PROJECT),'--','bash','-lc',shell],check=True)


async def fixture_responder(mailbox,queue):
    """Exercise real mailbox IO with explicitly scripted, non-policy responses."""
    number=0
    try:
        save(mailbox/'ready.json',{'model_run':False,'source':'scripted_mailbox_fixture'})
        while not (mailbox/'stop').exists():
            request=mailbox/f'request-{number}.json'
            if not request.exists():
                await asyncio.sleep(.02);continue
            data=json.loads(request.read_text())
            if not data['images'] or any(not Path(p).is_file() for p in data['images']):
                raise ValueError('Fixture did not receive actual saved images')
            if not queue: raise ValueError('Unexpected scripted fixture request')
            save(mailbox/f'response-{number}.json',{'response':queue.pop(0),'finish_reason':'stop',
                'source':'scripted_mailbox_fixture','model_run':False})
            number+=1
    except Exception:
        save(mailbox/'error.json',{'error':traceback.format_exc()})
        raise
    finally:
        save(mailbox/'stopped.json',{'source':'scripted_mailbox_fixture','requests':number})


async def run(config,output,report,model,cpu_fixture):
    # Imports are CPU preparation and occur before a GPU worker is dispatched.
    import lite.gym as gym
    from lite.agents.factory import make
    from lite.agents.core.agent.logger import TrajectoryLogger
    from transformers import AutoProcessor
    from cua_defense.mailbox import MailboxClient
    from cua_defense.runtime import VerifiedTaskEnv
    from cua_defense.tasks import register_row
    from cua_defense.replay import checkpoint
    processor=AutoProcessor.from_pretrained(model['processor_path'],local_files_only=True)
    deadline=time.monotonic()+(15 if cpu_fixture else config['budget_minutes'])*60
    mailbox=(output/'mailbox') if cpu_fixture else SHARED/('m1-'+uuid.uuid4().hex[:12])
    mailbox.mkdir(parents=True,exist_ok=False)
    save(output/'mailbox.json',{'path':str(mailbox),'source_snapshot':not cpu_fixture})
    save(mailbox/'expected_model.json',report['model_assets'])
    client=MailboxClient(mailbox,timeout=min(180,config['episode_timeout_seconds']))
    queue=[];fixture=None
    (output/'episodes').mkdir(exist_ok=True)
    done={r['case_id'] for r in report['results']}
    try:
        if cpu_fixture:
            fixture=asyncio.create_task(fixture_responder(mailbox,queue))
        else:
            launch(model,mailbox,deadline-time.monotonic())
            report['gpu_dispatched']=True;save(output/'receipt.json',report)
        startup_deadline=min(deadline,time.monotonic()+600)
        while not (mailbox/'ready.json').exists():
            if (mailbox/'error.json').exists(): raise RuntimeError((mailbox/'error.json').read_text())
            if (mailbox/'worker.exit.json').exists(): raise RuntimeError('Worker exited before ready')
            if time.monotonic()>=startup_deadline:
                raise TimeoutError('Worker startup deadline exhausted')
            await asyncio.sleep(.2)
        report['worker_ready']=json.loads((mailbox/'ready.json').read_text())
        if report['worker_ready']['model_run'] != (not cpu_fixture):
            raise ValueError('Unexpected worker mode')
        for job in report['planned_jobs']:
            if job['case_id'] in done: continue
            if deadline-time.monotonic()<15:
                report['stop_reason']='SESSION_BUDGET';break
            task,row=pilot_task(job['family'],job['condition'])
            directory=output/'episodes'/job['case_id'];directory.mkdir(exist_ok=False)
            record={**job,'execution_valid':False,'training_eligible':False,'model_run':not cpu_fixture,
                    'contract_sha256':digest_json(row),'policy_trace_count':0}
            env=None;score=None;started=time.monotonic();turn=0
            try:
                env=VerifiedTaskEnv(gym.make(register_row(row),max_steps=config['max_steps'],seed=job['seed'],
                    loop_detect=0,extra_tools=['response','terminate']),
                    '/home/user/Desktop/'+task['filename'],task['after'].encode())
                if cpu_fixture:
                    if queue: raise ValueError('Unconsumed fixture actions')
                    queue.extend(control_actions(task,'safe_completion'))
                underlying=client.generator(directory,seed=job['seed'],sampling=config['sampling_params'],
                                            capture_policy=not cpu_fixture)
                async def generate(**data):
                    nonlocal turn
                    await checkpoint(env,row,data['images'][-1],directory,turn)
                    images=save_inputs(data['images'],directory,turn)
                    (directory/f'prompt-{turn}.txt').write_text(data['prompt'])
                    try: window=await geometry(env._computer.interface)
                    except Exception as error:
                        window={'status':'UNKNOWN','error':str(error)}
                    save(directory/f'geometry-{turn}.json',window)
                    answer=await underlying(**{**data,'images':images})
                    if not cpu_fixture: record['policy_trace_count']+=1
                    turn+=1
                    return answer
                agent=make(model['model_id'],env=env,processor=processor,generate_fn=generate)
                await asyncio.wait_for(agent.sample(env,hooks=[TrajectoryLogger(directory,save_data=True,
                    save_video=False)]),min(config['episode_timeout_seconds'],deadline-time.monotonic()))
                summary=json.loads((directory/'summary.json').read_text())
                score=env.final_task_score
                record['final_target_sha256']=env.final_target_sha256
                if score is None: raise ValueError('Final target read unavailable: '+str(env.final_read_error))
                record['native_score']=summary['episode_return']
                if summary['episode_return']!=score: raise ValueError('Independent utility disagrees with native evaluator')
                if cpu_fixture and queue: raise ValueError('Fixture did not execute all controls')
                record['execution_valid']=True
            except Exception:
                record['error']=traceback.format_exc()
            finally:
                risk=None
                if env is not None:
                    try:
                        await env.close();risk=env.risk
                    except Exception:
                        record['close_error']=traceback.format_exc();record['execution_valid']=False
                record.update(risk=risk,labels=outcome(score,risk,record['execution_valid']),
                              seconds=round(time.monotonic()-started,3),turns=turn)
                save(directory/'receipt.json',record)
                report['results'].append(record);report['analysis']=analysis(report['results'],report['planned_jobs'])
                save(output/'receipt.json',report)
                print(json.dumps({k:record.get(k) for k in ['case_id','execution_valid','labels','error']}),flush=True)
            if not record['execution_valid']:
                report['stop_reason']='FIRST_INVALID_EPISODE';break
    finally:
        client.close();(mailbox/'stop').touch()
        if fixture:
            await fixture
            report['worker_exit']={'status':'SCRIPTED_RESPONDER_STOPPED','gpu_used':False}
        else:
            for _ in range(75):
                if (mailbox/'worker.exit.json').exists(): break
                await asyncio.sleep(.2)
            report['worker_exit']=json.loads((mailbox/'worker.exit.json').read_text()) if (mailbox/'worker.exit.json').exists() else {'status':'UNKNOWN_STOP_REQUESTED'}
        # Raw IO stays once in its original mailbox; no wholesale duplicate copy.
        save(output/'receipt.json',report)
    report['analysis']=analysis(report['results'],report['planned_jobs'])
    report['status']='PANEL_COMPLETE' if len(report['results'])==len(report['planned_jobs']) and all(r['execution_valid'] for r in report['results']) else 'PANEL_INCOMPLETE'


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--cpu-scripted',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--qualification',type=Path)
    args=parser.parse_args()
    config=json.loads((PROJECT/'configs/cua/m1_pilot.json').read_text())
    source=source_manifest(PROJECT)
    planned=jobs(config,args.cpu_scripted)
    report={'schema_version':1,'phase':'CPU_M1_ACCEPTANCE' if args.cpu_scripted else 'M1',
        'status':'PLAN_ONLY','model_run':False,'gpu_dispatched':False,'training_updates':0,'training_eligible':False,
        'created_at':datetime.now(timezone.utc).isoformat(),'config':config,'source_sha256':source,
        'protocol':PROTOCOL,'task_manifest':task_manifest(config),'planned_jobs':planned,'results':[],
        'causal_branch_eligible':False,'full_environment_equivalence':'NOT_ESTABLISHED'}
    if not (args.run or args.cpu_scripted):
        print(json.dumps(report,indent=2));return 0
    if platform.system()!='Linux' or not ROOT.is_dir(): raise SystemExit('QZ_CPU_REQUIRED')
    if args.output is None: raise SystemExit('--output is required for execution')
    if args.run:
        if args.qualification is None: raise SystemExit('CPU qualification receipt required')
        gate=json.loads(args.qualification.read_text())
        if gate.get('status')!='READY_FOR_GPU_MEASUREMENT' or gate['source_sha256']!=source:
            raise SystemExit('CPU qualification or source fingerprint mismatch')
    model=json.loads((PROJECT/'configs/cua/models.json').read_text())[config['model']]
    report['model_assets']=model_manifest(model['processor_path'])
    output=args.output.resolve();output.relative_to(ROOT.resolve())
    if args.resume:
        if args.cpu_scripted: raise SystemExit('CPU fixture retries require a new directory')
        old=json.loads((output/'receipt.json').read_text())
        for key in ['source_sha256','config','model_assets','planned_jobs']:
            if old[key]!=report[key]: raise SystemExit('Resume identity differs: '+key)
        if old['status'] not in ['PANEL_INCOMPLETE','PANEL_COMPLETE','FAILED']:
            raise SystemExit('Previous controller must have a terminal receipt before resume')
        report['results']=old['results']
        save(output/('prior-receipt-'+uuid.uuid4().hex[:8]+'.json'),old)
    else: output.mkdir(exist_ok=False,parents=True)
    with (output/'.controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        report['status']='STARTED';report['model_run']=args.run;save(output/'receipt.json',report)
        if len(report['results'])==len(planned):
            report['status']='NO_UNATTEMPTED_EPISODES';save(output/'receipt.json',report);return 0
        try: asyncio.run(run(config,output,report,model,args.cpu_scripted))
        except Exception:
            report.update(status='FAILED',error=traceback.format_exc())
        report['analysis']=analysis(report['results'],planned)
        save(output/'receipt.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['results','planned_jobs','source_sha256','task_manifest','model_assets','worker_ready']},indent=2))
    return 0 if report['status']=='PANEL_COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
