#!/usr/bin/env python3
"""One command for a bounded M1 session; no GPU work without --run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

from scripts.qz_cua_dind import ROOT, checked_path
from scripts.qz_cua_worker_session import supervise


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--cpu-scripted',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--qualification',type=Path)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    qualification=args.qualification or project/'validation/cua_m1_cpu_ready.json'
    if not (args.run or args.cpu_scripted):
        print(json.dumps({'status':'PLAN_ONLY','gpu_dispatched':False,'gpu_budget_minutes':30,
            'registered_model_episodes':32,'qualification_exists':qualification.is_file(),
            'next':'--run starts one GPU worker only after qualification and live relay checks'}));return 0
    if platform.system()!='Linux' or '--91bb3a6ab7a7-' not in platform.node():
        raise SystemExit('THIS_QZ_CPU_NOTEBOOK_REQUIRED')
    out=checked_path(args.output or ROOT/'.runtime/cua-host/prepared'/
        ('m1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')))
    if out.exists() and not args.resume: raise SystemExit('NEW_OUTPUT_REQUIRED')
    session=out.with_name(out.name+'.session-'+datetime.now(timezone.utc).strftime('%H%M%S'))
    session.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'.runtime/cua-dind/cua-test-qi--91bb3a6ab7a7-oxj4j3ju6p/disk/cua-python-cache-v1'
    from scripts.qz_cua_runtime_cache import bind_cached_runtime
    runtime=bind_cached_runtime(cache)/'venv/bin/python'
    os.environ.update(PYTHONPATH=str(project)+':'+str(project/'references/cua_lite'),
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        DOCKER_HOST='unix:///var/run/docker.sock')
    command=[str(runtime),'-u','-B','-m','scripts.qz_cua_m1',
             '--cpu-scripted' if args.cpu_scripted else '--run','--output',str(out)]
    if args.run: command+=['--qualification',str(qualification)]
    if args.resume: command+=['--resume']
    os.chdir(project)
    report=supervise(command,session,1200 if args.cpu_scripted else 2100)
    print(json.dumps({'output':str(out),'controller_exit':report,'session':str(session)}),flush=True)
    return 0 if report['returncode']==0 else 1


if __name__=='__main__': raise SystemExit(main())
