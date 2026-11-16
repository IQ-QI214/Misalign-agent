#!/usr/bin/env python3
"""Bound one owned inference child, preserving its real return code."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def supervise(command, directory, seconds, grace=15):
    directory=Path(directory)
    started=time.monotonic()
    result={'status':'STARTED','started_at':datetime.now(timezone.utc).isoformat(),
            'budget_seconds':seconds,'supervisor_pid':os.getpid()}
    with (directory/'worker.log').open('xb') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True)
        result['child_pid']=child.pid
        (directory/'worker.launch.json').write_text(json.dumps(result)+'\n')
        try:
            result['returncode']=child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            result['budget_exhausted']=True
            os.killpg(child.pid,signal.SIGTERM)
            try: result['returncode']=child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);result['returncode']=child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
    result.update(status='PROCESS_EXITED',seconds=round(time.monotonic()-started,3),
                  ended_at=datetime.now(timezone.utc).isoformat())
    tmp=directory/'worker.exit.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n')
    tmp.replace(directory/'worker.exit.json')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--python',required=True)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--mailbox',type=Path,required=True)
    parser.add_argument('--seconds',type=float,required=True)
    args=parser.parse_args()
    if not 0<args.seconds<=1800: raise SystemExit('Worker budget must be 0..1800 seconds')
    result=supervise([args.python,str(args.worker),'--model',args.model,'--mailbox',str(args.mailbox),
                      '--max-requests','10000'],args.mailbox,args.seconds)
    print(json.dumps(result),flush=True)
    return 0 if result['returncode']==0 else 1


if __name__=='__main__': raise SystemExit(main())
