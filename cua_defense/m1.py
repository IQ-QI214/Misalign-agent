"""Fixed pilot tasks and descriptive signal statistics; no training updates."""
from collections import defaultdict
import hashlib
import itertools
import json
import math
import shlex

from cua_defense.observation import stable_task
from cua_defense.tasks import ATTACKS, NOTICE
from scripts.qz_cua_judge_controls import TASKS
from scripts.qz_cua_m0 import task_row


def jobs(config, cpu_fixture=False):
    repeats=1 if cpu_fixture else config['repetitions']
    return [{'case_id':f'{family}__{condition}__r{repeat}', 'family':family,
             'condition':condition,'repeat':repeat,
             'seed':config['seed']+10000*family_index+100*condition_index+10*repeat}
            for (family_index,family),(condition_index,condition),repeat in itertools.product(
                enumerate(config['families']),enumerate(config['conditions']),range(repeats))]


def pilot_task(family,condition):
    task=next(t for t in TASKS if t['name']==family)
    row=task_row(task)
    content='' if condition=='clean' else ATTACKS[condition]
    row['task_id']='m1_'+family+'_'+condition
    code='from pathlib import Path; Path('+repr(NOTICE)+').write_text('+repr(content+'\n')+')'
    row['metadata']['config'].insert(1,{'type':'execute','parameters':{
        'command':'python3 -c '+shlex.quote(code),'shell':True}})
    launch=next(x for x in row['metadata']['config'] if x['type']=='launch')
    target='/home/user/Desktop/'+task['filename']
    launch['parameters']['command']=['gnome-terminal','--','bash','-lc',
        'cat '+shlex.quote(target)+'; printf "\\nExternal file content:\\n"; cat '+shlex.quote(NOTICE)+
        '; exec bash --noprofile --norc']
    row['metadata']['others']['research_checkpoint_paths'].append(NOTICE)
    return task,stable_task(row)


def standardize(values):
    mean=sum(values)/len(values)
    sd=math.sqrt(sum((x-mean)**2 for x in values)/len(values))
    return [0.0]*len(values) if sd==0 else [(x-mean)/sd for x in values]


def analysis(results,planned):
    groups=defaultdict(list)
    for r in results:
        groups[(r['family'],r['condition'])].append(r)
    expected=defaultdict(int)
    for job in planned: expected[(job['family'],job['condition'])]+=1
    rows=[]
    for key,n in expected.items():
        actual=groups[key]
        valid=[r for r in actual if r.get('execution_valid') and r.get('labels',{}).get('sum_reward') is not None]
        complete=len(actual)==n and len(valid)==n
        row={'family':key[0],'condition':key[1],'registered':n,'attempted':len(actual),
             'valid':len(valid),'group_complete':complete,
             'safe_completions':sum(r['labels']['safe_completion'] is True for r in valid),
             'violations':sum(r['labels']['safety'] is False for r in valid)}
        if complete and n>=2:
            rewards=[r['labels']['sum_reward'] for r in valid]
            safety=[2*int(r['labels']['safety'])-1 for r in valid]
            utility=[2*int(r['labels']['utility'])-1 for r in valid]
            scalar=standardize(rewards)
            component=[(s+u)/2 for s,u in zip(standardize(safety),standardize(utility))]
            buckets=defaultdict(set)
            for r in valid:
                x=r['labels'];buckets[x['sum_reward']].add((x['safety'],x['utility']))
            row.update(rewards=rewards,scalar_relative_advantages=scalar,
                component_relative_advantages=component,
                reward_uniform=len(set(rewards))==1,
                same_reward_different_outcomes=any(len(b)>1 for b in buckets.values()))
        rows.append(row)
    return {'groups':rows,'registered':len(planned),'attempted':len(results),
            'unattempted':len(planned)-len(results),
            'scope':'Descriptive only; no incomplete group is normalized and no training benefit is inferred.'}


def task_manifest(config):
    return [{'family':family,'condition':condition,
             'contract_sha256':hashlib.sha256(json.dumps(pilot_task(family,condition)[1],sort_keys=True).encode()).hexdigest()}
            for family,condition in itertools.product(config['families'],config['conditions'])]
