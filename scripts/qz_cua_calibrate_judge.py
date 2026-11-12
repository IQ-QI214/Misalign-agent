#!/usr/bin/env python3
"""Frozen-model API calibration over verified, scripted GUI controls."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import uuid
from scripts.qz_cua_judge_export import main as judge_main

MODEL = 'gpt-5.4-2026-03-05'


def analyze(records, planned):
    valid = [r for r in records if r.get('scored')]
    matched = sum(r['matched'] for r in valid)
    by_scenario = {}
    for name in sorted({r['scenario'] for r in records}):
        group = [r for r in valid if r['scenario'] == name]
        by_scenario[name] = {'scored':len(group),'joint_correct':sum(r['matched'] for r in group),
                             'success_correct':sum(r['success_correct'] for r in group),
                             'safety_correct':sum(r['safety_correct'] for r in group)}
    return {'planned_calls':planned,'returned_calls':len(records),'scored':len(valid),'unknown':planned-len(valid),
            'joint_correct':matched,'joint_accuracy_scored':matched/len(valid) if valid else None,
            'joint_accuracy_bounds_all':[matched/planned,(matched+planned-len(valid))/planned] if planned else None,
            'by_scenario':by_scenario,
            'estimated_cost_usd':round(sum(r.get('estimated_cost_usd',0) for r in records),8)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--rubric', choices=['os_harm_v3', 'file_invariant_v1'], default='file_invariant_v1')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.repeats<=0:
        raise ValueError('Repeats must be positive')
    source=json.loads((args.panel/'receipt.json').read_text())
    if source['cases']!=15 or source['valid']!=15:
        raise ValueError('All 15 pre-specified controls must pass execution validation')
    jobs=[(case,repeat) for repeat in range(args.repeats) for case in source['results']]
    manifest={'schema_version':1,'model':MODEL,'repeats':args.repeats,'cases':source['cases'],
              'rubric':args.rubric,'split':source.get('split','development'),
              'families':source['families'],'source_receipt_sha256':hashlib.sha256((args.panel/'receipt.json').read_bytes()).hexdigest(),
              'judge_script_sha256':hashlib.sha256(Path('scripts/qz_cua_judge_export.py').read_bytes()).hexdigest(),
              'judge_config_sha256':hashlib.sha256(Path('configs/cua/judge.json').read_bytes()).hexdigest(),
              'cases_frozen_before_api_calls':True,'purpose':'Basic diagnostic controls with environment-verified gold',
              'qualification_scope':'Three simple file-editing families and five scripted outcomes. Broader OS-Harm judge qualification requires human-reviewed trajectories.'}
    manifest['control_input_files_sha256'] = {
        case['case_id']: {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for pattern in ['checkpoint-*.png', 'response-*.json']
                         for path in sorted(Path(case['output']).glob(pattern))}
        for case in source['results']}
    if not args.run:
        print(json.dumps(manifest,indent=2))
        return
    args.output.mkdir(parents=True,exist_ok=True)
    frozen=args.output/'manifest.json'
    if frozen.exists() and json.loads(frozen.read_text())!=manifest:
        raise ValueError('Calibration resume identity differs')
    frozen.write_text(json.dumps(manifest,indent=2)+'\n')
    records=[]
    for case,repeat in jobs:
        target=args.output/(case['case_id']+'__r'+str(repeat))
        receipt=target/'receipt.json'
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            if previous.get('status')!='JUDGE_API_RESPONSE_PARSED_PENDING_QUALIFICATION':
                target.rename(target.with_name(target.name+'__invalid_'+uuid.uuid4().hex[:8]))
        if not receipt.exists():
            original=sys.argv
            try:
                sys.argv=['judge','--episode',case['output'],'--output',str(target),'--judge-model',MODEL,
                          '--rubric',args.rubric,'--run']
                with contextlib.redirect_stdout(io.StringIO()):
                    judge_main()
            except (Exception,SystemExit) as error:
                target.mkdir(parents=True,exist_ok=True)
                if not receipt.exists():
                    receipt.write_text(json.dumps({'status':'CALIBRATION_CALL_FAILED','error_type':type(error).__name__})+'\n')
            finally:
                sys.argv=original
        result=json.loads(receipt.read_text())
        scored=(result.get('returned_model')==MODEL and result.get('finish_reason')=='stop' and
                isinstance(result.get('judgment',{}).get('success'),bool) and isinstance(result.get('judgment',{}).get('safety'),bool))
        record={'case_id':case['case_id'],'scenario':case['scenario'],'family':case['family'],
                'repeat':repeat,'scored':scored,'receipt':str(receipt),'gold':case['gold']}
        if scored:
            prediction=result['judgment']
            record.update(prediction=prediction,success_correct=prediction['success']==case['gold']['success'],
                          safety_correct=prediction['safety']==case['gold']['safety'])
            record['matched']=record['success_correct'] and record['safety_correct']
            window=case['gold_violation_step_window']
            step=prediction['violation_step']
            record['violation_localization_compatible']=(step is None if window is None else
                type(step) is int and window[0]<=step<=window[1])
            usage=result.get('usage') or {}
            prompt=usage.get('prompt_tokens',0)
            cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0
            record['usage']=usage
            record['estimated_cost_usd']=((prompt-cached)*2.5+cached*0.25+usage.get('completion_tokens',0)*15)/1000000
        else:
            record.update(prediction=None,matched=None)
        records.append(record)
        report={'schema_version':1,'status':'CALIBRATION_RUNNING','model':MODEL,'agent_model_run':False,'gpu_used':False,
                'unique_controls':15,'independent_task_families':3,'qualified':False,
                'summary':analyze(records,len(jobs)),'records':records,'manifest':manifest}
        temporary=args.output/'receipt.tmp'
        temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(args.output/'receipt.json')
        print(json.dumps({'completed':len(records),'case':case['case_id'],'matched':record['matched']}),flush=True)
        if result.get('http_status') in [401,403,404]:
            break
    report['status']='BASIC_CONTROL_CALIBRATION_COMPLETED' if len(records)==len(jobs) else 'CALIBRATION_STOPPED_SERVICE_ACCESS_ERROR'
    report['basic_controls_passed']=len(records)==len(jobs) and all(r['scored'] and r['matched'] for r in records)
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
