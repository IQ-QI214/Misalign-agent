#!/usr/bin/env python3
"""CPU qualification for the first bounded M1 GPU measurement, never a GPU launch."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import unittest

from cua_defense.assets import model_manifest,source_manifest
from scripts.qz_cua_m0 import ROOT,PROJECT,save


def require(ok,message):
    if not ok: raise ValueError(message)


def qualify(args,report):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    require(torch.version.cuda is None,'Use the existing CPU Torch runtime for this acceptance')
    require(transformers.__version__=='5.21.0','Test the exact prepared GPU Transformers version')
    report['runtime']={'python':sys.version,'torch':torch.__version__,
                       'transformers':transformers.__version__,'cuda':torch.version.cuda}
    os.environ['CUA_REQUIRE_NATIVE']='1'
    names=['test_cua_observation','test_cua_policy_trace','test_cua_m1','test_cua_runtime',
           'test_cua_wire_format','test_cua_measurement','test_cua_pipeline']
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.'+n for n in names])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    report['tests']={'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                     'skipped':len(result.skipped),'details':stream.getvalue()}
    require(result.wasSuccessful() and not result.skipped,'Native CPU tests must pass without skipped tests')
    obs=json.loads(args.observation.read_text())
    obs=obs.get('receipt',obs)
    require(obs['status']=='PROBE_COMPLETE' and obs['valid']==6 and obs['matched_pairs']==4
            and obs['observation_qualified'],'Observation qualification missing')
    require(obs['source_sha256']['cua_defense/observation.py']==report['source_sha256']['cua_defense/observation.py'],
            'Observation implementation changed after qualification')
    report['observation']={'receipt_sha256':hashlib.sha256(args.observation.read_bytes()).hexdigest(),
                           'valid':6,'matched_pairs':4,'scope':'Versioned synthetic terminal model inputs only'}
    fixture=json.loads((args.fixture/'receipt.json').read_text())
    require(fixture['phase']=='CPU_M1_ACCEPTANCE' and not fixture['gpu_dispatched'],'Expected CPU fixture run')
    require(fixture['status']=='PANEL_COMPLETE' and len(fixture['results'])==8,'CPU mailbox panel incomplete')
    require(all(r['execution_valid'] and r['labels']['safe_completion'] is True and
                not r['model_run'] and r['policy_trace_count']==0 for r in fixture['results']),
            'Scripted outcome, source or policy trace status is wrong')
    changed={k:{'executed':old,'current':report['source_sha256'].get(k)}
             for k,old in fixture['source_sha256'].items() if report['source_sha256'].get(k)!=old}
    # The fixture does not run HF inference. Its GPU-only paths are checked above
    # with real CPU generation, while the worker preflight remains GPU-pending.
    require(set(changed)<= {'scripts/qz_cua_hf_worker.py','cua_defense/inference.py'},
            'Executed CPU pipeline changed since the fixture')
    report['fixture_source_changes']=changed
    case_ids={r['case_id'] for r in fixture['results']}
    require(case_ids=={r['case_id'] for r in fixture['planned_jobs']},'Fixture registration mismatch')
    prompt_hashes={};tokens=[];actions=0
    model=json.loads((PROJECT/'configs/cua/models.json').read_text())['qwen35_9b']
    processor=AutoProcessor.from_pretrained(model['processor_path'],local_files_only=True)
    for record in fixture['results']:
        directory=args.fixture/'episodes'/record['case_id']
        prompt=(directory/'prompt-0.txt').read_text()
        prompt_hashes.setdefault(record['family'],set()).add(hashlib.sha256(prompt.encode()).hexdigest())
        for path in directory.glob('response-*.json'):
            data=json.loads(path.read_text())
            require(data['source']=='scripted_mailbox_fixture','CPU fixture was mislabeled as model data')
            encoded=processor.tokenizer.encode(data['response'],add_special_tokens=False)
            require(processor.decode(encoded,skip_special_tokens=True)==data['response'],
                    'Action text does not roundtrip through the real model tokenizer')
            actions+=1
        last=record['turns']-1
        images=[Image.open(p).convert('RGB') for p in sorted(directory.glob(f'input-{last}-*.png'))]
        data=processor(text=[(directory/f'prompt-{last}.txt').read_text()],images=images,
                       return_tensors='pt',padding=True)
        require(data['input_ids'].shape[-1]<32768,'Pilot prefill limit exceeded in the fixture')
        tokens.append({'case_id':record['case_id'],'input_tokens':int(data['input_ids'].shape[-1]),
                       'images':len(images),'image_grid_thw':data['image_grid_thw'].tolist()})
        for image in images:image.close()
    require(all(len(v)==1 for v in prompt_hashes.values()),'Condition labels leaked into the text prompt')
    require(actions==32,'Expected eight four-turn scripted mailbox episodes')
    report['fixture']={'receipt_sha256':hashlib.sha256((args.fixture/'receipt.json').read_bytes()).hexdigest(),
                      'registered':8,'valid':8,'mailbox_responses':actions,
                      'prompt_equal_across_conditions':True,'tokenizer_roundtrip':True,
                      'tokenization':tokens,'model_run':False}
    report['model_assets']=model_manifest(model['processor_path'])
    require(report['model_assets']==fixture['model_assets'],'Model metadata changed during preparation')
    report['gpu_requirements']={'cards':1,'reference_device':'H100 80GB','python':'3.12',
        'transformers':'5.21.0','historical_torch':'2.8.0a0+5228986c39.nv25.06',
        'gpu_worker_minutes':30,'startup_included':True,'termination_grace_seconds':15,
        'actual_cuda_and_memory_validation':'PENDING_REAL_GPU'}
    report['pending_gpu_checks']=['actual GPU image Python/Torch/CUDA compatibility',
        '9B weights load and one real generation with aligned probabilities',
        'model-driven actions and task capability','memory, latency and complete rollout throughput']
    report['status']='READY_FOR_GPU_MEASUREMENT'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--observation',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if platform.system()!='Linux' or not ROOT.is_dir():raise SystemExit('QZ_CPU_REQUIRED')
    args.output.resolve().relative_to(ROOT.resolve())
    if args.output.exists():raise SystemExit('NEW_RECEIPT_REQUIRED')
    report={'schema_version':1,'status':'STARTED','created_at':datetime.now(timezone.utc).isoformat(),
            'gpu_used':False,'model_9b_loaded':False,'training_updates':0,
            'source_sha256':source_manifest(PROJECT)}
    try:qualify(args,report)
    except Exception:report.update(status='NOT_QUALIFIED',error=traceback.format_exc())
    save(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','model_assets']},indent=2),flush=True)
    return 0 if report['status']=='READY_FOR_GPU_MEASUREMENT' else 1


if __name__=='__main__':raise SystemExit(main())
