"""Dev-only CPU fixtures; synthetic completions are not model behavior."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from rsd import dev_baseline as baseline
from rsd.evaluation import digest, text_digest


def rows(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in data))


@pytest.fixture
def prepared(tmp_path):
    task = []
    gold = []
    for i, (benchmark, answer_type, answer, labels) in enumerate((('gsm8k','numeric','3',[]),('arc_challenge','choice','B',['A','B','C']))):
        common = {'id':f'task-{i}','source_id':f'task-source-{i}','split':'dev','benchmark':benchmark,
                  'answer_type':answer_type,'choice_labels':labels,'prompt_sha256':text_digest(f'Fixture task {i}')}
        task.append({**common,'prompt':f'Fixture task {i}','family':benchmark})
        gold.append({**common,'expected_answer':answer})
    safety = [{'id':f'safety-{i}','source_id':f'safety-source-{i}','split':'dev','benchmark':benchmark,
               'metric':metric,'family':benchmark,'prompt':f'Fixture safety {i}'}
              for i,(benchmark,metric) in enumerate((('strongreject','safety'),('xstest_safe','overrefusal')))]
    for name, data in (('dev.prompts.jsonl',task),('dev.gold.jsonl',gold),('safety.jsonl',safety)):
        rows(tmp_path/'data'/name,data)
    recipe={'model':copy.deepcopy(baseline.BASE),'train_file':'data/train.prompts.jsonl'}
    plan={'schema_version':'rsd-exploration-v2-plan','training_enabled':False,'candidate_shared_recipe':recipe,
          'candidate_shared_recipe_sha256':digest(recipe),
          'evaluation_generation':{'do_sample':False,'thinking':False,'max_new_tokens':512,'max_prompt_tokens':2048,'seed':0},
          'datasets':{'task_artifacts':{n:{'sha256':baseline.sha(tmp_path/'data'/n)} for n in ('dev.prompts.jsonl','dev.gold.jsonl')},
                      'safety_dev':{'file':'data/safety.jsonl','sha256':baseline.sha(tmp_path/'data/safety.jsonl')}}}
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    output=tmp_path/'bundle';manifest=baseline.prepare(path,output,root=tmp_path,purpose='cpu_fixture')
    return output,manifest,path,tmp_path


class Tokenizer:
    eos_token_id=1
    pad_token_id=0
    chat_template='synthetic-template'
    def get_vocab(self):return {'a':2}


class Backend:
    def __init__(self, manifest, responses=None, error_at=None):
        self.tokenizer=Tokenizer()
        self.protocol=baseline.generation_protocol(self.tokenizer,dtype='float32',generation=manifest['generation'])
        self.identity={'kind':'cpu_fixture','checkpoint_id':'fixture:untrained-backend'}
        self.calls=[];self.responses=responses or ['FINAL: 3','FINAL: B','Refusal fixture','Benign fixture'];self.error_at=error_at
    def generate(self,prompt):
        assert isinstance(prompt,str) and 'expected_answer' not in prompt
        index=len(self.calls);self.calls.append(prompt)
        if index==self.error_at:raise RuntimeError('synthetic interrupted generation')
        result=self.responses[index]
        if isinstance(result,dict):return result
        return {'response':result,'n_tokens':4,'stop_reason':'eos','error':None}


def test_prepare_opens_only_three_dev_assets_and_keeps_execution_disabled(prepared):
    root,manifest,_,source=prepared
    assert not (source/'data/train.prompts.jsonl').exists()
    assert not (source/'data/sealed').exists()
    assert manifest['n_planned']==4 and manifest['model_calls_executed']==0
    assert not manifest['train_or_confirmation_data_opened']
    config=json.loads((root/'execution-config.json').read_text())
    assert config['execution_enabled'] is False and config['model_path'] is None
    loaded=baseline.load_bundle(root)
    assert loaded[0]==manifest
    inputs=[json.loads(line) for line in (root/'generation-inputs.jsonl').read_text().splitlines()]
    assert [r['cohort'] for r in inputs]==['task','task','safety','safety']
    assert all(not {'expected_answer','answer','teacher_context','feedback'} & r.keys() for r in inputs)


def test_prepare_and_check_do_not_import_tensor_runtime(prepared):
    code='import sys; from rsd.dev_baseline import load_bundle; load_bundle(sys.argv[1]); assert "torch" not in sys.modules; assert "transformers" not in sys.modules; print("CPU metadata only")'
    result=subprocess.run([sys.executable,'-c',code,str(prepared[0])],capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('damage',['test_split','gold_in_prompt','duplicate_source','hash','sampling'])
def test_bad_dev_or_protocol_fails_before_output(prepared,damage):
    _,_,plan_path,root=prepared;plan=json.loads(plan_path.read_text())
    if damage=='sampling':plan['evaluation_generation']['do_sample']=True
    elif damage=='hash':plan['datasets']['task_artifacts']['dev.prompts.jsonl']['sha256']='0'*64
    else:
        path=root/'data/dev.prompts.jsonl';data=[json.loads(x) for x in path.read_text().splitlines()]
        if damage=='test_split':data[0]['split']='test'
        elif damage=='gold_in_prompt':data[0]['expected_answer']='3'
        else:data[1]['source_id']=data[0]['source_id']
        rows(path,data);plan['datasets']['task_artifacts']['dev.prompts.jsonl']['sha256']=baseline.sha(path)
    plan_path.write_text(json.dumps(plan));out=root/'bad'
    with pytest.raises(ValueError):baseline.prepare(plan_path,out,root=root,purpose='cpu_fixture')
    assert not out.exists()


def test_movable_self_contained_bundle_and_immutable_output(prepared,tmp_path):
    root,_,plan_path,source=prepared
    copied=tmp_path/'moved';shutil.copytree(root,copied)
    shutil.rmtree(source/'data');plan_path.unlink()
    assert baseline.load_bundle(copied)[0]['n_planned']==4
    with pytest.raises(FileExistsError):
        baseline.execute(copied,copied,backend=Backend(baseline.load_bundle(copied)[0]),purpose='cpu_fixture',device='cpu',dtype='float32',execution_enabled=True)


def test_disabled_or_wrong_purpose_never_calls_backend(prepared,tmp_path):
    root,manifest,_,_=prepared;backend=Backend(manifest)
    with pytest.raises(ValueError,match='disabled'):
        baseline.execute(root,tmp_path/'off',backend=backend)
    with pytest.raises(ValueError,match='purpose'):
        baseline.execute(root,tmp_path/'bad',backend=backend,execution_enabled=True)
    assert not backend.calls and not (tmp_path/'off').exists()


def test_injected_complete_baseline_scores_final_answers_and_defers_safety(prepared,tmp_path):
    root,manifest,_,_=prepared;backend=Backend(manifest);out=tmp_path/'run'
    result=baseline.execute(root,out,backend=backend,purpose='cpu_fixture',device='cpu',dtype='float32',execution_enabled=True)
    assert result['n_attempted']==4 and result['n_missing']==0 and result['n_not_generated']==0
    assert len(backend.calls)==4 and result['optimizer_updates']==0 and result['weights_exported'] is False
    assert all(group['accuracy_valid']==1 for group in result['task_summary']['groups'])
    assert result['safety_judge_status']=='NOT_RUN' and result['safety_responses_present']==2
    assert len((out/'safety-predictions.jsonl').read_text().splitlines())==2
    assert not list(out.glob('*.safetensors'))
    assert baseline.summarize(root,out)['n_attempted']==4


def test_fatal_error_preserves_all_slots_and_does_not_retry(prepared,tmp_path):
    root,manifest,_,_=prepared;backend=Backend(manifest,error_at=1);out=tmp_path/'failed'
    result=baseline.execute(root,out,backend=backend,purpose='cpu_fixture',device='cpu',dtype='float32',execution_enabled=True)
    assert len(backend.calls)==2 and result['n_recorded']==4
    assert result['n_attempted']==2 and result['n_not_generated']==2
    records=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
    assert [r['stop_reason'] for r in records]==['eos','error','not_generated','not_generated']
    assert result['task_summary']['n_unknown']==1 and result['execution_error']
    assert result['groups']['strongreject']['generated_tokens_mean'] is None
    assert result['groups']['strongreject']['n_with_generated_tokens']==0
    assert result['groups']['gsm8k']['generated_tokens_p95']==4
    assert json.loads((out/'receipt.json').read_text())['status']=='FAILED_ATTEMPT_RETAINED'
    with pytest.raises(FileExistsError):baseline.execute(root,out,backend=backend,purpose='cpu_fixture',device='cpu',dtype='float32',execution_enabled=True)


def test_truncation_empty_unknown_and_format_failures_keep_denominators(prepared,tmp_path):
    root,manifest,_,_=prepared
    outputs=[{'response':'FINAL: 3','n_tokens':512,'stop_reason':'length','error':None},
             'B',{'response':'','n_tokens':1,'stop_reason':'eos','error':None},
             {'response':'partial','n_tokens':8,'stop_reason':'unknown','error':None}]
    result=baseline.execute(root,tmp_path/'bad-outputs',backend=Backend(manifest,outputs),purpose='cpu_fixture',device='cpu',dtype='float32',execution_enabled=True)
    assert result['n_attempted']==4 and result['task_summary']['n_unknown']==2
    assert result['groups']['strongreject']['n_complete_nonempty']==0
    assert result['groups']['xstest_safe']['stop_reasons']=={'unknown':1}


@pytest.mark.parametrize('damage',['duplicate','prompt','value','identity','stop'])
def test_saved_prediction_binding_and_validity_are_rechecked(prepared,tmp_path,damage):
    root,manifest,_,_=prepared;out=tmp_path/'run'
    baseline.execute(root,out,backend=Backend(manifest),purpose='cpu_fixture',device='cpu',dtype='float32',execution_enabled=True)
    path=out/'predictions.jsonl';data=[json.loads(x) for x in path.read_text().splitlines()]
    if damage=='duplicate':data[1]=data[0]
    elif damage=='prompt':data[0]['prompt_sha256']='0'*64
    elif damage=='value':data[0]['valid']=False
    elif damage=='identity':data[0]['checkpoint_id']='other'
    else:data[0]['stop_reason']='length'
    rows(path,data)
    with pytest.raises(ValueError):baseline.summarize(root,out)


def test_protocol_matches_existing_milestone_exactly(prepared):
    from rsd.milestone_eval import build_generation_protocol
    tok=Tokenizer()
    assert baseline.generation_protocol(tok,dtype='float32',generation=prepared[1]['generation'])==build_generation_protocol(tok,dtype='float32',max_new_tokens=512,max_prompt_tokens=2048,seed=0)


@pytest.mark.parametrize('damage',['artifact','count','enabled','generation','symlink'])
def test_bundle_tampering_and_silent_enable_fail(prepared,tmp_path,damage):
    root,manifest,_,_=prepared
    if damage=='count':manifest['n_planned']=2
    elif damage=='artifact':(root/'task-prompts.jsonl').write_text('[]')
    elif damage=='symlink':
        p=root/'task-gold.jsonl';other=tmp_path/'moved-gold';p.rename(other);p.symlink_to(other)
    else:
        p=root/'execution-config.json';v=json.loads(p.read_text())
        if damage=='enabled':v['execution_enabled']=True
        else:v['generation']['max_new_tokens']=32
        p.write_text(json.dumps(v));manifest['files_sha256'][p.name]=baseline.sha(p)
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):baseline.load_bundle(root)


@pytest.mark.parametrize('damage',['sampling','budget','thinking','zero_eos','short_length','too_many_tokens'])
def test_saved_protocol_and_token_termination_are_revalidated(prepared,tmp_path,damage):
    root,manifest,_,_=prepared;out=tmp_path/'run'
    baseline.execute(root,out,backend=Backend(manifest),purpose='cpu_fixture',device='cpu',dtype='float32',execution_enabled=True)
    contract=json.loads((out/'run-contract.json').read_text())
    records=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
    if damage in {'sampling','budget','thinking'}:
        key,value={'sampling':('do_sample',True),'budget':('max_new_tokens',32),'thinking':('enable_thinking',True)}[damage]
        contract['generation_protocol'][key]=value
        for record in records:
            record['generation_protocol']=contract['generation_protocol']
            record['contract_sha256']=digest(contract)
        (out/'run-contract.json').write_text(json.dumps(contract))
    elif damage=='zero_eos':records[0]['n_tokens']=0
    elif damage=='short_length':records[0].update(stop_reason='length',finish_reason='length',valid=False)
    else:records[0]['n_tokens']=513
    rows(out/'predictions.jsonl',records)
    with pytest.raises(ValueError):baseline.summarize(root,out)


def test_malformed_backend_result_is_retained_and_remaining_slots_not_generated(prepared,tmp_path):
    root,manifest,_,_=prepared
    malformed={'response':'unfinished','n_tokens':4,'stop_reason':'length','error':None}
    result=baseline.execute(root,tmp_path/'bad-driver',backend=Backend(manifest,[malformed]),purpose='cpu_fixture',device='cpu',dtype='float32',execution_enabled=True)
    assert result['n_attempted']==1 and result['n_not_generated']==3
    saved=[json.loads(x) for x in (tmp_path/'bad-driver/predictions.jsonl').read_text().splitlines()]
    assert saved[0]['invalid_backend_result']==malformed and saved[0]['valid'] is False


@pytest.mark.parametrize('extra',['model.safetensors','pytorch_model.bin','pytorch_model.bin.index.json',
                                  'adapter_config.json','adapter_model.safetensors','added_tokens.json',
                                  'special_tokens_map.json','tokenizer.model','chat_template.json','chat_templates'])
def test_alternate_weights_adapters_and_tokenizer_overrides_are_rejected(tmp_path,extra):
    model=tmp_path/'model';model.mkdir()
    mapping={f'tensor-{i}':f'model-{i}.safetensors' for i in range(4)}
    (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))
    for name in mapping.values():(model/name).write_bytes(b'fixture')
    assert baseline._validate_model_directory(model)==set(mapping.values())
    (model/extra).write_text('unverified')
    with pytest.raises(ValueError,match='Unpinned alternate'):
        baseline._validate_model_directory(model)


def test_model_and_output_overlap_fails_before_loading(monkeypatch,tmp_path):
    monkeypatch.setattr(baseline,'load_bundle',lambda _:({'purpose':'research'},[],[],[]))
    model=tmp_path/'model';model.mkdir()
    with pytest.raises(ValueError,match='overlap'):
        baseline.execute('unused',model/'output',model_path=model,execution_enabled=True)
    assert not (model/'output').exists()


def test_runtime_caches_are_project_local_without_chdir(monkeypatch,tmp_path):
    cwd=Path.cwd();project=tmp_path/'project';project.mkdir();out=project/'run'
    monkeypatch.setattr(baseline,'ROOT',project)
    monkeypatch.setattr(baseline,'AUTHORIZED_QZ_ROOT',project)
    monkeypatch.setattr(baseline.os,'environ',{})
    monkeypatch.setattr(baseline.tempfile,'tempdir',None)
    baseline._runtime_paths(out)
    assert Path.cwd()==cwd
    assert all(Path(baseline.os.environ[k]).is_relative_to(out) for k in ('HF_HOME','TORCH_HOME','TRITON_CACHE_DIR','TMPDIR'))
    assert baseline.os.environ['HF_HUB_OFFLINE']=='1'
    with pytest.raises(ValueError,match='inside this project'):
        baseline._runtime_paths(tmp_path/'outside')


def test_research_write_boundary_is_not_inferred_from_checkout_name(monkeypatch,tmp_path):
    project=tmp_path/'root-checkout';project.mkdir()
    monkeypatch.setattr(baseline,'ROOT',project)
    with pytest.raises(ValueError,match='authorized user workspace'):
        baseline._runtime_paths(project/'run')
    assert not (project/'run').exists()
