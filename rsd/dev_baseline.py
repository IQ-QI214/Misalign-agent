"""Prepare or explicitly execute a dev-only, inference-only baseline.

Preparation imports no tensor/CUDA runtime. Research execution is opt-in,
local-files-only, and verifies the pinned initial model bytes before loading.
CPU injected backends are labeled fixtures and cannot become research runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

from .confirmation import no_symlink, write_new, validate_task_pair
from .evaluation import digest, read_jsonl, text_digest, validate_dataset
from .task_evaluation import score_records as score_tasks

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_QZ_ROOT = Path('/inspire/hdd/global_user/wenming-253108090054/zhujiaqi')
VERSION = "rsd-dev-baseline-v1"
BASE = {"repository": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
CODE = ("dev_baseline.py", "modeling.py", "evaluation.py", "task_evaluation.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _json(path):
    return json.loads(no_symlink(path).read_text(encoding='utf-8'))


def _codes():
    return {name: sha(Path(__file__).with_name(name)) for name in CODE}


def _source(root, name):
    path = Path(name)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('Baseline source must be a relative path within the repository')
    return no_symlink(Path(root) / path)


def _copy(source, target):
    with target.open('xb') as stream:
        stream.write(source.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o600)


def _write_rows(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            _append(stream, row)
    path.chmod(0o600)


def _append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def _validate_data(task, gold, safety, *, purpose):
    validate_task_pair(task, gold, 'dev')
    validate_dataset(safety)
    if not safety or any(r.get('split') != 'dev' for r in safety):
        raise ValueError('Baseline safety inputs must be dev only')
    counts = Counter(r['benchmark'] for r in task + safety)
    if purpose == 'research' and counts != {'gsm8k': 128, 'arc_challenge': 128, 'strongreject': 65, 'xstest_safe': 45}:
        raise ValueError('Research baseline requires the full 256 task + 110 safety dev cohort')
    ids, sources = set(), set()
    for row in task + safety:
        if row['id'] in ids or row['source_id'] in sources:
            raise ValueError('Baseline cohorts contain duplicate IDs/source clusters')
        if any(k in row for k in ('expected_answer', 'answer', 'answerKey', 'verified_solution', 'teacher_context', 'feedback')):
            raise ValueError('Generation inputs contain gold or teacher feedback')
        if (row in safety and ((row['benchmark'], row['metric']) not in
                               {('strongreject', 'safety'), ('xstest_safe', 'overrefusal')})):
            raise ValueError('Unexpected safety benchmark/metric')
        ids.add(row['id']); sources.add(row['source_id'])
    return dict(counts)


def _inputs(task, safety):
    return [{"id": row['id'], "source_id": row['source_id'], "benchmark": row['benchmark'],
             "cohort": cohort, "split": "dev", "prompt": row['prompt'],
             "prompt_sha256": text_digest(row['prompt'])}
            for cohort, group in (('task', task), ('safety', safety)) for row in group]


def _asset_lock(plan, root, purpose):
    model = plan['candidate_shared_recipe']['model']
    if purpose == 'cpu_fixture':
        return {'purpose': purpose, 'model': model, 'weight_hashes': [], 'metadata_sha256': {}}, {}
    if model != BASE:
        raise ValueError('Research baseline is limited to the pinned initial Qwen3.5-9B')
    weight_path = root / 'validation/gpu_qwen35_9b_v1.json'
    token_path = root / 'validation/reasoning_token_budget_v2.json'
    weight, token = _json(weight_path), _json(token_path)
    if ({k: weight['model'][k] for k in BASE} != model
            or token['model_repository'] != model['repository'] or token['model_revision'] != model['revision']):
        raise ValueError('Historical asset evidence refers to another base model')
    hashes = sorted(v['sha256'] for v in weight['model']['weights'].values())
    metadata = token['tokenizer_files_sha256']
    if (len(hashes) != 4 or set(metadata) != {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'}
            or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in hashes + list(metadata.values()))):
        raise ValueError('Incomplete pinned base weight/tokenizer evidence')
    return {'purpose': purpose, 'model': model, 'weight_hashes': hashes,
            'metadata_sha256': metadata, 'eos_token_id': token['eos_token_id']}, {
                'model-evidence.json': weight_path, 'tokenizer-evidence.json': token_path}


def prepare(plan_path, output_dir, *, root=ROOT, purpose='research'):
    if purpose not in {'research', 'cpu_fixture'}:
        raise ValueError('Explicit research or cpu_fixture purpose required')
    root, plan_path, output = no_symlink(root), no_symlink(plan_path), no_symlink(output_dir)
    plan = _json(plan_path)
    if plan.get('schema_version') != 'rsd-exploration-v2-plan' or plan.get('training_enabled') is not False:
        raise ValueError('Baseline preparation requires the disabled exploration plan')
    recipe, generation = plan['candidate_shared_recipe'], plan['evaluation_generation']
    if digest(recipe) != plan['candidate_shared_recipe_sha256']:
        raise ValueError('Shared recipe identity differs')
    if (generation.get('do_sample') is not False or generation.get('thinking') is not False
            or generation.get('max_new_tokens') != 512 or generation.get('max_prompt_tokens') != 2048
            or type(generation.get('seed')) is not int or generation['seed'] != 0):
        raise ValueError('Baseline must use the planned greedy 512/2048, seed0, thinking-off protocol')
    task_dir = Path(recipe['train_file']).parent  # Path only; training data is never opened.
    artifacts = plan['datasets']['task_artifacts']
    sources = {'task-prompts.jsonl': (_source(root, task_dir / 'dev.prompts.jsonl'), artifacts['dev.prompts.jsonl']['sha256']),
               'task-gold.jsonl': (_source(root, task_dir / 'dev.gold.jsonl'), artifacts['dev.gold.jsonl']['sha256']),
               'safety-prompts.jsonl': (_source(root, plan['datasets']['safety_dev']['file']), plan['datasets']['safety_dev']['sha256'])}
    for path, expected in sources.values():
        if sha(path) != expected:
            raise ValueError('Pinned dev source hash mismatch')
    task, gold, safety = (read_jsonl(sources[k][0]) for k in ('task-prompts.jsonl', 'task-gold.jsonl', 'safety-prompts.jsonl'))
    counts = _validate_data(task, gold, safety, purpose=purpose)
    lock, evidence = _asset_lock(plan, root, purpose)
    if output.exists():
        raise FileExistsError('Use a new baseline preparation directory')
    output.mkdir(parents=True, mode=0o700)
    for name, (path, _) in sources.items():
        _copy(path, output / name)
    for name, path in evidence.items():
        _copy(path, output / name)
    _copy(plan_path, output / 'source-plan.json')
    write_new(output / 'asset-lock.json', lock)
    inputs = _inputs(task, safety)
    _write_rows(output / 'generation-inputs.jsonl', inputs)
    write_new(output / 'execution-config.json', {'version': VERSION, 'execution_enabled': False,
        'purpose': purpose, 'model_path': None, 'device': 'cuda:0' if purpose == 'research' else 'cpu',
        'dtype': 'bfloat16' if purpose == 'research' else 'float32', 'generation': generation,
        'training_enabled': False, 'judge_execution_enabled': False})
    manifest = {'version': VERSION, 'purpose': purpose, 'state': 'PREPARED_EXECUTION_DISABLED',
                'source_plan_sha256': sha(plan_path), 'dev_source_sha256': {k: v[1] for k, v in sources.items()},
                'n_task': len(task), 'n_safety': len(safety), 'n_planned': len(inputs), 'counts': counts,
                'input_order_sha256': digest([r['id'] for r in inputs]), 'generation': generation,
                'model': recipe['model'], 'asset_lock_sha256': digest(lock), 'code_sha256': _codes(),
                'files_sha256': {p.name: sha(p) for p in sorted(output.iterdir())},
                'train_or_confirmation_data_opened': False, 'model_calls_executed': 0,
                'judge_calls_executed': 0, 'scientific_conclusion': None}
    write_new(output / 'manifest.json', manifest)
    return manifest


def load_bundle(bundle_dir):
    root = no_symlink(bundle_dir)
    manifest = _json(root / 'manifest.json')
    if (manifest.get('version') != VERSION or manifest.get('purpose') not in {'research', 'cpu_fixture'}
            or manifest.get('code_sha256') != _codes()):
        raise ValueError('Baseline bundle version/purpose/code identity differs')
    expected = {'task-prompts.jsonl', 'task-gold.jsonl', 'safety-prompts.jsonl', 'source-plan.json',
                'asset-lock.json', 'generation-inputs.jsonl', 'execution-config.json'}
    if manifest['purpose'] == 'research': expected |= {'model-evidence.json', 'tokenizer-evidence.json'}
    if set(manifest.get('files_sha256', {})) != expected:
        raise ValueError('Baseline artifact inventory differs')
    for name, value in manifest['files_sha256'].items():
        if sha(no_symlink(root / name)) != value:
            raise ValueError('Baseline artifact hash mismatch')
    task, gold, safety = (read_jsonl(root / n) for n in ('task-prompts.jsonl', 'task-gold.jsonl', 'safety-prompts.jsonl'))
    counts = _validate_data(task, gold, safety, purpose=manifest['purpose'])
    inputs = _inputs(task, safety)
    plan, config, lock = (_json(root / n) for n in ('source-plan.json', 'execution-config.json', 'asset-lock.json'))
    generation = manifest.get('generation', {})
    if (plan.get('schema_version') != 'rsd-exploration-v2-plan' or plan.get('training_enabled') is not False
            or digest(plan['candidate_shared_recipe']) != plan.get('candidate_shared_recipe_sha256')
            or generation.get('do_sample') is not False or generation.get('thinking') is not False
            or generation.get('max_new_tokens') != 512 or generation.get('max_prompt_tokens') != 2048
            or type(generation.get('seed')) is not int or generation['seed'] != 0):
        raise ValueError('Baseline plan/generation contract differs from fixed dev protocol')
    if manifest['purpose'] == 'research':
        weight, token = (_json(root / name) for name in ('model-evidence.json', 'tokenizer-evidence.json'))
        expected_lock = {'purpose': 'research', 'model': BASE,
                         'weight_hashes': sorted(v['sha256'] for v in weight['model']['weights'].values()),
                         'metadata_sha256': token['tokenizer_files_sha256'], 'eos_token_id': token['eos_token_id']}
        if (manifest['model'] != BASE or lock != expected_lock
                or {k: weight['model'][k] for k in BASE} != BASE
                or token['model_repository'] != BASE['repository'] or token['model_revision'] != BASE['revision']):
            raise ValueError('Baseline asset lock differs from pinned historical source evidence')
    if (manifest['source_plan_sha256'] != sha(root / 'source-plan.json')
            or manifest['generation'] != plan['evaluation_generation'] or config.get('generation') != manifest['generation']
            or config.get('execution_enabled') is not False or config.get('purpose') != manifest['purpose']
            or config.get('training_enabled') is not False or config.get('judge_execution_enabled') is not False
            or manifest['model'] != plan['candidate_shared_recipe']['model'] or lock['model'] != manifest['model']
            or digest(lock) != manifest['asset_lock_sha256']
            or manifest['counts'] != counts or manifest['n_task'] != len(task) or manifest['n_safety'] != len(safety)
            or manifest['n_planned'] != len(inputs) or manifest['input_order_sha256'] != digest([r['id'] for r in inputs])
            or read_jsonl(root / 'generation-inputs.jsonl') != inputs
            or any(sha(root / name) != value for name, value in manifest['dev_source_sha256'].items())):
        raise ValueError('Baseline manifest input/config/count binding mismatch')
    return manifest, task, gold, safety


def generation_protocol(tokenizer, *, dtype, generation):
    if (type(tokenizer.eos_token_id) is not int or tokenizer.eos_token_id < 0
            or type(tokenizer.pad_token_id) is not int or tokenizer.pad_token_id < 0 or not tokenizer.chat_template):
        raise ValueError('Explicit tokenizer EOS/PAD/template required')
    return {'profile': 'rsd-text-eval-v2', 'enable_thinking': False,
            'max_new_tokens': generation['max_new_tokens'], 'max_prompt_tokens': generation['max_prompt_tokens'],
            'temperature': 0.0, 'do_sample': False, 'top_p': 1.0, 'top_k': 0, 'num_beams': 1,
            'repetition_penalty': 1.0, 'seed': generation['seed'], 'dtype': dtype,
            'pad_token_id': tokenizer.pad_token_id, 'eos_token_id': tokenizer.eos_token_id,
            'eos_token_id_source': 'tokenizer.eos_token_id', 'tokenizer_vocab_sha256': digest(tokenizer.get_vocab()),
            'tokenizer_template_sha256': digest(tokenizer.chat_template)}


def _runtime_paths(output):
    """Route third-party writes under the authorized project output, no chdir."""
    if not output.is_relative_to(ROOT):
        raise ValueError('Research execution output must be inside this project')
    if not output.is_relative_to(AUTHORIZED_QZ_ROOT):
        raise ValueError('QZ writes must stay inside the authorized user workspace')
    paths = {'HF_HOME':'huggingface', 'HF_HUB_CACHE':'huggingface/hub', 'HF_MODULES_CACHE':'huggingface/modules',
             'HF_DATASETS_CACHE':'huggingface/datasets', 'XDG_CACHE_HOME':'cache', 'XDG_CONFIG_HOME':'config',
             'XDG_DATA_HOME':'share', 'XDG_STATE_HOME':'state', 'TORCH_HOME':'torch',
             'TORCH_EXTENSIONS_DIR':'torch_extensions', 'TORCHINDUCTOR_CACHE_DIR':'torchinductor',
             'TRITON_CACHE_DIR':'triton', 'TRITON_DUMP_DIR':'triton_dump', 'TRITON_OVERRIDE_DIR':'triton_override',
             'CUDA_CACHE_PATH':'cuda', 'NUMBA_CACHE_DIR':'numba', 'MPLCONFIGDIR':'config/matplotlib',
             'WANDB_DIR':'state/wandb', 'WANDB_CACHE_DIR':'cache/wandb', 'WANDB_CONFIG_DIR':'config/wandb',
             'TMPDIR':'tmp', 'TMP':'tmp', 'TEMP':'tmp'}
    for name, relative in paths.items():
        path = no_symlink(output/'runtime'/relative); path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
                      PYTHONDONTWRITEBYTECODE='1', DO_NOT_TRACK='1')
    tempfile.tempdir = os.environ['TMPDIR']


class _LocalQwenBackend:
    def __init__(self, bundle_dir, manifest, model_path, device, dtype):
        from .evaluation import checkpoint_identity
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir() or not re.fullmatch(r'cuda:[0-9]+', device) or dtype != 'bfloat16':
            raise ValueError('Research baseline requires a local model, explicit CUDA device and bfloat16')
        _validate_model_directory(path)
        lock = _json(Path(bundle_dir) / 'asset-lock.json')
        identity = checkpoint_identity(path)
        weights = sorted(value for name, value in identity['files_sha256'].items() if name.endswith('.safetensors'))
        if weights != lock['weight_hashes'] or any(sha(path / n) != v for n, v in lock['metadata_sha256'].items()):
            raise ValueError('Local checkpoint differs from pinned initial Qwen weights/config/tokenizer')
        for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}.items():
            os.environ[key] = value
        import torch
        from transformers import set_seed
        from .modeling import load_text_model, load_tokenizer
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU pretrained fallback')
        self.tokenizer = load_tokenizer(path)
        self.protocol = generation_protocol(self.tokenizer, dtype=dtype, generation=manifest['generation'])
        if self.protocol['eos_token_id'] != lock['eos_token_id']:
            raise ValueError('Tokenizer EOS differs from pinned token audit')
        self.model = load_text_model(path, dtype=dtype).to(device).eval()
        self.identity, self.device, self.load_report = identity, device, self.model.load_report
        set_seed(self.protocol['seed'])

    def generate(self, prompt):
        import torch
        from transformers import GenerationConfig
        p, tok = self.protocol, self.tokenizer
        rendered = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        encoded = tok(rendered, return_tensors='pt', add_special_tokens=False).to(self.device)
        n_prompt = encoded.input_ids.shape[1]
        common = {'prompt_tokens': n_prompt, 'rendered_prompt_sha256': text_digest(rendered)}
        if n_prompt > p['max_prompt_tokens']:
            return {**common, 'response': '', 'n_tokens': 0, 'stop_reason': 'error', 'error': 'prompt_over_budget'}
        config = GenerationConfig(max_new_tokens=p['max_new_tokens'], do_sample=False, use_cache=True,
            pad_token_id=p['pad_token_id'], eos_token_id=p['eos_token_id'], num_beams=1,
            num_return_sequences=1, repetition_penalty=1.0)
        with torch.inference_mode():
            output = self.model.generate(**encoded, generation_config=config, synced_gpus=False)
        tokens = output[0, n_prompt:].detach().cpu().tolist()
        finish = 'eos' if tokens and tokens[-1] == p['eos_token_id'] else 'length' if len(tokens) >= p['max_new_tokens'] else 'unknown'
        return {**common, 'response': tok.decode(tokens, skip_special_tokens=True), 'n_tokens': len(tokens),
                'stop_reason': finish, 'error': None}


def _validate_model_directory(path):
    """Prevent alternate load entries from bypassing the pinned sharded base."""
    path = Path(path)
    index = path/'model.safetensors.index.json'
    if not index.is_file() or index.is_symlink():
        raise ValueError('Pinned base requires its single safetensors shard index')
    mapping = json.loads(index.read_text()).get('weight_map', {})
    if not isinstance(mapping,dict) or not mapping:
        raise ValueError('Invalid base shard map')
    shards=set(mapping.values())
    if len(shards)!=4 or any(not isinstance(n,str) or Path(n).name!=n or not n.endswith('.safetensors') for n in shards):
        raise ValueError('Pinned base requires exactly four local safetensors shards')
    forbidden={'model.safetensors','pytorch_model.bin','pytorch_model.bin.index.json','adapter_config.json',
               'added_tokens.json','special_tokens_map.json','tokenizer.model','chat_template.json','chat_templates'}
    for child in path.iterdir():
        if (child.name in forbidden or child.name.startswith('adapter')
                or child.name.endswith('.bin') or child.name.endswith('.bin.index.json')
                or child.name.endswith('.safetensors') and child.name not in shards):
            raise ValueError(f'Unpinned alternate model/tokenizer/adapter entry: {child.name}')
    if any(not (path/name).is_file() or (path/name).is_symlink() for name in shards):
        raise ValueError('Base shard is missing or redirected')
    return shards


def _validate_record(record, source, contract):
    cap = contract['generation_protocol']['max_new_tokens']
    if (record.get('id') != source['id'] or record.get('source_id') != source['source_id']
            or record.get('cohort') != source['cohort'] or record.get('benchmark') != source['benchmark']
            or record.get('prompt_sha256') != source['prompt_sha256'] or record.get('contract_sha256') != digest(contract)
            or record.get('checkpoint_id') != contract['model_identity']['checkpoint_id']
            or record.get('generation_protocol') != contract['generation_protocol']
            or record.get('purpose') != contract['purpose'] or not isinstance(record.get('response'), str)
            or record.get('response_sha256') != text_digest(record['response'])
            or type(record.get('n_tokens')) is not int or not 0 <= record['n_tokens'] <= cap
            or type(record.get('generation_attempted')) is not bool
            or (record.get('error') is not None and (not isinstance(record['error'], str) or not record['error']))
            or record.get('stop_reason') not in {'eos', 'length', 'unknown', 'error', 'not_generated'}
            or record.get('finish_reason') != record['stop_reason']):
        raise ValueError('Baseline prediction identity/content binding differs')
    if record['stop_reason']=='not_generated' and (record['generation_attempted'] or record['n_tokens'] != 0 or record['response']):
        raise ValueError('Unattempted baseline slot cannot contain generated content')
    if (not record['generation_attempted'] and record['stop_reason'] != 'not_generated'
            or record['stop_reason']=='eos' and record['n_tokens']==0
            or record['stop_reason']=='length' and record['n_tokens']!=cap
            or record.get('reached_token_limit') is not (record['n_tokens']>=cap)):
        raise ValueError('Baseline token count/termination/attempt metadata disagree')
    valid = bool(record['response'].strip()) and record['stop_reason'] == 'eos' and record.get('error') is None
    if record.get('valid') is not valid:
        raise ValueError('Baseline completion validity disagrees with stop/error/text')


def _validate_contract(contract, manifest, bundle_dir):
    if (contract.get('version') != VERSION or contract.get('bundle_manifest_sha256') != sha(Path(bundle_dir)/'manifest.json')
            or contract.get('purpose') != manifest['purpose'] or contract.get('code_sha256') != _codes()
            or contract.get('weights_exported') is not False or contract.get('optimizer_updates') != 0):
        raise ValueError('Baseline run uses different bundle/purpose/code')
    p = contract.get('generation_protocol', {})
    expected = {'profile':'rsd-text-eval-v2','enable_thinking':False,'max_new_tokens':512,'max_prompt_tokens':2048,
                'temperature':0.0,'do_sample':False,'top_p':1.0,'top_k':0,'num_beams':1,
                'repetition_penalty':1.0,'seed':0,'eos_token_id_source':'tokenizer.eos_token_id'}
    if (any(digest(p.get(k)) != digest(v) for k,v in expected.items())
            or any(type(p.get(k)) is not int or p[k] < 0 for k in ('eos_token_id','pad_token_id'))
            or any(not isinstance(p.get(k),str) or not re.fullmatch('[0-9a-f]{64}',p[k])
                   for k in ('tokenizer_vocab_sha256','tokenizer_template_sha256'))):
        raise ValueError('Saved baseline generation protocol differs from fixed evaluation')
    identity = contract.get('model_identity', {})
    if manifest['purpose']=='cpu_fixture':
        if (identity.get('kind')!='cpu_fixture' or contract.get('device')!='cpu'
                or contract.get('model_execution_kind')!='synthetic_injected_cpu_fixture'
                or p.get('dtype')!='float32'):
            raise ValueError('CPU baseline fixture identity/device differs')
    else:
        lock = _json(Path(bundle_dir)/'asset-lock.json')
        files = identity.get('files_sha256', {})
        weights = sorted(v for k,v in files.items() if k.endswith('.safetensors'))
        if (identity.get('kind')!='local_files' or identity.get('checkpoint_id')!='sha256:'+digest(files)
                or weights != lock['weight_hashes']
                or any(files.get(k)!=v for k,v in lock['metadata_sha256'].items())
                or p.get('eos_token_id')!=lock['eos_token_id'] or p.get('dtype')!='bfloat16'
                or not re.fullmatch(r'cuda:[0-9]+',contract.get('device',''))
                or contract.get('model_execution_kind')!='local_pretrained'):
            raise ValueError('Saved research baseline model/tokenizer identity differs from pinned base')


def _quantile(values, probability):
    if not values:return None
    values=sorted(values); pos=(len(values)-1)*probability; lo=int(pos); hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def summarize(bundle_dir, run_dir):
    manifest, task, gold, safety = load_bundle(bundle_dir)
    output = no_symlink(run_dir); contract = _json(output / 'run-contract.json')
    _validate_contract(contract, manifest, bundle_dir)
    inputs = _inputs(task, safety)
    records = read_jsonl(output / 'predictions.jsonl')
    if len(records) > len(inputs): raise ValueError('Too many baseline attempts')
    for record, source in zip(records, inputs): _validate_record(record, source, contract)
    task_rows = [r for r in records if r['cohort'] == 'task']
    safety_rows = [r for r in records if r['cohort'] == 'safety']
    scored = score_tasks(gold, task_rows)
    groups = {}
    for benchmark in manifest['counts']:
        rows = [r for r in records if r['benchmark'] == benchmark]
        attempted = [r for r in rows if r['generation_attempted']]
        tokens = [r['n_tokens'] for r in attempted if r['n_tokens']>0]
        groups[benchmark] = {'n_planned': manifest['counts'][benchmark], 'n_recorded': len(rows),
            'n_attempted': sum(r['generation_attempted'] for r in rows),
            'n_not_generated': sum(not r['generation_attempted'] for r in rows),
            'n_missing': manifest['counts'][benchmark]-len(rows), 'n_complete_nonempty': sum(r['valid'] for r in rows),
            'stop_reasons': dict(Counter(r['stop_reason'] for r in rows)), 'n_empty': sum(not r['response'].strip() for r in rows),
            'n_errors': sum(r.get('error') is not None for r in rows),
            'empty_rate_per_attempt': sum(not r['response'].strip() for r in attempted)/len(attempted) if attempted else None,
            'truncation_rate_per_attempt': sum(r['stop_reason']=='length' for r in attempted)/len(attempted) if attempted else None,
            'length_denominator': 'attempted records with observed positive generated-token count',
            'n_with_generated_tokens': len(tokens),
            'generated_tokens_sum': sum(tokens), 'generated_tokens_mean': sum(tokens)/len(tokens) if tokens else None,
            'generated_tokens_p50': _quantile(tokens,.50), 'generated_tokens_p95': _quantile(tokens,.95),
            'generated_tokens_p99': _quantile(tokens,.99),
            'generated_tokens_max': max(tokens) if tokens else None}
    summary = {'version': VERSION, 'purpose': manifest['purpose'], 'n_planned': len(inputs),
               'n_recorded': len(records), 'n_attempted': sum(r['generation_attempted'] for r in records),
               'n_not_generated': sum(not r['generation_attempted'] for r in records),
               'n_missing': len(inputs)-len(records), 'groups': groups,
               'task_summary': scored, 'safety_judge_status': 'NOT_RUN', 'safety_judgments_created': 0,
               'safety_responses_present': len(safety_rows), 'weights_exported': False, 'optimizer_updates': 0,
               'status': 'ALL_ATTEMPTS_RECORDED' if len(records)==len(inputs) and all(r['generation_attempted'] for r in records) else 'PARTIAL_ATTEMPTS_RETAINED',
               'scientific_conclusion': None, 'source_predictions_sha256': sha(output / 'predictions.jsonl')}
    return summary


def execute(bundle_dir, output_dir, *, model_path=None, device='cuda:0', dtype='bfloat16',
            execution_enabled=False, backend=None, purpose='research'):
    if execution_enabled is not True:
        raise ValueError('Baseline execution is disabled; explicit execution_enabled=True is required')
    manifest, task, _, safety = load_bundle(bundle_dir)
    if purpose != manifest['purpose'] or purpose not in {'research', 'cpu_fixture'}:
        raise ValueError('Execution purpose differs from prepared bundle')
    if backend is not None and (purpose != 'cpu_fixture' or device != 'cpu'):
        raise ValueError('Injected backends are CPU fixtures only')
    if purpose == 'cpu_fixture' and backend is None:
        raise ValueError('CPU fixture requires an explicit injected backend; no pretrained fallback')
    output = no_symlink(output_dir)
    if output.exists(): raise FileExistsError('Baseline attempts cannot be overwritten or retried in place')
    if backend is None:
        if model_path is None: raise ValueError('Explicit verified local model path required')
        model_root = Path(model_path).expanduser().resolve()
        if output.is_relative_to(model_root) or model_root.is_relative_to(output):
            raise ValueError('Baseline outputs must not overlap the read-only model directory')
        _runtime_paths(output)
        try:
            backend = _LocalQwenBackend(bundle_dir, manifest, model_path, device, dtype)
        except BaseException as exc:
            write_new(output/'startup-error.json', {'version': VERSION, 'purpose': purpose,
                'status': 'FAILED_BEFORE_GENERATION', 'n_planned': manifest['n_planned'], 'n_attempted': 0,
                'bundle_manifest_sha256': sha(Path(bundle_dir)/'manifest.json'), 'error': f'{type(exc).__name__}: {exc}'})
            raise
    protocol = generation_protocol(backend.tokenizer, dtype=dtype, generation=manifest['generation'])
    if backend.protocol != protocol:
        raise ValueError('Backend generation protocol differs from planned evaluation')
    if not isinstance(backend.identity, dict) or not isinstance(backend.identity.get('checkpoint_id'), str):
        raise ValueError('Backend lacks checkpoint identity')
    if purpose == 'cpu_fixture' and backend.identity.get('kind') != 'cpu_fixture':
        raise ValueError('Injected identity must explicitly remain a CPU fixture')
    output.mkdir(parents=True, mode=0o700, exist_ok=purpose=='research')
    contract = {'version': VERSION, 'purpose': purpose, 'bundle_manifest_sha256': sha(Path(bundle_dir)/'manifest.json'),
                'model_identity': backend.identity, 'generation_protocol': protocol, 'device': device,
                'code_sha256': _codes(), 'weights_exported': False, 'optimizer_updates': 0,
                'load_report': getattr(backend, 'load_report', None), 'model_execution_kind': 'local_pretrained' if purpose=='research' else 'synthetic_injected_cpu_fixture'}
    write_new(output/'run-contract.json', contract)
    _validate_contract(contract, manifest, bundle_dir)
    inputs = _inputs(task, safety)
    error, started = None, time.monotonic()
    with (output/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for source in inputs:
            record = {k: source[k] for k in ('id','source_id','cohort','benchmark','prompt_sha256')}
            record.update(purpose=purpose, contract_sha256=digest(contract), checkpoint_id=backend.identity['checkpoint_id'],
                generation_protocol=protocol, response='', n_tokens=0, stop_reason='error', error=None,
                generation_attempted=error is None)
            try:
                if error is None:
                    result = backend.generate(source['prompt'])  # Never gold, feedback, or benchmark metadata.
                    record.update({k: result[k] for k in ('response','n_tokens','stop_reason','error')})
                    for key in ('prompt_tokens','rendered_prompt_sha256'):
                        if key in result: record[key] = result[key]
                else:
                    record.update(stop_reason='not_generated', error='not_attempted_after_prior_failure')
            except BaseException as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
                error = record['error']
            try:
                record['finish_reason'] = record['stop_reason']
                record['response_sha256'] = text_digest(record['response'])
                record['valid'] = bool(record['response'].strip()) and record['stop_reason']=='eos' and record['error'] is None
                record['reached_token_limit'] = record['n_tokens'] >= protocol['max_new_tokens']
                _validate_record(record, source, contract)
            except (ValueError, TypeError, AttributeError) as exc:
                # Keep malformed backend evidence separately; never coerce it
                # into a completed generation or silently drop the attempt.
                raw = {k: record.get(k) for k in ('response','n_tokens','stop_reason','error')}
                error = f'Invalid backend result: {type(exc).__name__}: {exc}'
                record.update(invalid_backend_result=raw, response='', response_sha256=text_digest(''), n_tokens=0,
                    stop_reason='error', finish_reason='error', error=error, valid=False, reached_token_limit=False)
                _validate_record(record, source, contract)
            _append(stream, record)
    result = summarize(bundle_dir, output)
    write_new(output/'task-summary.json', result['task_summary'])
    write_new(output/'summary.json', {**result, 'elapsed_seconds': time.monotonic()-started, 'execution_error': error})
    _write_rows(output/'safety-predictions.jsonl', [r for r in read_jsonl(output/'predictions.jsonl') if r['cohort']=='safety'])
    _write_rows(output/'safety-dataset.jsonl', safety)
    write_new(output/'receipt.json', {'version': VERSION, 'status': 'FAILED_ATTEMPT_RETAINED' if error else result['status'],
        'purpose': purpose, 'n_attempted': result['n_attempted'], 'n_planned': len(inputs), 'execution_error': error,
        'artifacts_sha256': {p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        'judge_calls_executed': 0, 'optimizer_updates': 0, 'weights_exported': False, 'scientific_conclusion': None})
    return {**result, 'execution_error': error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--plan', default='configs/rsd/exploration_v2.plan.json'); p.add_argument('--output', required=True)
    p = commands.add_parser('check'); p.add_argument('--bundle', required=True)
    p = commands.add_parser('execute'); p.add_argument('--bundle', required=True); p.add_argument('--output', required=True)
    p.add_argument('--model-path', required=True); p.add_argument('--device', default='cuda:0'); p.add_argument('--enable-execution', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.plan, args.output)
    elif args.command == 'check': result = load_bundle(args.bundle)[0]
    else: result = execute(args.bundle, args.output, model_path=args.model_path, device=args.device, execution_enabled=args.enable_execution)
    print(json.dumps({k:result[k] for k in ('version','purpose','n_planned')}, ensure_ascii=False))
    if result.get('execution_error'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
