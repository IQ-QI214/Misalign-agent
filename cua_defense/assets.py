"""Small configuration hashes and shard metadata; never load model weights."""
import hashlib
import json
from pathlib import Path


def model_manifest(path):
    path=Path(path)
    index=path/'model.safetensors.index.json'
    files=sorted(set(json.loads(index.read_text())['weight_map'].values()))
    metadata={}
    for name in files:
        f=path/name
        stat=f.stat()
        if stat.st_size<=0: raise ValueError('Empty model shard')
        metadata[name]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    configs={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in
        ['config.json','tokenizer_config.json','tokenizer.json','preprocessor_config.json',
         'chat_template.jinja','model.safetensors.index.json']}
    return {'config_sha256':configs,'shards':metadata,'weight_bytes':sum(x['bytes'] for x in metadata.values()),
            'weight_content_sha256':'NOT_COMPUTED',
            'identity_scope':'Configuration content hashes and shard size/mtime only; not cryptographic proof of weight contents.'}


def source_manifest(project):
    project=Path(project)
    files=['scripts/qz_cua_m1.py','scripts/qz_cua_m1_session.py','scripts/qz_cua_worker_session.py','scripts/qz_cua_hf_worker.py','scripts/qz_cua_batch.py',
        'scripts/qz_cua_m0.py','scripts/qz_cua_judge_controls.py','scripts/qz_cua_focus_diagnostic.py',
        'configs/cua/m1_pilot.json','configs/cua/models.json','configs/cua/upstream.lock.json']
    files += [str(p.relative_to(project)) for p in sorted((project/'cua_defense').glob('*.py'))]
    return {name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in files}
