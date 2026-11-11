#!/usr/bin/env python3
"""QZ CPU preparation of frozen task variants and cross-workspace model assets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')
PROJECT = ROOT / 'Misalign-agent'
UPSTREAM = PROJECT / 'references/cua_lite'


def main():
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise SystemExit('QZ_LINUX_REQUIRED')
    from cua_defense.tasks import ATTACKS, load_rows, variant
    state = ROOT / '.runtime/cua-host/prepared'
    state.mkdir(exist_ok=True)
    rows = load_rows(UPSTREAM, PROJECT / 'configs/cua/development_tasks_v1.json')
    variants = [variant(row, family) for row in rows for family in ATTACKS]
    (state / 'risk_tasks.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in variants))
    report = {'schema_version': 1, 'time_utc': datetime.now(timezone.utc).isoformat(),
              'model_run': False, 'gpu_used': False, 'base_tasks': len(rows),
              'variants': len(variants), 'models': {}}
    models = json.loads((PROJECT / 'configs/cua/models.json').read_text())
    for name, spec in models.items():
        source = Path(spec['processor_path'])
        target = Path(spec['gpu_model_path'])
        if source != target and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.staging')
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(source, temporary)
            temporary.rename(target)
        weights = json.loads((source / 'model.safetensors.index.json').read_text())
        files = sorted(set(weights['weight_map'].values()))
        identities = {}
        for filename in ['config.json', 'tokenizer_config.json', *files]:
            digest = hashlib.file_digest((source / filename).open('rb'), 'sha256').hexdigest()
            if source != target:
                staged = hashlib.file_digest((target / filename).open('rb'), 'sha256').hexdigest()
                if staged != digest:
                    raise RuntimeError('Staged model file hash differs: ' + filename)
            identities[filename] = {'sha256': digest, 'bytes': (source / filename).stat().st_size}
        report['models'][name] = {**spec, 'files': identities, 'weights_sha256_checked': True}
        print('MODEL_ASSETS_PASSED=' + name, flush=True)
    report['risk_tasks_sha256'] = hashlib.sha256((state / 'risk_tasks.jsonl').read_bytes()).hexdigest()
    (state / 'assets.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PREPARED=' + str(state), flush=True)


if __name__ == '__main__':
    main()
