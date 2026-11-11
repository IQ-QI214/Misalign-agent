#!/usr/bin/env python3
"""Freeze external evaluation identities and an application-held-out OSWorld panel."""
import ast
import hashlib
import json
from pathlib import Path
import platform

ROOT = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq')
PROJECT = ROOT / 'Misalign-agent'
OS_HARM_COMMIT = 'c0fa95e75bafb00ac05d2eb4ac5418b9913475ee'


def main():
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise SystemExit('QZ_LINUX_REQUIRED')
    state = ROOT / '.runtime/cua-host/prepared'
    source = ROOT / '.runtime/cua-host/references/os-harm'
    tree = ast.parse((source / 'run.py').read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'all_injection_goals' for t in node.targets))
    goals = ast.literal_eval(assignment.value)
    index = json.loads((source / 'evaluation_examples/test_injection.json').read_text())
    cases = {}
    occurrences = 0
    for domain, ids in index.items():
        for task_id in ids:
            path = source / 'evaluation_examples/examples' / domain / (task_id + '.json')
            task = json.loads(path.read_text())
            for injection in task['injections']:
                for goal in injection['goals']:
                    occurrences += 1
                    key = '__'.join([domain, task_id, injection['type'], goal])
                    if key in cases:
                        cases[key]['source_occurrences'] += 1
                        continue
                    cases[key] = {'case_id': key, 'domain': domain, 'task_id': task_id,
                        'source_occurrences': 1, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'goal': goal, 'injection': {**injection, 'content': goals[goal], 'goal_id': goal},
                        'task': task, 'official_utility': None, 'official_safety': None,
                        'judge_status': 'PENDING_MODEL_EVALUATION'}
    (state / 'os_harm_cases.json').write_text(json.dumps(list(cases.values()), indent=2) + '\n')
    catalog = PROJECT / 'references/cua_lite/lite/gym/envs/lite/osworld/data/eval.jsonl'
    rows = [json.loads(line) for line in catalog.read_text().splitlines()]
    allowed = {'chrome', 'gimp', 'vlc', 'libreoffice_impress', 'thunderbird', 'vs_code'}
    kept = [r for r in rows if r['metadata']['others']['domain'] in allowed
            and not r['metadata']['others'].get('exclude_reason')]
    heldout = {'schema_version': 1, 'catalog': 'lite/gym/envs/lite/osworld/data/eval.jsonl',
        'catalog_sha256': hashlib.sha256(catalog.read_bytes()).hexdigest(),
        'selection': 'Retain the six listed applications after upstream exclusions; Calc/Writer development, OS bootstrap, and multi-app tasks are outside this panel.',
        'domains': sorted(allowed), 'task_count': len(kept),
        'tasks': [{'task_id': r['task_id'], 'domain': r['metadata']['others']['domain'],
                   'catalog_row_sha256': hashlib.sha256((json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n').encode()).hexdigest()}
                  for r in kept]}
    (state / 'osworld_heldout.json').write_text(json.dumps(heldout, indent=2) + '\n')
    summary = {'schema_version': 1, 'status': 'BENCHMARK_IDENTITIES_FROZEN', 'model_run': False,
        'os_harm_commit': OS_HARM_COMMIT, 'base_tasks': sum(map(len, index.values())),
        'source_occurrences': occurrences, 'unique_injected_cases': len(cases),
        'source_duplicate_occurrences': occurrences - len(cases),
        'case_ids': list(cases), 'injection_vectors': sorted({c['injection']['type'] for c in cases.values()}),
        'judge_required': True, 'official_scores': 'UNKNOWN_WITHOUT_QUALIFIED_JUDGE',
        'osworld_total_catalog': len(rows), 'osworld_heldout_tasks': len(kept)}
    (state / 'benchmark_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='case_ids'}, indent=2))


if __name__ == '__main__':
    main()
