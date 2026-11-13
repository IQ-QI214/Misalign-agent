#!/usr/bin/env python3
"""Select verified positives and reuse the pinned CUA-Lite SFT exporter."""
import argparse
import hashlib
import json
from pathlib import Path
import platform


def eligible(record):
    if record.get('data_role') not in ['train', 'development']:
        return False
    if not record.get('model_run') or not record.get('execution_valid') or record.get('task_score') != 1:
        return False
    if record['condition'] == 'clean':
        return True
    return record.get('safe_completion') is True and record.get('risk', {}).get('coverage_complete') is True


def convert_trajectory(path, record, model, project, *, diagnostic=False):
    prefix = record.get('supervise_from_step', 0)
    if type(prefix) is not int or prefix < 0:
        raise ValueError('Supervision start must be a nonnegative step index')
    if not diagnostic and not eligible(record):
        raise ValueError('Trajectory lacks verified positive eligibility')
    import pyarrow.parquet as pq
    from lite.agents.factory import LOCAL_AGENTS
    from lite.train.export.export_sft import _convert_sample
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    converted = []
    provenance = []
    agent_kwargs = {'enable_thinking': False}
    if record.get('protocol', {}).get('shift_click'):
        from cua_defense.desktop_actions import make_shift_click_action_space
        agent_kwargs['action_space'] = make_shift_click_action_space()
    for raw in pq.read_table(path).to_pylist():
        row = _convert_sample(raw, agent_id=LOCAL_AGENTS[model['model_id']]['agent_id'],
                              agent_kwargs=agent_kwargs, model_id=model['processor_path'],
                              image_root=str(project / 'references/cua_lite'), strict=True)
        row.pop('_error')
        row['steps'] = [step for step in row['steps'][prefix:] if step['status'] == 'completed']
        if not row['steps']:
            continue
        metadata = json.loads(row['metadata'])
        metadata['others'].update(
            research_training_eligible=not diagnostic,
            research_job_id=record['job_id'],
            research_source_sha256=digest,
            research_repair_parent=record.get('repair_parent'),
            research_repair_mode=record.get('repair_mode'),
            research_supervise_from_step=prefix,
        )
        row['metadata'] = json.dumps(metadata)
        converted.append(row)
        provenance.append({'job_id': record['job_id'], 'source': str(path),
            'source_sha256': digest, 'steps': len(row['steps']),
            'supervised_tokens': sum(len(step['response_tokens']) for step in row['steps']),
            'supervise_from_step': prefix, 'repair_parent': record.get('repair_parent'),
            'repair_mode': record.get('repair_mode')})
    return converted, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--diagnostic-input', type=Path)
    parser.add_argument('--model', choices=['qwen35_9b', 'qwen3vl_4b'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('QZ_LINUX_REQUIRED')
    if bool(args.batch) == bool(args.diagnostic_input):
        raise ValueError('Choose one batch or diagnostic input')
    from lite.utils.parquet import write_records_to_parquet
    project = Path(__file__).resolve().parents[1]
    model = json.loads((project / 'configs/cua/models.json').read_text())[args.model]
    records = []
    diagnostic = bool(args.diagnostic_input)
    if diagnostic:
        receipt = args.diagnostic_input.parent / 'receipt.json'
        record = {'job_id': 'diagnostic', 'supervise_from_step': 0}
        if receipt.exists():
            record.update(json.loads(receipt.read_text()))
        inputs = [(args.diagnostic_input, record)]
    else:
        for path in args.batch.glob('episodes/*/receipt.json'):
            record = json.loads(path.read_text())
            records.append(record)
        inputs = [(Path(r['output']) / 'trajectory.parquet', r) for r in records if eligible(r)]
    converted = []
    provenance = []
    for path, record in inputs:
        rows, sources = convert_trajectory(path, record, model, project, diagnostic=diagnostic)
        converted.extend(rows)
        provenance.extend(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if converted:
        write_records_to_parquet(converted, args.output)
    report = {'schema_version': 1, 'status': 'SFT_EXPORT_PASSED' if converted else 'NO_ELIGIBLE_POSITIVES',
              'diagnostic_only': diagnostic, 'training_eligible': not diagnostic and bool(converted),
              'source_episodes': len(records), 'rows': len(converted), 'provenance': provenance,
              'supervised_tokens': sum(p['supervised_tokens'] for p in provenance),
              'output': str(args.output) if converted else None}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
