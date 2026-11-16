#!/usr/bin/env python3
"""Aggregate existing M0 receipts and audit saved pixels; never run a desktop."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from cua_defense.measurement import compare_prefix, outcome, summarize_controls


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_hash(image):
    return hashlib.sha256((image.mode + str(image.size)).encode() + image.tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(paths):
    from PIL import Image, ImageChops

    runs = [(p.resolve(), json.loads((p / 'receipt.json').read_text())) for p in paths]
    require(len({p.name for p, _ in runs}) == len(runs), 'Duplicate run ID')
    planned = runs[0][1]['planned_ids']
    config_hash = runs[0][1]['config_sha256']
    attempts, valid, invalid = [], {}, []
    fixed_sources = None
    for directory, report in runs:
        require(report['planned_ids'] == planned and report['config_sha256'] == config_hash,
                'Registered panel or configuration changed')
        require(not any(report[k] for k in ('model_run', 'gpu_used', 'api_calls',
                                           'training_updates', 'training_eligible')), 'Non-M0 run')
        lifecycle_file = directory / 'lifecycle.json'
        lifecycle = json.loads(lifecycle_file.read_text()) if lifecycle_file.exists() else None
        exit_file = directory.with_suffix('.exit.json')
        exited = json.loads(exit_file.read_text()) if exit_file.exists() else None
        require(report['status'] in ('M0_CONTROLS_COMPLETE', 'M0_SUBSET_COMPLETE', 'M0_INCOMPLETE')
                or lifecycle and lifecycle['status'] == 'EXITED_WITHOUT_FINAL_RECEIPT',
                'Run is not known to be terminal')
        counts = summarize_controls(report['results'], planned)
        if 'counts' in report:
            require(counts == report['counts'], 'Recorded counts disagree')
        attempts.append({
            'run_id': directory.name, 'receipt_sha256': sha(directory / 'receipt.json'),
            'receipt_status': report['status'], 'lifecycle': lifecycle,
            'exit_record_sha256': sha(exit_file) if exited else None,
            'exit': {k: exited.get(k) for k in ('status', 'returncode', 'seconds',
                    'started_at', 'ended_at')} if exited else None,
            'counts': counts, 'selected_count': len(report.get('execution_subset', planned)),
            'stop_reason': report.get('stop_reason'), 'source_sha256': report['source_sha256'],
            'reference_run_id': Path(report['reference_run']).name if report.get('reference_run') else None,
            'reference_receipt_sha256': report.get('reference_receipt_sha256'),
        })
        if report.get('reference_run'):
            ref = Path(report['reference_run'])
            require(any(p == ref.resolve() for p, _ in runs), 'Reference run missing from audit')
            require(sha(ref / 'receipt.json') == report['reference_receipt_sha256'],
                    'Reference receipt changed')
        for row in report['results']:
            require(row['labels'] == outcome(row['task_score'], row.get('risk'), row['execution_valid']),
                    'Labels do not reproduce')
            case_file = directory / row['case_id'] / 'receipt.json'
            require(json.loads(case_file.read_text()) == row, 'Per-control receipt disagrees')
            if not row['execution_valid']:
                invalid.append({'run_id': directory.name, 'case_id': row['case_id'],
                    'labels': row['labels'], 'seconds': row['seconds'],
                    'error_last_line': row.get('error', '').strip().splitlines()[-1],
                    'receipt_sha256': sha(case_file)})
                continue
            require(row['control_match'] and row['risk']['coverage_complete'] is True,
                    'Valid execution did not match control')
            require(row['case_id'] not in valid, 'Multiple valid results: do not choose the best')
            sources = {k: v for k, v in report['source_sha256'].items() if k != 'scripts/qz_cua_m0.py'}
            if fixed_sources is None:
                fixed_sources = sources
            require(sources == fixed_sources, 'Successful controls used different measurement/protocol sources')
            valid[row['case_id']] = (directory, row)

    controls, pixels, non_image_matches = [], [], []
    comparisons = Counter()
    for case_id in planned:
        if case_id not in valid:
            continue
        directory, row = valid[case_id]
        reference_id = f"{row['family']}__r0__safe_completion"
        ref_dir, reference = valid[reference_id]
        prefix = row['branch_prefix']
        comparison = row['prefix_comparison']
        if case_id == reference_id:
            require(comparison['status'] == 'REFERENCE', 'Reference selection changed')
        else:
            reproduced = compare_prefix(reference['branch_prefix'], prefix)
            require(all(comparison[k] == v for k, v in reproduced.items()), 'Prefix comparison changed')
            require(comparison['reference_case_id'] == reference_id, 'Reference selection changed')
            non_image_matches.append(all(value is True for key, value in reproduced['checks'].items()
                                         if key != 'images_sha256'))
            frames = []
            require(len(prefix['images_sha256']) == 2, 'Unexpected history length')
            for turn in range(2):
                with Image.open(ref_dir / reference_id / f'checkpoint-{turn}.png') as a, \
                        Image.open(directory / case_id / f'checkpoint-{turn}.png') as b:
                    require(image_hash(a) == reference['branch_prefix']['images_sha256'][turn]
                            and image_hash(b) == prefix['images_sha256'][turn],
                            'Saved image differs from the recorded model input')
                    require(a.mode == b.mode and a.size == b.size, 'Image mode or dimensions changed')
                    channels = ImageChops.difference(a, b).split()
                    mask = channels[0]
                    for channel in channels[1:]:
                        mask = ImageChops.lighter(mask, channel)
                    mask = mask.point(lambda value: 255 if value else 0)
                    cut = int(a.height * .04)
                    total = mask.histogram()[255]
                    top = mask.crop((0, 0, a.width, cut)).histogram()[255]
                    frames.append({'turn': turn, 'size': list(a.size), 'bbox': mask.getbbox(),
                                   'changed_pixels': total, 'top_4pct_pixels': top, 'body_pixels': total - top})
            pixels.append({'case_id': case_id, 'reference_case_id': reference_id, 'frames': frames})
        comparisons[comparison['status']] += 1
        controls.append({'case_id': case_id, 'run_id': directory.name, 'scenario': row['scenario'],
                         'labels': row['labels'], 'seconds': row['seconds'],
                         'prefix_status': comparison['status'],
                         'receipt_sha256': sha(directory / case_id / 'receipt.json')})

    all_frames = [frame for item in pixels for frame in item['frames']]
    scenario_counts = {s: sum(c['scenario'] == s for c in controls)
                       for s in runs[0][1]['config']['scenarios']}
    body_changes = sum(f['body_pixels'] for f in all_frames)
    return {
        'schema_version': 1, 'phase': 'M0', 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'SCRIPTED_CONTROL_PANEL_COMPLETE' if len(valid) == len(planned) else 'PANEL_INCOMPLETE',
        'scope': 'Two synthetic desktop file-edit families, six scripted outcomes, two repeats. '
                 'Not autonomous model trajectories, OSWorld benchmark scores, or training evidence.',
        'model_run': False, 'gpu_used': False, 'api_calls': 0, 'training_updates': 0,
        'training_eligible': False, 'config_sha256': config_hash,
        'registered_unique_controls': len(planned), 'unique_valid_controls': len(valid),
        'missing_valid_control_ids': [k for k in planned if k not in valid],
        'physical_control_attempts': sum(a['counts']['attempted'] for a in attempts),
        'invalid_control_attempts': len(invalid), 'attempts': attempts,
        'invalid_controls_preserved': invalid, 'controls': controls,
        'valid_controls_per_scenario': scenario_counts,
        'prefix_audit': {'population': 'Unique valid controls only; invalid attempts remain above.',
            'counts': dict(comparisons), 'image_comparisons': len(pixels),
            'non_image_checks_all_match': bool(non_image_matches) and all(non_image_matches),
            'frames_audited': len(all_frames), 'body_changed_pixels_total': body_changes,
            'comparisons_with_body_difference': sum(any(f['body_pixels'] for f in p['frames']) for p in pixels),
            'all_differences_within_top_4pct': bool(all_frames) and body_changes == 0,
            'saved_pixel_hashes_match_recorded_model_inputs': True,
            'diagnostic_only_no_input_was_cropped': True, 'pixel_details': pixels},
        'full_environment_equivalence': 'NOT_ESTABLISHED', 'candidate_A_causal_branch_ready': False,
        'aggregation_policy': 'One valid execution per registered ID required; duplicate valid results '
                              'are rejected. Invalid attempts and startup exits remain in the audit. '
                              'Only runner scheduling may differ among valid attempts; task config, '
                              'measurement and action protocol hashes must agree.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.runs)
    report['audit_script_sha256'] = sha(Path(__file__))
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'registered_unique_controls',
        'unique_valid_controls', 'physical_control_attempts', 'invalid_control_attempts')}))
    print(json.dumps({k: v for k, v in report['prefix_audit'].items() if k != 'pixel_details'}))


if __name__ == '__main__':
    main()
