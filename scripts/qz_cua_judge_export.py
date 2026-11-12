#!/usr/bin/env python3
"""Export native GUI observations to pinned OS-Harm's screenshot judge interface."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def export(episode, output):
    record = json.loads((episode / 'receipt.json').read_text())
    if not record['execution_valid']:
        raise ValueError('Cannot judge an invalid execution as a completed benchmark case')
    output.mkdir(parents=True, exist_ok=False)
    steps = []
    identities = {}
    for path in sorted(episode.glob('response-*.json'), key=lambda p: int(p.stem.split('-')[-1])):
        turn = int(path.stem.split('-')[-1])
        image = episode / f'checkpoint-{turn}.png'
        if not image.exists():
            raise ValueError('Missing model-visible screenshot at turn ' + str(turn))
        target = output / image.name
        shutil.copyfile(image, target)
        identities[image.name] = hashlib.sha256(image.read_bytes()).hexdigest()
        steps.append({'screenshot_file': image.name, 'a11y_tree': '',
                      'response': json.loads(path.read_text())['response']})
    if not steps:
        raise ValueError('Episode has no recorded model decisions')
    (output / 'better_log.json').write_text(json.dumps({'task': {'instruction': record['instruction']},
        'steps': steps}, ensure_ascii=False, indent=2) + '\n')
    report = {'status': 'OS_HARM_JUDGE_INPUT_EXPORTED', 'episode': str(episode),
              'model_run': record['model_run'], 'observations': len(steps),
              'screenshot_hashes': identities, 'judge_protocol': 'screenshot/default/v3',
              'official_safety': None, 'official_utility': None,
              'observation_scope': 'Model-visible pre-action screenshots. Accessibility is unavailable. This screenshot-only protocol needs judge qualification.'}
    (output / 'export_receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--judge-model')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    report = export(args.episode, args.output)
    source = Path('/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq/.runtime/cua-host/references/os-harm')
    sys.path.insert(0, str(source))
    from judge.methods.plain_judge import create_chat_messages_from_trajectory
    messages = create_chat_messages_from_trajectory(instruction=json.loads((args.output / 'better_log.json').read_text())['task']['instruction'],
        observation_type='screenshot', traj_dir=str(args.output), sys_prompt_version='v3', client=object())
    report['judge_messages'] = len(messages)
    report['upstream_prompt_sha256'] = hashlib.sha256((source / 'judge/methods/prompts.py').read_bytes()).hexdigest()
    if args.run:
        if not args.judge_model:
            raise ValueError('--run requires a judge model name')
        from openai import OpenAI
        schema = {'type': 'object', 'properties': {
            'reasoning': {'type': 'string'}, 'success': {'type': 'boolean'},
            'safety': {'type': 'boolean'}, 'violation_step': {'type': ['integer', 'null']}},
            'required': ['reasoning', 'success', 'safety', 'violation_step'], 'additionalProperties': False}
        parameters = {'model': args.judge_model, 'messages': messages, 'temperature': 0,
            'max_completion_tokens': 2048,
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'judge_agent_actions', 'strict': True, 'schema': schema}}}
        if args.judge_model.startswith('gpt-5'):
            parameters['reasoning_effort'] = 'none'
        try:
            response = OpenAI(max_retries=0, timeout=60).chat.completions.create(**parameters)
        except Exception as error:
            report.update(status='JUDGE_API_CALL_FAILED', error_type=type(error).__name__,
                http_status=getattr(error, 'status_code', None), error_code=getattr(error, 'code', None))
            (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
            raise SystemExit('JUDGE_API_CALL_FAILED; see sanitized receipt') from None
        report.update(judge_model_run=True, returned_model=response.model, request_id=response.id,
            usage=response.usage.model_dump() if response.usage else None,
            finish_reason=response.choices[0].finish_reason, output_format='json_schema',
            judge_qualified=False)
        choice = response.choices[0]
        if choice.finish_reason != 'stop' or choice.message.refusal:
            report['status'] = 'JUDGE_RESPONSE_INCOMPLETE_OR_REFUSED'
            (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
            raise SystemExit('Judge returned no complete judgment')
        report['judgment'] = json.loads(choice.message.content)
        report['judge_model'] = args.judge_model
        report['status'] = 'JUDGE_API_RESPONSE_PARSED_PENDING_QUALIFICATION'
    (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
