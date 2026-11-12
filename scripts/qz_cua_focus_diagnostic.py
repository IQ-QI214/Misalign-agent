#!/usr/bin/env python3
"""Replay saved GUI actions with optional source timing and a Writer focus control."""
import argparse
import asyncio
from collections import Counter
import hashlib
import io
import itertools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile

from scripts.qz_cua_batch import PROJECT, ROOT, write_json


def fonts(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        document = ET.fromstring(archive.read('word/document.xml'))
    namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    return dict(Counter(node.get('{' + namespace['w'] + '}ascii', 'UNSPECIFIED')
                        for node in document.findall('.//w:rFonts', namespace)))


async def check(episode, output, repetitions=1, match_timing=False, original_only=False):
    import lite.gym as gym
    from lite.agents.factory import make
    from lite.agents.core.agent.logger import TrajectoryLogger
    from transformers import AutoProcessor
    from cua_defense.tasks import load_rows, register_row
    from cua_defense.replay import checkpoint, verify_checkpoint, file_fingerprint, focus_state
    source = json.loads((episode / 'receipt.json').read_text())
    task_id = source['task_id']
    if not original_only and task_id != 'synth_writer_f_writer_93__doc_font_0001':
        raise ValueError('This diagnostic is specific to the Origami font task')
    rows = load_rows(PROJECT / 'references/cua_lite', PROJECT / 'configs/cua/development_tasks_v1.json')
    row = next(row for row in rows if row['task_id'] == task_id)
    target = row['metadata']['evaluator']['result']['path']
    model = json.loads((PROJECT / 'configs/cua/models.json').read_text())['qwen35_9b']
    processor = AutoProcessor.from_pretrained(model['processor_path'], local_files_only=True)
    responses = [json.loads(path.read_text())['response'] for path in
                 sorted(episode.glob('response-*.json'), key=lambda path: int(path.stem.split('-')[-1]))]
    waits = [json.loads((episode / f'turn_{turn:04d}/05_timing.json').read_text())['predict']
             for turn in range(len(responses))] if match_timing else [0] * len(responses)
    focus = ('<tool_call>\n<function=computer_use>\n<parameter=action>left_click</parameter>\n'
             '<parameter=coordinate>[500, 400]</parameter>\n</function>\n</tool_call>')
    records = []
    conditions = [('original_replay', responses, waits),
                  ('document_focus_then_replay', [focus, *responses], [0, *waits])]
    if original_only:
        conditions = conditions[:1]
    for repeat, (name, sequence, delays) in itertools.product(range(repetitions), conditions):
        directory = output / (name if repetitions == 1 else f'{name}-{repeat}')
        directory.mkdir()
        env = gym.make(register_row(row), max_steps=len(sequence) + 1, seed=20261112,
                       loop_detect=0, extra_tools=['response', 'terminate'])
        turn = 0
        record = {'condition': name, 'repeat': repeat, 'match_prediction_timing': match_timing,
                  'model_run': False, 'gpu_used': False, 'training_eligible': False, 'turns': []}
        original_evaluator = env.unwrapped._evaluate_final_fn

        async def evaluate_and_capture(task, computer, actions=None, debug=False):
            score = await original_evaluator(task, computer, actions, debug)
            # Native sample() closes its computer before returning to the caller.
            # Capture after grading while its execution handle is still alive.
            record['post_evaluation_files'] = []
            try:
                entries = [row['metadata']['evaluator']['result']]
                expected = row['metadata']['evaluator'].get('expected', {})
                entries += expected if isinstance(expected, list) else [expected]
                for index, entry in enumerate(entries):
                    if entry.get('type') != 'vm_file':
                        continue
                    data = await computer.interface.read_bytes(entry['path'])
                    filename = f'post_evaluation_{index}' + Path(entry['path']).suffix
                    (directory / filename).write_bytes(data)
                    item = {'path': entry['path'], 'snapshot': filename,
                            'fingerprint': file_fingerprint(data)}
                    if filename.endswith('.xlsx'):
                        import openpyxl
                        workbook = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
                        item['sheet_values'] = {sheet.title: list(sheet.values) for sheet in workbook}
                        workbook.close()
                    record['post_evaluation_files'].append(item)
            except Exception as error:
                record['file_capture_error'] = f'{type(error).__name__}: {error}'
            return score

        env.unwrapped._evaluate_final_fn = evaluate_and_capture
        async def generate(**data):
            nonlocal turn
            started = time.monotonic()
            observation = {'turn': turn, 'focus': await focus_state(env._computer.interface),
                           'source_predict_seconds': delays[turn]}
            if turn == 0:
                point = await checkpoint(env, row, data['images'][-1], directory, turn)
                record['initial_state_check'] = verify_checkpoint(episode, point, directory, turn)
            if turn == len(sequence) - 1:
                result = await env._computer.interface.read_bytes(target)
                (directory / ('before_termination' + Path(target).suffix)).write_bytes(result)
                if target.endswith('.docx'):
                    record['document_fonts'] = fonts(result)
                record['document_fingerprint'] = file_fingerprint(result)
                original = json.loads((episode / f'checkpoint-{len(responses)-1}.json').read_text())
                record['original_final_fingerprint'] = original['files'][target]
                data['images'][-1].save(directory / 'before_termination.png')
            text = sequence[turn]
            await asyncio.sleep(max(0, delays[turn] - (time.monotonic() - started)))
            observation['replay_predict_seconds'] = time.monotonic() - started
            record['turns'].append(observation)
            turn += 1
            return {'response': text, 'finish_reason': 'stop'}
        try:
            agent = make(model['model_id'], env=env, processor=processor, generate_fn=generate)
            await asyncio.wait_for(agent.sample(env, hooks=[TrajectoryLogger(directory, save_data=True, save_video=False)]),
                                   180 + sum(delays))
            record['summary'] = json.loads((directory / 'summary.json').read_text())
            record['task_score'] = record['summary']['episode_return']
            record['execution_valid'] = True
        except Exception as error:
            record.update(execution_valid=False, error_type=type(error).__name__, error=str(error))
        finally:
            await env.close()
        records.append(record)
        write_json(directory / 'receipt.json', record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return {'schema_version': 1, 'status': 'CPU_FOCUS_COUNTERFACTUAL_COMPLETED',
            'gpu_used': False, 'model_run': False, 'source_episode': str(episode),
            'source_sha256': hashlib.sha256((episode / 'receipt.json').read_bytes()).hexdigest(),
            'records': records, 'output': str(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--match-timing', action='store_true')
    parser.add_argument('--original-only', action='store_true')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    if not ROOT.is_dir():
        raise SystemExit('QZ_CPU_REQUIRED')
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {'script_sha256_at_start': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'replay_sha256_at_start': hashlib.sha256(
                    (PROJECT / 'cua_defense/replay.py').read_bytes()).hexdigest()}
    write_json(args.output / 'launch.json', identity)
    record = asyncio.run(check(args.episode, args.output, args.repetitions, args.match_timing, args.original_only))
    write_json(args.output / 'receipt.json', {**record, **identity})


if __name__ == '__main__':
    main()
