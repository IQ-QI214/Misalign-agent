#!/usr/bin/env python3
"""Replay one failed Writer trajectory and test an explicit document-focus click."""
import argparse
import asyncio
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from scripts.qz_cua_batch import PROJECT, ROOT, write_json


def fonts(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        document = ET.fromstring(archive.read('word/document.xml'))
    namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    return dict(Counter(node.get('{' + namespace['w'] + '}ascii', 'UNSPECIFIED')
                        for node in document.findall('.//w:rFonts', namespace)))


async def check(episode, output):
    import lite.gym as gym
    from lite.agents.factory import make
    from lite.agents.core.agent.logger import TrajectoryLogger
    from transformers import AutoProcessor
    from cua_defense.tasks import load_rows, register_row
    from cua_defense.replay import checkpoint, verify_checkpoint, file_fingerprint
    source = json.loads((episode / 'receipt.json').read_text())
    task_id = 'synth_writer_f_writer_93__doc_font_0001'
    if source['task_id'] != task_id:
        raise ValueError('This diagnostic is specific to the Origami font task')
    rows = load_rows(PROJECT / 'references/cua_lite', PROJECT / 'configs/cua/development_tasks_v1.json')
    row = next(row for row in rows if row['task_id'] == task_id)
    model = json.loads((PROJECT / 'configs/cua/models.json').read_text())['qwen35_9b']
    processor = AutoProcessor.from_pretrained(model['processor_path'], local_files_only=True)
    responses = [json.loads(path.read_text())['response'] for path in
                 sorted(episode.glob('response-*.json'), key=lambda path: int(path.stem.split('-')[-1]))]
    focus = ('<tool_call>\n<function=computer_use>\n<parameter=action>left_click</parameter>\n'
             '<parameter=coordinate>[500, 400]</parameter>\n</function>\n</tool_call>')
    records = []
    for name, sequence in [('original_replay', responses), ('document_focus_then_replay', [focus, *responses])]:
        directory = output / name
        directory.mkdir()
        env = gym.make(register_row(row), max_steps=len(sequence) + 1, seed=20261112,
                       loop_detect=0, extra_tools=['response', 'terminate'])
        turn = 0
        record = {'condition': name, 'model_run': False, 'gpu_used': False, 'training_eligible': False}
        async def generate(**data):
            nonlocal turn
            if turn == 0:
                point = await checkpoint(env, row, data['images'][-1], directory, turn)
                record['initial_state_check'] = verify_checkpoint(episode, point, directory, turn)
            if turn == len(sequence) - 1:
                target = row['metadata']['evaluator']['result']['path']
                result = await env._computer.interface.read_bytes(target)
                (directory / 'before_termination.docx').write_bytes(result)
                record['document_fonts'] = fonts(result)
                record['document_fingerprint'] = file_fingerprint(result)
                original = json.loads((episode / f'checkpoint-{len(responses)-1}.json').read_text())
                record['original_final_fingerprint'] = original['files'][target]
                data['images'][-1].save(directory / 'before_termination.png')
            text = sequence[turn]
            turn += 1
            return {'response': text, 'finish_reason': 'stop'}
        try:
            agent = make(model['model_id'], env=env, processor=processor, generate_fn=generate)
            await asyncio.wait_for(agent.sample(env, hooks=[TrajectoryLogger(directory, save_data=True, save_video=False)]), 180)
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
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'records': records, 'output': str(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not ROOT.is_dir():
        raise SystemExit('QZ_CPU_REQUIRED')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'receipt.json', asyncio.run(check(args.episode, args.output)))


if __name__ == '__main__':
    main()
