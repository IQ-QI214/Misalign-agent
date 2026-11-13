#!/usr/bin/env python3
"""Read-only audit of environment frames, model input images and preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.qz_cua_batch import ROOT, write_json


def pixel_signature(image):
    rgb = image.convert('RGB')
    return {'size': list(rgb.size), 'sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--processor-check', action='store_true')
    args = parser.parse_args()
    if not ROOT.is_dir():
        raise SystemExit('QZ_CPU_REQUIRED')
    from PIL import Image
    from lite.agents.models.qwen3_5.adapter import Qwen3_5DesktopUseAdapter
    from lite.core.metadata import LiteCUAMetadata
    adapter = Qwen3_5DesktopUseAdapter(metadata=LiteCUAMetadata(dims=('desktop', 'use')))
    assert pixel_signature(Image.new('RGB', (2, 2), 'black')) != pixel_signature(Image.new('RGB', (2, 2), 'white'))
    receipt = json.loads((args.run / 'receipt.json').read_text())
    archive = next(args.run.glob('model_io-*'))
    records = []
    first_request = first_image = first_response = None
    for row in receipt['results']:
        directory = Path(row['output'])
        checks = []
        for turn in range(row['summary']['n_turns']):
            response = json.loads((directory / f'response-{turn}.json').read_text())
            request = json.loads((archive / f"request-{response['request_id']}.json").read_text())
            model_path = archive / Path(request['images'][-1]).name
            prompt_path = sorted((directory / f'turn_{turn:04d}/prompt_images').glob('*.png'))[-1]
            with Image.open(model_path) as model_image, Image.open(prompt_path) as prompt_image:
                signature = pixel_signature(model_image)
                item = {'turn': turn, 'request_id': response['request_id'],
                        'raw_observation_size': list(prompt_image.size), 'model_image_size': list(model_image.size),
                        'model_matches_logged_observation': signature == pixel_signature(adapter.process_image(prompt_image))}
            if turn:
                previous = sorted((directory / f'turn_{turn-1:04d}/env_result_images').glob('*.png'))[-1]
                with Image.open(previous) as previous_image:
                    item['model_matches_previous_action_result'] = signature == pixel_signature(adapter.process_image(previous_image))
            checks.append(item)
            if first_request is None:
                first_request, first_image, first_response = request, model_path, response
        records.append({'task_id': row['task_id'], 'protocol': row['protocol']['name'], 'checks': checks})
    report = {'schema_version': 1, 'gpu_used': False, 'model_run': False,
              'run_receipt_sha256': hashlib.sha256((args.run / 'receipt.json').read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Exact RGB pixels after the native Qwen3.5 adapter image transformation, comparing the latest observation, model input and previous action result',
              'records': records}
    if args.processor_check:
        import transformers
        from transformers import AutoProcessor
        model = json.loads((args.run / 'config.json').read_text())['model']
        processor = AutoProcessor.from_pretrained(model['processor_path'], local_files_only=True)
        with Image.open(first_image) as source_image:
            features = processor(text=[first_request['prompt']], images=[source_image.convert('RGB')],
                                 return_tensors='pt', padding=True)
        grid = features['image_grid_thw'].tolist()[0]
        patch = processor.image_processor.patch_size
        report['processor'] = {'transformers': transformers.__version__,
            'image_grid_thw': grid, 'patch_size': patch,
            'merge_size': processor.image_processor.merge_size,
            'effective_image_size': [grid[2] * patch, grid[1] * patch],
            'recomputed_input_tokens': features['input_ids'].shape[-1],
            'recorded_gpu_input_tokens': first_response['input_tokens'],
            'input_token_count_matches': features['input_ids'].shape[-1] == first_response['input_tokens']}
    checks = [item for row in records for item in row['checks']]
    report['summary'] = {'observations': len(checks),
        'logged_observation_mismatches': sum(not item['model_matches_logged_observation'] for item in checks),
        'action_result_transitions': sum('model_matches_previous_action_result' in item for item in checks),
        'action_result_mismatches': sum(not item.get('model_matches_previous_action_result', True) for item in checks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({'summary': report['summary'], 'processor': report.get('processor')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
