#!/usr/bin/env python3
"""Single-client HF inference worker over a QZ shared-directory mailbox."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--mailbox', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('QZ_LINUX_REQUIRED')
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    def write(name, value):
        target = args.mailbox / name
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False) + '\n')
        temporary.replace(target)

    try:
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa').to('cuda:0').eval()
        write('ready.json', {'torch': torch.__version__, 'transformers': transformers.__version__,
                             'device': torch.cuda.get_device_name(0), 'model': str(args.model),
                             'started_utc': datetime.now(timezone.utc).isoformat()})
        for step in range(20):
            request = args.mailbox / f'request-{step}.json'
            deadline = time.monotonic() + 600
            while not request.exists():
                if (args.mailbox / 'stop').exists():
                    return
                if time.monotonic() > deadline:
                    raise TimeoutError('Mailbox request timed out')
                time.sleep(0.25)
            data = json.loads(request.read_text())
            images = [Image.open(path).convert('RGB') for path in data['images']]
            inputs = processor(text=[data['prompt']], images=images or None,
                               return_tensors='pt', padding=True).to(model.device)
            length = inputs['input_ids'].shape[-1]
            if length > 32768:
                raise ValueError(f'Input exceeds pilot token budget: {length}')
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                           pad_token_id=processor.tokenizer.eos_token_id)
            ids = generated[0, length:]
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            write(f'response-{step}.json', {
                'response': processor.decode(ids, skip_special_tokens=True),
                'finish_reason': 'stop' if int(ids[-1]) in eos else 'length',
                'input_tokens': length, 'output_tokens': len(ids)})
    except Exception:
        write('error.json', {'error': traceback.format_exc()})
        raise
    finally:
        write('stopped.json', {'time_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
