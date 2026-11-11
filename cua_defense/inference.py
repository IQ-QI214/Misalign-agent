"""Model generation shared by standalone workers and in-training evaluations."""
import json
from pathlib import Path
import time


def answer_request(model, processor, data):
    import torch
    from PIL import Image
    torch.manual_seed(data.get('seed', 0))
    images = [Image.open(path).convert('RGB') for path in data['images']]
    inputs = processor(text=[data['prompt']], images=images or None, return_tensors='pt', padding=True).to(model.device)
    length = inputs['input_ids'].shape[-1]
    if length > 32768:
        raise ValueError(f'Input exceeds pilot token budget: {length}')
    sampling = data.get('sampling_params', {})
    temperature = float(sampling.get('temperature', 0.0))
    generation = {'max_new_tokens': int(sampling.get('max_new_tokens', 1024)),
                  'do_sample': temperature > 0, 'pad_token_id': processor.tokenizer.eos_token_id}
    if temperature > 0:
        generation['temperature'] = temperature
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation)
    ids = generated[0, length:]
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    return {'response': processor.decode(ids, skip_special_tokens=True),
            'finish_reason': 'stop' if len(ids) and int(ids[-1]) in eos else 'length',
            'input_tokens': length, 'output_tokens': len(ids)}


def serve_requests(model, processor, mailbox, *, max_requests=10000, idle_timeout=300, deadline=None):
    mailbox = Path(mailbox)
    for number in range(max_requests):
        request = mailbox / f'request-{number}.json'
        idle_deadline = time.monotonic() + idle_timeout
        while not request.exists():
            if (mailbox / 'stop').exists():
                return
            if time.monotonic() > idle_deadline or (deadline is not None and time.monotonic() > deadline):
                raise TimeoutError('Mailbox request timed out')
            time.sleep(0.25)
        if (mailbox / 'stop').exists() or (deadline is not None and time.monotonic() > deadline):
            return
        result = answer_request(model, processor, json.loads(request.read_text()))
        target = mailbox / f'response-{number}.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False) + '\n')
        temporary.replace(target)
