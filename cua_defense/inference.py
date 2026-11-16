"""Model generation shared by standalone workers and in-training evaluations."""
import json
import hashlib
from pathlib import Path
import time


def eos_token_ids(model, processor):
    configured = model.generation_config.eos_token_id
    tokens = configured if isinstance(configured, list) else [configured]
    tokens = [token for token in tokens if token is not None]
    if processor.tokenizer.eos_token_id is not None:
        tokens.append(processor.tokenizer.eos_token_id)
    if not tokens:
        raise ValueError('Generation requires a model or tokenizer EOS token')
    return list(dict.fromkeys(tokens))


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
    eos = eos_token_ids(model, processor)
    generation = {'max_new_tokens': int(sampling.get('max_new_tokens', 1024)),
                  'do_sample': temperature > 0, 'pad_token_id': processor.tokenizer.eos_token_id,
                  'eos_token_id': eos}
    if temperature > 0:
        generation['temperature'] = temperature
    capture = data.get('capture_policy', False)
    if capture:
        from transformers import GenerationConfig
        if (temperature != 1.0 or sampling.get('top_p',1.0)!=1.0 or sampling.get('top_k',0)!=0
                or not 1 <= generation['max_new_tokens'] <= 512):
            raise ValueError('Policy capture requires temperature=1, top_p=1, top_k=0 and 1..512 new tokens')
        # Fresh explicit config prevents inherited top-k/min-p/repetition penalties.
        config = GenerationConfig(**generation, top_p=1.0,top_k=0,min_p=0.0,typical_p=1.0,
            bos_token_id=model.generation_config.bos_token_id,
            num_beams=1,num_return_sequences=1,repetition_penalty=1.0,
            no_repeat_ngram_size=0,epsilon_cutoff=0.0,eta_cutoff=0.0,
            return_dict_in_generate=True,output_logits=True,output_scores=True)
        # Resolve and record inherited defaults using the pinned HF implementation.
        # A raw-vs-sampling probability check below rejects hidden distribution changes.
        effective, _ = model._prepare_generation_config(config)
        generation = {'generation_config':effective}
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation)
    ids = (generated.sequences if capture else generated)[0, length:]
    result = {'response': processor.decode(ids, skip_special_tokens=True),
            'finish_reason': 'stop' if len(ids) and int(ids[-1]) in eos else 'length',
            'input_tokens': length, 'output_tokens': len(ids),
            'eos_token_ids': eos, 'last_token_id': int(ids[-1]) if len(ids) else None}
    if capture:
        from cua_defense.policy_trace import extract_trace
        result['policy_trace']=extract_trace(inputs,generated)
        result['generation_config']=generation['generation_config'].to_dict()
        result['prompt_sha256']=hashlib.sha256(data['prompt'].encode()).hexdigest()
        result['request_sha256']=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
        result['image_pixel_sha256']=[hashlib.sha256((im.mode+str(im.size)).encode()+im.tobytes()).hexdigest()
                                      for im in images]
    for image in images:
        image.close()
    return result


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
        try:
            result = answer_request(model, processor, json.loads(request.read_text()))
        except ValueError as error:
            result = {'error': f'{type(error).__name__}: {error}'}
        target = mailbox / f'response-{number}.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False) + '\n')
        temporary.replace(target)
