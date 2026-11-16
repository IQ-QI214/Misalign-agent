"""Validate token-aligned behavior probabilities before any future RL admission."""
import math


def validate_trace(trace):
    if trace.get('schema_version') != 1 or trace.get('sampling_kind') != 'untruncated_temperature_one':
        raise ValueError('Unsupported behavior probability contract')
    ids = trace['generated_token_ids']
    old, behavior = trace['old_policy_logprobs'], trace['behavior_logprobs']
    if not ids or len(ids) != len(old) or len(ids) != len(behavior):
        raise ValueError('Token/probability lengths disagree or generation is empty')
    if any(type(token) is not int or token < 0 for token in ids + trace['prompt_token_ids']):
        raise ValueError('Invalid token ID')
    if any(not math.isfinite(v) or v > 1e-5 for v in old + behavior):
        raise ValueError('Nonfinite or positive log probability')
    if any(abs(a-b) > 1e-5 for a,b in zip(old,behavior)):
        raise ValueError('Generation processors changed the sampled behavior distribution')
    if trace.get('training_eligible') is not False:
        raise ValueError('Measurement traces do not grant training eligibility')
    return {'token_count':len(ids), 'probability_alignment_valid':True,
            'training_eligible':False}


def extract_trace(inputs, generated):
    import torch
    length = inputs['input_ids'].shape[-1]
    if generated.sequences.shape[0] != 1 or not torch.equal(
            generated.sequences[0,:length], inputs['input_ids'][0]):
        raise ValueError('Only one decoder-only sequence with an unchanged prefix is supported')
    ids=generated.sequences[0,length:]
    if len(ids)!=len(generated.logits) or len(ids)!=len(generated.scores):
        raise ValueError('Generation score history is incomplete')
    def selected_logprobs(scores):
        return [float(torch.log_softmax(logits[0].float(),dim=-1)[token].item())
                for logits,token in zip(scores,ids)]
    trace={'schema_version':1,'sampling_kind':'untruncated_temperature_one',
           'prompt_token_ids':inputs['input_ids'][0].tolist(),'generated_token_ids':ids.tolist(),
           'old_policy_logprobs':selected_logprobs(generated.logits),
           'behavior_logprobs':selected_logprobs(generated.scores),
           'includes_eos_if_generated':True,'training_eligible':False,
           'probability_definition':'Per generated token conditioned on the exact prompt and prior generated tokens; normalized raw logits must equal normalized sampling scores.'}
    if 'image_grid_thw' in inputs:
        trace['image_grid_thw']=inputs['image_grid_thw'].tolist()
    validate_trace(trace)
    return trace
