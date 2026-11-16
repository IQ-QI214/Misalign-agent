import copy
import os
import unittest
from cua_defense.policy_trace import validate_trace


class TraceValidationTests(unittest.TestCase):
    def setUp(self):
        self.trace={'schema_version':1,'sampling_kind':'untruncated_temperature_one',
                    'prompt_token_ids':[1,2],'generated_token_ids':[3,4],
                    'old_policy_logprobs':[-.2,-1.3],'behavior_logprobs':[-.2,-1.3],
                    'training_eligible':False}

    def test_aligned_trace_still_not_training_data(self):
        self.assertFalse(validate_trace(self.trace)['training_eligible'])

    def test_missing_probability_rejected(self):
        self.trace['old_policy_logprobs'].pop()
        with self.assertRaises(ValueError): validate_trace(self.trace)

    def test_changed_distribution_rejected(self):
        self.trace['behavior_logprobs'][0] -= .1
        with self.assertRaises(ValueError): validate_trace(self.trace)

    def test_nonfinite_rejected(self):
        for value in [float('nan'),float('inf'),float('-inf')]:
            trace=copy.deepcopy(self.trace);trace['old_policy_logprobs'][0]=value
            with self.assertRaises(ValueError): validate_trace(trace)


class NativeGenerationTests(unittest.TestCase):
    def test_tiny_cpu_model_matches_independent_forward(self):
        try:
            import torch
            from transformers import GPT2Config, GPT2LMHeadModel, BatchEncoding
        except ImportError:
            if os.environ.get('CUA_REQUIRE_NATIVE') == '1':
                raise
            self.skipTest('Run native generation acceptance in the existing QZ CPU runtime')
        from cua_defense.inference import answer_request
        torch.manual_seed(78)
        model=GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=32,n_embd=16,n_layer=1,
            n_head=2,eos_token_id=31,bos_token_id=1,pad_token_id=31)).eval()
        model.generation_config.top_k=2
        model.generation_config.min_p=.8
        model.generation_config.temperature=.3
        class Processor:
            class tokenizer:
                eos_token_id=31
            def __call__(self,**kwargs):
                return BatchEncoding({'input_ids':torch.tensor([[1,5,9]]),
                                      'attention_mask':torch.ones((1,3),dtype=torch.long)})
            def decode(self,ids,**kwargs): return ' '.join(map(str,ids.tolist()))
        result=answer_request(model,Processor(),{'prompt':'tiny CPU acceptance','images':[],
            'seed':102,'capture_policy':True,'sampling_params':{'temperature':1.0,'top_k':0,'top_p':1.0,
                                                               'max_new_tokens':4}})
        trace=result['policy_trace']
        tokens=torch.tensor([trace['prompt_token_ids']+trace['generated_token_ids']])
        with torch.no_grad(): logits=model(tokens).logits.float()
        expected=[float(logits[0,2+i].log_softmax(-1)[t]) for i,t in enumerate(trace['generated_token_ids'])]
        for a,b in zip(expected,trace['old_policy_logprobs']): self.assertAlmostEqual(a,b,places=5)
        self.assertEqual(len(trace['generated_token_ids']),result['output_tokens'])
        self.assertEqual(trace['prompt_token_ids'],[1,5,9])
        self.assertEqual(result['generation_config']['top_k'],0)
        self.assertEqual(result['generation_config']['min_p'],0.0)
        self.assertEqual(result['generation_config']['temperature'],1.0)


if __name__=='__main__': unittest.main()
