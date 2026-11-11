"""HF SFT consumer of CUA-Lite's tokenized multimodal step representation."""
import io
import hashlib
import shutil
import json
from pathlib import Path


def step_inputs(step, images, processor, *, diagnostic_resize=None):
    import torch
    from PIL import Image
    visible = [Image.open(io.BytesIO(images[i])).convert('RGB') for i in step['image_indices']]
    if diagnostic_resize:
        visible = [image.resize((diagnostic_resize, diagnostic_resize)) for image in visible]
    batch = processor(text=[step['prompt']], images=visible or None, return_tensors='pt', padding=False)
    response = torch.tensor([step['response_tokens']], dtype=torch.long)
    if response.numel() == 0:
        raise ValueError('SFT target has no response tokens')
    prompt_length = batch['input_ids'].shape[-1]
    batch['input_ids'] = torch.cat((batch['input_ids'], response), dim=-1)
    batch['attention_mask'] = torch.cat((batch['attention_mask'], torch.ones_like(response)), dim=-1)
    for key in ['token_type_ids', 'mm_token_type_ids']:
        if key in batch:
            batch[key] = torch.cat((batch[key], torch.zeros_like(response)), dim=-1)
    batch['labels'] = torch.cat((torch.full((1, prompt_length), -100, dtype=torch.long), response), dim=-1)
    if batch['input_ids'].shape[-1] > 32768:
        raise ValueError('SFT sequence exceeds the prepared 32768-token limit')
    if not torch.equal(batch['labels'][batch['labels'] != -100], response.flatten()):
        raise ValueError('Prompt/target masking differs from the exported response')
    return dict(batch)


class StepDataset:
    def __init__(self, path, processor, *, diagnostic=False):
        import pyarrow.parquet as pq
        self.rows = pq.read_table(path).to_pylist()
        self.processor = processor
        self.indices = []
        for row_index, row in enumerate(self.rows):
            meta = json.loads(row['metadata'])
            if not diagnostic and meta.get('others', {}).get('research_training_eligible') is not True:
                raise ValueError('Training input lacks validated positive eligibility')
            for step_index, step in enumerate(row['steps']):
                if step['status'] == 'completed':
                    self.indices.append((row_index, step_index))
        if not self.indices:
            raise ValueError('No complete supervised steps')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row_index, step_index = self.indices[index]
        row = self.rows[row_index]
        return step_inputs(row['steps'][step_index], row['processed_images'], self.processor)


def single_example_collator(rows):
    if len(rows) != 1:
        raise ValueError('V1 uses microbatch size 1 and gradient accumulation')
    return rows[0]


def cpu_smoke(data, processor_path, output):
    import tempfile
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    dataset = StepDataset(data, processor, diagnostic=True)
    row_index, step_index = dataset.indices[0]
    row = dataset.rows[row_index]
    inputs = step_inputs(row['steps'][step_index], row['processed_images'], processor, diagnostic_resize=64)
    config = AutoConfig.from_pretrained(processor_path, local_files_only=True)
    text = config.text_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 32, 64, 1
    text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 2, 8
    text.rope_parameters = {'rope_type': 'default', 'rope_theta': 1000000.0, 'mrope_section': [1, 1, 2], 'partial_rotary_factor': 1.0}
    if config.model_type == 'qwen3_5':
        text.num_hidden_layers = 2
        text.layer_types = ['linear_attention', 'full_attention']
        text.linear_num_key_heads = text.linear_num_value_heads = 2
        text.linear_key_head_dim = text.linear_value_head_dim = 8
        text.partial_rotary_factor = 1.0
    vision = config.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size = 1, 32, 64
    vision.num_heads, vision.out_hidden_size = 4, 32
    vision.deepstack_visual_indexes = [0]
    torch.manual_seed(17)
    base = AutoModelForImageTextToText.from_config(config, attn_implementation='sdpa')
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
        target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    loss = model(**inputs).loss
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable):
        raise RuntimeError('Tiny multimodal SFT produced non-finite values')
    optimizer.step()
    changed = sum(not torch.equal(old, new) for old, new in zip(before, trainable))
    if not changed:
        raise RuntimeError('Tiny LoRA update did not change parameters')
    model.eval()
    with torch.no_grad():
        saved_loss = model(**inputs).loss
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        model.save_pretrained(temporary)
        model.load_adapter(temporary, adapter_name='reload_check')
        model.set_adapter('reload_check')
        with torch.no_grad():
            inplace_loss = model(**inputs).loss
        if not torch.allclose(saved_loss, inplace_loss, atol=1e-5, rtol=1e-5):
            raise RuntimeError('In-process reloaded adapter differs')
        torch.manual_seed(17)
        restored_base = AutoModelForImageTextToText.from_config(config, attn_implementation='sdpa')
        restored = PeftModel.from_pretrained(restored_base, temporary).eval()
        with torch.no_grad():
            restored_loss = restored(**inputs).loss
        if not torch.allclose(saved_loss, restored_loss, atol=1e-5, rtol=1e-5):
            raise RuntimeError('Reloaded adapter differs')
    report = {'schema_version': 1, 'status': 'CPU_TINY_MULTIMODAL_SFT_PASSED',
              'device': 'cpu', 'model_type': config.model_type, 'diagnostic_only': True, 'real_model_weights_loaded': False,
              'optimizer_steps': 1, 'changed_trainable_tensors': changed,
              'loss_before_update': float(loss.detach()), 'loss_after_reload': float(restored_loss),
              'supervised_tokens': int((inputs['labels'] != -100).sum()),
              'prompt_tokens_masked': int((inputs['labels'] == -100).sum()),
              'vision_pixels_present': 'pixel_values' in inputs,
              'adapter_reload_verified': True, 'training_weights_retained': False}
    (Path(output) / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def train(data, model_path, output, *, steps, learning_rate, accumulation, seed, eval_root=None, milestones=(), budget_minutes=60):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training requires a CUDA device; use --cpu-smoke for the CPU contract check')
    set_seed(seed)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    dataset = StepDataset(data, processor)
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa')
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        task_type='CAUSAL_LM'))
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if '.visual.' in name):
        raise RuntimeError('Vision encoder unexpectedly trainable')
    arguments = TrainingArguments(output_dir=str(output), max_steps=steps, learning_rate=learning_rate,
        per_device_train_batch_size=1, gradient_accumulation_steps=accumulation, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy='no', logging_steps=1, report_to='none', remove_unused_columns=False,
        seed=seed, data_seed=seed, dataloader_num_workers=0)
    model.config.use_cache = False
    class ExposureCounter:
        tokens = 0
        examples = 0
    exposure = ExposureCounter()
    class ExposureTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            exposure.tokens += int((inputs['labels'] != -100).sum())
            exposure.examples += 1
            return loss
    trainer = ExposureTrainer(model=model, args=arguments, train_dataset=dataset,
                              data_collator=single_example_collator)
    evaluator = None
    if eval_root is not None:
        from cua_defense.milestones import MilestoneEvaluation
        evaluator = MilestoneEvaluation(eval_root, processor, milestones, budget_minutes)
        trainer.add_callback(evaluator)
    result = trainer.train()
    trainer.state.save_to_json(str(Path(output) / 'trainer_state.json'))
    adapter_path = Path(output) / 'adapter'
    trainer.save_model(str(adapter_path))
    model.eval()
    probe = {key: value.to(model.device) for key, value in dataset[0].items()}
    with torch.no_grad():
        original_loss = model(**probe).loss
    model.load_adapter(str(adapter_path), adapter_name='reload_check')
    model.set_adapter('reload_check')
    with torch.no_grad():
        restored_loss = model(**probe).loss
    reload_verified = bool(torch.allclose(original_loss, restored_loss, atol=1e-3, rtol=1e-3))
    identities = {str(path.relative_to(adapter_path)): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                  for path in adapter_path.rglob('*') if path.is_file()}
    retirement = {'adapter_reload_verified': reload_verified, 'files_sha256': identities,
                  'evaluation_milestones_completed': sorted(evaluator.completed) if evaluator else []}
    (Path(output) / 'adapter_receipt.json').write_text(json.dumps(retirement, indent=2) + '\n')
    if not reload_verified:
        raise RuntimeError('Final adapter reload changed probe loss; preserved for diagnosis')
    shutil.rmtree(adapter_path)
    if trainer.is_world_process_zero():
        (Path(output) / 'training_result.json').write_text(json.dumps({
            'data_sha256': hashlib.file_digest(Path(data).open('rb'), 'sha256').hexdigest(),
            'status': 'COMPLETED' if trainer.state.global_step == steps and (evaluator is None or set(milestones) <= evaluator.completed) else 'PARTIAL',
            'metrics': result.metrics, 'trained_target_tokens': exposure.tokens, 'trained_microbatches': exposure.examples,
            'completed_eval_milestones': sorted(evaluator.completed) if evaluator else [], 'supervised_steps': len(dataset), 'optimizer_steps': trainer.state.global_step, 'requested_steps': steps,
            'model_path': str(model_path), 'data_path': str(data),
            'checkpoint_policy': 'No periodic weights. Final adapter was reloaded, hashed, and retired after in-process milestone evaluations.'}, indent=2) + '\n')
