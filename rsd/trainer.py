"""Thin research adapter over the pinned OPSD/TRL training framework.

OPSD provides initialization, loss formula and Trainer infrastructure. We override
the math-only input path, full-parameter frozen teacher, trajectory source, and
token alignment. Transformers owns optimizer/distribution/checkpoint/RNG resume.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import time
import uuid

import torch
from torch.utils.checkpoint import checkpoint
from transformers import Trainer, TrainerCallback

from .collator import completion_mask
from .upstream import import_trainer

OPSDTrainer = import_trainer()


@contextmanager
def inference_mode(model):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def masked_distillation_loss(student_logits, teacher_logits, mask, direction, chunk_size=32):
    if student_logits.shape != teacher_logits.shape or mask.shape != student_logits.shape[:-1]:
        raise ValueError("Student/teacher token alignment differs")
    if direction not in ("forward", "reverse") or chunk_size < 1:
        raise ValueError("Invalid KL configuration")
    n = mask.sum()
    if n.item() == 0:
        raise ValueError("No valid completion tokens")
    beta = 0 if direction == "forward" else 1
    total = student_logits.new_zeros((), dtype=torch.float32)
    # Recompute float32 vocabulary intermediates during backward to avoid keeping
    # them for every chunk. The upstream KL formula itself is unchanged.
    for start in range(0, student_logits.shape[1], chunk_size):
        s = student_logits[:, start:start + chunk_size]
        t = teacher_logits[:, start:start + chunk_size].detach()
        labels = torch.where(mask[:, start:start + chunk_size], 0, -100)
        def loss_chunk(a, b, labels):
            return OPSDTrainer.generalized_jsd_loss(
                a.float(), b.float(), labels=labels, beta=beta, temperature=1.0,
                reduction="sum", top_k=None, token_clip=None,
            )
        total = total + checkpoint(loss_chunk, s, t, labels, use_reentrant=False)
    return total / n


class TransferTrainer(OPSDTrainer):
    def __init__(self, *args, teacher_mode="recursive", trajectory_mode="on_policy",
                 frozen_teacher=None, loss_chunk_size=32, kl_direction="reverse", **kwargs):
        if teacher_mode not in ("recursive", "frozen"):
            raise ValueError("Unknown teacher mode")
        if trajectory_mode not in ("on_policy", "off_policy"):
            raise ValueError("Unknown trajectory mode")
        if (teacher_mode == "frozen") != (frozen_teacher is not None):
            raise ValueError("Frozen mode requires an independent initial teacher")
        self.teacher_mode = teacher_mode
        self.trajectory_mode = trajectory_mode
        self.loss_chunk_size = loss_chunk_size
        if kl_direction not in ("forward", "reverse"):
            raise ValueError("Unknown KL direction")
        self.kl_direction = kl_direction
        self.attempt_id = uuid.uuid4().hex
        # Deliberately bypass upstream LoRA-only fixed_teacher switch.
        super().__init__(*args, fixed_teacher=False, reason_first=False,
                         jsd_token_clip=None, top_k_loss=None, **kwargs)
        self.frozen_teacher = frozen_teacher
        if frozen_teacher is not None:
            frozen_teacher.requires_grad_(False)
            frozen_teacher.eval().to(self.accelerator.device)
        self.model_accepts_loss_kwargs = False
        self._trace_path = Path(self.args.output_dir) / f"trajectory_rank{self.accelerator.process_index}.jsonl"
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)

    def _teacher(self, student):
        return self.frozen_teacher if self.frozen_teacher is not None else self.accelerator.unwrap_model(student)

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        student = self.accelerator.unwrap_model(model)
        source = student if self.trajectory_mode == "on_policy" else self._teacher(model)
        role = "student" if self.trajectory_mode == "on_policy" else "teacher"
        started = time.monotonic()
        with inference_mode(source):
            output = source.generate(
                input_ids=inputs[f"{role}_prompts"],
                attention_mask=inputs[f"{role}_prompt_attention_mask"],
                generation_config=self.generation_config, use_cache=True,
            )
        answer = output[:, inputs[f"{role}_prompt_length"]:]
        if answer.shape[1] == 0:
            raise ValueError("Generation produced no tokens")
        mask = completion_mask(answer, self.generation_config.eos_token_id)
        inputs["completion_mask"] = mask
        for role in ("student", "teacher"):
            inputs[f"{role}_input_ids"] = torch.cat([inputs[f"{role}_prompts"], answer], dim=1)
            inputs[f"{role}_attention_mask"] = torch.cat(
                [inputs[f"{role}_prompt_attention_mask"], mask.long()], dim=1,
            )
        trace = {
            "step_before_update": self.state.global_step, "rank": self.accelerator.process_index,
            "ids": inputs["record_ids"], "source_ids": inputs["source_ids"],
            "teacher_mode": self.teacher_mode,
            "teacher_step": 0 if self.teacher_mode == "frozen" else self.state.global_step,
            "trajectory_mode": self.trajectory_mode, "feedback": inputs["feedback"],
            "kl_direction": self.kl_direction, "attempt_id": self.attempt_id,
            "completion_ids": answer.detach().cpu().tolist(),
            "completion": self.processing_class.batch_decode(answer, skip_special_tokens=True),
            "valid_tokens": mask.sum(-1).tolist(),
            "generation_seconds": time.monotonic() - started,
        }
        loss = Trainer.training_step(self, model, inputs, num_items_in_batch)
        trace["loss"] = float(loss.detach())
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Teacher forward happens first, with no gradient and no KV cache. Both
        # distributions are scored along identical, unmodified answer token IDs.
        teacher = self._teacher(model)
        with inference_mode(teacher):
            t = teacher(input_ids=inputs["teacher_input_ids"],
                        attention_mask=inputs["teacher_attention_mask"], use_cache=False).logits
            t = t[:, inputs["teacher_prompt_length"] - 1:-1]
        s = model(input_ids=inputs["student_input_ids"],
                  attention_mask=inputs["student_attention_mask"], use_cache=False).logits
        s = s[:, inputs["student_prompt_length"] - 1:-1]
        loss = masked_distillation_loss(s, t, inputs["completion_mask"],
                                       self.kl_direction,
                                       self.loss_chunk_size)
        return (loss, {"loss": loss}) if return_outputs else loss

    def log(self, logs, start_time=None):
        # Upstream's textual/mixed-policy accumulator assumes its own training_step.
        Trainer.log(self, logs, start_time)


class ReceiptCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            source = Path(args.output_dir) / "run_contract.json"
            if source.exists():
                target = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "run_contract.json"
                target.write_bytes(source.read_bytes())

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            receipt = {"global_step": state.global_step, "epoch": state.epoch,
                       "max_memory_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
            p = Path(args.output_dir) / "progress.json"
            temp = p.with_suffix(".tmp")
            temp.write_text(json.dumps(receipt, indent=2) + "\n")
            temp.replace(p)
