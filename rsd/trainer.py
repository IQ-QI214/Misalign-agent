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
TRAINING_GENERATION_POLICY = {
    "version": "tokenizer_eos_v1",
    "eos_token_id_source": "tokenizer.eos_token_id",
    "completion_mask": "through_first_eos_inclusive",
}
ACCUMULATION_POLICY = {
    "version": "equal_completion_mean_v2",
    "microbatch_size": 1,
    "loss": "Mean KL over valid tokens of each completion; equal mean of microsteps and distributed ranks.",
    "scaling": "Transformers Trainer divides once by current GAS; Accelerate GAS is 1; DeepSpeed scale_wrt_gas is false.",
    "partial_window": "reject before generation/backward",
    "teacher_version": "Initial weights for frozen; current student before optimizer update for recursive.",
}


def synchronize_device(model):
    """Phase wall times include device completion, not just queued CUDA work."""
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TrainingTelemetryCallback(TrainerCallback):
    """One receipt per optimizer update, with observed per-rank and global dose.

    This callback runs on every rank. Collectives merge only small counters and
    source IDs; no predictions or tensors of model weights are gathered.
    """
    def __init__(self, trainer):
        self.trainer = trainer
        self.started = None
        self.global_sources = set()
        self.global_exposures = 0
        self.global_tokens = 0

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        trainer = self.trainer
        if len(trainer._window_traces) != args.gradient_accumulation_steps:
            raise RuntimeError("Optimizer update lacks a complete accumulation window")
        synchronize_device(trainer.model)
        self.started = time.monotonic()

    def on_optimizer_step(self, args, state, control, **kwargs):
        trainer = self.trainer
        synchronize_device(trainer.model)
        elapsed = time.monotonic() - self.started
        if trainer.accelerator.optimizer_step_was_skipped:
            raise RuntimeError("Optimizer step was skipped; refusing to advance teacher/dose identity")
        rows = trainer._window_traces
        local = {"rank": trainer.accelerator.process_index,
                 "sample_exposures": sum(len(r["ids"]) for r in rows),
                 "valid_tokens": sum(sum(r["valid_tokens"]) for r in rows),
                 "source_ids": [source for r in rows for source in r["source_ids"]]}
        gathered = [local]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        if len(gathered) != trainer.accelerator.num_processes:
            raise RuntimeError("Observed telemetry world size differs from Accelerator")
        if {r["rank"] for r in gathered} != set(range(len(gathered))):
            raise RuntimeError("Missing or duplicate rank telemetry")
        expected_per_rank = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if any(r["sample_exposures"] != expected_per_rank for r in gathered):
            raise RuntimeError("Observed sample exposure differs from declared full batch")
        exposures = sum(r["sample_exposures"] for r in gathered)
        tokens = sum(r["valid_tokens"] for r in gathered)
        self.global_exposures += exposures
        self.global_tokens += tokens
        self.global_sources.update(source for r in gathered for source in r["source_ids"])
        receipt = {
            "schema_version": "rsd-optimizer-telemetry-v2", "attempt_id": trainer.attempt_id,
            "optimizer_step_completed": state.global_step + 1,
            "rank": trainer.accelerator.process_index, "world_size": len(gathered),
            "microsteps_per_rank": len(rows), "observed_global_sample_exposures": exposures,
            "observed_global_valid_tokens": tokens,
            "attempt_global_sample_exposures": self.global_exposures,
            "attempt_global_valid_tokens": self.global_tokens,
            "attempt_global_unique_sources": len(self.global_sources),
            "teacher_step_used": rows[0]["teacher_step"], "teacher_mode": trainer.teacher_mode,
            "rank_mean_unscaled_loss": sum(r["loss_before_accumulation_scale"] for r in rows) / len(rows),
            "rank_generation_seconds": sum(r["generation_seconds"] for r in rows),
            "rank_forward_backward_seconds": sum(r["forward_backward_seconds"] for r in rows),
            "rank_teacher_forward_seconds": sum(r["teacher_forward_seconds"] for r in rows),
            "rank_student_forward_seconds": sum(r["student_forward_seconds"] for r in rows),
            "rank_loss_construction_seconds": sum(r["loss_construction_seconds"] for r in rows),
            "rank_optimizer_seconds": elapsed,
            "timing_scope": "Per-rank wall clock with CUDA synchronization; optimizer excludes clipping, scheduler and telemetry gathering. Forward/backward includes framework overhead and checkpoint recomputation.",
            "cuda_max_memory_allocated": torch.cuda.max_memory_allocated(trainer.accelerator.device) if trainer.accelerator.device.type == "cuda" else None,
            "cuda_max_memory_reserved": torch.cuda.max_memory_reserved(trainer.accelerator.device) if trainer.accelerator.device.type == "cuda" else None,
            "loss_policy": ACCUMULATION_POLICY,
            "attempt_scope": "Counters reset on each process attempt; resumed run totals require joining attempts.",
        }
        path = Path(args.output_dir) / f"optimizer_rank{trainer.accelerator.process_index}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n")
        trainer._window_traces = []


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
        # OPSD copies the model's original EOS before Trainer.train() aligns
        # model configs. Pin this independent sampling config to the tokenizer;
        # both generation and the completion loss mask use this exact value.
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise ValueError("Training requires tokenizer.eos_token_id; model-default fallback is disabled")
        self.generation_config.eos_token_id = eos_token_id
        self.training_generation_policy = {
            **TRAINING_GENERATION_POLICY, "eos_token_id": eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id,
        }
        self.frozen_teacher = frozen_teacher
        if frozen_teacher is not None:
            frozen_teacher.requires_grad_(False)
            frozen_teacher.eval().to(self.accelerator.device)
        self.model_accepts_loss_kwargs = False
        # Our loss is a per-completion mean, not an already GAS-normalized sum.
        # Keep the upstream Trainer's one division, even if a library default
        # or another loss configuration changes later.
        self.loss_is_scaled_for_ga = False
        if self.args.per_device_train_batch_size != 1:
            raise ValueError("TransferTrainer requires per-device microbatch=1")
        if type(self.args.gradient_accumulation_steps) is not int or self.args.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError("Accelerator GAS must be 1: Trainer owns accumulation loss scaling")
        self._window_traces = []
        self._attempt_microsteps = 0
        self._last_phase_metrics = None
        self.add_callback(TrainingTelemetryCallback(self))
        self._trace_path = Path(self.args.output_dir) / f"trajectory_rank{self.accelerator.process_index}.jsonl"
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)

    def _teacher(self, student):
        return self.frozen_teacher if self.frozen_teacher is not None else self.accelerator.unwrap_model(student)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        rows, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
        if len(rows) != self.args.gradient_accumulation_steps:
            raise ValueError("Incomplete accumulation window; dataset must divide into full effective batches")
        return rows, num_items

    def training_step(self, model, inputs, num_items_in_batch=None):
        gas = getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
        if gas != self.args.gradient_accumulation_steps or self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError("Partial or multiply-scaled accumulation window is not supported")
        if len(self._window_traces) >= gas:
            raise RuntimeError("New microstep arrived before previous optimizer-window receipt")
        inputs = self._prepare_inputs(inputs)
        if len(inputs["record_ids"]) != 1 or len(inputs["source_ids"]) != 1:
            raise ValueError("TransferTrainer requires exactly one completion per microstep")
        student = self.accelerator.unwrap_model(model)
        source = student if self.trajectory_mode == "on_policy" else self._teacher(model)
        role = "student" if self.trajectory_mode == "on_policy" else "teacher"
        synchronize_device(source)
        started = time.monotonic()
        with inference_mode(source):
            output = source.generate(
                input_ids=inputs[f"{role}_prompts"],
                attention_mask=inputs[f"{role}_prompt_attention_mask"],
                generation_config=self.generation_config, use_cache=True,
            )
        synchronize_device(source)
        generation_seconds = time.monotonic() - started
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
            "generation_policy": self.training_generation_policy,
            "completion_ids": answer.detach().cpu().tolist(),
            "completion": self.processing_class.batch_decode(answer, skip_special_tokens=True),
            "valid_tokens": mask.sum(-1).tolist(),
            "generation_seconds": generation_seconds,
            "microstep_in_optimizer_step": len(self._window_traces) + 1,
            "microsteps_per_optimizer_step": gas,
            "attempt_microstep": self._attempt_microsteps + 1,
            "synchronize_gradients": bool(self.accelerator.sync_gradients),
            "loss_policy": ACCUMULATION_POLICY,
        }
        self._last_phase_metrics = None
        synchronize_device(student)
        started = time.monotonic()
        loss = Trainer.training_step(self, model, inputs, num_items_in_batch)
        synchronize_device(student)
        trace["forward_backward_seconds"] = time.monotonic() - started
        trace["loss"] = float(loss.detach())
        if not torch.isfinite(loss.detach()).all().item():
            raise FloatingPointError("Nonfinite training loss; optimizer update is not allowed")
        trace.update(self._last_phase_metrics or {
            "loss_before_accumulation_scale": float(loss.detach()) * gas,
            "teacher_forward_seconds": None, "student_forward_seconds": None,
            "loss_construction_seconds": None,
        })
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
        self._window_traces.append(trace)
        self._attempt_microsteps += 1
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Teacher forward happens first, with no gradient and no KV cache. Both
        # distributions are scored along identical, unmodified answer token IDs.
        teacher = self._teacher(model)
        synchronize_device(teacher)
        started = time.monotonic()
        with inference_mode(teacher):
            t = teacher(input_ids=inputs["teacher_input_ids"],
                        attention_mask=inputs["teacher_attention_mask"], use_cache=False).logits
            t = t[:, inputs["teacher_prompt_length"] - 1:-1]
        synchronize_device(teacher)
        teacher_seconds = time.monotonic() - started
        started = time.monotonic()
        s = model(input_ids=inputs["student_input_ids"],
                  attention_mask=inputs["student_attention_mask"], use_cache=False).logits
        s = s[:, inputs["student_prompt_length"] - 1:-1]
        synchronize_device(model)
        student_seconds = time.monotonic() - started
        started = time.monotonic()
        loss = masked_distillation_loss(s, t, inputs["completion_mask"],
                                       self.kl_direction,
                                       self.loss_chunk_size)
        synchronize_device(model)
        if not torch.isfinite(loss.detach()).all().item():
            raise FloatingPointError("Nonfinite distillation loss; backward is not allowed")
        self._last_phase_metrics = {
            "loss_before_accumulation_scale": float(loss.detach()),
            "teacher_forward_seconds": teacher_seconds,
            "student_forward_seconds": student_seconds,
            "loss_construction_seconds": time.monotonic() - started,
        }
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
