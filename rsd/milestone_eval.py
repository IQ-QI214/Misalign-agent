"""Dev-only evaluation of resident students; no checkpoint export or judge load.

Rank-zero generation is supported for replicated models (DDP / ZeRO-2), not
FSDP, ZeRO-3 or tensor-parallel weights. Recoverable Python errors are exchanged
between ranks; process death / failed collectives still require the launcher's
distributed timeout. In-memory identities are sampled fingerprints, not full
model hashes and not reloadable checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from transformers import GenerationConfig, TrainerCallback

from .evaluation import digest, judge_inputs, read_jsonl, score_records, text_digest, validate_dataset, write_json


class MilestoneEvaluationError(RuntimeError):
    pass


def build_milestone_contract(task_dataset, benchmark_dataset, steps=(0, 1, 2), *,
                             max_new_tokens=512, max_prompt_tokens=1024, seed=0,
                             probe_samples=32):
    """Canonical settings for the training contract; does not read/load models."""
    if not steps or any(type(s) is not int or s < 0 for s in steps) or list(steps) != sorted(set(steps)):
        raise ValueError("Milestone steps must be nonempty, sorted, unique nonnegative integers")
    if any(type(n) is not int or n <= 0 for n in (max_new_tokens, max_prompt_tokens, probe_samples)):
        raise ValueError("Milestone budgets/probe samples must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Milestone generation seed must be a uint32")
    if not task_dataset or not benchmark_dataset:
        raise ValueError("Both task and benchmark dev cohorts must be nonempty")
    validate_dataset(list(task_dataset) + list(benchmark_dataset))
    if any(r["split"] != "dev" for r in list(task_dataset) + list(benchmark_dataset)):
        raise ValueError("Milestone evaluation is dev-only; test is forbidden")
    if any(r["metric"] != "exact_match" for r in task_dataset):
        raise ValueError("Task cohort must contain exact_match records only")
    if any(r["metric"] not in {"safety", "overrefusal"} for r in benchmark_dataset):
        raise ValueError("Benchmark cohort must contain safety/overrefusal records only")
    return {"version": "resident-student-milestones-v1", "steps": list(steps),
            "task_dataset_sha256": digest(sorted(task_dataset, key=lambda r: r["id"])),
            "benchmark_dataset_sha256": digest(sorted(benchmark_dataset, key=lambda r: r["id"])),
            "task_order_sha256": digest([r["id"] for r in task_dataset]),
            "benchmark_order_sha256": digest([r["id"] for r in benchmark_dataset]),
            "n_task": len(task_dataset), "n_benchmark": len(benchmark_dataset),
            "max_new_tokens": max_new_tokens, "max_prompt_tokens": max_prompt_tokens,
            "seed": seed, "probe_samples": probe_samples,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "judge_execution": "deferred_until_training_releases_gpu",
            "identity_scope": "run_contract_step_and_sampled_parameters_not_full_weight_sha"}


def build_generation_protocol(tokenizer, *, dtype, max_new_tokens=512, max_prompt_tokens=1024, seed=0):
    """The exact rsd.evaluation rsd-text-eval-v2 greedy generation policy."""
    if type(tokenizer.eos_token_id) is not int or tokenizer.eos_token_id < 0:
        raise ValueError("Milestone evaluation requires tokenizer.eos_token_id")
    if type(tokenizer.pad_token_id) is not int or tokenizer.pad_token_id < 0:
        raise ValueError("Milestone evaluation requires tokenizer.pad_token_id")
    if not tokenizer.chat_template:
        raise ValueError("Milestone evaluation requires the existing chat template")
    return {"profile": "rsd-text-eval-v2", "enable_thinking": False,
            "max_new_tokens": max_new_tokens, "max_prompt_tokens": max_prompt_tokens,
            "temperature": 0.0, "do_sample": False, "top_p": 1.0, "top_k": 0,
            "num_beams": 1, "repetition_penalty": 1.0, "seed": seed,
            "dtype": str(dtype).removeprefix("torch."), "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id, "eos_token_id_source": "tokenizer.eos_token_id",
            "tokenizer_vocab_sha256": digest(tokenizer.get_vocab()),
            "tokenizer_template_sha256": digest(tokenizer.chat_template)}


def parameter_fingerprint(model, samples_per_parameter=32):
    """Hash evenly spaced samples, never serialize a weight tensor or claim full SHA."""
    entries = {}
    for name, parameter in model.named_parameters():
        if parameter.is_meta or getattr(parameter, "ds_id", None) is not None:
            raise ValueError("Partitioned/meta parameters are unsupported for rank-zero evaluation")
        if not parameter.numel():
            continue
        count = min(samples_per_parameter, parameter.numel())
        indices = torch.arange(count, device=parameter.device, dtype=torch.int64)
        if count > 1:
            indices = indices * (parameter.numel() - 1) // (count - 1)
        values = parameter.detach().reshape(-1).index_select(0, indices).cpu().contiguous()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite sampled student parameters")
        entries[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                         "numel": parameter.numel(), "n_samples": count,
                         "sample_sha256": hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest()}
    if not entries:
        raise ValueError("Resident student has no parameters")
    return {"method": "all_parameters_evenly_spaced_v1", "samples_per_parameter": samples_per_parameter,
            "n_tensors": len(entries), "sampled_coordinates": sum(r["n_samples"] for r in entries.values()),
            "tensors": entries, "sha256": digest(entries), "full_model_sha256": None}


def _resident_model(model):
    # The preferred getter is trainer.accelerator.unwrap_model(trainer.model_wrapped).
    # Never use TRL unwrap_model_for_generation: it can toggle checkpointing settings.
    while isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if hasattr(model, "zero_optimization_stage"):
        if model.zero_optimization_stage() > 2:
            raise ValueError("ZeRO-3 rank-zero generation is unsupported")
        model = model.module
    if any("FullySharded" in type(m).__name__ or hasattr(m, "_fsdp_wrapped_module") for m in model.modules()):
        raise ValueError("FSDP rank-zero generation is unsupported")
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type not in {"cpu", "cuda"}:
        raise ValueError("A complete resident model on one CPU/CUDA device is required")
    if getattr(model, "_tp_size", None) not in (None, 1) or any(type(p).__name__ == "DTensor" for p in model.parameters()):
        raise ValueError("Tensor-parallel generation is unsupported")
    return model, next(iter(devices))


@contextmanager
def preserved_runtime_state(model, device):
    """Restore RNG, every module's mode, config/cache settings and registered buffers.

    Only the model's CUDA device is touched. Seeding below deliberately avoids
    torch.manual_seed / transformers.set_seed, which seed *all* CUDA devices.
    Other devices' RNGs are never touched or initialized by this callback.
    Gradient-checkpointing enable/disable methods are never called.
    """
    py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    modules = [(m, m.training) for m in model.modules()]
    configs = {}
    buffers = []
    for module, _ in modules:
        for name in ("config", "generation_config"):
            value = getattr(module, name, None)
            if value is not None and id(value) not in configs:
                configs[id(value)] = (value, copy.deepcopy(value.__dict__))
        buffers.append((module, dict(module._buffers),
                        {k: v.detach().cpu().clone() for k, v in module._buffers.items() if v is not None}))
    try:
        yield
    finally:
        try:
            for config, snapshot in configs.values():
                config.__dict__.clear()
                config.__dict__.update(snapshot)
            with torch.no_grad():
                for module, original, snapshots in buffers:
                    module._buffers.clear()
                    module._buffers.update(original)
                    for name, value in snapshots.items():
                        original[name].copy_(value)
            for module, training in modules:
                module.training = training  # Preserve heterogeneous submodule modes exactly.
        finally:
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)


def _seed_evaluation(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            _append(stream, row)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MilestoneEvalCallback(TrainerCallback):
    def __init__(self, *, output_dir, run_contract, task_dataset, benchmark_dataset,
                 tokenizer, steps=(0, 1, 2), model_getter=None, max_new_tokens=512,
                 max_prompt_tokens=1024, seed=0, probe_samples=32):
        if not isinstance(run_contract, dict) or not run_contract:
            raise ValueError("A nonempty frozen run contract is required")
        self.run_contract = copy.deepcopy(run_contract)
        self.contract_sha = digest(self.run_contract)
        self.task_dataset, self.benchmark_dataset = copy.deepcopy(task_dataset), copy.deepcopy(benchmark_dataset)
        self.plan = build_milestone_contract(task_dataset, benchmark_dataset, steps,
                                             max_new_tokens=max_new_tokens, max_prompt_tokens=max_prompt_tokens,
                                             seed=seed, probe_samples=probe_samples)
        self.output_dir = Path(output_dir)
        self.tokenizer, self.model_getter = tokenizer, model_getter
        self.steps = tuple(steps)

    def _directory(self, step):
        return self.output_dir / f"step-{step:06d}"

    def _validated_existing(self, step, expected_manifest=None):
        folder = self._directory(step)
        receipt = json.loads((folder / "receipt.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        if receipt.get("status") != "completed":
            raise ValueError("Partial/failed milestone cannot be overwritten or silently retried")
        if manifest.get("plan") != self.plan or manifest.get("run_contract_sha256") != self.contract_sha:
            raise ValueError("Existing milestone uses another evaluation/run contract")
        identity = manifest["checkpoint_identity"]
        if identity["global_step"] != step or identity["run_contract_sha256"] != self.contract_sha:
            raise ValueError("Existing milestone step identity differs")
        body = {k: v for k, v in identity.items() if k != "checkpoint_id"}
        if identity["checkpoint_id"] != "in-memory:" + digest(body):
            raise ValueError("Invalid in-memory identity fingerprint")
        if expected_manifest is not None and manifest != expected_manifest:
            raise ValueError("Existing milestone differs from current model/protocol identity")
        if manifest["generation_protocol"] != build_generation_protocol(self.tokenizer,
                dtype=manifest["generation_protocol"]["dtype"], max_new_tokens=self.plan["max_new_tokens"],
                max_prompt_tokens=self.plan["max_prompt_tokens"], seed=self.plan["seed"]):
            raise ValueError("Existing milestone tokenizer/generation protocol differs")
        required = {"manifest.json", "task-dataset.jsonl", "benchmark-dataset.jsonl",
                    "task-predictions.jsonl", "benchmark-predictions.jsonl", "task-summary.json",
                    "benchmark-summary-pending.json", "judge-pending.json"}
        if set(receipt.get("artifact_sha256", {})) != required:
            raise ValueError("Incomplete milestone artifact inventory")
        if any(_sha(folder / name) != sha for name, sha in receipt["artifact_sha256"].items()):
            raise ValueError("Milestone artifact hash mismatch")
        return receipt

    @property
    def completed_steps(self):
        completed = []
        for step in self.steps:
            if self._directory(step).exists():
                self._validated_existing(step)
                completed.append(step)
        return tuple(completed)

    def assert_complete(self):
        if self.completed_steps != self.steps:
            raise MilestoneEvaluationError(f"Required milestones incomplete: {self.completed_steps}, expected {self.steps}")
        return self.completed_steps

    def _generate(self, model, device, dataset, path, identity, protocol):
        with path.open("x", encoding="utf-8") as stream:
            for row in dataset:
                record = {"id": row["id"], "checkpoint_id": identity["checkpoint_id"],
                          "checkpoint_identity": identity, "checkpoint_label": f"in-memory-step-{identity['global_step']}",
                          "global_step": identity["global_step"], "run_contract_sha256": self.contract_sha,
                          "model_revision": self.run_contract.get("config", {}).get("model_revision"),
                          "generation_protocol": protocol, "prompt_sha256": text_digest(row["prompt"]),
                          "response": "", "valid": False, "stop_reason": "error"}
                try:
                    rendered = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": row["prompt"]}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
                    inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
                    record["rendered_prompt_sha256"] = text_digest(rendered)
                    if inputs.input_ids.shape[1] > protocol["max_prompt_tokens"]:
                        record["error"] = "prompt_over_budget"
                    else:
                        config = GenerationConfig(max_new_tokens=protocol["max_new_tokens"], do_sample=False,
                            use_cache=True, pad_token_id=protocol["pad_token_id"], eos_token_id=protocol["eos_token_id"],
                            num_beams=1, num_return_sequences=1, repetition_penalty=1.0)
                        with torch.no_grad():
                            output = model.generate(**inputs, generation_config=config, synced_gpus=False)
                        tokens = output[0, inputs.input_ids.shape[1]:].detach().cpu().tolist()
                        response = self.tokenizer.decode(tokens, skip_special_tokens=True)
                        record.update(response=response, valid=bool(response.strip()), n_tokens=len(tokens),
                                      reached_token_limit=len(tokens) >= protocol["max_new_tokens"],
                                      stop_reason="eos" if tokens and tokens[-1] == protocol["eos_token_id"] else "length" if len(tokens) >= protocol["max_new_tokens"] else "unknown")
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    _append(stream, record)
                    raise
                _append(stream, record)

    def _evaluate_rank_zero(self, step, model, device):
        protocol = build_generation_protocol(self.tokenizer, dtype=next(model.parameters()).dtype,
                    max_new_tokens=self.plan["max_new_tokens"], max_prompt_tokens=self.plan["max_prompt_tokens"], seed=self.plan["seed"])
        probe = parameter_fingerprint(model, self.plan["probe_samples"])
        identity = {"kind": "resident_student_sampled_fingerprint", "run_contract_sha256": self.contract_sha,
                    "global_step": step, "parameter_probe": probe, "full_model_sha256": None,
                    "reloadable_checkpoint": False}
        identity["checkpoint_id"] = "in-memory:" + digest(identity)
        manifest = {"plan": self.plan, "run_contract_sha256": self.contract_sha,
                    "checkpoint_identity": identity, "generation_protocol": protocol}
        folder = self._directory(step)
        if folder.exists():
            self._validated_existing(step, manifest)
            return
        folder.mkdir(parents=True, exist_ok=False)
        self._created_step = step
        write_json(folder / "manifest.json", manifest)
        _jsonl(folder / "task-dataset.jsonl", self.task_dataset)
        _jsonl(folder / "benchmark-dataset.jsonl", self.benchmark_dataset)
        receipt = {"status": "running", "global_step": step, "checkpoint_id": identity["checkpoint_id"],
                   "judge_status": "pending", "weights_exported": False,
                   "n_task_planned": len(self.task_dataset), "n_benchmark_planned": len(self.benchmark_dataset)}
        write_json(folder / "receipt.json", receipt)
        try:
            _seed_evaluation(self.plan["seed"], device)
            model.eval()
            self._generate(model, device, self.task_dataset, folder / "task-predictions.jsonl", identity, protocol)
            self._generate(model, device, self.benchmark_dataset, folder / "benchmark-predictions.jsonl", identity, protocol)
            if parameter_fingerprint(model, self.plan["probe_samples"]) != probe:
                raise ValueError("Evaluation changed sampled student parameters")
            receipt["status"] = "completed"
        except BaseException as exc:
            receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            task = read_jsonl(folder / "task-predictions.jsonl") if (folder / "task-predictions.jsonl").exists() else []
            benchmark = read_jsonl(folder / "benchmark-predictions.jsonl") if (folder / "benchmark-predictions.jsonl").exists() else []
            write_json(folder / "task-summary.json", score_records(self.task_dataset, task, checkpoint_id=identity["checkpoint_id"]))
            write_json(folder / "benchmark-summary-pending.json", score_records(self.benchmark_dataset, benchmark, checkpoint_id=identity["checkpoint_id"]))
            write_json(folder / "judge-pending.json", {"status": "pending", "execute_after_training_releases_gpu": True,
                "checkpoint_id": identity["checkpoint_id"], "n_planned": len(self.benchmark_dataset),
                "n_eligible": len(judge_inputs(self.benchmark_dataset, benchmark, identity["checkpoint_id"])),
                "n_predictions": len(benchmark), "judgments_created": 0})
            receipt.update(n_task_predictions=len(task), n_benchmark_predictions=len(benchmark),
                n_invalid_predictions=sum(p.get("valid") is False for p in task + benchmark))
            receipt["artifact_sha256"] = {p.name: _sha(p) for p in folder.iterdir() if p.name != "receipt.json" and p.is_file()}
            write_json(folder / "receipt.json", receipt)

    def _event(self, args, state, control, callback_model):
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        error = None
        self._created_step = None
        try:
            if self.steps[-1] > state.max_steps:
                raise ValueError("Milestone schedule exceeds training max_steps")
            if state.global_step in self.steps:
                model, device = _resident_model(self.model_getter() if self.model_getter else callback_model)
                with preserved_runtime_state(model, device):
                    if rank == 0:
                        self._evaluate_rank_zero(state.global_step, model, device)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        local = {"step": state.global_step, "plan_sha256": digest(self.plan), "run_contract_sha256": self.contract_sha, "error": error}
        statuses = [local]
        if distributed:
            statuses = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(statuses, local)
        headers = {(s["step"], s["plan_sha256"], s["run_contract_sha256"]) for s in statuses}
        failures = [f"rank {i}: {s['error']}" for i, s in enumerate(statuses) if s["error"]]
        if len(headers) != 1:
            failures.append("Ranks disagree on milestone step/contract")
        if failures:
            if rank == 0 and self._created_step == state.global_step:
                path = self._directory(state.global_step) / "receipt.json"
                try:
                    receipt = json.loads(path.read_text()) if path.exists() else {"global_step": state.global_step}
                    receipt.update(status="failed", synchronized_errors=failures)
                    write_json(path, receipt)
                except Exception:
                    pass  # Every rank still raises the shared failure; never claim completion.
            raise MilestoneEvaluationError("; ".join(failures))
        return control

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        return self._event(args, state, control, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        return self._event(args, state, control, model)
