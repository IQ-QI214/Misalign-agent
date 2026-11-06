"""Train the independent RSD adaptation. --dry-run never loads model weights."""

import argparse
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import time

from .upstream import ROOT, verify_upstream


def normalize_save_strategy(config):
    config = dict(config)
    config.setdefault("save_strategy", "steps")
    if config["save_strategy"] not in ("steps", "no"):
        raise ValueError("save_strategy must be steps or no")
    return config


def load_config(path, arm=None):
    config = normalize_save_strategy(json.loads(Path(path).read_text()))
    if arm:
        matrix = json.loads((ROOT / "configs/rsd/matrix.json").read_text())
        from .data import validate_matrix
        validate_matrix(matrix, ROOT)
        entries = matrix["arms"]
        if isinstance(entries, list):
            entries = {row.get("id", row.get("name", row.get("arm_id"))): row for row in entries}
        if arm not in entries:
            raise ValueError(f"Unknown matrix arm: {arm}")
        if config["train_file"] != entries[arm]["train_file"]:
            raise ValueError("Matrix and config refer to different training data")
        config.update({k: v for k, v in entries[arm].items() if k in ("teacher_mode", "feedback_mode")})
        config["run_name"] = arm
        config["output_dir"] = str(Path(config["output_dir"]) / arm)
    if config["teacher_mode"] not in ("frozen", "recursive"):
        raise ValueError("teacher_mode must be frozen or recursive")
    if config["trajectory_mode"] not in ("on_policy", "off_policy"):
        raise ValueError("trajectory_mode must be on_policy or off_policy")
    if config["kl_direction"] not in ("forward", "reverse"):
        raise ValueError("kl_direction must be explicit and independent of sampling mode")
    if config["per_device_train_batch_size"] != 1 or config["gradient_accumulation_steps"] != 1:
        raise ValueError("v1 requires microbatch=1 and accumulation=1")
    if config["max_steps"] < 1 or config["max_completion_length"] < 1:
        raise ValueError("Invalid step/length budget")
    if config["max_prompt_length"] + config["max_completion_length"] > config["max_length"]:
        raise ValueError("Prompt+completion exceeds declared context")
    return config


def inspect_training_data(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Training file is empty")
    ids = set()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Training input contains non-training records")
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("id", "source_id", "prompt")):
            raise ValueError("Missing training identity or prompt")
        if row["id"] in ids:
            raise ValueError("Duplicate training id")
        ids.add(row["id"])
    return rows


def make_training_args(config, output_dir, use_cpu=False):
    from trl.experimental.gold import GOLDConfig
    config = normalize_save_strategy(config)
    return GOLDConfig(
        output_dir=str(output_dir), max_steps=config["max_steps"],
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=config["learning_rate"], lr_scheduler_type="constant", warmup_steps=0,
        weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        max_grad_norm=config["max_grad_norm"],
        bf16=not use_cpu, fp16=False, use_cpu=use_cpu,
        gradient_checkpointing=not use_cpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=config["max_length"], max_completion_length=config["max_completion_length"],
        temperature=config["generation_temperature"], top_p=1.0, top_k=0,
        beta=1.0 if config["kl_direction"] == "reverse" else 0.0,
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_drop_last=True, packing=False,
        logging_steps=1, save_strategy=config["save_strategy"], save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 3), save_only_model=False,
        report_to="none", seed=config["training_seed"], data_seed=config["training_seed"],
        use_vllm=False, log_completions=False, disable_dropout=True,
        deepspeed=None if use_cpu else str(ROOT / config["deepspeed"]),
        ddp_find_unused_parameters=False,
    )


def prepare_contract(output, identity, resume=None):
    """Called by rank zero only, before broadcasting its success to every rank."""
    output = Path(output).resolve()
    contract = output / "run_contract.json"
    if contract.exists():
        if json.loads(contract.read_text()) != identity:
            raise ValueError("Output contains a different experiment contract")
        if not resume:
            raise ValueError("Existing run requires an explicit checkpoint resume")
    if resume:
        checkpoint = Path(resume).resolve()
        if checkpoint.parent != output or not checkpoint.name.startswith("checkpoint-"):
            raise ValueError("Resume must belong to this run output")
        checkpoint_contract = checkpoint / "run_contract.json"
        if not checkpoint_contract.is_file() or json.loads(checkpoint_contract.read_text()) != identity:
            raise ValueError("Checkpoint has no matching experiment contract")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if checkpoint.name != f"checkpoint-{state['global_step']}":
            raise ValueError("Checkpoint step identity mismatch")
    if not contract.exists():
        output.mkdir(parents=True, exist_ok=True)
        temp = output / "run_contract.tmp"
        temp.write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n")
        temp.replace(contract)


def capture_parameter_probe(model, samples_per_parameter=1024):
    """Snapshot evenly spaced coordinates of every trainable tensor on CPU.

    Includes each tensor's first and last coordinate. This observes model
    weights, not optimizer master weights, and deliberately avoids a full copy.
    """
    import torch

    if not isinstance(samples_per_parameter, int) or samples_per_parameter < 1:
        raise ValueError("samples_per_parameter must be a positive integer")
    tensors = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        count = min(samples_per_parameter, parameter.numel())
        indices = torch.arange(count, device=parameter.device, dtype=torch.int64)
        if count > 1:
            indices = indices * (parameter.numel() - 1) // (count - 1)
        values = parameter.detach().reshape(-1).index_select(0, indices)
        tensors[name] = {
            "shape": list(parameter.shape), "dtype": str(parameter.dtype),
            "numel": parameter.numel(), "indices": indices.cpu(),
            "values": values.to(device="cpu", dtype=torch.float32).clone(),
        }
    if not tensors:
        raise ValueError("Parameter probe requires nonempty trainable tensors")
    return {"samples_per_parameter": samples_per_parameter, "tensors": tensors}


def compare_parameter_probes(initial, final):
    """Return sampled model-weight changes; zero is not proof of no update."""
    import math
    import torch

    if initial["samples_per_parameter"] != final["samples_per_parameter"] or initial["tensors"].keys() != final["tensors"].keys():
        raise ValueError("Parameter probe tensor identity differs")
    details = {}
    squared_delta = 0.0
    largest_delta = 0.0
    total_samples = total_changed = total_nonfinite = total_numel = 0
    for name, before in initial["tensors"].items():
        after = final["tensors"][name]
        if any(before[key] != after[key] for key in ("shape", "dtype", "numel")) or not torch.equal(before["indices"], after["indices"]):
            raise ValueError(f"Parameter probe coordinates differ: {name}")
        a, b = before["values"], after["values"]
        finite = torch.isfinite(a) & torch.isfinite(b)
        nonfinite = int((~finite).sum())
        changed = int(((a != b) & finite).sum())
        delta = b.double() - a.double()
        norm_squared = float(delta.square().sum()) if not nonfinite else None
        max_delta = float(delta.abs().max()) if not nonfinite else None
        details[name] = {
            "numel": before["numel"], "sampled_coordinates": a.numel(),
            "changed_coordinates": changed, "nonfinite_coordinates": nonfinite,
            "l2_delta": math.sqrt(norm_squared) if norm_squared is not None else None,
            "max_abs_delta": max_delta,
        }
        total_numel += before["numel"]
        total_samples += a.numel()
        total_changed += changed
        total_nonfinite += nonfinite
        if norm_squared is not None:
            squared_delta += norm_squared
            largest_delta = max(largest_delta, max_delta)
    return {
        "method": "all_trainable_tensors_evenly_spaced_v1",
        "samples_per_parameter": initial["samples_per_parameter"],
        "trainable_tensors": len(details), "trainable_coordinates": total_numel,
        "sampled_coordinates": total_samples, "changed_coordinates": total_changed,
        "changed_tensors": sum(row["changed_coordinates"] > 0 for row in details.values()),
        "nonfinite_coordinates": total_nonfinite,
        "l2_delta": math.sqrt(squared_delta) if not total_nonfinite else None,
        "max_abs_delta": largest_delta if not total_nonfinite else None,
        "evidence_scope": "Sampled model weights only; excludes optimizer master weights and unsampled coordinates.",
        "zero_delta_interpretation": "No visible change in these samples; BF16 rounding or changes elsewhere may hide optimizer updates.",
        "tensors": details,
    }


def run(config, resume=None, use_cpu=False):
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from accelerate import PartialState
    from .modeling import load_text_model, load_tokenizer
    from .collator import TransferCollator
    from .trainer import TransferTrainer, ReceiptCallback, TRAINING_GENERATION_POLICY

    config = normalize_save_strategy(config)
    verify_upstream()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not use_cpu and world != config["expected_world_size"]:
        raise ValueError(f"Expected {config['expected_world_size']} processes, got {world}")
    set_seed(config["training_seed"])
    train_path = (ROOT / config["train_file"]).resolve()
    rows = inspect_training_data(train_path)
    output = (ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "upstream": verify_upstream(), "world_size": world,
                "deepspeed_sha256": hashlib.sha256((ROOT / config["deepspeed"]).read_bytes()).hexdigest(),
                "effective_global_batch": world, "profile": "independent_qwen35_rsd_v1",
                "generation_policy": dict(TRAINING_GENERATION_POLICY),
                "template": {"version": "explicit_privileged_prefix_v1", "thinking": False,
                             "context_mode": config.get("context_mode", "feedback_only"),
                             "placement": config.get("context_placement", "user")}}
    distributed = PartialState(cpu=use_cpu)
    rank = distributed.process_index
    decision = [None]
    if distributed.is_main_process:
        try:
            prepare_contract(output, identity, resume)
        except Exception as exc:
            decision[0] = str(exc)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(decision, src=0)
    if decision[0] is not None:
        raise ValueError(decision[0])
    args = make_training_args(config, output, use_cpu)
    model_path = config["model_name_or_path"]
    revision = config.get("model_revision")
    dtype = "float32" if use_cpu else "bfloat16"
    tokenizer = load_tokenizer(model_path, revision)
    model = load_text_model(model_path, revision, dtype=dtype)
    teacher = load_text_model(model_path, revision, dtype=dtype) if config["teacher_mode"] == "frozen" else None
    initial_probe = capture_parameter_probe(model)
    trainer = TransferTrainer(
        model=model, args=args, train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
        data_collator=TransferCollator(tokenizer, config["max_prompt_length"], config["feedback_mode"], False,
                                      config.get("context_mode", "feedback_only"),
                                      config.get("context_placement", "user")),
        teacher_mode=config["teacher_mode"], trajectory_mode=config["trajectory_mode"],
        kl_direction=config["kl_direction"],
        frozen_teacher=teacher, loss_chunk_size=config["loss_chunk_size"], callbacks=[ReceiptCallback()],
        student_thinking=False, teacher_thinking=False,
    )
    if rank == 0:
        versions = {}
        for package in ("torch", "transformers", "trl", "accelerate", "datasets", "deepspeed", "flash-linear-attention", "causal-conv1d"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        (output / "environment.json").write_text(json.dumps({
            "packages": versions, "load_report": model.load_report,
            "training_generation_policy": trainer.training_generation_policy,
        }, indent=2) + "\n")
    training_started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=resume)
    training_finished = time.monotonic()
    trainer.save_model(str(output / "final"))
    trainer.save_state()
    export_finished = time.monotonic()
    final_probe = capture_parameter_probe(trainer.accelerator.unwrap_model(trainer.model))
    if rank == 0:
        probe_report = compare_parameter_probes(initial_probe, final_probe)
        probe_report["baseline"] = "model_loaded_before_optional_resume"
        probe_report["resume_requested"] = resume is not None
        report = {"status": "training_completed", "global_step": trainer.state.global_step,
                  "metrics": result.metrics, "parameter_probe_l2_delta": probe_report["l2_delta"],
                  "parameter_probe": probe_report,
                  "evaluation_status": "not_run", "scientific_conclusion": None}
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    cuda_device = trainer.accelerator.device if not use_cpu and trainer.accelerator.device.type == "cuda" else None
    resources = {
        "status": "training_and_export_completed", "rank": rank,
        "effective_world_size": trainer.accelerator.num_processes,
        "global_step": trainer.state.global_step, "attempt_id": trainer.attempt_id,
        "resume_requested": resume is not None,
        "device": str(trainer.accelerator.device),
        "device_name": torch.cuda.get_device_name(cuda_device) if cuda_device is not None else None,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(cuda_device) if cuda_device is not None else None,
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(cuda_device) if cuda_device is not None else None,
        "memory_scope": "PyTorch CUDA allocator only; excludes other processes, non-PyTorch allocations and host RAM.",
        "memory_peak_window": "Since process start or last allocator peak reset; this receipt does not reset counters.",
        "train_and_periodic_checkpoints_seconds": training_finished - training_started,
        "final_export_and_trainer_state_seconds": export_finished - training_finished,
        "timing_scope": "Rank-local monotonic wall clock without added barriers or CUDA synchronization; excludes loading and parameter probes.",
    }
    (output / f"resources_rank{rank}.json").write_text(json.dumps(resources, indent=2) + "\n")
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rsd/qwen35_9b_2h100.json")
    parser.add_argument("--arm", choices=["F0", "R0", "F1", "R1"])
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.arm)
    rows = inspect_training_data(ROOT / config["train_file"])
    verify_upstream()
    if args.dry_run:
        print(json.dumps({"status": "configuration_validated_only", "config": config,
                          "n_train": len(rows), "gpu_verified": False}, indent=2, ensure_ascii=False))
        return
    run(config, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
