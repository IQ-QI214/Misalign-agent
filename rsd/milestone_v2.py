"""Resident dev scoring and unscored confirmation responses for EXP V2.

No confirmation gold is loaded by preparation/callback. Existing v1 callbacks
remain dev-only. This callback reuses v1 generation and runtime preservation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .confirmation import (child, digest, load_plan, no_symlink, sha, validate_run_contract,
                           validate_task_pair, write_new)
from .evaluation import read_jsonl, score_records as score_benchmark, text_digest
from .milestone_eval import (MilestoneEvalCallback, build_generation_protocol,
                             parameter_fingerprint, _jsonl, _seed_evaluation)
from .task_evaluation import score_records as score_task


SPEC_DATASET_MAP = {"task_dataset": "task_dev", "task_gold": "task_dev_gold", "benchmark_dataset": "benchmark_dev",
                    "confirmation_task_dataset": "task_test", "confirmation_benchmark_dataset": "benchmark_test"}


def prepare_v2_milestone_data(config, root):
    spec = config["milestone_eval"]
    if spec.get("profile") != "reasoning-v2":
        raise ValueError("Expected reasoning-v2 milestone profile")
    experiment = config.get("experiment", {})
    plan_path = no_symlink(Path(root) / experiment.get("plan_file", ""))
    if sha(plan_path) != experiment.get("plan_file_sha256"):
        raise ValueError("Experiment plan bytes changed")
    plan, datasets = load_plan(plan_path, root)
    # The run contract is constructed later by train.run. Check the same config
    # bindings here so dry-run rejects wrong arms/recipes before model loading.
    pseudo_contract = {"config": config, "train_sha256": sha(no_symlink(Path(root) / config["train_file"]))}
    run = validate_run_contract(pseudo_contract, plan, sha(plan_path), root=root)
    if spec.get("steps") != plan["dev_steps"]:
        raise ValueError("Dev nodes differ from frozen plan")
    for spec_key, plan_key in (("confirmation_task_steps", "confirmation_task_steps"),
                               ("confirmation_benchmark_steps", "confirmation_benchmark_steps")):
        if spec.get(spec_key) != plan[plan_key]:
            raise ValueError("Confirmation nodes differ from plan")
    for key in ("max_new_tokens", "max_prompt_tokens", "seed"):
        if spec.get(key) != plan["generation"][key]:
            raise ValueError("Generation budget differs from plan")
    for key in ("do_sample", "thinking"):
        if key in spec and spec[key] != plan["generation"][key]:
            raise ValueError("Generation sampling/thinking differs from plan")
    for key, dataset_key in SPEC_DATASET_MAP.items():
        actual = no_symlink(Path(root) / spec.get(key, ""))
        expected = no_symlink(Path(root) / plan["datasets"][dataset_key]["path"])
        if actual != expected or sha(actual) != plan["datasets"][dataset_key]["sha256"]:
            raise ValueError("Configured dataset differs from confirmation plan")
    validate_task_pair(datasets["task_dev"], datasets["task_dev_gold"], "dev")
    if any(row.get("teacher_context") != plan["shared_recipe"].get("teacher_context") for row in datasets["train"]):
        raise ValueError("Training teacher cue differs from shared recipe")
    return {"datasets": datasets, "plan": plan, "root": str(no_symlink(root)),
            "plan_path": str(plan_path), "run": run,
            "contract": {"version": "resident-student-milestones-v2", "profile": "reasoning-v2",
                         "plan_file_sha256": sha(plan_path), "plan_canonical_sha256": digest(plan),
                         "run_id": run["run_id"], "shared_recipe_sha256": plan["shared_recipe_sha256"],
                         "dataset_sha256": {k: v["sha256"] for k, v in plan["datasets"].items()},
                         "scope": "dev scores plus sealed unscored confirmation; test gold never loaded"}}


def required_artifacts(plan, step):
    names = {"manifest.json", "started.json", "task-dataset.jsonl", "task-gold.jsonl", "benchmark-dataset.jsonl",
             "task-predictions.jsonl", "benchmark-predictions.jsonl", "task-summary.json", "benchmark-summary-pending.json"}
    for cohort, key in (("task", "confirmation_task_steps"), ("benchmark", "confirmation_benchmark_steps")):
        if step in plan[key]:
            names.update((f"sealed/{cohort}-dataset.jsonl", f"sealed/{cohort}-predictions.jsonl"))
    return names


def _check_predictions(rows, dataset, identity, protocol, contract_sha, step):
    if [row.get("id") for row in rows] != [row["id"] for row in dataset]:
        raise ValueError("Prediction coverage/order differs from frozen cohort")
    for prediction, source in zip(rows, dataset):
        if (prediction.get("checkpoint_identity") != identity
                or prediction.get("checkpoint_id") != identity["checkpoint_id"]
                or prediction.get("run_contract_sha256") != contract_sha
                or prediction.get("global_step") != step
                or prediction.get("generation_protocol") != protocol
                or prediction.get("prompt_sha256") != text_digest(source["prompt"])
                or type(prediction.get("valid")) is not bool
                or prediction.get("stop_reason") not in {"eos", "length", "unknown", "error"}
                or prediction.get("valid") is True and prediction.get("stop_reason") == "error"
                or not isinstance(prediction.get("response"), str)):
            raise ValueError("Raw prediction identity/step/prompt/protocol mismatch")


def validate_node(folder, plan, datasets, contract_sha, plan_sha, step):
    folder = no_symlink(folder)
    if child(folder, "failure.json").exists():
        raise ValueError("Failed milestone cannot be released")
    receipt = json.loads(child(folder, "receipt.json").read_text())
    if receipt.get("status") != "completed" or receipt.get("global_step") != step:
        raise ValueError("Partial/failed/wrong-step milestone")
    if (receipt.get("confirmation_scored") is not False
            or receipt.get("n_task_predictions") != len(datasets["task_dev"])
            or receipt.get("n_benchmark_predictions") != len(datasets["benchmark_dev"])):
        raise ValueError("Receipt coverage or confirmation seal state differs")
    expected_counts = {cohort: len(datasets[f"{cohort}_test"])
                       for cohort, key in (("task", "confirmation_task_steps"), ("benchmark", "confirmation_benchmark_steps"))
                       if step in plan[key]}
    if receipt.get("confirmation_counts") != expected_counts:
        raise ValueError("Confirmation planned counts differ")
    required = required_artifacts(plan, step)
    if set(receipt.get("artifact_sha256", {})) != required:
        raise ValueError("Milestone artifact inventory differs")
    for name, expected in receipt["artifact_sha256"].items():
        if sha(child(folder, name)) != expected:
            raise ValueError("Milestone artifact hash mismatch")
    # Confirmation subdirectory may contain only the declared raw datasets and
    # responses. A score/summary here means the operational seal was violated.
    sealed = child(folder, "sealed")
    if sealed.exists():
        actual_sealed = {"sealed/" + p.name for p in sealed.iterdir()}
        if actual_sealed != {name for name in required if name.startswith("sealed/")}:
            raise ValueError("Unexpected files inside confirmation seal")
    manifest = json.loads(child(folder, "manifest.json").read_text())
    identity, protocol = manifest["checkpoint_identity"], manifest["generation_protocol"]
    if (manifest.get("plan_file_sha256") != plan_sha or manifest.get("plan_canonical_sha256") != digest(plan)
            or manifest.get("run_contract_sha256") != contract_sha
            or identity.get("run_contract_sha256") != contract_sha or identity.get("global_step") != step
            or identity.get("full_model_sha256") is not None or identity.get("reloadable_checkpoint") is not False):
        raise ValueError("Milestone run/plan/checkpoint identity mismatch")
    identity_body = {k: v for k, v in identity.items() if k != "checkpoint_id"}
    if identity.get("checkpoint_id") != "in-memory:" + digest(identity_body):
        raise ValueError("Invalid checkpoint fingerprint")
    if receipt.get("checkpoint_id") != identity["checkpoint_id"]:
        raise ValueError("Receipt checkpoint differs from manifest")
    for key in ("max_new_tokens", "max_prompt_tokens", "seed", "do_sample"):
        if protocol.get(key) != plan["generation"][key]:
            raise ValueError("Generation differs from planned protocol")
    if protocol.get("enable_thinking") is not False or protocol.get("profile") != "rsd-text-eval-v2":
        raise ValueError("Generation profile changed")
    for cohort, dataset_key, prefix in (("task", "task_dev", ""), ("benchmark", "benchmark_dev", ""),
                                        ("task", "task_test", "sealed/"), ("benchmark", "benchmark_test", "sealed/")):
        dataset_name, predictions_name = prefix + cohort + "-dataset.jsonl", prefix + cohort + "-predictions.jsonl"
        if dataset_name not in required:
            continue
        raw_dataset = read_jsonl(child(folder, dataset_name))
        if raw_dataset != datasets[dataset_key]:
            raise ValueError("Milestone dataset differs from frozen source")
        predictions = read_jsonl(child(folder, predictions_name))
        _check_predictions(predictions, raw_dataset, identity, protocol, contract_sha, step)
    if read_jsonl(child(folder, "task-gold.jsonl")) != datasets["task_dev_gold"]:
        raise ValueError("Dev gold differs from frozen source")
    return receipt, manifest


class ReasoningMilestoneCallback(MilestoneEvalCallback):
    """V2 callback; inherited event code preserves RNG/modes and rank agreement."""
    def __init__(self, *, output_dir, run_contract, prepared, tokenizer, model_getter=None, probe_samples=32):
        self.prepared = copy.deepcopy(prepared)
        self.experiment_plan = prepared["plan"]
        # CPU dry-run can inspect draft plans; actual callback construction
        # requires explicit freeze (research or clearly named CPU fixture).
        plan, _ = load_plan(prepared["plan_path"], prepared["root"], frozen=True)
        if plan != self.experiment_plan:
            raise ValueError("Prepared plan changed")
        self.run_contract, self.contract_sha = copy.deepcopy(run_contract), digest(run_contract)
        run = validate_run_contract(run_contract, plan, prepared["contract"]["plan_file_sha256"],
                                    root=prepared["root"], require_runtime=True)
        if run != prepared["run"]:
            raise ValueError("Prepared run changed")
        self.datasets = prepared["datasets"]
        self.task_dataset, self.benchmark_dataset = self.datasets["task_dev"], self.datasets["benchmark_dev"]
        self.plan = {**prepared["contract"], "steps": plan["dev_steps"], **plan["generation"], "probe_samples": probe_samples}
        self.steps = tuple(plan["dev_steps"])
        self.output_dir = no_symlink(output_dir)
        self.tokenizer, self.model_getter = tokenizer, model_getter

    def _validated_existing(self, step, expected_manifest=None):
        receipt, manifest = validate_node(self._directory(step), self.experiment_plan, self.datasets,
                                           self.contract_sha, self.plan["plan_file_sha256"], step)
        if expected_manifest is not None and manifest != expected_manifest:
            raise ValueError("Existing milestone differs from current resident model/protocol")
        expected = build_generation_protocol(self.tokenizer, dtype=manifest["generation_protocol"]["dtype"],
                    max_new_tokens=self.plan["max_new_tokens"], max_prompt_tokens=self.plan["max_prompt_tokens"], seed=self.plan["seed"])
        if manifest["generation_protocol"] != expected:
            raise ValueError("Existing milestone tokenizer differs")
        return receipt

    def _evaluate_rank_zero(self, step, model, device):
        protocol = build_generation_protocol(self.tokenizer, dtype=next(model.parameters()).dtype,
                    max_new_tokens=self.plan["max_new_tokens"], max_prompt_tokens=self.plan["max_prompt_tokens"], seed=self.plan["seed"])
        probe = parameter_fingerprint(model, self.plan["probe_samples"])
        identity = {"kind": "resident_student_sampled_fingerprint", "run_contract_sha256": self.contract_sha,
                    "global_step": step, "parameter_probe": probe, "full_model_sha256": None, "reloadable_checkpoint": False}
        identity["checkpoint_id"] = "in-memory:" + digest(identity)
        manifest = {"version": "resident-student-milestones-v2", "plan_file_sha256": self.plan["plan_file_sha256"],
                    "plan_canonical_sha256": digest(self.experiment_plan), "run_contract_sha256": self.contract_sha,
                    "checkpoint_identity": identity, "generation_protocol": protocol}
        folder = no_symlink(self._directory(step))
        if folder.exists():
            self._validated_existing(step, manifest)
            return
        folder.mkdir(parents=True, exist_ok=False)
        write_new(folder / "manifest.json", manifest)
        write_new(folder / "started.json", {"status": "running", "global_step": step,
                                            "checkpoint_id": identity["checkpoint_id"]})
        try:
            for key, name in (("task_dev", "task-dataset.jsonl"), ("task_dev_gold", "task-gold.jsonl"),
                              ("benchmark_dev", "benchmark-dataset.jsonl")):
                _jsonl(folder / name, self.datasets[key])
            _seed_evaluation(self.plan["seed"], device)
            model.eval()
            self._generate(model, device, self.task_dataset, folder / "task-predictions.jsonl", identity, protocol)
            self._generate(model, device, self.benchmark_dataset, folder / "benchmark-predictions.jsonl", identity, protocol)
            task = read_jsonl(folder / "task-predictions.jsonl")
            benchmark = read_jsonl(folder / "benchmark-predictions.jsonl")
            task_summary = score_task(self.datasets["task_dev_gold"],
                      [{**p, "finish_reason": p.get("stop_reason", "unknown")} for p in task])
            task_summary.update(run_contract_sha256=self.contract_sha, checkpoint_id=identity["checkpoint_id"],
                                global_step=step, generation_protocol_sha256=digest(protocol),
                                task_gold_sha256=self.experiment_plan["datasets"]["task_dev_gold"]["sha256"],
                                task_gold_canonical_sha256=digest(self.datasets["task_dev_gold"]),
                                scorer_code_sha256=self.experiment_plan["code_sha256"]["task_evaluation.py"])
            write_new(folder / "task-summary.json", task_summary)
            write_new(folder / "benchmark-summary-pending.json", score_benchmark(self.benchmark_dataset, benchmark,
                       checkpoint_id=identity["checkpoint_id"]))
            counts = {}
            for cohort, key in (("task", "confirmation_task_steps"), ("benchmark", "confirmation_benchmark_steps")):
                if step not in self.experiment_plan[key]:
                    continue
                sealed = folder / "sealed"
                sealed.mkdir(mode=0o700, exist_ok=True)
                dataset = self.datasets[f"{cohort}_test"]
                _jsonl(sealed / f"{cohort}-dataset.jsonl", dataset)
                self._generate(model, device, dataset, sealed / f"{cohort}-predictions.jsonl", identity, protocol)
                counts[cohort] = len(dataset)
            if parameter_fingerprint(model, self.plan["probe_samples"]) != probe:
                raise ValueError("Evaluation changed sampled parameters")
            required = required_artifacts(self.experiment_plan, step)
            receipt = {"status": "completed", "global_step": step, "checkpoint_id": identity["checkpoint_id"],
                       "n_task_predictions": len(task), "n_benchmark_predictions": len(benchmark),
                       "confirmation_counts": counts, "confirmation_scored": False, "judge_status": "pending",
                       "weights_exported": False, "artifact_sha256": {name: sha(child(folder, name)) for name in sorted(required)}}
            write_new(folder / "receipt.json", receipt)
            self._validated_existing(step, manifest)
        except BaseException as exc:
            write_new(folder / "failure.json", {"status": "failed", "global_step": step,
                                                "error": f"{type(exc).__name__}: {exc}"})
            raise
