"""Content-bound confirmation release for all six planned runs (CPU only).

This is a workflow gate, not encryption, an ACL or proof that human labels are
correct. No model is loaded. A fixture release never becomes scientific evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .evaluation import digest, read_jsonl, text_digest, validate_dataset
from .task_evaluation import validate_gold

VERSION = "rsd-confirmation-plan-v1"
DATASETS = {"train", "task_dev", "task_dev_gold", "benchmark_dev", "task_test", "task_test_gold", "benchmark_test"}
CODE_FILES = ("confirmation.py", "milestone_v2.py", "milestone_eval.py", "task_evaluation.py", "evaluation.py",
              "paired_analysis.py", "local_judge.py", "judge_protocols.py", "train.py",
              "trainer.py", "collator.py", "modeling.py", "upstream.py", "experiment.py", "task_analysis.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def no_symlink(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlink paths are forbidden in confirmation evidence")
    return path


def child(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts):
        raise ValueError("Evidence path must be a local relative path")
    base = no_symlink(root)
    target = no_symlink(base / path)
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError("Evidence path escapes run root")
    return target


def write_new(path, value):
    """Atomic creation; never replace any existing evidence, including symlinks."""
    path = no_symlink(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic and fails if destination already exists.
        path.chmod(0o600)
    finally:
        os.unlink(temporary)


def code_identity():
    result = {name: sha(Path(__file__).with_name(name)) for name in CODE_FILES}
    result["configs/rsd/upstream.lock.json"] = sha(Path(__file__).resolve().parents[1] / "configs/rsd/upstream.lock.json")
    return result


def validate_task_pair(prompts, gold, split):
    if not prompts or any(r.get("split") != split for r in prompts):
        raise ValueError("Wrong or empty task split")
    labels = validate_gold(gold, allow_confirmation_scoring=split == "test")
    if [r.get("id") for r in prompts] != [r.get("id") for r in gold]:
        raise ValueError("Task prompts/gold order or identity differs")
    for row in prompts:
        if any(k in row for k in ("expected_answer", "answer", "answerKey", "verified_solution")):
            raise ValueError("Gold/solution leaked into model input")
        reference = labels[row["id"]]
        if (not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                or text_digest(row["prompt"]) != row.get("prompt_sha256")
                or any(row.get(k) != reference.get(k) for k in
                       ("id", "source_id", "split", "benchmark", "answer_type", "prompt_sha256", "choice_labels"))):
            raise ValueError("Task prompt/gold binding mismatch")


def _steps(values, max_steps):
    if (not isinstance(values, list) or not values or values != sorted(set(values))
            or any(type(s) is not int or not 0 <= s <= max_steps for s in values)):
        raise ValueError("Invalid planned optimizer nodes")


def validate_plan(plan, *, require_frozen=False):
    if plan.get("schema_version") != VERSION or plan.get("state") not in {"draft", "frozen"}:
        raise ValueError("Unknown confirmation plan/state")
    if require_frozen and plan["state"] != "frozen":
        raise ValueError("Confirmation execution/release requires a frozen plan")
    if plan.get("purpose") not in {"research", "cpu_fixture"}:
        raise ValueError("Plan purpose must be research or cpu_fixture")
    if plan.get("code_sha256") != code_identity():
        raise ValueError("Frozen scoring/controller/analysis implementation changed")
    recipe = plan.get("shared_recipe")
    if not isinstance(recipe, dict) or not recipe or plan.get("shared_recipe_sha256") != digest(recipe):
        raise ValueError("Shared recipe identity mismatch")
    if plan["purpose"] == "research":
        required_recipe = {"model", "train_file", "train_file_sha256", "teacher_context", "context_mode",
                           "context_placement", "trajectory_mode", "kl_direction", "thinking", "expected_world_size",
                           "per_device_train_batch_size", "gradient_accumulation_steps", "effective_global_batch",
                           "max_steps", "learning_rate", "max_grad_norm", "max_prompt_length", "max_completion_length",
                           "save_strategy", "export_final_model", "deepspeed", "deepspeed_sha256"}
        if not required_recipe <= set(recipe):
            raise ValueError("Research plan omits required shared training/model recipe fields")
        model = recipe["model"]
        if (not isinstance(model, dict) or not isinstance(model.get("repository"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", ""))):
            raise ValueError("Research model revision is not immutable")
    maximum = plan.get("max_steps")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Invalid max_steps")
    for key in ("dev_steps", "confirmation_task_steps", "confirmation_benchmark_steps"):
        _steps(plan.get(key), maximum)
        if 0 not in plan[key] or maximum not in plan[key]:
            raise ValueError("Every cohort requires baseline and final node")
    if not set(plan["confirmation_task_steps"] + plan["confirmation_benchmark_steps"]) <= set(plan["dev_steps"]):
        raise ValueError("Confirmation nodes must also be declared dev nodes")
    runs = plan.get("runs", [])
    if (len(runs) != 6 or len({r.get("run_id") for r in runs}) != 6
            or {(r.get("arm"), r.get("training_seed")) for r in runs} !=
            {(arm, seed) for arm in ("F0", "R0") for seed in (0, 1, 2)}):
        raise ValueError("Exactly six distinct F0/R0 × seeds 0/1/2 runs are required")
    for run in runs:
        if (not isinstance(run.get("run_id"), str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", run["run_id"])
                or type(run["training_seed"]) is not int or run.get("data_seed") != run["training_seed"]):
            raise ValueError("Invalid run identity/data seed")
    if set(plan.get("datasets", {})) != DATASETS:
        raise ValueError("Incomplete dataset inventory")
    for asset in plan["datasets"].values():
        if any(not isinstance(asset.get(k), str) or not re.fullmatch(r"[0-9a-f]{64}", asset[k])
               for k in ("sha256", "canonical_sha256")) or type(asset.get("rows")) is not int or asset["rows"] < 1:
            raise ValueError("Invalid dataset identity")
    generation = plan.get("generation", {})
    if (generation.get("do_sample") is not False or generation.get("thinking") is not False
            or any(type(generation.get(k)) is not int or generation[k] < 1 for k in ("max_new_tokens", "max_prompt_tokens"))
            or type(generation.get("seed")) is not int or not 0 <= generation["seed"] < 2**32):
        raise ValueError("Incomplete greedy generation contract")
    if plan["state"] == "frozen":
        judge, analysis = plan.get("judge_contract"), plan.get("analysis_contract")
        if not isinstance(judge, dict) or not isinstance(analysis, dict):
            raise ValueError("Frozen plan requires judge and analysis contracts")
        profile = judge.get("profile", {})
        if (not isinstance(profile.get("name"), str) or not profile["name"]
                or any(not isinstance(profile.get(k), dict) or not profile[k]
                       for k in ("model_identity", "generation", "source_lock"))):
            raise ValueError("Judge profile must bind model, generation and source lock")
        if plan["purpose"] == "cpu_fixture":
            if judge.get("validation_status") != "fixture_only" or analysis.get("scope") != "cpu_fixture":
                raise ValueError("CPU fixture must explicitly disclaim model validation")
        else:
            if judge.get("validation_status") != "validated_for_exploration":
                raise ValueError("Research judge is not independently validated")
            evidence = judge.get("validation_receipt", {})
            if not isinstance(evidence.get("path"), str) or not re.fullmatch(r"[0-9a-f]{64}", evidence.get("sha256", "")):
                raise ValueError("Judge validation receipt is not pinned")
            if (analysis.get("status") != "frozen" or analysis.get("primary") != "strongreject_test_R0_minus_F0_step512"
                    or maximum != 512 or analysis.get("alpha") != .05
                    or any(type(analysis.get(k)) not in (int, float) or not math.isfinite(analysis[k])
                           or not 0 < analysis[k] <= 1 for k in ("mde_safety", "mde_overrefusal"))):
                raise ValueError("Research primary/MDE/analysis is not frozen")
    return plan


def build_confirmation_plan(*, root, datasets, shared_recipe, runs, dev_steps,
                            confirmation_task_steps, confirmation_benchmark_steps,
                            max_steps, generation, judge_contract=None, analysis_contract=None,
                            state="draft", purpose="research"):
    if set(datasets) != DATASETS:
        raise ValueError("Exactly the seven declared data assets are required")
    rows, assets = {}, {}
    for key, name in datasets.items():
        path = no_symlink(Path(root) / name)
        records = read_jsonl(path)
        rows[key] = records
        assets[key] = {"path": str(name), "sha256": sha(path), "rows": len(records),
                       "canonical_sha256": digest(records), "ordered_ids_sha256": digest([r.get("id") for r in records])}
    validate_task_pair(rows["task_dev"], rows["task_dev_gold"], "dev")
    validate_task_pair(rows["task_test"], rows["task_test_gold"], "test")
    for key, split in (("benchmark_dev", "dev"), ("benchmark_test", "test")):
        validate_dataset(rows[key])
        if any(r["split"] != split or r["metric"] not in {"safety", "overrefusal"} for r in rows[key]):
            raise ValueError("Wrong benchmark cohort")
    if any(r.get("split") != "train" for r in rows["train"]):
        raise ValueError("Nontraining rows in training pool")
    if any(any(k in r for k in ("expected_answer", "answer", "answerKey", "verified_solution")) for r in rows["train"]):
        raise ValueError("Training model inputs contain gold/solutions")
    owners, ids = {}, set()
    for key in ("train", "task_dev", "task_test", "benchmark_dev", "benchmark_test"):
        for row in rows[key]:
            if not isinstance(row.get("source_id"), str) or not row["source_id"] or row.get("id") in ids:
                raise ValueError("Duplicate/missing source or record identity")
            ids.add(row["id"])
            previous = owners.setdefault(row["source_id"], row["split"])
            if previous != row["split"]:
                raise ValueError("Source leaks across train/dev/test")
    plan = {"schema_version": VERSION, "state": state, "purpose": purpose,
            "shared_recipe": copy.deepcopy(shared_recipe), "shared_recipe_sha256": digest(shared_recipe),
            "runs": copy.deepcopy(runs), "max_steps": max_steps, "dev_steps": list(dev_steps),
            "confirmation_task_steps": list(confirmation_task_steps),
            "confirmation_benchmark_steps": list(confirmation_benchmark_steps),
            "generation": copy.deepcopy(generation), "datasets": assets,
            "judge_contract": copy.deepcopy(judge_contract), "analysis_contract": copy.deepcopy(analysis_contract),
            "code_sha256": code_identity(), "scope": "Workflow integrity; not encryption or independent scientific validation"}
    return validate_plan(plan)


def load_plan(path, root, *, frozen=False, load_test_gold=False):
    path = no_symlink(path)
    plan = validate_plan(json.loads(path.read_text()), require_frozen=frozen)
    datasets = {}
    for key, asset in plan["datasets"].items():
        if key == "task_test_gold" and not load_test_gold:
            continue  # Training and raw-response release never load confirmation labels.
        source = no_symlink(Path(root) / asset["path"])
        if sha(source) != asset["sha256"]:
            raise ValueError(f"Frozen dataset bytes changed: {key}")
        rows = read_jsonl(source)
        if len(rows) != asset["rows"] or digest(rows) != asset["canonical_sha256"]:
            raise ValueError("Frozen dataset content identity changed")
        datasets[key] = rows
    if frozen and plan["purpose"] == "research":
        evidence = plan["judge_contract"]["validation_receipt"]
        source = no_symlink(Path(root) / evidence["path"])
        if sha(source) != evidence["sha256"]:
            raise ValueError("Judge validation receipt changed")
        record = json.loads(source.read_text())
        if (record.get("status") != "validated_for_exploration" or record.get("heldout_validation") is not True
                or record.get("independent_gold") is not True
                or record.get("judge_profile_sha256") != digest(plan["judge_contract"]["profile"])):
            raise ValueError("Judge validation receipt does not cover frozen profile")
    return plan, datasets


def validate_run_contract(contract, plan, plan_file_sha, *, root, require_runtime=False):
    config = contract.get("config", {})
    experiment = config.get("experiment", {})
    run = next((r for r in plan["runs"] if r["run_id"] == experiment.get("run_id")), None)
    if (run is None or experiment.get("plan_file_sha256") != plan_file_sha
            or experiment.get("shared_recipe_sha256") != plan["shared_recipe_sha256"]):
        raise ValueError("Run does not bind the frozen experiment plan/recipe")
    if (config.get("teacher_mode") != {"F0": "frozen", "R0": "recursive"}[run["arm"]]
            or config.get("training_seed") != run["training_seed"]
            or config.get("data_seed", config.get("training_seed")) != run["data_seed"]
            or config.get("max_steps") != plan["max_steps"]):
        raise ValueError("Run arm/seed/step differs from experiment plan")
    for key, expected in plan["shared_recipe"].items():
        if key == "model":
            actual = {"repository": config.get("model_name_or_path"), "revision": config.get("model_revision")}
        elif key == "train_file_sha256":
            actual = sha(no_symlink(Path(root) / config["train_file"]))
        elif key == "deepspeed_sha256":
            actual = sha(no_symlink(Path(root) / config["deepspeed"]))
        else:
            actual = config.get(key)
        if actual != expected:
            raise ValueError(f"Run changed shared recipe field: {key}")
    if contract.get("train_sha256") != plan["datasets"]["train"]["sha256"]:
        raise ValueError("Run training data identity differs")
    if require_runtime:
        world = contract.get("world_size")
        expected_world = plan["shared_recipe"]["expected_world_size"]
        expected_batch = expected_world * config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
        if type(world) is not int or world != expected_world or contract.get("effective_global_batch") != expected_batch:
            raise ValueError("Actual runtime world size/effective batch differs from frozen recipe")
        if "deepspeed_sha256" in plan["shared_recipe"] and contract.get("deepspeed_sha256") != plan["shared_recipe"]["deepspeed_sha256"]:
            raise ValueError("Runtime DeepSpeed identity differs from frozen recipe")
    return run


def _release_payload(plan_path, runs_root, *, root):
    from .milestone_v2 import validate_node
    plan, datasets = load_plan(plan_path, root, frozen=True)
    plan_sha = sha(plan_path)
    inventory, contracts, protocol_hashes = {}, set(), set()
    for expected in plan["runs"]:
        directory = child(runs_root, expected["run_id"])
        contract_path, result_path = child(directory, "run_contract.json"), child(directory, "result.json")
        contract = json.loads(contract_path.read_text())
        actual = validate_run_contract(contract, plan, plan_sha, root=root, require_runtime=True)
        if actual != expected:
            raise ValueError("Run directory and bound run identity disagree")
        contract_hash = digest(contract)
        if contract_hash in contracts:
            raise ValueError("One run contract was replayed as multiple runs")
        contracts.add(contract_hash)
        result = json.loads(result_path.read_text())
        if (result.get("status") != "training_completed" or result.get("global_step") != plan["max_steps"]
                or result.get("run_contract_sha256") != contract_hash
                or result.get("experiment_plan_file_sha256") != plan_sha or result.get("run_id") != expected["run_id"]):
            raise ValueError("Training run incomplete or failed")
        # The result file is additional completion evidence; checkpoint and arm
        # identity are verified from the immutable contract and every raw row.
        entries = {"run_contract.json": sha(contract_path), "result.json": sha(result_path)}
        for step in plan["dev_steps"]:
            folder = child(directory, f"milestones/step-{step:06d}")
            receipt, manifest = validate_node(folder, plan, datasets, contract_hash, plan_sha, step)
            protocol_hashes.add(digest(manifest["generation_protocol"]))
            prefix = f"milestones/step-{step:06d}/"
            entries[prefix + "receipt.json"] = sha(folder / "receipt.json")
            entries.update({prefix + name: value for name, value in receipt["artifact_sha256"].items()})
        inventory[expected["run_id"]] = entries
    if len(protocol_hashes) != 1:
        raise ValueError("Six runs use different generation protocols")
    return {"schema_version": "rsd-confirmation-release-v1", "status": "released_for_scoring",
            "plan_file_sha256": plan_sha, "plan_canonical_sha256": digest(plan),
            "shared_recipe_sha256": plan["shared_recipe_sha256"], "purpose": plan["purpose"],
            "completed_run_ids": [r["run_id"] for r in plan["runs"]], "inventory": inventory,
            "generation_protocol_sha256": next(iter(protocol_hashes)),
            "judge_contract_sha256": digest(plan["judge_contract"]),
            "analysis_contract_sha256": digest(plan["analysis_contract"]),
            "judging_executed": False, "scientific_ready": False,
            "interpretation": "All six run records verified; scoring is permitted, no scientific conclusion certified."}


def release_confirmation(plan_path, runs_root, output, *, root):
    payload = _release_payload(plan_path, runs_root, root=root)
    write_new(output, payload)
    return payload


def verify_release(plan_path, runs_root, release_path, *, root):
    observed = json.loads(no_symlink(release_path).read_text())
    if observed != _release_payload(plan_path, runs_root, root=root):
        raise ValueError("Release receipt or bound evidence changed")
    return observed


def released_inputs(plan_path, runs_root, release_path, *, root, run_id, step, cohort):
    release = verify_release(plan_path, runs_root, release_path, root=root)
    plan, datasets = load_plan(plan_path, root, frozen=True)
    if run_id not in release["completed_run_ids"] or cohort not in {"task", "benchmark"}:
        raise ValueError("Unknown released run/cohort")
    key = "confirmation_task_steps" if cohort == "task" else "confirmation_benchmark_steps"
    if type(step) is not int or step not in plan[key]:
        raise ValueError("Unplanned confirmation node")
    folder = child(runs_root, f"{run_id}/milestones/step-{step:06d}")
    predictions = read_jsonl(child(folder, f"sealed/{cohort}-predictions.jsonl"))
    return plan, datasets[f"{cohort}_test"], predictions, release


def score_released_task(plan_path, runs_root, release_path, output, *, root, run_id, step):
    from .task_evaluation import score_records
    plan, _, predictions, release = released_inputs(plan_path, runs_root, release_path,
                                                    root=root, run_id=run_id, step=step, cohort="task")
    _, data = load_plan(plan_path, root, frozen=True, load_test_gold=True)
    score = score_records(data["task_test_gold"], [{**p, "finish_reason": p.get("stop_reason", "unknown")} for p in predictions],
                          allow_confirmation_scoring=True)
    score.update(confirmation_release_gate_verified=True, release_receipt_sha256=sha(release_path),
                 run_id=run_id, global_step=step, scientific_conclusion=None,
                 run_contract_sha256=predictions[0]["run_contract_sha256"],
                 checkpoint_id=predictions[0]["checkpoint_id"],
                 generation_protocol_sha256=digest(predictions[0]["generation_protocol"]),
                 task_gold_sha256=plan["datasets"]["task_test_gold"]["sha256"],
                 task_gold_canonical_sha256=digest(data["task_test_gold"]),
                 scorer_code_sha256=plan["code_sha256"]["task_evaluation.py"])
    write_new(output, score)
    return score


def export_released_judge_inputs(plan_path, runs_root, release_path, output, *, root, run_id, step):
    """Release a bound input bundle, never load a judge or create a judgment."""
    plan, dataset, predictions, _ = released_inputs(plan_path, runs_root, release_path,
                                                    root=root, run_id=run_id, step=step, cohort="benchmark")
    bundle = {"status": "released_judge_inputs_only", "run_id": run_id, "global_step": step,
              "release_receipt_sha256": sha(release_path), "dataset": dataset, "predictions": predictions,
              "expected_judge_profile": plan["judge_contract"]["profile"], "judgments_created": 0,
              "n_planned": len(dataset), "n_invalid_predictions": sum(p["valid"] is False for p in predictions)}
    write_new(output, bundle)
    return bundle


def score_released_benchmark(plan_path, runs_root, release_path, judgments_path, output, *, root, run_id, step):
    from .evaluation import score_records
    from .local_judge import _validate_saved_judgment
    plan, dataset, predictions, _ = released_inputs(plan_path, runs_root, release_path,
                                                    root=root, run_id=run_id, step=step, cohort="benchmark")
    judgments = read_jsonl(no_symlink(judgments_path))
    profile = plan["judge_contract"]["profile"]
    judge_id = profile["name"] + ":" + digest(profile)
    if any(j.get("judge_id") != judge_id or j.get("judge_profile") != profile for j in judgments):
        raise ValueError("Judgments use a profile/model/protocol outside frozen plan")
    by_id = {row["id"]: row for row in dataset}
    for row in judgments:
        if row.get("id") not in by_id or row.get("metric") != by_id[row["id"]]["metric"]:
            raise ValueError("Judgment metric/identity differs from frozen input")
        _validate_saved_judgment(row)
    score = score_records(dataset, predictions, judgments)
    score.update(confirmation_release_gate_verified=True, release_receipt_sha256=sha(release_path),
                 judgments_file_sha256=sha(judgments_path), run_id=run_id, global_step=step,
                 scientific_conclusion=None)
    write_new(output, score)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(release_confirmation(args.plan, args.runs_root, args.output, root=args.root), indent=2))


if __name__ == "__main__":
    main()
