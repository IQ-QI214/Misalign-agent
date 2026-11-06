"""Materialize six disabled EXP V2 configs on CPU; never launch a process.

This produces a draft confirmation plan. Judge validation, actual throughput,
and scientific protocol freeze are external prerequisites, not implied here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .evaluation import digest, read_jsonl
from .train import load_config, prepare_milestone_data, validate_training_geometry
from .upstream import ROOT


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def checked_inputs(plan, root):
    if plan.get("schema_version") != "rsd-exploration-v2-plan" or plan.get("training_enabled") is not False:
        raise ValueError("Only a disabled exploration-v2 preparation plan is accepted")
    recipe = plan["candidate_shared_recipe"]
    if digest(recipe) != plan["candidate_shared_recipe_sha256"]:
        raise ValueError("Candidate shared recipe hash mismatch")
    files = {recipe["train_file"]: recipe["train_file_sha256"], recipe["deepspeed"]: recipe["deepspeed_sha256"]}
    for name, item in plan["datasets"]["task_artifacts"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Task artifact path escapes data directory")
        files[str(Path(recipe["train_file"]).parent / name)] = item["sha256"]
    for key in ("safety_dev", "safety_test"):
        spec = plan["datasets"][key]
        files[spec["file"]] = spec["sha256"]
    for key in ("source_lock", "task_audit"):
        files[plan["datasets"][key]] = plan["datasets"][key + "_file_sha256"]
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError(f"Preparation input hash differs: {name}")
    runs = plan["planned_run_order"]
    seeds = plan["training_seeds"]
    if (seeds != [0, 1, 2] or len(runs) != 6 or len({r["run_id"] for r in runs}) != 6
            or {(r["arm"], r["training_seed"]) for r in runs} != {(a, s) for a in ("F0", "R0") for s in seeds}
            or any(r["data_seed"] != r["training_seed"] for r in runs)):
        raise ValueError("Require the six predeclared F0/R0 seed pairs")
    for run in runs:
        if not isinstance(run["run_id"], str) or not run["run_id"] or Path(run["run_id"]).name != run["run_id"] or run["run_id"] in (".", ".."):
            raise ValueError("Run id must be a safe nonempty basename")
    return recipe, files


def materialize(plan_path, output_dir, *, root=ROOT):
    from .confirmation import build_confirmation_plan
    root = Path(root).resolve()
    plan_path, output = Path(plan_path).resolve(), Path(output_dir).absolute()
    plan = json.loads(plan_path.read_text())
    recipe, input_hashes = checked_inputs(plan, root)
    if output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise ValueError("Preparation output cannot traverse a symlink")
    if output.exists():
        raise FileExistsError("Preparation output must be new; preserve earlier attempts")
    tasks = str(Path(recipe["train_file"]).parent)
    datasets = {"train": recipe["train_file"], "task_dev": tasks + "/dev.prompts.jsonl",
                "task_dev_gold": tasks + "/dev.gold.jsonl", "benchmark_dev": plan["datasets"]["safety_dev"]["file"],
                "task_test": tasks + "/sealed/test.prompts.jsonl", "task_test_gold": tasks + "/sealed/test.gold.jsonl",
                "benchmark_test": plan["datasets"]["safety_test"]["file"]}
    generation = {k: plan["evaluation_generation"][k] for k in ("max_new_tokens", "max_prompt_tokens", "seed", "do_sample", "thinking")}
    confirmation = build_confirmation_plan(root=root, datasets=datasets, shared_recipe=recipe,
        runs=plan["planned_run_order"], dev_steps=plan["dev_nodes"],
        confirmation_task_steps=plan["confirmation_task_nodes"],
        confirmation_benchmark_steps=plan["confirmation_safety_nodes"], max_steps=recipe["max_steps"],
        generation=generation, state="draft")
    output.mkdir(parents=True)
    plan_output = output / "confirmation-plan.json"
    write_new(plan_output, confirmation)
    def relative_or_absolute(path):
        return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    paths = []
    for entry in plan["planned_run_order"]:
        config = {k: v for k, v in recipe.items() if k != "model"}
        config.update(model_name_or_path=recipe["model"]["repository"], model_revision=recipe["model"]["revision"],
            run_name=entry["run_id"], teacher_mode="frozen" if entry["arm"] == "F0" else "recursive",
            training_seed=entry["training_seed"], data_seed=entry["data_seed"],
            feedback_mode="neutral", output_dir="runs/" + entry["run_id"], training_enabled=False,
            experiment={"plan_file": relative_or_absolute(plan_output), "plan_file_sha256": file_sha(plan_output),
                        "run_id": entry["run_id"], "shared_recipe_sha256": confirmation["shared_recipe_sha256"]},
            milestone_eval={"profile": "reasoning-v2", "steps": plan["dev_nodes"],
                            "task_dataset": datasets["task_dev"], "task_gold": datasets["task_dev_gold"],
                            "benchmark_dataset": datasets["benchmark_dev"],
                            "confirmation_task_dataset": datasets["task_test"],
                            "confirmation_benchmark_dataset": datasets["benchmark_test"],
                            "confirmation_task_steps": plan["confirmation_task_nodes"],
                            "confirmation_benchmark_steps": plan["confirmation_safety_nodes"], **generation})
        path = output / (entry["run_id"] + ".json")
        write_new(path, config)
        config = load_config(path)
        dose = validate_training_geometry(config, read_jsonl(root / config["train_file"]), recipe["expected_world_size"])
        # Use the helper's explicit root for test fixtures / separate checkouts.
        from .milestone_v2 import prepare_v2_milestone_data
        prepared = prepare_v2_milestone_data(config, root)
        paths.append({"run_id": entry["run_id"], "arm": entry["arm"], "seed": entry["training_seed"],
                      "file": path.name, "sha256": file_sha(path), "training_enabled": False,
                      "dose_contract": dose, "milestone_contract_sha256": digest(prepared["contract"])})
    receipt = {"version": "rsd-experiment-cpu-preparation-v2", "status": "SIX_DISABLED_CONFIGS_VALIDATED_CPU",
               "source_plan_sha256": file_sha(plan_path), "input_file_sha256": input_hashes,
               "confirmation_plan_sha256": file_sha(plan_output), "confirmation_plan_state": "draft",
               "runs": paths, "model_loads": 0, "training_updates": 0, "gpu_executions": 0,
               "research_ready": False, "scientific_conclusion": None}
    write_new(output / "preparation-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/rsd/exploration_v2.plan.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = materialize(args.plan, args.output)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "research_ready": False}))


if __name__ == "__main__":
    main()
