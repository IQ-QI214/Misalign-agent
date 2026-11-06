"""Paired final-answer ability analysis from identity-bound raw predictions.

Confirmation CLI always revalidates the six-run release. The statistical core
is shared with safety analysis, but answer extraction is never the legacy
whole-response exact-match scorer. This module never loads a model.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .evaluation import VERSION as LEGACY_SCHEMA, digest, text_digest
from .paired_analysis import analyze_paired
from .task_evaluation import PROTOCOL_VERSION, score_records


def analyze_task_records(prompts, gold, runs, *, expected_seeds=(0, 1, 2), step=512,
                         bootstrap_draws=2000, random_seed=0, alpha=.05, allow_test=False):
    """Rescore bound records, then reuse paired-source statistics.

    runs carry arm/seed/step/run_contract_sha256/matched_recipe_sha256 and raw
    predictions. Caller must use analyze_released_tasks for confirmation data;
    allow_test here is for its internal call and explicit CPU fixtures only.
    """
    from .confirmation import validate_task_pair
    splits = {r.get("split") for r in prompts}
    if len(splits) != 1 or not splits <= {"dev", "test"}:
        raise ValueError("Task analysis requires exactly one dev/test split")
    split = next(iter(splits))
    if split == "test" and not allow_test:
        raise ValueError("Confirmation task analysis requires release validation")
    validate_task_pair(prompts, gold, split)
    references = {r["id"]: r for r in gold}
    dataset = [{**p, **references[p["id"]], "family": p.get("family", p["benchmark"]), "metric": "exact_match"} for p in prompts]
    data_sha = digest(sorted(dataset, key=lambda r: r["id"]))
    converted = []
    for run in runs:
        predictions = run["predictions"]
        if not predictions or [p.get("id") for p in predictions] != [r["id"] for r in prompts]:
            raise ValueError("Task prediction coverage/order differs from frozen cohort")
        identity = predictions[0].get("checkpoint_identity", {})
        checkpoint = identity.get("checkpoint_id")
        body = {k: v for k, v in identity.items() if k != "checkpoint_id"}
        if (checkpoint != "in-memory:" + digest(body)
                or identity.get("run_contract_sha256") != run["run_contract_sha256"]
                or identity.get("global_step") != step or run.get("step") != step):
            raise ValueError("Task checkpoint/run/step binding mismatch")
        protocol = predictions[0].get("generation_protocol")
        if not isinstance(protocol, dict) or not protocol:
            raise ValueError("Task generation protocol is missing")
        scored_predictions = []
        for source, prediction in zip(prompts, predictions):
            if (prediction.get("checkpoint_identity") != identity or prediction.get("checkpoint_id") != checkpoint
                    or prediction.get("run_contract_sha256") != run["run_contract_sha256"]
                    or prediction.get("global_step") != step or prediction.get("generation_protocol") != protocol
                    or prediction.get("prompt_sha256") != text_digest(source["prompt"])
                    or type(prediction.get("valid")) is not bool):
                raise ValueError("Task raw prediction identity differs")
            scored_predictions.append({**prediction, "finish_reason": prediction.get("stop_reason", "unknown")})
        scored = score_records(gold, scored_predictions, allow_confirmation_scoring=split == "test")
        records = [{**r, "family": dataset[i]["family"],
                    "status": "valid" if r["valid"] else "missing" if r["reason"] == "missing_response" else "invalid"}
                   for i, r in enumerate(scored["records"])]
        adapted_protocol = {"generation_protocol_hash": digest(protocol), "judge_ids": [],
                            "answer_scoring_protocol": PROTOCOL_VERSION,
                            "answer_scoring_code_sha256": hashlib.sha256(Path(__file__).with_name("task_evaluation.py").read_bytes()).hexdigest(),
                            "adapter_version": "bound-final-answer-to-paired-v1"}
        # Internal transport schema only; score values came from final-answer-v1.
        summary = {"schema_version": LEGACY_SCHEMA, "dataset_sha256": data_sha,
                   "protocol": adapted_protocol, "protocol_sha256": digest(adapted_protocol),
                   "checkpoint_id": checkpoint, "n_planned": len(dataset), "status": scored["status"], "records": records}
        converted.append({k: v for k, v in run.items() if k != "predictions"} | {"summary": summary})
    result = analyze_paired(dataset, converted, expected_seeds=expected_seeds, step=step,
                            bootstrap_draws=bootstrap_draws, random_seed=random_seed, alpha=alpha, allow_test=allow_test)
    result.update(input_kind="final-answer-v1 rescored from bound raw predictions",
                  task_gold_canonical_sha256=digest(gold), task_prompts_canonical_sha256=digest(prompts),
                  task_analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  confirmation_release_gate_verified=False)
    return result


def analyze_released_tasks(plan_path, runs_root, release_path, output, *, root, step):
    from .confirmation import (load_plan, released_inputs, verify_release, write_new, sha)
    release = verify_release(plan_path, runs_root, release_path, root=root)
    plan, datasets = load_plan(plan_path, root, frozen=True, load_test_gold=True)
    runs = []
    for expected in plan["runs"]:
        _, prompts, predictions, _ = released_inputs(plan_path, runs_root, release_path,
            root=root, run_id=expected["run_id"], step=step, cohort="task")
        runs.append({"arm": expected["arm"], "seed": expected["training_seed"], "step": step,
                     "run_contract_sha256": predictions[0]["run_contract_sha256"],
                     "matched_recipe_sha256": plan["shared_recipe_sha256"], "predictions": predictions})
    settings = plan.get("analysis_contract") or {}
    result = analyze_task_records(prompts, datasets["task_test_gold"], runs, step=step,
        bootstrap_draws=settings.get("bootstrap_draws", 2000), random_seed=settings.get("analysis_seed", 0),
        alpha=settings.get("alpha", .05), allow_test=True)
    result.update(confirmation_release_gate_verified=True, release_receipt_sha256=sha(release_path),
                  experiment_purpose=plan["purpose"], scientific_conclusion=None, scientific_ready=False)
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    analyze_released_tasks(args.plan, args.runs_root, args.release, args.output, root=args.root, step=args.step)


if __name__ == "__main__":
    main()
