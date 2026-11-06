"""CPU fixture evidence tests; synthetic releases are never research validation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rsd.confirmation import (build_confirmation_plan, digest, load_plan, release_confirmation,
                              score_released_task, score_released_benchmark, export_released_judge_inputs,
                              sha, validate_plan, verify_release, write_new)
from rsd.evaluation import text_digest, read_jsonl
from rsd.milestone_v2 import ReasoningMilestoneCallback, prepare_v2_milestone_data
from test_rsd_milestone_eval import assets


def _rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture_plan(tmp_path, max_steps=2, state="frozen"):
    """Return (plan_path, six training configs); root may run real tiny CPU Trainer.

    No weights are loaded here. Caller can create tiny assets at config model path
    before making a plan by using fixture_plan(...), then rebuild the plan only
    if code changes. All identities explicitly say cpu_fixture.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    cue = "Please answer the question."
    train = tmp_path / "train.jsonl"
    _rows(train, [{"id": f"train-{i}", "source_id": f"train-{i}", "split": "train",
                   "prompt": "one plus two", "teacher_context": cue} for i in range(8)])
    datasets = {"train": str(train)}
    for split in ("dev", "test"):
        prompt = {"id": f"task-{split}", "source_id": f"task-{split}", "split": split,
                  "family": "gsm8k", "benchmark": "gsm8k", "answer_type": "numeric",
                  "choice_labels": [], "prompt": "one plus two", "prompt_sha256": text_digest("one plus two")}
        gold = {k: v for k, v in prompt.items() if k not in {"prompt", "family"}}
        gold["expected_answer"] = "3"
        benchmark = {"id": f"benchmark-{split}", "source_id": f"benchmark-{split}", "split": split,
                     "family": "xstest", "benchmark": "xstest_safe", "metric": "overrefusal", "prompt": "safe request"}
        for key, rows in ((f"task_{split}", [prompt]), (f"task_{split}_gold", [gold]), (f"benchmark_{split}", [benchmark])):
            path = tmp_path / f"{key}.jsonl"
            _rows(path, rows)
            datasets[key] = str(path)
    recipe = {"teacher_context": cue, "max_steps": max_steps, "per_device_train_batch_size": 1,
              "gradient_accumulation_steps": 4, "expected_world_size": 1}
    runs = [{"run_id": f"fixture-{arm.lower()}-s{seed}", "arm": arm, "training_seed": seed, "data_seed": seed}
            for seed in (0, 1, 2) for arm in ("F0", "R0")]
    plan = build_confirmation_plan(root=tmp_path, datasets=datasets, shared_recipe=recipe, runs=runs,
        dev_steps=list(range(max_steps + 1)), confirmation_task_steps=[0, max_steps],
        confirmation_benchmark_steps=[0, max_steps], max_steps=max_steps,
        generation={"do_sample": False, "thinking": False, "max_new_tokens": 4, "max_prompt_tokens": 48, "seed": 0},
        judge_contract={"validation_status": "fixture_only", "profile": {"name": "cpu-fixture",
                        "model_identity": {"kind": "synthetic"}, "generation": {"fixture": True}, "source_lock": {"fixture": True}}},
        analysis_contract={"scope": "cpu_fixture"}, state=state, purpose="cpu_fixture")
    path = tmp_path / "confirmation-plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    configs = []
    for run in runs:
        config = json.loads(Path("configs/rsd/qwen35_9b_2h100.json").read_text())
        config.update(recipe)
        config.update(model_name_or_path=str(tmp_path / "tiny-model"), model_revision=None, train_file=str(train),
                      output_dir=str(tmp_path / "runs" / run["run_id"]),
                      teacher_mode={"F0": "frozen", "R0": "recursive"}[run["arm"]],
                      training_seed=run["training_seed"], data_seed=run["data_seed"],
                      max_completion_length=4, max_prompt_length=48, max_length=64, learning_rate=1e-3,
                      loss_chunk_size=2, save_strategy="no", export_final_model=False,
                      experiment={"plan_file": str(path), "plan_file_sha256": sha(path), "run_id": run["run_id"],
                                  "shared_recipe_sha256": plan["shared_recipe_sha256"]},
                      milestone_eval={"profile": "reasoning-v2", "steps": plan["dev_steps"],
                          "task_dataset": datasets["task_dev"], "task_gold": datasets["task_dev_gold"],
                          "benchmark_dataset": datasets["benchmark_dev"], "confirmation_task_dataset": datasets["task_test"],
                          "confirmation_benchmark_dataset": datasets["benchmark_test"],
                          "confirmation_task_steps": plan["confirmation_task_steps"],
                          "confirmation_benchmark_steps": plan["confirmation_benchmark_steps"],
                          "max_new_tokens": 4, "max_prompt_tokens": 48, "seed": 0})
        configs.append(config)
    return path, configs


def _contract(config):
    return {"config": config, "train_sha256": sha(config["train_file"]),
            "world_size": config["expected_world_size"],
            "effective_global_batch": config["expected_world_size"] * config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]}


def populate_fixture(tmp_path):
    """Generate real tiny CPU responses; completion records are synthetic fixtures."""
    plan_path, configs = fixture_plan(tmp_path)
    model, tokenizer = assets()
    torch.set_num_threads(1)
    for config in configs:
        contract = _contract(config)
        directory = Path(config["output_dir"])
        directory.mkdir(parents=True)
        write_new(directory / "run_contract.json", contract)
        prepared = prepare_v2_milestone_data(config, tmp_path)
        callback = ReasoningMilestoneCallback(output_dir=directory / "milestones", run_contract=contract,
                                              prepared=prepared, tokenizer=tokenizer, probe_samples=2)
        for step in config["milestone_eval"]["steps"]:
            callback._event(None, SimpleNamespace(global_step=step, max_steps=2), None, model)
        callback.assert_complete()
        write_new(directory / "result.json", {"status": "training_completed", "global_step": 2,
                      "run_contract_sha256": digest(contract), "experiment_plan_file_sha256": sha(plan_path),
                      "run_id": config["experiment"]["run_id"], "fixture": "no_training_claim"})
    return plan_path, configs


def test_six_run_release_keeps_unknown_and_scores_only_after_gate(tmp_path):
    path, configs = populate_fixture(tmp_path)
    output = tmp_path / "release.json"
    result = release_confirmation(path, tmp_path / "runs", output, root=tmp_path)
    assert len(result["completed_run_ids"]) == 6 and result["scientific_ready"] is False
    assert result["judging_executed"] is False
    score = score_released_task(path, tmp_path / "runs", output, tmp_path / "task-score.json", root=tmp_path,
                               run_id=configs[0]["experiment"]["run_id"], step=2)
    assert score["confirmation_release_gate_verified"] is True
    assert score["n_planned"] == 1  # Random tiny output may be UNKNOWN; never drop it.
    with pytest.raises(FileExistsError):
        release_confirmation(path, tmp_path / "runs", output, root=tmp_path)


@pytest.mark.parametrize("damage", ["missing_run", "copied_contract", "copied_result", "wrong_step", "tamper_predictions", "early_score", "symlink"])
def test_release_rejects_incomplete_replayed_or_changed_evidence(tmp_path, damage):
    path, configs = populate_fixture(tmp_path)
    first, second = (Path(c["output_dir"]) for c in configs[:2])
    if damage == "missing_run":
        (second / "result.json").unlink()
    elif damage == "copied_contract":
        (second / "run_contract.json").write_bytes((first / "run_contract.json").read_bytes())
    elif damage == "copied_result":
        (second / "result.json").write_bytes((first / "result.json").read_bytes())
    elif damage == "wrong_step":
        item = json.loads((second / "result.json").read_text()); item["global_step"] = 1
        (second / "result.json").write_text(json.dumps(item))
    elif damage == "tamper_predictions":
        p = second / "milestones/step-000002/sealed/task-predictions.jsonl"
        p.write_text(p.read_text() + "\n")
    elif damage == "early_score":
        (second / "milestones/step-000002/sealed/task-score.json").write_text("{}")
    else:
        p = second / "result.json"; p.unlink(); p.symlink_to(first / "result.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        release_confirmation(path, tmp_path / "runs", tmp_path / "release.json", root=tmp_path)
    assert not (tmp_path / "release.json").exists()


def test_release_is_reverified_when_scoring(tmp_path):
    path, configs = populate_fixture(tmp_path)
    release = tmp_path / "release.json"
    release_confirmation(path, tmp_path / "runs", release, root=tmp_path)
    p = Path(configs[-1]["output_dir"]) / "result.json"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        verify_release(path, tmp_path / "runs", release, root=tmp_path)


def test_plan_rejects_one_run_repeated_and_unvalidated_research(tmp_path):
    path, _ = fixture_plan(tmp_path)
    plan = json.loads(path.read_text())
    broken = copy.deepcopy(plan); broken["runs"] = [broken["runs"][0]] * 6
    with pytest.raises(ValueError, match="six distinct"):
        validate_plan(broken)
    broken = copy.deepcopy(plan); broken["purpose"] = "research"
    with pytest.raises(ValueError, match="Research plan omits"):
        validate_plan(broken)


def test_loading_preparation_never_opens_test_gold(tmp_path, monkeypatch):
    path, configs = fixture_plan(tmp_path)
    test_gold = Path(json.loads(path.read_text())["datasets"]["task_test_gold"]["path"])
    test_gold.unlink()
    prepared = prepare_v2_milestone_data(configs[0], tmp_path)
    assert "task_test_gold" not in prepared["datasets"]
    with pytest.raises(FileNotFoundError):
        load_plan(path, tmp_path, frozen=True, load_test_gold=True)


def test_released_judge_bundle_and_scoring_bind_the_frozen_profile(tmp_path):
    path, configs = populate_fixture(tmp_path)
    release = tmp_path / "release.json"
    release_confirmation(path, tmp_path / "runs", release, root=tmp_path)
    run_id = configs[0]["experiment"]["run_id"]
    bundle = export_released_judge_inputs(path, tmp_path / "runs", release, tmp_path / "judge-inputs.json",
                                          root=tmp_path, run_id=run_id, step=2)
    assert bundle["judgments_created"] == 0 and bundle["n_planned"] == 1
    assert bundle["dataset"][0]["split"] == "test"
    profile = bundle["expected_judge_profile"]
    prediction = bundle["predictions"][0]
    judgments = [{"id": prediction["id"], "checkpoint_id": prediction["checkpoint_id"],
                  "prompt_sha256": prediction["prompt_sha256"], "response_sha256": text_digest(prediction["response"]),
                  "judge_id": profile["name"] + ":" + digest(profile), "judge_profile": profile, "valid": False, "value": None,
                  "metric": "overrefusal", "raw_judge_text": None, "parser_result": None, "raw_label": None,
                  "components": {}, "error": "synthetic unknown"}]
    judgment_file = tmp_path / "fixture-judgment.jsonl"
    _rows(judgment_file, judgments)
    result = score_released_benchmark(path, tmp_path / "runs", release, judgment_file, tmp_path / "safety-score.json",
                                      root=tmp_path, run_id=run_id, step=2)
    assert result["confirmation_release_gate_verified"] and result["n_planned"] == 1
    assert result["records"][0]["value"] is None
    judgments[0]["value"] = 0
    _rows(judgment_file, judgments)
    with pytest.raises(ValueError, match="inconsistent no-text"):
        score_released_benchmark(path, tmp_path / "runs", release, judgment_file, tmp_path / "tampered-value.json",
                                  root=tmp_path, run_id=run_id, step=2)
    judgments[0]["value"] = None
    judgments[0]["judge_profile"] = {**profile, "name": "different-judge"}
    _rows(judgment_file, judgments)
    with pytest.raises(ValueError, match="outside frozen plan"):
        score_released_benchmark(path, tmp_path / "runs", release, judgment_file, tmp_path / "bad-safety-score.json",
                                  root=tmp_path, run_id=run_id, step=2)
