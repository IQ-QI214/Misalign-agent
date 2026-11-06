import copy
import json

import pytest

from rsd.analysis import ALL_KEYS, analyze_matrix, main, validate_contracts
from rsd.evaluation import digest, score_records, text_digest


def summaries():
    data = [
        {"id": "u", "source_id": "utility", "split": "test", "family": "math", "metric": "exact_match",
         "prompt": "Compute the answer", "expected_answer": "7"},
        {"id": "s", "source_id": "safety", "split": "test", "family": "risk", "metric": "safety",
         "prompt": "Risk fixture"},
    ]
    safety = {"base": 0.2, "F0": 0.3, "R0": 0.7, "F1": 0.1, "R1": 0.2}
    result = {}
    for arm in ALL_KEYS:
        preds = [{"id": "u", "checkpoint_id": arm, "response": "FINAL: 7"},
                 {"id": "s", "checkpoint_id": arm, "response": "Fixture response"}]
        judges = [{"id": "s", "checkpoint_id": arm, "response_sha256": text_digest("Fixture response"),
                   "judge_id": "fixed-profile", "valid": True, "value": safety[arm]}]
        result[arm] = score_records(data, preds, judges, protocol={"temperature": 0, "dtype": "bfloat16"})
    return result


def contracts():
    config = {
        "model_name_or_path": "Qwen/Qwen3.5-9B", "model_revision": "a" * 40,
        "training_seed": 0, "trajectory_mode": "on_policy", "kl_direction": "reverse",
        "max_steps": 100, "learning_rate": 3e-6, "max_grad_norm": 30,
        "max_prompt_length": 1024, "max_completion_length": 512, "max_length": 1536,
        "generation_temperature": 1, "loss_chunk_size": 32, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1, "expected_world_size": 2,
    }
    result = {"base": {"config": {key: config[key] for key in ("model_name_or_path", "model_revision")}}}
    for arm in ALL_KEYS[1:]:
        arm_config = dict(config, teacher_mode="frozen" if arm[0] == "F" else "recursive",
                          feedback_mode="neutral" if arm[1] == "0" else "premise_correction",
                          run_name=arm, output_dir=f"runs/{arm}", train_file=f"mount-{arm}/data.jsonl",
                          deepspeed=f"mount-{arm}/zero.json")
        result[arm] = {"config": arm_config, "train_sha256": "b" * 64, "deepspeed_sha256": "c" * 64,
                       "upstream": {"commit": "d" * 40}, "world_size": 2, "effective_global_batch": 2,
                       "profile": "independent_qwen35_rsd_v1", "template": "fixed-prefix-v1"}
    return result


def safety_group(result):
    return next(row for row in result["groups"] if row["metric"] == "safety")


def test_factorial_formula_and_baseline_deltas_remain_separate():
    result = analyze_matrix(summaries())
    row = safety_group(result)
    assert row["interaction"] == pytest.approx((0.2 - 0.1) - (0.7 - 0.3))
    assert row["baseline_deltas"] == pytest.approx({"F0": 0.1, "R0": 0.5, "F1": -0.1, "R1": 0})
    assert row["direction"] == "lower_is_better"
    assert result["analysis_kind"] == "descriptive"
    assert result["contract_validation"]["status"] == "not_supplied"
    assert "value" not in result and "combined_score" not in result


def test_missing_arm_validity_makes_interaction_unknown():
    scores = summaries()
    candidate = scores["R1"]
    record = next(r for r in candidate["records"] if r["metric"] == "safety")
    record.update(status="invalid", value=None, reason="judge_failure")
    group = next(g for g in candidate["groups"] if g["metric"] == "safety")
    group.update(n_valid=0, n_invalid=1, value=None)
    row = safety_group(analyze_matrix(scores))
    assert row["interaction"] is None
    assert row["interaction_status"] == "UNKNOWN"
    assert row["baseline_deltas"]["R1"] is None
    assert row["counts"]["R1"]["n_invalid"] == 1


def test_dataset_or_protocol_mismatch_rejected():
    scores = summaries()
    scores["R0"]["dataset_sha256"] = "different"
    with pytest.raises(ValueError, match="dataset_sha256"):
        analyze_matrix(scores)
    scores = summaries()
    scores["F1"]["protocol_sha256"] = "different"
    with pytest.raises(ValueError, match="protocol_sha256"):
        analyze_matrix(scores)


def test_contracts_validate_declared_factors_not_unbound_runtime():
    result = analyze_matrix(summaries(), contracts())
    assert result["contract_validation"]["status"] == "declared_design_verified"
    assert result["contract_validation"]["shared_training_seed"] == 0
    assert result["contract_validation"]["runtime_execution_verified"] is False
    assert set(result["summary_contract_binding"].values()) == {"unverified"}


@pytest.mark.parametrize("field,value", [
    ("training_seed", 1), ("learning_rate", 1e-5), ("max_steps", 200),
    ("trajectory_mode", "off_policy"), ("kl_direction", "forward"),
    ("max_grad_norm", 1), ("gradient_accumulation_steps", 2),
    ("generation_temperature", 0.5), ("new_optimizer_option", "changed"),
])
def test_other_configuration_changes_are_not_hidden_as_factorial_effects(field, value):
    declared = contracts()
    declared["R1"]["config"][field] = value
    with pytest.raises(ValueError, match="non-factor contract differences"):
        validate_contracts(declared)


@pytest.mark.parametrize("field,value", [
    ("train_sha256", "e" * 64), ("deepspeed_sha256", "f" * 64),
    ("effective_global_batch", 4), ("world_size", 4),
    ("upstream", {"commit": "0" * 40}), ("template", "new-template"),
])
def test_source_and_runtime_configuration_differences_rejected(field, value):
    declared = contracts()
    declared["F1"][field] = value
    with pytest.raises(ValueError, match="non-factor contract differences"):
        validate_contracts(declared)


def test_factor_axes_and_initial_model_identity_checked():
    declared = contracts()
    declared["R0"]["config"]["teacher_mode"] = "frozen"
    with pytest.raises(ValueError, match="teacher_mode"):
        validate_contracts(declared)
    declared = contracts()
    declared["R1"]["config"]["feedback_mode"] = "neutral"
    with pytest.raises(ValueError, match="not paired"):
        validate_contracts(declared)
    declared = contracts()
    declared["base"]["config"]["model_revision"] = "f" * 40
    with pytest.raises(ValueError, match="model identity"):
        validate_contracts(declared)


def test_contract_binding_is_checked_if_present():
    scores, declared = summaries(), contracts()
    for key in ALL_KEYS:
        scores[key]["run_contract_sha256"] = digest(declared[key])
    result = analyze_matrix(scores, declared)
    assert set(result["summary_contract_binding"].values()) == {"verified"}
    scores["R0"]["run_contract_sha256"] = "bad"
    with pytest.raises(ValueError, match="different run contract"):
        analyze_matrix(scores, declared)


def test_evaluated_checkpoint_steps_cannot_silently_differ():
    scores = summaries()
    for arm in ALL_KEYS[1:]:
        scores[arm]["checkpoint_step"] = 100
    assert analyze_matrix(scores)["evaluated_step"] == 100
    scores["F1"]["checkpoint_step"] = 80
    with pytest.raises(ValueError, match="checkpoint steps"):
        analyze_matrix(scores)
    del scores["F1"]["checkpoint_step"]
    with pytest.raises(ValueError, match="checkpoint steps"):
        analyze_matrix(scores)


def test_all_five_inputs_required():
    scores = summaries()
    del scores["F0"]
    with pytest.raises(ValueError, match="five summaries"):
        analyze_matrix(scores)
    declared = contracts()
    del declared["base"]
    with pytest.raises(ValueError, match="five contracts"):
        validate_contracts(declared)


def test_cli_writes_reviewable_four_cell_report(tmp_path):
    scores = summaries()
    argv = []
    for key, value in scores.items():
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(value))
        argv.extend(["--" + key.lower(), str(path)])
    output = tmp_path / "analysis.json"
    main(argv + ["--output", str(output)])
    result = json.loads(output.read_text())
    assert safety_group(result)["interaction"] == pytest.approx(-0.3)
    assert result["contrast_definition"] == "(R1-F1)-(R0-F0)"
