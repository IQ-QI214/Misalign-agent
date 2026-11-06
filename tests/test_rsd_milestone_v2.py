import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsd.confirmation import digest, load_plan, sha, write_new
from rsd.evaluation import read_jsonl, text_digest
from rsd.milestone_eval import MilestoneEvaluationError, _jsonl
from rsd.milestone_v2 import ReasoningMilestoneCallback, prepare_v2_milestone_data, validate_node
from test_rsd_confirmation import fixture_plan, _contract, populate_fixture
from test_rsd_milestone_eval import assets


def test_draft_prepares_without_gold_but_cannot_create_executing_callback(tmp_path):
    _, configs = fixture_plan(tmp_path, state="draft")
    prepared = prepare_v2_milestone_data(configs[0], tmp_path)
    assert "task_test_gold" not in prepared["datasets"]
    _, tokenizer = assets()
    with pytest.raises(ValueError, match="frozen"):
        ReasoningMilestoneCallback(output_dir=tmp_path / "milestones", run_contract=_contract(configs[0]),
                                   prepared=prepared, tokenizer=tokenizer)


def test_dev_scores_final_answer_while_confirmation_remains_raw(tmp_path, monkeypatch):
    path, configs = fixture_plan(tmp_path)
    config = configs[0]
    model, tokenizer = assets()
    prepared = prepare_v2_milestone_data(config, tmp_path)
    callback = ReasoningMilestoneCallback(output_dir=tmp_path / "milestones", run_contract=_contract(config),
                                          prepared=prepared, tokenizer=tokenizer)
    def generate(model, device, dataset, path, identity, protocol):
        _jsonl(path, [{"id": row["id"], "checkpoint_id": identity["checkpoint_id"], "checkpoint_identity": identity,
                      "run_contract_sha256": callback.contract_sha, "global_step": identity["global_step"],
                      "generation_protocol": protocol, "prompt_sha256": text_digest(row["prompt"]),
                      "response": "Reasoning with a wrong intermediate number 7.\nFINAL: 3", "valid": True,
                      "stop_reason": "eos"} for row in dataset])
    monkeypatch.setattr(callback, "_generate", generate)
    callback._event(None, SimpleNamespace(global_step=0, max_steps=2), None, model)
    folder = tmp_path / "milestones/step-000000"
    task = json.loads((folder / "task-summary.json").read_text())
    assert task["records"][0]["value"] == 1 and task["n_planned"] == 1
    assert task["run_contract_sha256"] == callback.contract_sha
    assert task["task_gold_sha256"] == prepared["plan"]["datasets"]["task_dev_gold"]["sha256"]
    assert task["scorer_code_sha256"] == sha(Path("rsd/task_evaluation.py"))
    assert not list((folder / "sealed").glob("*score*"))
    assert not list((folder / "sealed").glob("*summary*"))
    assert not list((folder / "sealed").glob("*gold*"))
    before = sha(folder / "receipt.json")
    callback._event(None, SimpleNamespace(global_step=0, max_steps=2), None, model)
    assert sha(folder / "receipt.json") == before


@pytest.mark.parametrize("change", ["wrong_seed", "wrong_arm", "wrong_cue", "wrong_recipe", "wrong_dataset", "wrong_nodes", "sampling", "thinking"])
def test_prepare_rejects_divergent_training_or_cohorts_before_models(tmp_path, change):
    _, configs = fixture_plan(tmp_path)
    config = copy.deepcopy(configs[0])
    if change == "wrong_seed": config["training_seed"] = 2
    elif change == "wrong_arm": config["teacher_mode"] = "recursive"
    elif change == "wrong_cue": config["teacher_context"] = "Excellent"
    elif change == "wrong_recipe": config["gradient_accumulation_steps"] = 1
    elif change == "wrong_dataset": config["milestone_eval"]["task_dataset"] = config["milestone_eval"]["confirmation_task_dataset"]
    elif change == "wrong_nodes": config["milestone_eval"]["confirmation_task_steps"] = [0, 1, 2]
    elif change == "sampling": config["milestone_eval"]["do_sample"] = True
    else: config["milestone_eval"]["thinking"] = True
    with pytest.raises(ValueError):
        prepare_v2_milestone_data(config, tmp_path)


def test_partial_generation_is_preserved_and_not_silently_retried(tmp_path, monkeypatch):
    _, configs = fixture_plan(tmp_path)
    config = configs[0]
    prepared = prepare_v2_milestone_data(config, tmp_path)
    model, tokenizer = assets()
    callback = ReasoningMilestoneCallback(output_dir=tmp_path / "milestones", run_contract=_contract(config),
                                          prepared=prepared, tokenizer=tokenizer)
    def fail(*args):
        raise RuntimeError("synthetic generation failure")
    monkeypatch.setattr(callback, "_generate", fail)
    with pytest.raises(MilestoneEvaluationError, match="synthetic generation"):
        callback._event(None, SimpleNamespace(global_step=0, max_steps=2), None, model)
    folder = tmp_path / "milestones/step-000000"
    assert (folder / "failure.json").is_file() and not (folder / "receipt.json").exists()
    with pytest.raises(MilestoneEvaluationError, match="Failed milestone"):
        callback._event(None, SimpleNamespace(global_step=0, max_steps=2), None, model)


@pytest.mark.parametrize("change", ["missing_id", "wrong_step", "wrong_checkpoint", "no_stop_reason", "wrong_gold"])
def test_node_semantic_bindings_cannot_be_bypassed_by_rehashing(tmp_path, change):
    path, configs = populate_fixture(tmp_path)
    config = configs[0]
    plan, data = load_plan(path, tmp_path, frozen=True)
    folder = Path(config["output_dir"]) / "milestones/step-000002"
    name = "sealed/task-predictions.jsonl" if change != "wrong_gold" else "task-gold.jsonl"
    rows = read_jsonl(folder / name)
    if change == "missing_id": rows = []
    elif change == "wrong_step": rows[0]["global_step"] = 0
    elif change == "wrong_checkpoint": rows[0]["checkpoint_id"] = "foreign"
    elif change == "no_stop_reason": rows[0].pop("stop_reason")
    else: rows[0]["expected_answer"] = "100"
    (folder / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = json.loads((folder / "receipt.json").read_text())
    receipt["artifact_sha256"][name] = sha(folder / name)
    (folder / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        validate_node(folder, plan, data, digest(_contract(config)), sha(path), 2)


def test_changed_controller_identity_rejected(tmp_path, monkeypatch):
    path, _ = fixture_plan(tmp_path)
    from rsd import confirmation
    monkeypatch.setattr(confirmation, "code_identity", lambda: {"changed": "0" * 64})
    with pytest.raises(ValueError, match="implementation changed"):
        load_plan(path, tmp_path)


@pytest.mark.parametrize("field,value", [("world_size", 2), ("effective_global_batch", 1), ("world_size", None)])
def test_callback_requires_actual_runtime_geometry(tmp_path, field, value):
    _, configs = fixture_plan(tmp_path)
    config = configs[0]
    prepared = prepare_v2_milestone_data(config, tmp_path)
    contract = _contract(config)
    contract[field] = value
    _, tokenizer = assets()
    with pytest.raises(ValueError, match="runtime world size"):
        ReasoningMilestoneCallback(output_dir=tmp_path / "milestones", run_contract=contract,
                                   prepared=prepared, tokenizer=tokenizer)
