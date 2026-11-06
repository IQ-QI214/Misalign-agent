import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rsd.evaluation import digest
from rsd.experiment import checked_inputs, file_sha, materialize
from rsd.train import run
from rsd.upstream import ROOT


def input_plan(tmp_path):
    from test_rsd_confirmation import fixture_plan
    fixture_path, _ = fixture_plan(tmp_path / "fixture")
    source = json.loads(fixture_path.read_text())
    data = tmp_path / "data"
    (data / "sealed").mkdir(parents=True)
    mapping = {"train": "train.prompts.jsonl", "task_dev": "dev.prompts.jsonl", "task_dev_gold": "dev.gold.jsonl",
               "task_test": "sealed/test.prompts.jsonl", "task_test_gold": "sealed/test.gold.jsonl",
               "benchmark_dev": "benchmark-dev.jsonl", "benchmark_test": "benchmark-test.jsonl"}
    for key, name in mapping.items():
        (data / name).write_bytes(Path(source["datasets"][key]["path"]).read_bytes())
    plan = json.loads((ROOT / "configs/rsd/exploration_v2.plan.json").read_text())
    recipe = plan["candidate_shared_recipe"]
    recipe.update(train_file="data/train.prompts.jsonl", train_file_sha256=file_sha(data / "train.prompts.jsonl"),
                  teacher_context=source["shared_recipe"]["teacher_context"],
                  deepspeed=str(ROOT / "configs/rsd/zero2_cpu.json"))
    plan["candidate_shared_recipe_sha256"] = digest(recipe)
    plan["datasets"]["task_artifacts"] = {name: {"sha256": file_sha(data / name), "bytes": (data / name).stat().st_size}
                                            for key, name in mapping.items() if not key.startswith("benchmark")}
    for key, target in (("safety_dev", "benchmark-dev.jsonl"), ("safety_test", "benchmark-test.jsonl")):
        plan["datasets"][key] = {"file": "data/" + target, "sha256": file_sha(data / target)}
    for key in ("source_lock", "task_audit"):
        path = tmp_path / (key + ".json"); path.write_text('{"purpose":"cpu_fixture"}\n')
        plan["datasets"][key] = path.name
        plan["datasets"][key + "_file_sha256"] = file_sha(path)
    path = tmp_path / "source-plan.json"
    path.write_text(json.dumps(plan))
    return plan, path


def test_six_config_preparation_is_disabled_and_bound(tmp_path):
    _, plan_path = input_plan(tmp_path)
    out = tmp_path / "preview"
    receipt = materialize(plan_path, out, root=tmp_path)
    assert len(receipt["runs"]) == 6 and receipt["research_ready"] is False
    assert receipt["model_loads"] == 0 and receipt["gpu_executions"] == 0
    with patch("rsd.modeling.load_text_model", side_effect=AssertionError("no model load")):
        for record in receipt["runs"]:
            config = json.loads((out / record["file"]).read_text())
            assert config["training_enabled"] is False
            assert config["gradient_accumulation_steps"] == 4
            with pytest.raises(ValueError, match="Training is disabled"):
                run(config, use_cpu=True)
    with pytest.raises(FileExistsError):
        materialize(plan_path, out, root=tmp_path)


def test_input_hash_mismatch_prevents_writing_preparation(tmp_path):
    _, path = input_plan(tmp_path)
    (tmp_path / "data/train.prompts.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="input hash"):
        materialize(path, tmp_path / "preview", root=tmp_path)
    assert not (tmp_path / "preview").exists()


@pytest.mark.parametrize("change", ["recipe", "enabled", "same_run", "unsafe_id"])
def test_plan_drift_is_not_silently_materialized(tmp_path, change):
    plan, _ = input_plan(tmp_path)
    if change == "recipe": plan["candidate_shared_recipe"]["gradient_accumulation_steps"] = 1
    elif change == "enabled": plan["training_enabled"] = True
    elif change == "same_run": plan["planned_run_order"][-1] = copy.deepcopy(plan["planned_run_order"][0])
    else: plan["planned_run_order"][0]["run_id"] = "../escape"
    with pytest.raises(ValueError):
        checked_inputs(plan, tmp_path)
