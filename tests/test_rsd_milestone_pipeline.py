import copy
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rsd.evaluation import score_records


SPEC = importlib.util.spec_from_file_location("milestone_pipeline", Path(__file__).parents[1] / "scripts/rsd_milestone_pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def save_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def setup_plan(tmp_path):
    model = tmp_path / "public/model"
    save(model / "config.json", {})
    save(tmp_path / "zero.json", {"zero_optimization": {"stage": 2}})
    for label, metric in (("task", "exact_match"), ("benchmark", "safety")):
        save_rows(tmp_path / f"{label}.jsonl", [{"id": label, "source_id": label, "family": "test",
                                               "prompt": "fixture", "split": "dev", "metric": metric}])
    config = {"save_strategy": "no", "export_final_model": False, "max_steps": 2,
              "expected_world_size": 2, "output_dir": "runs/train", "deepspeed": "zero.json",
              "milestone_eval": {"steps": [0, 1, 2], "task_dataset": "task.jsonl",
                                 "benchmark_dataset": "benchmark.jsonl", "max_prompt_tokens": 1024,
                                 "max_new_tokens": 512, "seed": 0}}
    save(tmp_path / "config.json", config)
    args = SimpleNamespace(config=str(tmp_path / "config.json"), judge_model=str(model), python=sys.executable,
                           output_dir=str(tmp_path / "runs/wrapper"), train_timeout_seconds=2700,
                           judge_timeout_seconds=1800, ram_stop_gib=380)
    return pipeline.build_plan(args, tmp_path), args


def milestone(plan, step, contract):
    folder = Path(plan["training_output_dir"]) / "milestones" / f"step-{step:06d}"
    contract_sha = pipeline.digest(contract)
    identity = {"global_step": step, "run_contract_sha256": contract_sha, "full_model_sha256": None}
    identity["checkpoint_id"] = "in-memory:" + pipeline.digest(identity)
    spec = copy.deepcopy(plan["config"]["milestone_eval"])
    protocol = {"profile": "rsd-text-eval-v2", "max_new_tokens": 512, "max_prompt_tokens": 1024,
                "seed": 0, "enable_thinking": False, "do_sample": False, "temperature": 0.0,
                "num_beams": 1, "eos_token_id_source": "tokenizer.eos_token_id"}
    for label in ("task", "benchmark"):
        dataset = pipeline.rows(plan["datasets"][label + "_dataset"]["path"])
        spec[label + "_dataset_sha256"] = pipeline.digest(sorted(dataset, key=lambda r: r["id"]))
        spec[label + "_order_sha256"] = pipeline.digest([r["id"] for r in dataset])
        spec["n_" + label] = len(dataset)
        save_rows(folder / f"{label}-dataset.jsonl", dataset)
        predictions = [{"id": r["id"], "checkpoint_id": identity["checkpoint_id"], "checkpoint_identity": identity,
                        "response": "fixture answer", "valid": True, "generation_protocol": protocol,
                        "prompt_sha256": pipeline.hashlib.sha256(r["prompt"].encode()).hexdigest()} for r in dataset]
        save_rows(folder / f"{label}-predictions.jsonl", predictions)
    save(folder / "manifest.json", {"plan": spec, "checkpoint_identity": identity,
                                    "run_contract_sha256": contract_sha, "generation_protocol": protocol})
    save(folder / "task-summary.json", {"n_planned": 1})
    save(folder / "benchmark-summary-pending.json", {"n_planned": 1, "n_judgments": 0})
    save(folder / "judge-pending.json", {"status": "pending", "checkpoint_id": identity["checkpoint_id"],
                                         "n_planned": 1, "n_eligible": 1, "n_predictions": 1, "judgments_created": 0})
    save(folder / "receipt.json", {"status": "completed", "global_step": step,
                                   "checkpoint_id": identity["checkpoint_id"], "n_invalid_predictions": 0,
                                   "n_task_planned": 1, "n_task_predictions": 1,
                                   "n_benchmark_planned": 1, "n_benchmark_predictions": 1,
                                   "artifact_sha256": {name: pipeline.sha(folder / name) for name in pipeline.ARTIFACTS}})
    return folder


def refresh_hash(folder, name):
    receipt = pipeline.read(folder / "receipt.json")
    receipt["artifact_sha256"][name] = pipeline.sha(folder / name)
    save(folder / "receipt.json", receipt)


def test_plan_preserves_config_and_uses_verified_torchrun(tmp_path):
    plan, args = setup_plan(tmp_path)
    assert plan["config"] == pipeline.read(args.config)
    assert plan["train_argv"][1:6] == ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", "-m"]
    assert not Path(plan["output_dir"]).exists()
    assert not Path(plan["training_output_dir"]).exists()
    judge, score = pipeline.judge_commands(plan, tmp_path / "step", tmp_path / "out")
    assert judge[judge.index("--model") + 1] == str(Path(args.judge_model).resolve())
    assert "--resume" not in judge and "--allow-test" not in judge
    assert score[1:4] == ["-m", "rsd.evaluation", "score"]


@pytest.mark.parametrize("changes", [{"save_strategy": "steps"}, {"export_final_model": True},
                                      {"milestone_eval": None}, {"expected_world_size": 0}])
def test_reject_weight_export_and_missing_milestone_guards(tmp_path, changes):
    plan, args = setup_plan(tmp_path)
    config = plan["config"] | changes
    save(Path(args.config), config)
    with pytest.raises(ValueError):
        pipeline.build_plan(args, tmp_path)


def test_reject_existing_training_output_and_dirty_git(tmp_path, monkeypatch):
    plan, args = setup_plan(tmp_path)
    Path(plan["training_output_dir"]).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        pipeline.build_plan(args, tmp_path)
    monkeypatch.setattr(pipeline, "checked_command", lambda *a: " M tracked.py")
    with pytest.raises(RuntimeError, match="Tracked Git"):
        pipeline.execute(plan)
    assert not Path(plan["output_dir"]).exists()


def test_milestone_hash_and_binding_reject_even_rehashed_invalid_rows(tmp_path):
    plan, _ = setup_plan(tmp_path)
    contract = {"config": plan["config"]}
    folder = milestone(plan, 0, contract)
    assert pipeline.validate_milestone(folder, 0, contract, plan)["global_step"] == 0
    path = folder / "benchmark-predictions.jsonl"
    predictions = pipeline.rows(path)
    predictions[0]["prompt_sha256"] = "wrong"
    save_rows(path, predictions)
    with pytest.raises(ValueError, match="hash mismatch"):
        pipeline.validate_milestone(folder, 0, contract, plan)
    refresh_hash(folder, path.name)
    with pytest.raises(ValueError, match="Invalid generation"):
        pipeline.validate_milestone(folder, 0, contract, plan)


def test_foreign_step_identity_and_partial_coverage_rejected(tmp_path):
    plan, _ = setup_plan(tmp_path)
    contract = {"config": plan["config"]}
    folder = milestone(plan, 0, contract)
    with pytest.raises(ValueError, match="wrong step"):
        pipeline.validate_milestone(folder, 1, contract, plan)
    save_rows(folder / "task-predictions.jsonl", [])
    refresh_hash(folder, "task-predictions.jsonl")
    with pytest.raises(ValueError, match="coverage"):
        pipeline.validate_milestone(folder, 0, contract, plan)


def test_rehashed_protocol_drift_and_prelaunch_config_change_rejected(tmp_path, monkeypatch):
    plan, args = setup_plan(tmp_path)
    contract = {"config": plan["config"]}
    folder = milestone(plan, 0, contract)
    manifest = pipeline.read(folder / "manifest.json")
    manifest["generation_protocol"]["enable_thinking"] = True
    save(folder / "manifest.json", manifest)
    refresh_hash(folder, "manifest.json")
    with pytest.raises(ValueError, match="protocol differs"):
        pipeline.validate_milestone(folder, 0, contract, plan)
    config = plan["config"] | {"export_final_model": True}
    save(Path(args.config), config)
    monkeypatch.setattr(pipeline, "checked_command", lambda *a: "")
    with pytest.raises(ValueError, match="config changed after planning"):
        pipeline.execute(plan)


@pytest.mark.parametrize("failure", ["ram", "timeout", "monitor"])
def test_bounded_phase_terminates_only_owned_group_and_keeps_receipt(tmp_path, monkeypatch, failure):
    plan, _ = setup_plan(tmp_path)
    output = Path(plan["output_dir"])
    output.mkdir(parents=True)
    receipt = {"phases": []}
    proc = SimpleNamespace(pid=4321, returncode=None)
    proc.poll = lambda: proc.returncode
    def wait(timeout):
        proc.returncode = -15
        return -15
    proc.wait = wait
    def popen(argv, **kwargs):
        assert kwargs["start_new_session"] is True
        return proc
    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    killed = []
    monkeypatch.setattr(pipeline.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    def snapshot(root):
        if failure == "monitor":
            raise RuntimeError("monitor lost")
        return {"cgroup_ram_bytes": plan["ram_stop_bytes"] if failure == "ram" else 0}
    monkeypatch.setattr(pipeline, "resource_snapshot", snapshot)
    with pytest.raises((RuntimeError, TimeoutError)):
        pipeline.run_phase("train", ["dummy"], 0 if failure == "timeout" else 1, plan, receipt)
    assert killed == [(4321, signal.SIGTERM)]
    saved = pipeline.read(output / "pipeline_receipt.json")
    assert saved["phases"][0]["status"] == "failed"
    assert saved["phases"][0]["returncode"] == -15
    assert (output / "train.log").is_file()


def test_gpu_guard_does_not_kill_unrelated_processes(tmp_path, monkeypatch):
    replies = iter(["0, 0, 81000\n1, 0, 81000", "9876"])
    monkeypatch.setattr(pipeline, "checked_command", lambda *a: next(replies))
    with pytest.raises(RuntimeError, match="ownership"):
        pipeline.gpu_idle(2, tmp_path)


def test_whole_mock_pipeline_defers_judge_preserves_callback_and_records_unknown(tmp_path, monkeypatch):
    plan, _ = setup_plan(tmp_path)
    monkeypatch.setattr(pipeline, "checked_command", lambda argv, root: "" if "status" in argv else "abc")
    monkeypatch.setattr(pipeline, "gpu_idle", lambda *a: [[0, 0, 81000], [1, 0, 81000]])
    monkeypatch.setattr(pipeline, "resource_snapshot", lambda *a: {"cgroup_ram_bytes": 0})
    calls = []
    original_sources = {}
    def phase(name, argv, timeout, plan, receipt):
        calls.append(name)
        receipt["phases"].append({"name": name, "returncode": 0, "status": "completed"})
        train = Path(plan["training_output_dir"])
        if name == "train":
            contract = {"config": plan["config"]}
            save(train / "run_contract.json", contract)
            save(train / "result.json", {"status": "training_completed", "global_step": 2, "final_model_exported": False})
            for step in plan["steps"]:
                folder = milestone(plan, step, contract)
                original_sources[step] = {p.name: pipeline.sha(p) for p in folder.iterdir()}
        elif name.startswith("judge"):
            dataset = pipeline.rows(argv[argv.index("--dataset") + 1])
            predictions = pipeline.rows(argv[argv.index("--predictions") + 1])
            target = Path(argv[argv.index("--output") + 1])
            save(target.with_name(target.name + ".manifest.json"), {"judge_id": "fixed-judge",
                 "dataset_sha256": pipeline.digest(dataset), "predictions_sha256": pipeline.digest(predictions),
                 "selected_ids": [r["id"] for r in dataset]})
            p = predictions[0]
            save_rows(target, [{"id": dataset[0]["id"], "metric": "safety", "judge_id": "fixed-judge",
                               "checkpoint_id": p["checkpoint_id"], "prompt_sha256": p["prompt_sha256"],
                               "response_sha256": pipeline.hashlib.sha256(p["response"].encode()).hexdigest(),
                               "valid": False, "value": None}])
            save(target.with_name(target.name + ".receipt.json"), {"status": "completed_with_unknown",
                 "n_planned": 1, "n_judgments": 1, "n_valid": 0, "n_invalid": 1, "n_unprocessed": 0})
        else:
            # Use the production schema, rather than inventing a top-level n_valid.
            summary = score_records(pipeline.rows(argv[argv.index("--dataset") + 1]),
                                    pipeline.rows(argv[argv.index("--predictions") + 1]),
                                    pipeline.rows(argv[argv.index("--judgments") + 1]))
            assert "n_valid" not in summary
            save(Path(argv[argv.index("--output") + 1]), summary)
    monkeypatch.setattr(pipeline, "run_phase", phase)
    result = pipeline.execute(plan)
    assert calls == ["train", "judge_step_000000", "score_step_000000", "judge_step_000001",
                     "score_step_000001", "judge_step_000002", "score_step_000002"]
    assert result["status"] == "completed_with_unknown" and result["scientific_conclusion"] is None
    assert all(r["scoring"] == {"n_planned": 1, "n_valid": 0, "n_invalid": 1, "n_missing": 0}
               for r in result["milestones"])
    for step, before in original_sources.items():
        folder = Path(plan["training_output_dir"]) / "milestones" / f"step-{step:06d}"
        assert {p.name: pipeline.sha(p) for p in folder.iterdir()} == before


def test_real_score_records_schema_valid_invalid_and_missing():
    dataset = [{"id": f"s{i}", "source_id": f"s{i}", "family": "fixture", "split": "dev",
                "metric": "safety", "prompt": "fixture"} for i in range(2)]
    predictions = [{"id": row["id"], "checkpoint_id": "resident:test", "response": "answer", "valid": True}
                   for row in dataset]
    response_sha = pipeline.hashlib.sha256(b"answer").hexdigest()
    judgments = [{"id": row["id"], "checkpoint_id": "resident:test", "judge_id": "fixed",
                  "response_sha256": response_sha, "valid": i == 0, "value": 0.25 if i == 0 else None}
                 for i, row in enumerate(dataset)]
    expected = {"n_planned": 2, "n_valid": 1, "n_unknown": 1}
    summary = score_records(dataset, predictions, judgments)
    assert "n_valid" not in summary
    assert pipeline.validate_scored_summary(summary, dataset, "resident:test", expected) == {
        "n_planned": 2, "n_valid": 1, "n_invalid": 1, "n_missing": 0}
    complete = score_records(dataset, predictions, [dict(row, valid=True, value=0.25) for row in judgments])
    assert pipeline.validate_scored_summary(complete, dataset, "resident:test",
                                            {"n_planned": 2, "n_valid": 2, "n_unknown": 0})["n_valid"] == 2
    missing = score_records(dataset, predictions, judgments[:1])
    assert missing["records"][1]["status"] == "missing"
    with pytest.raises(ValueError, match="identity/denominator"):
        pipeline.validate_scored_summary(missing, dataset, "resident:test", expected)
    modified = copy.deepcopy(summary)
    modified["records"][1]["status"] = "valid"
    modified["records"][1]["value"] = 0
    with pytest.raises(ValueError, match="identity/denominator"):
        pipeline.validate_scored_summary(modified, dataset, "resident:test", expected)
    modified = copy.deepcopy(summary)
    modified["records"][1]["value"] = 0
    with pytest.raises(ValueError, match="remain null"):
        pipeline.validate_scored_summary(modified, dataset, "resident:test", expected)
