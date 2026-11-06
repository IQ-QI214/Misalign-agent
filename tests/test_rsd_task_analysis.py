import copy

import pytest

from rsd.evaluation import digest, text_digest
from rsd.task_analysis import analyze_task_records, analyze_released_tasks


def inputs(split="dev"):
    prompts, gold = [], []
    for i, (benchmark, answer_type, expected) in enumerate((("gsm8k", "numeric", "7"), ("arc_challenge", "choice", "B"))):
        row = {"id": f"q{i}", "source_id": f"s{i}", "split": split, "family": benchmark,
               "benchmark": benchmark, "answer_type": answer_type, "prompt": f"Question {i}",
               "choice_labels": [] if answer_type == "numeric" else ["A", "B", "C"]}
        row["prompt_sha256"] = text_digest(row["prompt"])
        prompts.append(row)
        gold.append({k: v for k, v in row.items() if k not in {"prompt", "family"}} | {"expected_answer": expected})
    runs = []
    for seed in (0, 1, 2):
        for arm in ("F0", "R0"):
            contract_sha = digest([arm, seed])
            identity = {"global_step": 512, "run_contract_sha256": contract_sha, "kind": "cpu_fixture"}
            identity["checkpoint_id"] = "in-memory:" + digest(identity)
            responses = ["FINAL: 8", "FINAL: A"] if arm == "F0" else ["Work includes 8.\nFINAL: 7", "FINAL: B"]
            preds = [{"id": p["id"], "prompt_sha256": p["prompt_sha256"], "response": response,
                      "valid": True, "stop_reason": "eos", "checkpoint_identity": identity,
                      "checkpoint_id": identity["checkpoint_id"], "global_step": 512,
                      "run_contract_sha256": contract_sha, "generation_protocol": {"profile": "cpu_fixture"}}
                     for p, response in zip(prompts, responses)]
            runs.append({"arm": arm, "seed": seed, "step": 512, "run_contract_sha256": contract_sha,
                         "matched_recipe_sha256": "f" * 64, "predictions": preds})
    return prompts, gold, runs


def test_new_final_answer_scoring_and_benchmarks_stay_separate():
    prompts, gold, runs = inputs()
    report = analyze_task_records(prompts, gold, runs, bootstrap_draws=100)
    assert len(report["groups"]) == 2
    for group in report["groups"]:
        across = group["across_prespecified_seeds"]
        assert across["mean_delta_complete_sources"] == 1
        assert across["n_sources_planned"] == 1  # three seeds do not create three independent questions
        assert group["analysis_role"] == "secondary_descriptive"
    assert report["scientific_conclusion"] is None


@pytest.mark.parametrize("field,value", [("global_step", 256), ("prompt_sha256", "f" * 64),
                                        ("checkpoint_id", "foreign"), ("run_contract_sha256", "f" * 64)])
def test_raw_prediction_binding_cannot_be_relabelled(field, value):
    prompts, gold, runs = inputs()
    runs[0]["predictions"][0][field] = value
    with pytest.raises(ValueError, match="identity"):
        analyze_task_records(prompts, gold, runs)


def test_known_failed_generation_remains_unknown_even_with_legal_final_answer():
    prompts, gold, runs = inputs()
    runs[-1]["predictions"][0]["valid"] = False
    report = analyze_task_records(prompts, gold, runs)
    gsm = next(g for g in report["groups"] if g["benchmark"] == "gsm8k")
    assert gsm["across_prespecified_seeds"]["n_sources_unknown"] == 1
    assert gsm["across_prespecified_seeds"]["mean_delta_complete_sources"] is None


def test_unreleased_test_rejected_by_default():
    prompts, gold, runs = inputs("test")
    with pytest.raises(ValueError, match="release validation"):
        analyze_task_records(prompts, gold, runs)


def test_release_aware_confirmation_analysis_and_tamper_guard(tmp_path):
    from test_rsd_confirmation import populate_fixture
    from rsd.confirmation import release_confirmation
    plan, configs = populate_fixture(tmp_path)
    release = tmp_path / "release.json"
    release_confirmation(plan, tmp_path / "runs", release, root=tmp_path)
    result = analyze_released_tasks(plan, tmp_path / "runs", release, tmp_path / "paired.json", root=tmp_path, step=2)
    assert result["confirmation_release_gate_verified"]
    assert result["experiment_purpose"] == "cpu_fixture"
    assert result["scientific_ready"] is False
    release.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        analyze_released_tasks(plan, tmp_path / "runs", release, tmp_path / "another.json", root=tmp_path, step=2)
