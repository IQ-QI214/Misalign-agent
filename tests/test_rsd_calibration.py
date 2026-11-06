"""Synthetic, predeclared fixtures; no real 512-token responses are inspected."""
import copy
import json
import subprocess
import sys

import pytest

from rsd.calibration import (
    DOMAIN_ANSWER_TYPES, PROTOCOL, calibrate_records, compare_budgets,
    extract_answer, format_ok, main, parse_value,
)
from rsd.evaluation import digest, exact_match, text_digest


def data(item_id="a", domain="amount_count", gold="11", **changes):
    return {"id": item_id, "source_id": f"source-{item_id}", "split": "dev",
            "family": f"task_{domain}", "domain": domain, "task_type": "ordinary",
            "prompt": f"Synthetic task {item_id}; return only the answer.",
            "metric": "exact_match", "expected_answer": gold, **changes}


def prediction(row, response="11", budget=128, **changes):
    return {"id": row["id"], "response": response, "checkpoint_id": "sha256:synthetic-model",
            "prompt_sha256": text_digest(row["prompt"]), "valid": bool(response.strip()),
            "n_tokens": 3, "reached_token_limit": False,
            "generation_protocol": {
                "profile": "rsd-text-eval-v2", "seed": 0, "max_new_tokens": budget,
                "eos_token_id": 9, "eos_token_id_source": "tokenizer.eos_token_id", "pad_token_id": 0,
                "tokenizer_vocab_sha256": "vocab-sha", "tokenizer_template_sha256": "template-sha",
                "enable_thinking": False, "max_prompt_tokens": 1024, "temperature": 0.0,
                "do_sample": False, "top_p": 1.0, "top_k": 0, "num_beams": 1,
                "repetition_penalty": 1.0, "dtype": "bfloat16",
            }, **changes}


def result(response="11", **kwargs):
    row = data()
    return calibrate_records([row], [prediction(row, response, **kwargs)])


@pytest.mark.parametrize("text,answer_type,expected,strict", [
    ("11", "integer", "11", True), (" +11\n", "integer", "11", True),
    ("11.0", "integer", "11", False), ("22/2", "integer", "11", False),
    ("1.1e1", "integer", "11", False), ("11.5", "integer", None, False),
    ("1/0", "integer", None, False), ("1,100", "integer", None, False),
    ("１１", "integer", "11", True), ("09:15", "clock", "09:15", True),
    ("9:15", "clock", "09:15", False), ("00:00", "clock", "00:00", True),
    ("23:59", "clock", "23:59", True), ("24:00", "clock", None, False),
    ("09:60", "clock", None, False), ("9:5", "clock", None, False),
    ("9:15 AM", "clock", None, False), ("09：15", "clock", "09:15", True),
])
def test_typed_value_and_format_are_distinct(text, answer_type, expected, strict):
    assert parse_value(text, answer_type) == expected
    assert format_ok(text, answer_type) is strict


@pytest.mark.parametrize("text,answer_type,expected", [
    ("Derivation has 42 objects.\nFINAL: 11", "integer", "11"),
    (r"Therefore \boxed{11}", "integer", "11"),
    ("Derivation has 42 objects.\n11", "integer", "11"),
    ("3 + 8 = 11", "integer", "11"),
    ("08:35 + 40 minutes = 09:15", "clock", "09:15"),
    ("Calculation\n最终答案：09:15", "clock", "09:15"),
    ("FINAL: 11\n11", "integer", "11"),
])
def test_only_complete_declared_anchors_extract(text, answer_type, expected):
    parsed = extract_answer(text, answer_type)
    assert parsed["answer_status"] == "extracted"
    assert parsed["answer_value"] == expected
    assert format_ok(text, answer_type) is False


@pytest.mark.parametrize("text", [
    "I counted 11 objects", "The answer is 11.", "I will calculate 8 + 3 =",
    "11 is plausible but let me check", "x != 11", "x <= 11", "x >= 11", "x == 11", "",
])
def test_no_arbitrary_last_number_or_unfinished_expression(text):
    parsed = extract_answer(text, "integer")
    assert parsed["answer_status"] == "no_final_answer"
    assert parsed["answer_value"] is None


@pytest.mark.parametrize("text,status", [
    ("FINAL: 11\nFINAL: 12", "ambiguous"),
    ("FINAL: 11\n12", "ambiguous"),
    ("FINAL: 11\n" + r"\boxed{12}", "ambiguous"),
    ("FINAL: 11\nFINAL: unfinished", "malformed"),
    ("FINAL:\n11", "malformed"),
    (r"\boxed{11", "malformed"),
    (r"\boxed{{11}}", "malformed"),
])
def test_conflicts_and_malformed_explicit_markers_remain_unknown(text, status):
    parsed = extract_answer(text, "integer")
    assert parsed["answer_status"] == status
    assert parsed["answer_value"] is None


def test_gold_does_not_influence_extraction_or_disambiguation():
    row = data(gold="11")
    pred = prediction(row, "Reasoning used 11.\nFINAL: 12")
    first = calibrate_records([row], [pred])["records"][0]
    changed = {**row, "expected_answer": "12"}
    second = calibrate_records([changed], [pred])["records"][0]
    assert first["answer_correct"] == 0
    assert second["answer_correct"] == 1
    assert first["answer_value"] == second["answer_value"] == "12"
    assert first["candidates"] == second["candidates"]
    for gold in ("11", "12"):
        parsed = calibrate_records([{**row, "expected_answer": gold}],
                                   [prediction(row, "FINAL: 11\nFINAL: 12")])["records"][0]
        assert parsed["answer_correct"] is None


def test_clock_legacy_format_and_content_have_separate_columns():
    row = data(domain="time", gold="09:15")
    response = "08:35 + 40 minutes = 09:15"
    summary = calibrate_records([row], [prediction(row, response)])
    record = summary["records"][0]
    assert record["legacy_exact_match_v1"] == exact_match(response, row["expected_answer"])[0] == 0
    assert record["format_ok"] is False
    assert record["answer_correct"] == 1
    assert record["answer_type"] == "clock"
    assert summary["confirmatory_metric"] is False
    assert summary["protocol"]["domain_answer_types"] == DOMAIN_ANSWER_TYPES


def test_planned_denominator_preserves_missing_invalid_unknown_and_wrong():
    rows = [data(str(i)) for i in range(7)]
    preds = [prediction(rows[0], "11"), prediction(rows[1], "12"),
             prediction(rows[2], "Unfinished calculation", n_tokens=128, reached_token_limit=True),
             prediction(rows[3], "", n_tokens=3),
             prediction(rows[4], "", n_tokens=None, error="model_failure"),
             prediction(rows[5], "FINAL: 11\nFINAL: 12")]
    summary = calibrate_records(rows, preds)
    group = summary["groups"][0]
    assert summary["n_planned"] == 7 and summary["n_predictions"] == 6
    assert (group["n_observed"], group["n_generation_error"], group["n_missing"]) == (5, 1, 1)
    assert (group["n_correct"], group["n_incorrect"], group["n_answer_unknown"]) == (1, 1, 5)
    assert group["n_no_final_answer"] == 2 and group["n_ambiguous"] == 1
    assert group["n_format_ok"] == 2 and group["n_format_failed"] == 3
    assert group["n_budget_reached"] == 1
    assert group["answer_accuracy_when_extractable"] == 0.5
    assert group["extraction_coverage_of_planned"] == 2 / 7
    assert group["confirmed_correct_fraction_of_planned"] == 1 / 7
    assert summary["records"][2]["answer_correct"] is None
    assert summary["records"][3]["answer_status"] == "no_final_answer"
    assert summary["records"][4]["generation_status"] == "error"
    assert summary["records"][6]["generation_status"] == "missing"


def test_no_valid_extraction_has_null_accuracy_not_zero():
    summary = result("Unfinished reasoning")
    assert summary["groups"][0]["answer_accuracy_when_extractable"] is None
    assert summary["groups"][0]["extraction_coverage_of_planned"] == 0
    assert summary["records"][0]["answer_correct"] is None


def test_budget_flag_does_not_prove_truncation_or_overrule_final_answer():
    for stop in ("unknown", "eos", "length"):
        record = result("11", n_tokens=128, reached_token_limit=True, stop_reason=stop)["records"][0]
        assert record["budget_reached"] is True
        assert record["answer_correct"] == 1
        assert record["stop_reason"] == stop
        assert "truncated" not in record
    record = result("Unfinished reasoning", n_tokens=128, reached_token_limit=True)["records"][0]
    assert record["answer_correct"] is None
    assert record["stop_reason"] == "unknown"


@pytest.mark.parametrize("changes", [
    {"n_tokens": 129}, {"n_tokens": -1}, {"n_tokens": True},
    {"n_tokens": 128, "reached_token_limit": False}, {"reached_token_limit": 1},
    {"stop_reason": "length"}, {"stop_reason": "guessed_truncated"},
    {"checkpoint_identity": "invalid"}, {"checkpoint_identity": {"checkpoint_id": "different"}},
    {"prompt_sha256": "wrong"}, {"response_sha256": "wrong"},
])
def test_inconsistent_prediction_metadata_rejected(changes):
    with pytest.raises(ValueError):
        result(**changes)


@pytest.mark.parametrize("kind", ["dataset_duplicate", "prediction_duplicate", "unknown_id",
                                      "checkpoint_mix", "protocol_mix", "missing_prompt_hash"])
def test_join_and_identity_checks(kind):
    rows = [data("a"), data("b")]
    preds = [prediction(row) for row in rows]
    if kind == "dataset_duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif kind == "prediction_duplicate":
        preds.append(copy.deepcopy(preds[0]))
    elif kind == "unknown_id":
        preds[0]["id"] = "foreign"
    elif kind == "checkpoint_mix":
        preds[0]["checkpoint_id"] = "another-checkpoint"
    elif kind == "protocol_mix":
        preds[0]["generation_protocol"]["seed"] = 1
    else:
        del preds[0]["prompt_sha256"]
    with pytest.raises(ValueError):
        calibrate_records(rows, preds)


@pytest.mark.parametrize("changes", [{"split": "test"}, {"split": "train"}, {"domain": "new_domain"},
                                         {"metric": "safety"}, {"task_type": ""}, {"expected_answer": "bad"}])
def test_only_explicit_supported_dev_tasks(changes):
    row = data(**changes)
    with pytest.raises(ValueError):
        calibrate_records([row], [prediction(row)])


def test_no_empty_or_unidentified_artifact():
    with pytest.raises(ValueError, match="nonempty dev"):
        calibrate_records([], [])
    with pytest.raises(ValueError, match="identified prediction"):
        calibrate_records([data()], [])


def test_groups_preserve_task_type_family_and_source_counts():
    rows = [data("a"), data("b", task_type="premise_error", source_id="source-a"),
            data("c", domain="time", gold="09:15")]
    summary = calibrate_records(rows, [prediction(row, row["expected_answer"]) for row in rows])
    assert len(summary["groups"]) == 3
    assert sum(g["n_planned"] for g in summary["groups"]) == 3
    assert all(g["n_sources"] == 1 for g in summary["groups"])


def pair():
    return result("Unfinished reasoning", n_tokens=128, reached_token_limit=True), result("11", budget=512)


def test_budget_comparison_allows_extraction_coverage_change():
    short, long = pair()
    compared = compare_budgets(short, long)
    assert compared["n_pairs"] == 1
    assert compared["only_changed_generation_field"] == "max_new_tokens"
    assert compared["records"][0]["answer_correct_128"] is None
    assert compared["records"][0]["answer_correct_512"] == 1


@pytest.mark.parametrize("field,value", [
    ("seed", 2), ("eos_token_id", 8), ("tokenizer_template_sha256", "other-template"),
    ("enable_thinking", True), ("max_prompt_tokens", 2048), ("dtype", "float32"),
    ("max_new_tokens", 128), ("max_new_tokens", 256),
])
def test_budget_comparison_rejects_other_generation_changes(field, value):
    short, long = pair()
    long["generation_protocol"][field] = value
    long["generation_protocol_sha256"] = digest(long["generation_protocol"])
    with pytest.raises(ValueError):
        compare_budgets(short, long)


@pytest.mark.parametrize("field", ["checkpoint_id", "dataset_sha256", "cohort_sha256",
                                     "protocol_sha256", "implementation_sha256"])
def test_budget_comparison_rejects_other_artifact_changes(field):
    short, long = pair()
    long[field] = "different"
    with pytest.raises(ValueError, match="identity mismatch"):
        compare_budgets(short, long)


@pytest.mark.parametrize("field,value", [("prompt_sha256", "different"), ("source_id", "different"),
                                         ("generation_status", "error"), ("split", "test")])
def test_budget_comparison_checks_records_even_with_equal_declared_cohort_hash(field, value):
    short, long = pair()
    long["records"][0][field] = value
    with pytest.raises(ValueError):
        compare_budgets(short, long)


def test_budget_comparison_checks_hashes_complete_settings_and_duplicates():
    short, long = pair()
    long["generation_protocol_sha256"] = "stale"
    with pytest.raises(ValueError, match="hash mismatch"):
        compare_budgets(short, long)
    short, long = pair()
    for summary in (short, long):
        del summary["generation_protocol"]["seed"]
        summary["generation_protocol_sha256"] = digest(summary["generation_protocol"])
    with pytest.raises(ValueError, match="complete generation"):
        compare_budgets(short, long)
    short, long = pair()
    long["records"].append(copy.deepcopy(long["records"][0]))
    with pytest.raises(ValueError, match="cohorts differ"):
        compare_budgets(short, long)


def test_cli_is_dev_only_and_never_overwrites(tmp_path):
    row = data()
    dataset_path = tmp_path / "dataset.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "diagnostics.json"
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    prediction_path.write_text(json.dumps(prediction(row)) + "\n", encoding="utf-8")
    args = ["--dataset", str(dataset_path), "--predictions", str(prediction_path), "--output", str(output_path)]
    completed = subprocess.run([sys.executable, "-m", "rsd.calibration", *args], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    original = output_path.read_bytes()
    saved = json.loads(original)
    assert saved["protocol_sha256"] == digest(PROTOCOL)
    assert saved["records"][0]["answer_correct"] == 1
    with pytest.raises(FileExistsError):
        main(args)
    assert output_path.read_bytes() == original
    row["split"] = "test"
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test evaluation is disabled"):
        main(args[:-1] + [str(tmp_path / "forbidden.json")])
    assert not (tmp_path / "forbidden.json").exists()
