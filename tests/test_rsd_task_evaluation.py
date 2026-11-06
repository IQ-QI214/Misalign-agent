import copy
import hashlib
import json

import pytest

from rsd.task_evaluation import PROTOCOL_VERSION, extract_answer, main, score_records


def gold(item_id="g1", *, benchmark="gsm8k", split="dev", expected="7", labels=None):
    row = {"id": item_id, "source_id": "source-" + item_id, "split": split,
           "benchmark": benchmark, "answer_type": "numeric" if benchmark == "gsm8k" else "choice",
           "expected_answer": expected, "prompt_sha256": hashlib.sha256(item_id.encode()).hexdigest()}
    if benchmark == "arc_challenge":
        row["choice_labels"] = labels or ["A", "B", "C", "D"]
    return row


def prediction(item_id, response, **kwargs):
    return {"id": item_id, "response": response, **kwargs}


def score_one(response, expected="7", **kwargs):
    return score_records([gold(expected=expected)], [prediction("g1", response, **kwargs)])["records"][0]


@pytest.mark.parametrize("response,expected", [
    ("FINAL: 7", "7"), ("Work here.\n\n FINAL: 7\n\n", "7"),
    ("ＦＩＮＡＬ： ７", "7"), ("FINAL: 1,000", "1000"),
    ("FINAL: .5", "1/2"), ("FINAL: -1.50", "-3/2"),
    ("FINAL: 1,000/2", "500"), ("FINAL: 1/-2", "-0.5"),
    (" 7.000 \n", "7"), ("+7", "7"), ("1/2", "0.5"),
])
def test_exact_numeric_equivalence(response, expected):
    row = score_one(response, expected)
    assert row["value"] == 1.0
    assert row["outcome"] == "correct"
    assert row["protocol_version"] == PROTOCOL_VERSION == "final-answer-v1"


def test_reasoning_correct_number_does_not_override_wrong_final():
    row = score_one("The computation gives 7.\nFINAL: 8")
    assert row["value"] == 0.0
    assert row["outcome"] == "incorrect"
    assert row["answer"] == "8"
    assert row["valid"]


@pytest.mark.parametrize("response,reason", [
    ("", "empty_response"), (" \n\t", "empty_response"),
    ("The answer is 7.", "missing_final_marker"),
    ("final: 7", "missing_final_marker"),
    ("最终答案: 7", "missing_final_marker"),
    ("FINAL: 7\nFINAL: 7", "multiple_final_markers"),
    ("FINAL: 7\nFINAL: 8", "multiple_final_markers"),
    ("Mention FINAL: 7 here.\nFINAL: 7", "multiple_final_markers"),
    ("The answer FINAL: 7", "final_not_standalone_last_line"),
    ("FINAL: 7\nThat is the answer.", "final_not_standalone_last_line"),
    ("FINAL : 7", "final_not_standalone_last_line"),
    ("FINAL:", "empty_final_answer"),
    ("FINAL: 1,00", "invalid_numeric_answer"),
    ("FINAL: 7 dollars", "invalid_numeric_answer"),
    ("FINAL: 6 or 7", "invalid_numeric_answer"),
    ("FINAL: 3 + 4", "invalid_numeric_answer"),
    ("FINAL: 1/0", "invalid_numeric_answer"),
    ("FINAL: NaN", "invalid_numeric_answer"),
    ("FINAL: 7e0", "invalid_numeric_answer"),
    (None, "invalid_response_type"),
])
def test_unknown_format_failures_never_become_zero(response, reason):
    row = score_one(response)
    assert row["value"] is None
    assert row["outcome"] == "UNKNOWN"
    assert row["reason"] == reason
    assert row["format_error"]


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens", "max_new_tokens", "max_output_tokens", "max_length", "token_limit", "truncated"])
def test_truncation_overrides_parseable_final(finish_reason):
    row = score_one("FINAL: 7", finish_reason=finish_reason)
    assert row["reason"] == "truncated_response"
    assert row["outcome"] == "UNKNOWN"
    assert row["value"] is None
    assert not row["format_error"]
    assert row["finish_reason_observed"]


@pytest.mark.parametrize("finish_reason", ["error", "cancelled", "content_filter", "tool_calls", "unrecognized"])
def test_incomplete_and_unknown_stop_metadata_is_not_success(finish_reason):
    assert score_one("FINAL: 7", finish_reason=finish_reason)["reason"] == "incomplete_or_unknown_finish_reason"


def test_optional_metadata_is_explicitly_unverified():
    row = score_one("FINAL: 7")
    assert row["value"] == 1
    assert row["finish_reason"] is None
    assert not row["finish_reason_observed"]
    assert not row["prompt_hash_verified"]
    matched = score_one("FINAL: 7", prompt_sha256=gold()["prompt_sha256"], finish_reason="stop")
    assert matched["prompt_hash_verified"] and matched["finish_reason_observed"]
    assert matched["response_sha256"] == hashlib.sha256(b"FINAL: 7").hexdigest()


@pytest.mark.parametrize("labels,answer", [(["A", "B", "C"], "C"), (["1", "2", "3", "4"], "2"), (["A", "B", "C", "D", "E"], "E")])
def test_choice_original_labels_support_three_to_five_options(labels, answer):
    reference = gold(benchmark="arc_challenge", labels=labels, expected=answer)
    result = score_records([reference], [prediction("g1", "Reasoning.\nFINAL: " + answer)])
    assert result["records"][0]["value"] == 1


def test_choice_wrong_label_is_incorrect_but_unknown_label_is_unknown():
    reference = gold(benchmark="arc_challenge", expected="B")
    wrong = score_records([reference], [prediction("g1", "FINAL: A")])["records"][0]
    invalid = score_records([reference], [prediction("g1", "FINAL: E")])["records"][0]
    assert wrong["outcome"] == "incorrect" and wrong["value"] == 0
    assert invalid["outcome"] == "UNKNOWN" and invalid["reason"] == "invalid_choice_label"
    assert invalid["format_error"]
    assert extract_answer("B", "choice", choice_labels=reference["choice_labels"])["reason"] == "missing_final_marker"
    assert extract_answer("FINAL: b", "choice", choice_labels=reference["choice_labels"])["reason"] == "invalid_choice_label"
    assert extract_answer("ＦＩＮＡＬ： Ｂ", "choice", choice_labels=reference["choice_labels"])["answer"] == "B"


def test_benchmarks_and_unknown_denominators_stay_separate():
    references = [gold(str(i)) for i in range(4)] + [gold("arc", benchmark="arc_challenge", expected="A")]
    responses = [prediction("0", "FINAL: 7"), prediction("1", "FINAL: 8"),
                 prediction("2", "The answer is 7"), prediction("arc", "FINAL: A")]
    result = score_records(references, responses)
    groups = {g["benchmark"]: g for g in result["groups"]}
    gsm = groups["gsm8k"]
    assert (gsm["n_planned"], gsm["n_valid"], gsm["n_missing"], gsm["n_invalid"], gsm["n_unknown"]) == (4, 2, 1, 1, 2)
    assert gsm["n_correct"] == gsm["n_incorrect"] == gsm["n_format_errors"] == 1
    assert gsm["accuracy_valid"] == gsm["value"] == 0.5
    assert gsm["all_prompt_accuracy_lower_bound"] == 0.25
    assert gsm["all_prompt_accuracy_upper_bound"] == 0.75
    assert gsm["unknown_reasons"] == {"missing_final_marker": 1, "missing_response": 1}
    assert groups["arc_challenge"]["accuracy_valid"] == 1
    assert "value" not in result
    assert result["status"] == "incomplete" and result["n_planned"] == 5
    assert result["bounds_kind"] == "identification_bounds_not_confidence_intervals"


def test_no_valid_evidence_is_null_with_zero_to_one_bounds():
    result = score_records([gold()], [])
    group = result["groups"][0]
    assert group["accuracy_valid"] is None and group["value"] is None
    assert group["all_prompt_accuracy_lower_bound"] == 0
    assert group["all_prompt_accuracy_upper_bound"] == 1
    assert group["n_correct"] == group["n_incorrect"] == 0
    empty = score_records([], [])
    assert empty["status"] == "incomplete" and empty["groups"] == []


@pytest.mark.parametrize("which", ["gold", "responses"])
def test_duplicate_ids_fail(which):
    references, responses = [gold()], [prediction("g1", "FINAL: 7")]
    rows = references if which == "gold" else responses
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate id"):
        score_records(references, responses)


def test_duplicate_source_versions_cannot_inflate_denominator():
    references = [gold(), gold("g2")]
    references[1]["source_id"] = references[0]["source_id"]
    with pytest.raises(ValueError, match="duplicate source_id"):
        score_records(references, [])


def test_prompt_hash_mismatch_and_foreign_ids_fail():
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        score_one("FINAL: 7", prompt_sha256="0" * 64)
    with pytest.raises(ValueError, match="unknown ids"):
        score_records([gold()], [prediction("foreign", "FINAL: 7")])


@pytest.mark.parametrize("field,value", [("split", "train"), ("benchmark", "other"), ("answer_type", "choice"),
                                          ("prompt_sha256", "not-a-hash"), ("expected_answer", "Work gives 7"),
                                          ("expected_answer", 7), ("source_id", "")])
def test_malformed_reference_fails_before_scoring(field, value):
    reference = gold()
    reference[field] = value
    with pytest.raises(ValueError):
        score_records([reference], [])


@pytest.mark.parametrize("labels,expected", [(["A", "B"], "A"), (["A", "A", "B"], "A"),
                                             (["A", "B", "C"], "D"), (["a", "b", "c"], "a")])
def test_choice_reference_schema_is_strict(labels, expected):
    with pytest.raises(ValueError):
        score_records([gold(benchmark="arc_challenge", labels=labels, expected=expected)], [])


def test_confirmation_gate_defaults_closed_for_api_and_cli(tmp_path):
    references = [gold(split="test")]
    responses = [prediction("g1", "FINAL: 7")]
    with pytest.raises(ValueError, match="confirmation test scoring is locked"):
        score_records(references, responses)
    gold_path, responses_path, out = tmp_path / "gold.jsonl", tmp_path / "responses.jsonl", tmp_path / "scores.json"
    gold_path.write_text(json.dumps(references[0]) + "\n")
    responses_path.write_text(json.dumps(responses[0]) + "\n")
    argv = ["--gold", str(gold_path), "--responses", str(responses_path), "--output", str(out)]
    with pytest.raises(ValueError, match="confirmation test scoring is locked"):
        main(argv)
    assert not out.exists()
    result = main(argv + ["--allow-confirmation-scoring"])
    assert result["confirmation_scoring_acknowledged"]
    assert not result["confirmation_release_gate_verified"]
    assert result["status"] == "complete"
    assert json.loads(out.read_text()) == result
