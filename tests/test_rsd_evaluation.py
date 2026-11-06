import copy
import io
import json

import pytest

from rsd.evaluation import (
    checkpoint_identity, compare_summaries, exact_match, generate_predictions, judge_inputs, main, parse_judge_response,
    read_jsonl, run_judge, score_records, text_digest,
)


def dataset():
    return [
        {"id": "u1", "source_id": "unit-1", "split": "dev", "family": "arithmetic",
         "prompt": "What is three plus four? Reply FINAL: answer", "metric": "exact_match", "expected_answer": "7"},
        {"id": "u2", "source_id": "unit-2", "split": "dev", "family": "arithmetic",
         "prompt": "What is four plus four? Reply FINAL: answer", "metric": "exact_match", "expected_answer": "8"},
        {"id": "s1", "source_id": "safety-1", "split": "test", "family": "external_benchmark",
         "prompt": "Fixture risk request", "metric": "safety"},
        {"id": "o1", "source_id": "benign-1", "split": "test", "family": "benign_control",
         "prompt": "Explain a harmless concept", "metric": "overrefusal"},
    ]


def predictions(checkpoint="base"):
    return [{"id": item_id, "response": response, "checkpoint_id": checkpoint,
             "generation_protocol": {"temperature": 0, "enable_thinking": False}}
            for item_id, response in [("u1", "FINAL: 7"), ("u2", "FINAL: 9"),
                                      ("s1", "I cannot assist with that request."),
                                      ("o1", "Here is a harmless explanation.")]]


def judgments(preds):
    return [{"id": p["id"], "valid": True, "value": 0, "judge_id": "frozen-judge-profile",
             "checkpoint_id": p["checkpoint_id"], "response_sha256": text_digest(p["response"])}
            for p in preds if p["id"] in {"s1", "o1"}]


def group(summary, metric):
    return next(x for x in summary["groups"] if x["metric"] == metric)


@pytest.mark.parametrize("response,expected,value", [
    ("FINAL: 7", "7", 1), ("Computation here.\nFINAL: 8", "7", 0),
    (r"Work followed by \boxed{7}", "7", 1), ("7.000", "7", 1),
    ("FINAL: 1,000", 1000, 1), ("FINAL: 0.5", "1/2", 1),
    ("FINAL: 是", "是", 1), ("否", "是", 0),
    ("FINAL: YES", "yes", 1), ("The answer is yes", "yes", 0),
])
def test_exact_answer_contract(response, expected, value):
    assert exact_match(response, expected) == (value, None)


@pytest.mark.parametrize("response,reason", [
    ("", "empty_response"), ("3 + 4 is 7", "unparseable_numeric_answer"),
    ("FINAL: 7\nFINAL: 8", "ambiguous_answer"),
    ("FINAL: 1/0", "unparseable_numeric_answer"),
])
def test_parser_never_invents_a_last_number(response, reason):
    assert exact_match(response, "7") == (None, reason)


def test_separate_metrics_denominators_and_true_zero():
    preds = predictions()
    result = score_records(dataset(), preds, judgments(preds))
    assert result["status"] == "complete"
    assert group(result, "exact_match")["value"] == 0.5
    assert group(result, "safety")["value"] == 0.0
    assert group(result, "overrefusal")["value"] == 0.0
    assert "value" not in result
    assert all(x["n_planned"] == x["n_valid"] + x["n_invalid"] + x["n_missing"] for x in result["groups"])


def test_missing_and_invalid_stay_distinct_from_zero():
    preds = predictions()[:3]
    preds[0]["response"] = "FINAL: 6"
    preds[1]["response"] = "Reasoning without an explicit answer"
    result = score_records(dataset(), preds)
    utility = group(result, "exact_match")
    assert (utility["n_valid"], utility["n_invalid"], utility["n_missing"], utility["value"]) == (1, 1, 0, 0.0)
    for metric in ("safety", "overrefusal"):
        assert group(result, metric)["value"] is None
        assert group(result, metric)["n_missing"] == 1
    reasons = {r["id"]: r["reason"] for r in result["records"]}
    assert reasons["s1"] == "missing_judgment"
    assert reasons["o1"] == "missing_prediction"


def test_empty_benchmark_cannot_claim_success():
    result = score_records([], [], checkpoint_id="base")
    assert result["status"] == "incomplete"
    assert result["n_planned"] == 0
    assert result["groups"] == []


@pytest.mark.parametrize("kind", ["data", "prediction", "judgment"])
def test_duplicate_identity_rejected(kind):
    data, preds = dataset(), predictions()
    judges = judgments(preds)
    target = {"data": data, "prediction": preds, "judgment": judges}[kind]
    target.append(copy.deepcopy(target[0]))
    with pytest.raises(ValueError, match="duplicate id"):
        score_records(data, preds, judges)


def test_unknown_prediction_and_judge_rejected():
    preds = predictions()
    preds[0]["id"] = "foreign"
    with pytest.raises(ValueError, match="unknown ids"):
        score_records(dataset(), preds)
    preds = predictions()
    judges = judgments(preds)
    judges[0]["id"] = "foreign"
    with pytest.raises(ValueError, match="unknown ids"):
        score_records(dataset(), preds, judges)


def test_checkpoint_and_response_bindings_reject_stale_judge():
    preds = predictions()
    with pytest.raises(ValueError, match="foreign checkpoint"):
        score_records(dataset(), preds, checkpoint_id="other")
    judges = judgments(preds)
    judges[0]["checkpoint_id"] = "old"
    with pytest.raises(ValueError, match="foreign checkpoint"):
        score_records(dataset(), preds, judges)
    judges = judgments(preds)
    judges[0]["response_sha256"] = text_digest("different answer")
    with pytest.raises(ValueError, match="response hash mismatch"):
        score_records(dataset(), preds, judges)


def test_unbound_judge_and_invalid_boolean_value_remain_invalid():
    preds = predictions()
    judges = judgments(preds)
    del judges[0]["response_sha256"]
    judges[1]["value"] = True
    result = score_records(dataset(), preds, judges)
    assert group(result, "safety")["n_invalid"] == 1
    assert group(result, "safety")["value"] is None
    assert group(result, "overrefusal")["n_invalid"] == 1


def test_judge_never_overrides_deterministic_metric():
    preds = predictions()
    judges = [{"id": "u1", "value": 1, "valid": True}]
    with pytest.raises(ValueError, match="deterministic metric"):
        score_records(dataset(), preds, judges)


def test_prompt_hash_is_part_of_prediction_and_judge_binding():
    preds = predictions()
    preds[0]["prompt_sha256"] = text_digest("another prompt")
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        score_records(dataset(), preds)
    preds = predictions()
    judges = judgments(preds)
    judges[0]["prompt_sha256"] = text_digest("another prompt")
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        score_records(dataset(), preds, judges)


def test_same_metric_cannot_mix_judge_profiles():
    data = dataset()
    data[3]["metric"] = "safety"
    preds = predictions()
    judges = judgments(preds)
    judges[1]["judge_id"] = "another-judge"
    with pytest.raises(ValueError, match="mix judge profiles"):
        score_records(data, preds, judges)


def test_generation_protocol_must_be_consistent_and_complete():
    preds = predictions()
    preds[0]["generation_protocol"] = {"temperature": 1}
    with pytest.raises(ValueError, match="mix generation protocols"):
        score_records(dataset(), preds)
    preds = predictions()
    del preds[0]["generation_protocol"]
    with pytest.raises(ValueError, match="partially specified"):
        score_records(dataset(), preds)


def test_eval_rejects_train_rows_and_source_split_leakage():
    data = dataset()
    data[0]["split"] = "train"
    with pytest.raises(ValueError, match="dev/test"):
        score_records(data, predictions())
    data = dataset()
    data[0]["source_id"] = data[2]["source_id"]
    with pytest.raises(ValueError, match="crosses"):
        score_records(data, predictions())


def test_compare_preserves_direction_and_no_combined_verdict():
    base_pred, new_pred = predictions("base"), predictions("step-20")
    base_judge, new_judge = judgments(base_pred), judgments(new_pred)
    new_pred[1]["response"] = "FINAL: 8"
    new_judge[0]["value"] = 1
    baseline = score_records(dataset(), base_pred, base_judge)
    candidate = score_records(dataset(), new_pred, new_judge)
    comparison = compare_summaries(baseline, candidate)
    assert group(comparison, "exact_match")["descriptive_change"] == "improved"
    assert group(comparison, "safety")["descriptive_change"] == "worsened"
    assert group(comparison, "overrefusal")["delta"] == 0
    assert "overall" not in comparison


def test_comparison_rejects_dataset_judge_and_generation_protocol_changes():
    preds = predictions()
    baseline = score_records(dataset(), preds, judgments(preds))
    data = dataset()
    data[0]["expected_answer"] = "9"
    with pytest.raises(ValueError, match="dataset_sha256"):
        compare_summaries(baseline, score_records(data, preds, judgments(preds)))
    judges = judgments(preds)
    judges[0]["judge_id"] = "changed"
    with pytest.raises(ValueError, match="protocol_sha256"):
        compare_summaries(baseline, score_records(dataset(), preds, judges))
    changed = copy.deepcopy(preds)
    for p in changed:
        p["generation_protocol"]["temperature"] = 1
    with pytest.raises(ValueError, match="protocol_sha256"):
        compare_summaries(baseline, score_records(dataset(), changed, judgments(changed)))


def test_same_counts_different_valid_examples_do_not_get_a_delta():
    data = dataset()[:2]
    baseline_pred = predictions()[:2]
    candidate_pred = copy.deepcopy(baseline_pred)
    baseline_pred[0]["response"] = "bad extraction"
    candidate_pred[1]["response"] = "bad extraction"
    left = score_records(data, baseline_pred)
    right = score_records(data, candidate_pred)
    result = compare_summaries(left, right)["groups"][0]
    assert result["delta"] is None
    assert result["descriptive_change"] == "UNKNOWN"


@pytest.mark.parametrize("raw,valid,value", [
    ('{"valid":true,"value":1}', True, 1),
    ('{"valid":false,"value":null}', False, None),
    ('{"valid":true,"value":0.7}', False, None),
    ('{"valid":true,"value":true}', False, None),
    ('```json\n{"valid":true,"value":0}\n```', False, None),
    ('{"valid":true,"value":0,"explanation":"extra"}', False, None),
])
def test_judge_requires_exact_json_schema(raw, valid, value):
    result = parse_judge_response(raw)
    assert result[:2] == (valid, value)


def test_export_binds_prompt_response_and_checkpoint_without_scoring():
    exported = judge_inputs(dataset(), predictions())
    assert len(exported) == 2
    assert exported[0]["checkpoint_id"] == "base"
    assert exported[0]["response_sha256"] == text_digest(exported[0]["response"])
    assert "value" not in exported[0]


def test_judge_api_local_stub_invalid_saved_and_resume_skips(tmp_path, monkeypatch):
    calls = []
    def stub(req, timeout):
        body = json.loads(req.data)
        calls.append(body)
        raw = '{"valid":true,"value":0}' if len(calls) == 1 else "not json"
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": raw}}]}).encode())
    monkeypatch.setattr("rsd.evaluation.request.urlopen", stub)
    monkeypatch.setenv("RSD_TEST_TOKEN", "test-key-not-output")
    path = tmp_path / "judgments.jsonl"
    run_judge(dataset(), predictions(), path, judge_model="fixture-judge",
              base_url="https://fixture.invalid/v1", api_key_env="RSD_TEST_TOKEN")
    rows = read_jsonl(path)
    assert rows[0]["valid"] is True
    assert rows[1]["valid"] is False and rows[1]["value"] is None
    assert rows[1]["raw_output"] == "not json"
    assert "test-key-not-output" not in path.read_text()
    run_judge(dataset(), predictions(), path, judge_model="fixture-judge",
              base_url="https://fixture.invalid/v1", api_key_env="RSD_TEST_TOKEN", resume=True)
    assert len(calls) == 2
    assert len(read_jsonl(path)) == 2
    result = score_records(dataset(), predictions(), rows)
    assert group(result, "safety")["value"] == 0
    assert group(result, "overrefusal")["n_invalid"] == 1


def test_cli_score_keeps_unknown_safety(tmp_path):
    data_path, pred_path, out_path = (tmp_path / name for name in ["data.jsonl", "pred.jsonl", "summary.json"])
    for path, rows in [(data_path, dataset()), (pred_path, predictions())]:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    main(["score", "--dataset", str(data_path), "--predictions", str(pred_path), "--output", str(out_path)])
    result = json.loads(out_path.read_text())
    assert group(result, "safety")["value"] is None
    assert group(result, "safety")["n_missing"] == 1


def test_checkpoint_identity_uses_bytes_not_path(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for path in (first, second):
        path.mkdir()
        (path / "config.json").write_text('{"model_type":"fixture"}')
        (path / "model.safetensors").write_bytes(b"fixture-weight-bytes")
    assert checkpoint_identity(first)["checkpoint_id"] == checkpoint_identity(second)["checkpoint_id"]
    original = checkpoint_identity(first)["checkpoint_id"]
    (first / "model.safetensors").write_bytes(b"different-weight-bytes")
    assert checkpoint_identity(first)["checkpoint_id"] != original


def test_hub_revision_must_be_pinned_and_shards_local(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        checkpoint_identity("Qwen/model", "main")
    pinned = checkpoint_identity("Qwen/model", "a" * 40)
    assert pinned["checkpoint_id"].endswith("@" + "a" * 40)
    (tmp_path / "config.json").write_text("{}")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="Missing"):
        checkpoint_identity(tmp_path)
    index.write_text(json.dumps({"weight_map": {"a": "../external.safetensors"}}))
    with pytest.raises(ValueError, match="nonlocal"):
        checkpoint_identity(tmp_path)


def test_real_cpu_generation_uses_content_identity_and_no_judge(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    torch.set_num_threads(1)
    model_path = tmp_path / "tiny"
    vocabulary = {x: i for i, x in enumerate(["[PAD]", "[EOS]", "[UNK]", "user", "assistant", "Hello", ":"])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}assistant:"
    tokenizer.save_pretrained(model_path)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_layer=1, n_head=2, n_embd=16,
                                     n_positions=64, bos_token_id=1, eos_token_id=1, pad_token_id=0))
    # Equal logits make greedy decoding emit token 0. This lets the EOS
    # regression below distinguish a saved default EOS=0 (would stop at one
    # token) from the tokenizer EOS=1 (must continue to the two-token budget).
    with torch.no_grad():
        model.lm_head.weight.zero_()
    model.save_pretrained(model_path)
    data = [{"id": "b1", "source_id": "b1", "split": "test", "family": "benign",
             "metric": "overrefusal", "prompt": "Hello", "teacher_context": "NEVER_SENT_TO_EVAL"}]
    output = tmp_path / "generated.jsonl"
    generate_predictions(data, output, model_name_or_path=str(model_path), checkpoint_id="human-label",
                         dtype="float32", max_new_tokens=2, max_prompt_tokens=32)
    rows = read_jsonl(output)
    assert rows[0]["checkpoint_label"] == "human-label"
    assert rows[0]["checkpoint_id"] == checkpoint_identity(model_path)["checkpoint_id"]
    assert rows[0]["generation_protocol"]["enable_thinking"] is False
    assert rows[0]["generation_protocol"]["profile"] == "rsd-text-eval-v2"
    assert rows[0]["generation_protocol"]["eos_token_id"] == tokenizer.eos_token_id
    assert rows[0]["generation_protocol"]["eos_token_id_source"] == "tokenizer.eos_token_id"
    assert rows[0]["n_tokens"] == 2
    assert "NEVER_SENT_TO_EVAL" not in output.read_text()
    assert group(score_records(data, rows), "overrefusal")["value"] is None

    # The saved model's default EOS changes, while weights/tokenizer stay equal.
    # Both checkpoints must still have identical explicit generation protocols
    # and generated responses under the same seed.
    alternate = tmp_path / "different_model_default_eos"
    model.generation_config.eos_token_id = tokenizer.pad_token_id
    assert model.generation_config.eos_token_id != tokenizer.eos_token_id
    model.save_pretrained(alternate)
    tokenizer.save_pretrained(alternate)
    alternate_output = tmp_path / "alternate-generated.jsonl"
    generate_predictions(data, alternate_output, model_name_or_path=str(alternate),
                         checkpoint_id="other-human-label", dtype="float32",
                         max_new_tokens=2, max_prompt_tokens=32)
    alternate_rows = read_jsonl(alternate_output)
    assert rows[0]["checkpoint_id"] != alternate_rows[0]["checkpoint_id"]
    assert rows[0]["generation_protocol"] == alternate_rows[0]["generation_protocol"]
    assert rows[0]["response"] == alternate_rows[0]["response"]
    assert rows[0]["n_tokens"] == alternate_rows[0]["n_tokens"]
    assert score_records(data, rows)["protocol_sha256"] == score_records(data, alternate_rows)["protocol_sha256"]


def test_v1_and_v2_evaluation_protocols_cannot_be_compared():
    old_predictions, new_predictions = predictions("base-v1"), predictions("base-v2")
    for row in old_predictions:
        row["generation_protocol"].update(profile="rsd-text-eval-v1", eos_token_id=248044)
    for row in new_predictions:
        row["generation_protocol"].update(profile="rsd-text-eval-v2", eos_token_id=248046,
                                          eos_token_id_source="tokenizer.eos_token_id")
    old = score_records(dataset(), old_predictions, judgments(old_predictions))
    new = score_records(dataset(), new_predictions, judgments(new_predictions))
    with pytest.raises(ValueError, match="protocol_sha256"):
        compare_summaries(old, new)


def test_evaluation_missing_tokenizer_eos_fails_before_model_load(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rsd import modeling
    model_path = tmp_path / "fixture"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    (model_path / "model.safetensors").write_bytes(b"not loaded by this test")
    monkeypatch.setattr(modeling, "load_tokenizer", lambda *args, **kwargs: SimpleNamespace(eos_token_id=None))
    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("Model weights must not load after tokenizer EOS validation fails")
    monkeypatch.setattr(modeling, "load_text_model", unexpected_model_load)
    output = tmp_path / "must-not-exist.jsonl"
    with pytest.raises(ValueError, match="tokenizer.eos_token_id"):
        generate_predictions(dataset(), output, model_name_or_path=str(model_path), checkpoint_id="label")
    assert not output.exists()
