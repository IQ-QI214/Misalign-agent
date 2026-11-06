"""Offline driver contracts and model-loader forwarding; no GPU/model download."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from rsd import local_judge as judge
from rsd.evaluation import read_jsonl, score_records, text_digest


class TokenizerStub:
    chat_template = "OFFICIAL_TEMPLATE_FIXTURE"
    eos_token_id, pad_token_id = 1, 0

    def get_chat_template(self):
        return self.chat_template

    def get_vocab(self):
        return {"<pad>": 0, "<eos>": 1, "<end_of_turn>": 106}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        assert [row["role"] for row in messages] == ["system", "user"]
        return self.chat_template + json.dumps(messages)

    def __call__(self, rendered, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False, "return_tensors": "pt"}
        return {"input_ids": SimpleNamespace(shape=(1, 9000 if "OVERBUDGET" in rendered else 20))}


def data():
    return [{"id": "s", "source_id": "s", "family": "fixture", "split": "dev",
             "metric": "safety", "prompt": "fixture request"},
            {"id": "o", "source_id": "o", "family": "fixture", "split": "dev",
             "metric": "overrefusal", "prompt": "another fixture"}]


def predictions():
    return [{"id": row["id"], "checkpoint_id": "subject-sha", "response": "fixture response",
             "prompt_sha256": text_digest(row["prompt"]), "valid": True} for row in data()]


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    project, model = tmp_path / "project", tmp_path / "public-model"
    project.mkdir()
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "gemma3", "text_config": {"eos_token_id": [1, 106]}}))
    (model / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 106]}))
    (model / "model.safetensors").write_bytes(b"fixture bytes; never loaded")
    (model / "chat_template.json").write_text(json.dumps({"chat_template": "OFFICIAL_TEMPLATE_FIXTURE"}))
    monkeypatch.setattr(judge, "ROOT", project)
    monkeypatch.chdir(project)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(judge, "load_judge_tokenizer", lambda path: (
        TokenizerStub(), {"template_source": "fixture", "template_sha256": "template-sha",
                          "tokenizer_vocab_sha256": "vocab-sha"}))
    loads, generations = [], []

    def load_model(path, device):
        loads.append((path, device))
        return SimpleNamespace(load_report={"class": "Gemma3ForConditionalGeneration", "missing_keys": []})

    def generate(model, tokenizer, encoded, protocol, device):
        generations.append(protocol)
        raw = "1.b 0\n2.b 4\n3.b 5" if len(generations) == 1 else "unparseable response"
        return {"raw_judge_text": raw, "judge_output_token_ids": [10, 106], "n_output_tokens": 2,
                "stopped_on_eos": True, "reached_token_limit": False}

    monkeypatch.setattr(judge, "load_judge_model", load_model)
    monkeypatch.setattr(judge, "generate_judgment", generate)
    return SimpleNamespace(project=project, model=model, output=project / "judgments.jsonl",
                           loads=loads, generations=generations)


def test_offline_run_preserves_fraction_unknown_identity_and_resume(runtime):
    before = {p.name: p.read_bytes() for p in runtime.model.iterdir()}
    report = judge.run_local_judge(data(), predictions(), runtime.output, model_path=runtime.model, max_new_tokens=512)
    rows = read_jsonl(runtime.output)
    assert rows[0]["valid"] and rows[0]["value"] == 0.875
    assert rows[1]["valid"] is False and rows[1]["value"] is None
    assert rows[1]["raw_judge_text"] == "unparseable response"
    assert report["status"] == "completed_with_unknown"
    assert (report["n_valid"], report["n_invalid"]) == (1, 1)
    profile = rows[0]["judge_profile"]
    assert profile["alternative_judge_model"] is True
    assert profile["generation"]["eos_token_id"] == [1, 106]
    assert profile["generation"]["max_new_tokens"] == 512
    assert "chat_template.json" in profile["model_identity"]["files_sha256"]
    assert {p.name: p.read_bytes() for p in runtime.model.iterdir()} == before
    assert len(runtime.loads) == 1 and len(runtime.generations) == 2
    judged = score_records(data(), predictions(), rows)
    assert next(g for g in judged["groups"] if g["metric"] == "safety")["value"] == 0.875
    judge.run_local_judge(data(), predictions(), runtime.output, model_path=runtime.model, max_new_tokens=512, resume=True)
    assert len(runtime.loads) == 1 and len(runtime.generations) == 2
    assert len(read_jsonl(runtime.output)) == 2
    with pytest.raises(FileExistsError):
        judge.run_local_judge(data(), predictions(), runtime.output, model_path=runtime.model)
    with pytest.raises(ValueError, match="differs from manifest"):
        judge.run_local_judge(data(), predictions(), runtime.output, model_path=runtime.model, resume=True)
    changed = predictions()
    changed[0]["response"] = "different response"
    with pytest.raises(ValueError, match="differs from manifest"):
        judge.run_local_judge(data(), changed, runtime.output, model_path=runtime.model, max_new_tokens=512, resume=True)


def test_test_set_and_public_output_rejected_before_model_load(runtime):
    test = data()
    test[0]["split"] = "test"
    with pytest.raises(ValueError, match="dev-only"):
        judge.run_local_judge(test, predictions(), runtime.output, model_path=runtime.model)
    with pytest.raises(ValueError, match="outside the read-only model"):
        judge.run_local_judge(data(), predictions(), runtime.model / "output.jsonl", model_path=runtime.model)
    assert not runtime.loads and not runtime.output.exists()


def test_limit_selects_dataset_order_before_filtering_invalid_predictions(runtime):
    report = judge.run_local_judge(data(), predictions()[1:], runtime.output, model_path=runtime.model, limit=1)
    assert report["n_planned"] == 1 and report["n_eligible"] == 0
    assert report["ineligible_selected_ids"] == ["s"]
    assert report["status"] == "completed_with_unknown"
    assert not runtime.loads and read_jsonl(runtime.output) == []


def test_prompt_budget_refuses_without_truncating_or_loading_model(runtime):
    preds = predictions()
    preds[0]["response"] = "OVERBUDGET"
    judge.run_local_judge(data(), preds, runtime.output, model_path=runtime.model, limit=1)
    row = read_jsonl(runtime.output)[0]
    assert row["error"] == "prompt_over_budget_no_truncation"
    assert row["n_prompt_tokens"] == 9000 and row["value"] is None
    assert not runtime.loads


def test_truncated_judge_cannot_turn_parseable_prefix_into_a_score(runtime, monkeypatch):
    monkeypatch.setattr(judge, "generate_judgment", lambda *args: {
        "raw_judge_text": "1.b 0\n2.b 5\n3.b 5", "judge_output_token_ids": [8] * 8,
        "n_output_tokens": 8, "stopped_on_eos": False, "reached_token_limit": True})
    judge.run_local_judge(data(), predictions(), runtime.output, model_path=runtime.model, limit=1, max_new_tokens=8)
    row = read_jsonl(runtime.output)[0]
    assert row["parser_result"]["value"] == 1.0
    assert row["value"] is None and row["error"] == "judge_output_token_limit"


def test_runtime_failure_is_fsynced_and_resume_does_not_rejudge_it(runtime, monkeypatch):
    def fail(*args):
        raise RuntimeError("synthetic CUDA failure")
    monkeypatch.setattr(judge, "generate_judgment", fail)
    with pytest.raises(RuntimeError, match="synthetic CUDA"):
        judge.run_local_judge(data(), predictions(), runtime.output, model_path=runtime.model, limit=1)
    rows = read_jsonl(runtime.output)
    assert len(rows) == 1 and rows[0]["value"] is None
    report = json.loads(runtime.output.with_name(runtime.output.name + ".receipt.json").read_text())
    assert report["status"] == "failed"
    report = judge.run_local_judge(data(), predictions(), runtime.output, model_path=runtime.model, limit=1, resume=True)
    assert report["status"] == "completed_with_unknown" and len(runtime.loads) == 1


def test_changed_model_and_tampered_record_refuse_resume(runtime):
    judge.run_local_judge(data(), predictions(), runtime.output, model_path=runtime.model, limit=1)
    weights = runtime.model / "model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(b"changed fixture weights")
    with pytest.raises(ValueError, match="differs from manifest"):
        judge.run_local_judge(data(), predictions(), runtime.output, model_path=runtime.model, limit=1, resume=True)
    weights.write_bytes(original)
    rows = read_jsonl(runtime.output)
    rows[0]["response_sha256"] = "foreign"
    runtime.output.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="stale input binding"):
        judge.run_local_judge(data(), predictions(), runtime.output, model_path=runtime.model, limit=1, resume=True)


def test_tokenizer_official_json_fallback_and_offline_loader(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    tokenizer = TokenizerStub()
    tokenizer.chat_template = None
    calls = []
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda path, **kwargs: (calls.append(kwargs), tokenizer)[1])
    (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "LOCAL_OFFICIAL_TEMPLATE"}))
    actual, identity = judge.load_judge_tokenizer(tmp_path)
    assert actual.chat_template == "LOCAL_OFFICIAL_TEMPLATE"
    assert identity["template_source"] == "local_chat_template.json"
    assert calls == [{"local_files_only": True, "trust_remote_code": False}]
    tokenizer.chat_template = None
    (tmp_path / "chat_template.json").write_text('{}')
    with pytest.raises(ValueError, match="chat_template string"):
        judge.load_judge_tokenizer(tmp_path)


@pytest.mark.parametrize("fault", [None, "missing_keys", "mismatched_keys", "unexpected_keys"])
def test_full_gemma_loader_strict_info_and_no_auto_dispatch(tmp_path, monkeypatch, fault):
    import torch
    from transformers import Gemma3ForConditionalGeneration
    calls = []
    class Model:
        def requires_grad_(self, value):
            assert value is False
            return self
        def eval(self): return self
        def to(self, device):
            self.device = device
            return self
        def parameters(self): return [SimpleNamespace(device=self.device)]
    def load(path, **kwargs):
        calls.append(kwargs)
        return Model(), {fault: ["bad.weight"]} if fault else {}
    monkeypatch.setattr(Gemma3ForConditionalGeneration, "from_pretrained", load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    if fault:
        with pytest.raises(RuntimeError, match="loading was incomplete"):
            judge.load_judge_model(tmp_path, "cuda:0")
    else:
        assert judge.load_judge_model(tmp_path, "cuda:0").load_report["full_multimodal_weights_loaded"]
    assert calls[0]["local_files_only"] and calls[0]["use_safetensors"]
    assert calls[0]["trust_remote_code"] is False
    assert calls[0]["output_loading_info"] and not calls[0]["ignore_mismatched_sizes"]
    assert calls[0]["dtype"] == torch.bfloat16 and "device_map" not in calls[0]
