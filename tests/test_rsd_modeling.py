"""Real CPU weight-loading and update checks; no downloaded model weights."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import GPT2Config, GPT2LMHeadModel, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from rsd.modeling import load_text_model, load_tokenizer


def tiny_qwen_config():
    return Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, linear_conv_kernel_dim=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128, pad_token_id=0, eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )


def test_qwen_text_forward_backward_save_reload(tmp_path):
    torch.manual_seed(5)
    original = Qwen3_5ForCausalLM(tiny_qwen_config())
    original.save_pretrained(tmp_path / "initial")
    model = load_text_model(tmp_path / "initial", dtype="float32")
    assert model.load_report["source_model_type"] == "qwen3_5_text"
    assert model.load_report["model_class"] == "Qwen3_5ForCausalLM"
    assert model.load_report["local_header_audited"]
    assert all(p.device.type == "cpu" for p in model.parameters())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    before = model.lm_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight)
    model.eval()
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
    model.save_pretrained(tmp_path / "updated")
    reloaded = load_text_model(tmp_path / "updated", dtype="float32")
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_official_multimodal_prefixes_load_as_text(tmp_path):
    """Use real text tensors with the published checkpoint's prefix layout."""
    original = Qwen3_5ForCausalLM(tiny_qwen_config())
    original.save_pretrained(tmp_path)
    weights = load_file(tmp_path / "model.safetensors")
    published = {
        (key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key): value
        for key, value in weights.items()
    }
    published["model.visual.test_unused.weight"] = torch.ones(2, 2)
    published["mtp.test_unused.weight"] = torch.ones(2, 2)
    save_file(published, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": original.config.to_dict(), "vision_config": {},
        "tie_word_embeddings": False,
    }))
    loaded = load_text_model(tmp_path, dtype="float32")
    assert loaded.load_report["source_model_type"] == "qwen3_5"
    assert loaded.load_report["loaded_model_type"] == "qwen3_5_text"
    assert len(loaded.load_report["omitted_local_non_text_keys"]) == 2
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["missing", "unexpected", "mismatch"])
def test_reject_incomplete_or_wrong_weights(tmp_path, fault):
    model = Qwen3_5ForCausalLM(tiny_qwen_config())
    model.save_pretrained(tmp_path)
    weights = load_file(tmp_path / "model.safetensors")
    if fault == "missing":
        weights.pop("lm_head.weight")
    elif fault == "unexpected":
        weights["unrecognized.weight"] = torch.ones(2)
    else:
        weights["lm_head.weight"] = torch.ones(2, 2)
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises((RuntimeError, ValueError)):
        load_text_model(tmp_path, dtype="float32")


def test_generic_tiny_causal_model_and_unsupported_family(tmp_path):
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_head=2, n_layer=1, n_positions=32))
    model.save_pretrained(tmp_path / "gpt2")
    loaded = load_text_model(tmp_path / "gpt2", dtype="float32")
    assert loaded.load_report["model_class"] == "GPT2LMHeadModel"
    bad = tmp_path / "unsupported"
    bad.mkdir()
    (bad / "config.json").write_text(json.dumps({"model_type": "vit", "hidden_size": 16}))
    with pytest.raises(ValueError, match="Unsupported causal"):
        load_text_model(bad, dtype="float32")


def test_tokenizer_forwards_revision_and_disables_remote_code(monkeypatch):
    from types import SimpleNamespace
    from transformers import AutoTokenizer

    captured = {}
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, padding_side="right")

    def fake_load(path, **kwargs):
        captured.update(path=path, **kwargs)
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_load)
    assert load_tokenizer("example/model", revision="abc123") is tokenizer
    assert captured == {"path": "example/model", "revision": "abc123", "trust_remote_code": False}
    assert tokenizer.padding_side == "left"
