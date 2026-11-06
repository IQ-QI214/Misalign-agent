"""Regression checks for sampled model-weight receipts; no model downloads."""

import pytest
import torch

from rsd.train import capture_parameter_probe, compare_parameter_probes


class ProbeModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(128, 32, dtype=dtype)
        self.lm_head = torch.nn.Linear(32, 128, bias=False, dtype=dtype)


@pytest.mark.parametrize("changed_tensor", ["embed_tokens", "lm_head"])
def test_detects_changes_old_first_embedding_prefix_misses(changed_tensor):
    model = ProbeModel()
    old_prefix = next(model.parameters()).detach().flatten()[:1024].clone()
    initial = capture_parameter_probe(model)
    with torch.no_grad():
        getattr(model, changed_tensor).weight[-1].add_(0.1)
    final = capture_parameter_probe(model)
    assert torch.equal(old_prefix, next(model.parameters()).detach().flatten()[:1024])
    result = compare_parameter_probes(initial, final)
    assert result["l2_delta"] > 0
    assert result["changed_tensors"] == 1
    assert result["tensors"][f"{changed_tensor}.weight"]["changed_coordinates"] > 0


def test_probe_is_deterministic_cpu_snapshot_and_skips_frozen_tensors():
    model = ProbeModel()
    model.lm_head.requires_grad_(False)
    a, b = capture_parameter_probe(model, 7), capture_parameter_probe(model, 7)
    assert list(a["tensors"]) == ["embed_tokens.weight"]
    row = a["tensors"]["embed_tokens.weight"]
    assert row["indices"].tolist() == [0, 682, 1365, 2047, 2730, 3412, 4095]
    assert row["values"].device.type == "cpu"
    assert row["values"].requires_grad is False
    assert compare_parameter_probes(a, b)["l2_delta"] == 0
    with torch.no_grad():
        model.embed_tokens.weight.add_(1)
    assert torch.equal(row["values"], b["tensors"]["embed_tokens.weight"]["values"])


def test_bf16_zero_is_not_reported_as_no_optimizer_update():
    model = ProbeModel(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1)
    initial = capture_parameter_probe(model, 16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1e-6)  # Rounds back to 1 in BF16 model weights.
    report = compare_parameter_probes(initial, capture_parameter_probe(model, 16))
    assert report["l2_delta"] == 0
    assert report["changed_coordinates"] == 0
    assert "BF16 rounding" in report["zero_delta_interpretation"]
    assert "optimizer master weights" in report["evidence_scope"]


def test_nonfinite_samples_are_explicit_not_json_nan():
    model = ProbeModel()
    initial = capture_parameter_probe(model)
    with torch.no_grad():
        model.lm_head.weight[-1, -1] = float("nan")
    report = compare_parameter_probes(initial, capture_parameter_probe(model))
    assert report["nonfinite_coordinates"] == 1
    assert report["l2_delta"] is None
    assert report["max_abs_delta"] is None


def test_mismatched_tensor_identity_and_invalid_budget_fail():
    model = ProbeModel()
    initial = capture_parameter_probe(model)
    model.lm_head = torch.nn.Linear(32, 64, bias=False)
    with pytest.raises(ValueError, match="coordinates differ"):
        compare_parameter_probes(initial, capture_parameter_probe(model))
    with pytest.raises(ValueError, match="positive integer"):
        capture_parameter_probe(model, 0)


def test_frozen_teacher_probe_declares_scope_without_enabling_gradients():
    model = ProbeModel().requires_grad_(False)
    initial = capture_parameter_probe(model, 7, trainable_only=False)
    report = compare_parameter_probes(initial, capture_parameter_probe(model, 7, trainable_only=False))
    assert report["parameter_scope"] == "all_parameters_including_frozen"
    assert report["observed_tensors"] == 2 and report["changed_coordinates"] == 0
    assert "trainable_tensors" not in report
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    model.requires_grad_(True)
    with pytest.raises(ValueError, match="identity differs"):
        compare_parameter_probes(initial, capture_parameter_probe(model, 7))
