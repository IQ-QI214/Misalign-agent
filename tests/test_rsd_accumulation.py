"""Real tiny-CPU accumulation parity; no downloaded model or GPU execution."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import socket

import pytest
import torch
from datasets import Dataset

from rsd.collator import TransferCollator, completion_mask
from rsd.train import make_training_args
from rsd.trainer import (ACCUMULATION_POLICY, TransferTrainer, TrainingTelemetryCallback,
                         inference_mode, masked_distillation_loss)
from test_rsd_training import tiny_assets


def trainer_fixture(tmp_path, *, teacher_mode="recursive", accumulation=4, n_rows=8, max_steps=2):
    torch.set_num_threads(1)
    model, tokenizer = tiny_assets(tmp_path / "model")
    model.config._name_or_path = str(tmp_path / "model")
    initial = copy.deepcopy(model)
    frozen = copy.deepcopy(model) if teacher_mode == "frozen" else None
    rows = [{"id": f"task-{i}", "source_id": f"source-{i}", "split": "train",
             "prompt": ("How many two plus one?" if i % 2 else "How many one plus two?"),
             "teacher_context": ("Excellent" if i % 3 else "Check premise")}
            for i in range(n_rows)]
    collator = TransferCollator(tokenizer)
    config = json.loads(Path("configs/rsd/qwen35_9b_2h100.json").read_text())
    config.update(max_steps=max_steps, max_completion_length=4, max_prompt_length=48,
                  max_length=64, gradient_accumulation_steps=accumulation,
                  learning_rate=.05, max_grad_norm=0., save_strategy="no")
    args = make_training_args(config, str(tmp_path / "run"), use_cpu=True)
    # This direct adapter test is independent of train.py config plumbing.
    args.gradient_accumulation_steps = accumulation
    args.disable_tqdm = True
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    captured = {"grads": []}
    original_step = optimizer.step

    def capture_step(*a, **kw):
        captured["grads"].append({name: p.grad.clone() for name, p in model.named_parameters()
                                  if p.grad is not None})
        return original_step(*a, **kw)

    optimizer.step = capture_step
    trainer = TransferTrainer(model=model, frozen_teacher=frozen, teacher_mode=teacher_mode,
                              args=args, train_dataset=Dataset.from_list(rows),
                              processing_class=tokenizer, data_collator=collator,
                              student_thinking=False, teacher_thinking=False,
                              optimizers=(optimizer, None), loss_chunk_size=2)
    return trainer, initial, tokenizer, rows, collator, captured


@pytest.mark.parametrize("teacher_mode", ["frozen", "recursive"])
def test_accumulation_four_matches_manual_batch_mean_and_teacher_update_timing(tmp_path, monkeypatch, teacher_mode):
    trainer, initial, tokenizer, rows, collator, captured = trainer_fixture(tmp_path, teacher_mode=teacher_mode)
    outputs = [[16, 1], [17, 16, 1], [16, 17, 16, 1], [17, 1]] * 2
    generated = []
    teacher_versions = []

    def generate(**kwargs):
        answer = torch.tensor([outputs[len(generated)]], dtype=torch.long)
        generated.append(answer)
        return torch.cat([kwargs["input_ids"], answer], dim=1)

    compute_loss = trainer.compute_loss

    def observe_teacher(model, inputs, **kwargs):
        teacher_versions.append({name: p.detach().clone() for name, p in trainer._teacher(model).named_parameters()})
        return compute_loss(model, inputs, **kwargs)

    monkeypatch.setattr(trainer.model, "generate", generate)
    monkeypatch.setattr(trainer, "compute_loss", observe_teacher)
    trainer.train()
    traces = [json.loads(line) for line in (tmp_path / "run/trajectory_rank0.jsonl").read_text().splitlines()]
    receipts = [json.loads(line) for line in (tmp_path / "run/optimizer_rank0.jsonl").read_text().splitlines()]
    assert len(traces) == 8 and len(receipts) == 2
    assert [row["step_before_update"] for row in traces] == [0] * 4 + [1] * 4
    assert [row["microstep_in_optimizer_step"] for row in traces] == [1, 2, 3, 4] * 2
    assert [row["synchronize_gradients"] for row in traces] == [False, False, False, True] * 2
    assert [row["teacher_step"] for row in traces] == ([0] * 8 if teacher_mode == "frozen" else [0] * 4 + [1] * 4)
    for begin in (0, 4):
        for snapshot in teacher_versions[begin:begin + 4]:
            assert all(torch.equal(snapshot[name], teacher_versions[begin][name]) for name in snapshot)
    moved = any(not torch.equal(teacher_versions[0][name], teacher_versions[4][name]) for name in teacher_versions[0])
    assert moved == (teacher_mode == "recursive")
    assert trainer.accelerator.gradient_accumulation_steps == 1
    assert trainer.loss_is_scaled_for_ga is False
    for trace in traces:
        assert trace["loss"] == pytest.approx(trace["loss_before_accumulation_scale"] / 4)
        assert trace["loss_policy"] == ACCUMULATION_POLICY
        assert trace["teacher_forward_seconds"] >= 0
        assert trace["student_forward_seconds"] >= 0
        assert trace["loss_construction_seconds"] >= 0
        assert trace["forward_backward_seconds"] >= trace["teacher_forward_seconds"] + trace["student_forward_seconds"]
    assert [r["observed_global_sample_exposures"] for r in receipts] == [4, 4]
    assert [r["observed_global_valid_tokens"] for r in receipts] == [11, 11]
    assert receipts[-1]["attempt_global_sample_exposures"] == 8
    assert receipts[-1]["attempt_global_unique_sources"] == 8
    assert receipts[-1]["attempt_global_valid_tokens"] == 22
    assert all(r["cuda_max_memory_allocated"] is None for r in receipts)
    # Independent reference: construct each completion's token-mean KL and do
    # one backward of the equal completion mean, rather than dividing 4 times.
    manual = initial
    reference_teacher = copy.deepcopy(initial) if teacher_mode == "frozen" else manual
    optimizer = torch.optim.SGD(manual.parameters(), lr=.05)
    row_by_id = {r["id"]: r for r in rows}
    for window in range(2):
        losses = []
        for trace in traces[window * 4:(window + 1) * 4]:
            inputs = collator([row_by_id[trace["ids"][0]]])
            answer = torch.tensor(trace["completion_ids"], dtype=torch.long)
            mask = completion_mask(answer, tokenizer.eos_token_id)
            student_inputs = torch.cat([inputs["student_prompts"], answer], dim=1)
            teacher_inputs = torch.cat([inputs["teacher_prompts"], answer], dim=1)
            with inference_mode(reference_teacher):
                t = reference_teacher(teacher_inputs, use_cache=False).logits[:, inputs["teacher_prompt_length"] - 1:-1]
            s = manual(student_inputs, use_cache=False).logits[:, inputs["student_prompt_length"] - 1:-1]
            losses.append(masked_distillation_loss(s, t, mask, "reverse", 2))
        torch.stack(losses).mean().backward()
        for name, parameter in manual.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, captured["grads"][window][name], atol=2e-7, rtol=2e-5)
        optimizer.step()
        optimizer.zero_grad()
    for name, parameter in manual.named_parameters():
        torch.testing.assert_close(parameter, dict(trainer.model.named_parameters())[name], atol=2e-7, rtol=2e-5)
    if teacher_mode == "frozen":
        assert all(p.grad is None for p in trainer.frozen_teacher.parameters())


def test_partial_accumulation_window_rejected_before_any_model_generation(tmp_path, monkeypatch):
    trainer, *_ = trainer_fixture(tmp_path, n_rows=3)
    calls = []
    monkeypatch.setattr(trainer.model, "generate", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="Incomplete accumulation window"):
        trainer.train()
    assert calls == []
    assert not (tmp_path / "run/trajectory_rank0.jsonl").exists()


def test_remainder_after_full_window_preserves_only_completed_update(tmp_path, monkeypatch):
    trainer, *_ = trainer_fixture(tmp_path, n_rows=5)
    monkeypatch.setattr(trainer.model, "generate", lambda **kwargs: torch.cat(
        [kwargs["input_ids"], torch.tensor([[16, 1]])], dim=1))
    with pytest.raises(ValueError, match="Incomplete accumulation window"):
        trainer.train()
    receipts = (tmp_path / "run/optimizer_rank0.jsonl").read_text().splitlines()
    assert len(receipts) == 1
    assert json.loads(receipts[0])["observed_global_sample_exposures"] == 4


def test_runtime_double_scaling_rejected_before_generation(tmp_path):
    trainer, _, _, rows, collator, _ = trainer_fixture(tmp_path)
    trainer.accelerator.gradient_accumulation_steps = 4
    with pytest.raises(ValueError, match="multiply-scaled"):
        trainer.training_step(trainer.model, collator([rows[0]]))


def test_nonfinite_loss_stops_before_backward(tmp_path, monkeypatch):
    trainer, _, _, rows, collator, _ = trainer_fixture(tmp_path)
    batch = collator([rows[0]])
    answer = torch.tensor([[16, 1]])
    batch["completion_mask"] = completion_mask(answer, 1)
    for role in ("student", "teacher"):
        batch[role + "_input_ids"] = torch.cat([batch[role + "_prompts"], answer], dim=1)
        batch[role + "_attention_mask"] = torch.ones_like(batch[role + "_input_ids"])
    monkeypatch.setattr("rsd.trainer.masked_distillation_loss", lambda *args: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(FloatingPointError, match="backward is not allowed"):
        trainer.compute_loss(trainer.model, batch)
    assert all(p.grad is None for p in trainer.model.parameters())


def test_optimizer_receipt_rejects_incomplete_window(tmp_path):
    trainer, *_ = trainer_fixture(tmp_path)
    callback = next(c for c in trainer.callback_handler.callbacks if isinstance(c, TrainingTelemetryCallback))
    with pytest.raises(RuntimeError, match="complete accumulation window"):
        callback.on_pre_optimizer_step(trainer.args, trainer.state, trainer.control)


def distributed_worker(folder, teacher_mode):
    """Invoked by torchrun on loopback, CPU/Gloo only, using random tiny weights."""
    rank = int(os.environ["RANK"])
    local = Path(folder) / f"rank-{rank}"
    local.mkdir()
    trainer, initial, tokenizer, rows, collator, captured = trainer_fixture(
        local, teacher_mode=teacher_mode, n_rows=16)
    sequence = [[16, 1], [17, 16, 1], [16, 17, 16, 1], [17, 1]]
    calls = []

    def generate(**kwargs):
        answer = torch.tensor([sequence[len(calls) % 4]], dtype=torch.long)
        calls.append(None)
        return torch.cat([kwargs["input_ids"], answer], dim=1)

    trainer.model.generate = generate
    trainer.train()
    traces = [json.loads(line) for line in (local / f"run/trajectory_rank{rank}.jsonl").read_text().splitlines()]
    receipts = [json.loads(line) for line in (local / f"run/optimizer_rank{rank}.jsonl").read_text().splitlines()]
    assert len(traces) == 8 and len(receipts) == 2
    assert [r["observed_global_sample_exposures"] for r in receipts] == [8, 8]
    assert [r["observed_global_valid_tokens"] for r in receipts] == [22, 22]
    assert receipts[-1]["attempt_global_unique_sources"] == 16
    gathered = [None, None]
    torch.distributed.all_gather_object(gathered, traces)
    manual = initial
    reference_teacher = copy.deepcopy(initial) if teacher_mode == "frozen" else manual
    optimizer = torch.optim.SGD(manual.parameters(), lr=.05)
    row_by_id = {r["id"]: r for r in rows}
    for window in range(2):
        losses = []
        for rank_traces in gathered:
            for trace in rank_traces[window * 4:(window + 1) * 4]:
                inputs = collator([row_by_id[trace["ids"][0]]])
                answer = torch.tensor(trace["completion_ids"], dtype=torch.long)
                with inference_mode(reference_teacher):
                    t = reference_teacher(torch.cat([inputs["teacher_prompts"], answer], dim=1), use_cache=False).logits[:, inputs["teacher_prompt_length"] - 1:-1]
                s = manual(torch.cat([inputs["student_prompts"], answer], dim=1), use_cache=False).logits[:, inputs["student_prompt_length"] - 1:-1]
                losses.append(masked_distillation_loss(s, t, completion_mask(answer, 1), "reverse", 2))
        assert len(losses) == 8
        torch.stack(losses).mean().backward()
        for name, parameter in manual.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, captured["grads"][window][name], atol=2e-7, rtol=2e-5)
        optimizer.step()
        optimizer.zero_grad()
    max_error = max(float((p.detach() - dict(trainer.model.named_parameters())[name].detach()).abs().max())
                    for name, p in manual.named_parameters())
    assert max_error < 2e-7
    result = {"backend": torch.distributed.get_backend(), "world_size": 2, "device": str(trainer.accelerator.device),
              "rank": rank, "teacher_mode": teacher_mode, "microbatch": 1, "gradient_accumulation_steps": 4,
              "optimizer_steps": 2, "effective_batch": 8, "global_exposures": 16,
              "global_tokens": 44, "manual_batch_reference_max_parameter_error": max_error}
    (local / "worker-result.json").write_text(json.dumps(result) + "\n")
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("teacher_mode", ["frozen", "recursive"])
def test_two_rank_cpu_gloo_accumulation_matches_global_batch_eight(tmp_path, teacher_mode):
    if not torch.distributed.is_gloo_available():
        pytest.skip("This PyTorch build has no CPU Gloo backend")
    root = Path(__file__).resolve().parents[1]
    worker = tmp_path / "worker.py"
    worker.write_text("from test_rsd_accumulation import distributed_worker\n"
                      "import sys\ndistributed_worker(sys.argv[1], sys.argv[2])\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ACCELERATE_USE_CPU="true",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "tests"), str(root / ".runtime/trl026")])
    # Static IPv4 avoids this Mac's broken IPv6 reverse-DNS path in elastic's
    # --standalone rendezvous. This remains a local CPU process group.
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = str(port_socket.getsockname()[1])
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
                             "--nnodes=1", "--node_rank=0", "--master_addr=127.0.0.1", "--master_port=" + port,
                             str(worker), str(tmp_path), teacher_mode], cwd=root, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    for rank in (0, 1):
        receipt = json.loads((tmp_path / f"rank-{rank}/worker-result.json").read_text())
        assert receipt["backend"] == "gloo" and torch.device(receipt["device"]).type == "cpu"
        assert receipt["global_exposures"] == 16 and receipt["global_tokens"] == 44
