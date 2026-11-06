import copy
from datetime import timedelta
import json
import multiprocessing
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments, set_seed

from rsd.evaluation import generate_predictions, read_jsonl, score_records
from rsd.milestone_eval import (
    MilestoneEvalCallback, MilestoneEvaluationError, build_milestone_contract,
    parameter_fingerprint, preserved_runtime_state,
)


def assets():
    vocab = {word: i for i, word in enumerate([
        "[PAD]", "[EOS]", "[UNK]", "user", "assistant", ":", "one", "plus", "two",
        "3", "4", "question", "answer", "safe", "request", "fixture",
    ])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}assistant:"
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_layer=1, n_head=2, n_embd=16,
                        n_positions=64, eos_token_id=1, bos_token_id=1, pad_token_id=0,
                        resid_pdrop=0.2, embd_pdrop=0.2, attn_pdrop=0.2, use_cache=False))
    return model, tokenizer


def cohorts():
    def row(item_id, metric):
        return {"id": item_id, "source_id": item_id, "family": "fixture", "split": "dev",
                "prompt": "one plus two", "metric": metric, **({"expected_answer": "3"} if metric == "exact_match" else {})}
    return [row("u", "exact_match")], [row("s", "safety"), row("o", "overrefusal")]


def callback(path, tokenizer, **kwargs):
    task, benchmark = cohorts()
    return MilestoneEvalCallback(output_dir=path, tokenizer=tokenizer, run_contract={"arm": "R0", "seed": 17},
                task_dataset=task, benchmark_dataset=benchmark, steps=(0, 1, 2),
                max_new_tokens=4, max_prompt_tokens=32, **kwargs)


def invoke(cb, model, step=0):
    return cb.on_step_end(SimpleNamespace(), SimpleNamespace(global_step=step, max_steps=2), SimpleNamespace(), model=model)


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def test_real_resident_generation_records_pending_safety_and_restores_state(tmp_path):
    torch.set_num_threads(1)
    set_seed(7)
    model, tokenizer = assets()
    model.train()
    model.transformer.h[0].attn.eval()  # Heterogeneous module states must survive.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    checkpointing = model.transformer._gradient_checkpointing_func
    modes = [m.training for m in model.modules()]
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    model.generation_config.eos_token_id = 2  # Evaluation must use tokenizer EOS=1.
    original_config, original_generation = copy.deepcopy(model.config.to_dict()), copy.deepcopy(model.generation_config.to_dict())
    cb = callback(tmp_path / "milestones", tokenizer)
    before = rng_snapshot()
    invoke(cb, model)
    assert_rng_equal(before, rng_snapshot())
    assert [m.training for m in model.modules()] == modes
    assert model.config.to_dict() == original_config and model.generation_config.to_dict() == original_generation
    assert model.transformer._gradient_checkpointing_func is checkpointing
    assert model.is_gradient_checkpointing
    assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
    folder = tmp_path / "milestones/step-000000"
    task, benchmark = cohorts()
    pred = read_jsonl(folder / "task-predictions.jsonl")
    assert pred[0]["checkpoint_id"].startswith("in-memory:")
    assert pred[0]["checkpoint_identity"]["full_model_sha256"] is None
    assert pred[0]["checkpoint_identity"]["reloadable_checkpoint"] is False
    assert pred[0]["generation_protocol"]["eos_token_id"] == 1
    assert pred[0]["generation_protocol"]["profile"] == "rsd-text-eval-v2"
    assert json.loads((folder / "task-summary.json").read_text()) == score_records(task, pred)
    pending = json.loads((folder / "benchmark-summary-pending.json").read_text())
    assert pending["n_planned"] == 2 and pending["n_judgments"] == 0
    assert all(g["value"] is None for g in pending["groups"])
    assert json.loads((folder / "judge-pending.json").read_text())["status"] == "pending"
    assert cb.completed_steps == (0,)
    with pytest.raises(MilestoneEvaluationError, match="incomplete"):
        cb.assert_complete()
    assert not list(tmp_path.rglob("*.safetensors"))


class RandomTrainingData(torch.utils.data.Dataset):
    """All three RNGs change training inputs, while model dropout uses torch RNG."""
    def __len__(self):
        return 4
    def __getitem__(self, index):
        token = 3 + (int(random.random() * 1000) + int(np.random.randint(1000)) + int(torch.randint(1000, ()).item())) % 12
        ids = torch.tensor([3, 6, token, 8, 5, 9])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def test_actual_trainer_updates_and_rng_identical_with_and_without_callback(tmp_path):
    torch.set_num_threads(1)
    outputs, rngs = [], []
    for enabled in (False, True):
        set_seed(101)
        model, tokenizer = assets()
        cb = callback(tmp_path / "milestones", tokenizer) if enabled else None
        args = TrainingArguments(output_dir=str(tmp_path / str(enabled)), max_steps=2,
            per_device_train_batch_size=1, learning_rate=0.001, use_cpu=True,
            save_strategy="no", report_to="none", disable_tqdm=True, seed=17,
            dataloader_num_workers=0, dataloader_pin_memory=False)
        trainer = Trainer(model=model, args=args, train_dataset=RandomTrainingData(),
                          callbacks=[cb] if cb else [], processing_class=tokenizer)
        if cb:
            cb.model_getter = lambda: trainer.accelerator.unwrap_model(trainer.model_wrapped)
        trainer.train()
        outputs.append({k: v.clone() for k, v in model.state_dict().items()})
        rngs.append(rng_snapshot())
        if cb:
            assert cb.assert_complete() == (0, 1, 2)
    assert all(torch.equal(value, outputs[1][name]) for name, value in outputs[0].items())
    assert_rng_equal(*rngs)


def test_resident_outputs_match_standalone_evaluation_v2(tmp_path):
    set_seed(9)
    model, tokenizer = assets()
    saved = tmp_path / "reference"
    model.save_pretrained(saved)
    tokenizer.save_pretrained(saved)
    task, benchmark = cohorts()
    generate_predictions(task + benchmark, tmp_path / "standalone.jsonl", model_name_or_path=saved,
                         checkpoint_id="reference", max_new_tokens=4, max_prompt_tokens=32,
                         device="cpu", dtype="float32", seed=0)
    cb = callback(tmp_path / "milestones", tokenizer)
    invoke(cb, model)
    folder = tmp_path / "milestones/step-000000"
    resident = read_jsonl(folder / "task-predictions.jsonl") + read_jsonl(folder / "benchmark-predictions.jsonl")
    standalone = read_jsonl(tmp_path / "standalone.jsonl")
    for current, reference in zip(resident, standalone, strict=True):
        assert {k: current[k] for k in ("id", "response", "n_tokens", "valid", "prompt_sha256", "generation_protocol")} == {
            k: reference[k] for k in ("id", "response", "n_tokens", "valid", "prompt_sha256", "generation_protocol")}


def test_failure_preserves_partial_denominators_and_rng_then_refuses_retry(tmp_path, monkeypatch):
    model, tokenizer = assets()
    cb = callback(tmp_path / "milestones", tokenizer)
    def fail(**kwargs):
        random.random(); np.random.random(); torch.rand(3)
        model.config.use_cache = True
        raise RuntimeError("synthetic generation failure")
    monkeypatch.setattr(model, "generate", fail)
    before = rng_snapshot()
    with pytest.raises(MilestoneEvaluationError, match="synthetic generation failure"):
        invoke(cb, model)
    assert_rng_equal(before, rng_snapshot())
    assert model.training and model.config.use_cache is False
    folder = tmp_path / "milestones/step-000000"
    receipt = json.loads((folder / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    task = json.loads((folder / "task-summary.json").read_text())
    bench = json.loads((folder / "benchmark-summary-pending.json").read_text())
    assert task["groups"][0]["n_invalid"] == 1 and task["groups"][0]["value"] is None
    assert sum(g["n_missing"] for g in bench["groups"]) == 2
    previous = (folder / "task-predictions.jsonl").read_bytes()
    with pytest.raises(MilestoneEvaluationError, match="Partial/failed"):
        invoke(cb, model)
    assert (folder / "task-predictions.jsonl").read_bytes() == previous


def test_empty_and_overbudget_responses_stay_invalid_without_judgments(tmp_path, monkeypatch):
    model, tokenizer = assets()
    def only_eos(**kwargs):
        assert kwargs["synced_gpus"] is False
        assert kwargs["generation_config"].eos_token_id == tokenizer.eos_token_id
        return torch.cat((kwargs["input_ids"], torch.tensor([[tokenizer.eos_token_id]])), dim=1)
    monkeypatch.setattr(model, "generate", only_eos)
    cb = callback(tmp_path / "empty", tokenizer)
    invoke(cb, model)
    folder = tmp_path / "empty/step-000000"
    task = json.loads((folder / "task-summary.json").read_text())
    assert task["groups"][0]["n_invalid"] == 1 and task["groups"][0]["value"] is None
    assert json.loads((folder / "judge-pending.json").read_text())["n_eligible"] == 0
    cb = callback(tmp_path / "long", tokenizer)
    cb.plan["max_prompt_tokens"] = 1
    invoke(cb, model)
    assert read_jsonl(tmp_path / "long/step-000000/task-predictions.jsonl")[0]["error"] == "prompt_over_budget"


def test_completed_duplicate_is_reused_but_corruption_or_model_change_refused(tmp_path, monkeypatch):
    model, tokenizer = assets()
    cb = callback(tmp_path / "milestones", tokenizer)
    invoke(cb, model)
    folder = tmp_path / "milestones/step-000000"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    monkeypatch.setattr(model, "generate", lambda **kwargs: pytest.fail("duplicate must not generate"))
    invoke(cb, model)
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    with torch.no_grad():
        next(model.parameters()).reshape(-1)[0] += 1
    with pytest.raises(MilestoneEvaluationError, match="model/protocol identity"):
        invoke(cb, model)
    assert json.loads((folder / "receipt.json").read_text())["status"] == "completed"
    (folder / "task-summary.json").write_text('{}')
    with pytest.raises(ValueError, match="hash mismatch"):
        _ = cb.completed_steps


@pytest.mark.parametrize("steps", [[], [1, 0], [0, 0], [-1], [True]])
def test_invalid_schedule_rejected(steps):
    task, benchmark = cohorts()
    with pytest.raises(ValueError, match="steps"):
        build_milestone_contract(task, benchmark, steps)


@pytest.mark.parametrize("fault", ["test", "duplicate", "empty", "metric"])
def test_dev_cohort_contract_rejected_before_any_generation(fault):
    task, benchmark = cohorts()
    if fault == "test":benchmark[0]["split"] = "test"
    elif fault == "duplicate":benchmark[0]["id"] = task[0]["id"]
    elif fault == "empty":task = []
    else:task[0]["metric"] = "safety"
    with pytest.raises(ValueError):build_milestone_contract(task, benchmark)


def test_buffers_modes_and_rng_restore_even_on_exception():
    model, _ = assets()
    model.register_buffer("fixture_buffer", torch.tensor([1., 2.]))
    model.transformer.eval()
    modes = [m.training for m in model.modules()]
    before = rng_snapshot()
    with pytest.raises(RuntimeError):
        with preserved_runtime_state(model, torch.device("cpu")):
            model.eval(); model.fixture_buffer.add_(2)
            model.register_buffer("new_buffer", torch.tensor(3.))
            model.config.use_cache = True
            random.random();np.random.random();torch.rand(3)
            raise RuntimeError("fail")
    assert_rng_equal(before, rng_snapshot())
    assert model.fixture_buffer.tolist() == [1., 2.] and not hasattr(model, "new_buffer")
    assert [m.training for m in model.modules()] == modes and model.config.use_cache is False


def test_only_selected_cuda_device_rng_is_accessed(monkeypatch):
    model, _ = assets()
    calls = []
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: (calls.append(("get", str(device))), torch.tensor([7], dtype=torch.uint8))[1])
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: calls.append(("set", str(device))))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("must not initialize other GPUs"))
    with preserved_runtime_state(model, torch.device("cuda:1")):
        pass
    assert calls == [("get", "cuda:1"), ("set", "cuda:1")]


def _distributed_worker(rank, rendezvous, folder, fail):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method="file://" + rendezvous,
                                        rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        set_seed(9)
        model, tokenizer = assets()
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        original = model.generate
        def generate(**kwargs):
            assert rank == 0, "nonzero rank must not generate"
            if fail == "rank0":raise RuntimeError("rank-zero synthetic failure")
            return original(**kwargs)
        model.generate = generate
        def get_model():
            if rank == 1 and fail == "rank1":raise RuntimeError("rank-one synthetic failure")
            return ddp
        cb = callback(Path(folder) / "milestones", tokenizer, model_getter=get_model)
        if rank == 1 and fail == "contract":cb.contract_sha = "different-contract"
        rng = rng_snapshot()
        try:
            invoke(cb, ddp)
            status = "completed"
        except MilestoneEvaluationError as exc:
            status = str(exc)
        assert_rng_equal(rng, rng_snapshot())
        (Path(folder) / f"rank-{rank}.json").write_text(json.dumps({"status": status, "training": model.training}))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("fail", [False, "rank0", "rank1", "contract"])
def test_two_real_gloo_ranks_only_rank0_generates_and_share_failure(tmp_path, fail):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_distributed_worker, args=(rank, str(tmp_path / "rdv"), str(tmp_path), fail)) for rank in (0, 1)]
    for process in processes:process.start()
    for process in processes:
        process.join(40)
        if process.is_alive():
            process.terminate();process.join(5)
            pytest.fail("distributed callback deadlocked")
        assert process.exitcode == 0
    a, b = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    assert a == b and a["training"] is True
    if fail:
        expected = {"rank0": "rank-zero synthetic failure", "rank1": "rank-one synthetic failure", "contract": "Ranks disagree"}[fail]
        assert expected in a["status"]
        assert json.loads((tmp_path / "milestones/step-000000/receipt.json").read_text())["status"] == "failed"
    else:
        assert a["status"] == "completed"
