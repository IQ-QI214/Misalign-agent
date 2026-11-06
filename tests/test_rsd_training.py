import copy
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, set_seed

from rsd.collator import TransferCollator, completion_mask
from rsd.trainer import TransferTrainer, masked_distillation_loss
from rsd.train import make_training_args, run, prepare_contract


def tiny_assets(path):
    path.mkdir()
    vocabulary = {x: i for i, x in enumerate([
        "[PAD]", "[EOS]", "[UNK]", "user", "assistant", "How", "many", "two", "plus", "one",
        "Excellent", "Teacher", "only", "feedback", "Check", "premise", "3", "4", "?", ":", "[", "]", "-",
    ])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}assistant:"
    tokenizer.save_pretrained(path)
    set_seed(7)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_layer=1, n_head=2, n_embd=16,
                                     n_positions=128, bos_token_id=1, eos_token_id=1, pad_token_id=0,
                                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    model.save_pretrained(path)
    return model, tokenizer


def test_kl_matches_definition_and_teacher_is_detached():
    torch.manual_seed(1)
    s = torch.randn(1, 5, 7, requires_grad=True)
    t = torch.randn(1, 5, 7, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False]])
    for direction in ("forward", "reverse"):
        actual = masked_distillation_loss(s, t, mask, direction, 2)
        lp, lq = s.log_softmax(-1), t.detach().log_softmax(-1)
        expected = ((lq.exp() * (lq - lp)) if direction == "forward" else (lp.exp() * (lp - lq))).sum(-1)[mask].mean()
        assert torch.allclose(actual, expected, atol=1e-6)
        actual.backward(retain_graph=True)
        assert t.grad is None
        assert s.grad[:, 3:].abs().sum() == 0
        s.grad = None


def test_eos_is_trained_even_if_pad_equals_eos():
    ids = torch.tensor([[4, 1, 1, 1], [5, 6, 7, 8]])
    assert completion_mask(ids, 1).tolist() == [[True, True, False, False], [True] * 4]


def test_prompt_gold_never_visible_and_long_prompt_fails(tmp_path):
    _, tokenizer = tiny_assets(tmp_path / "model")
    row = {"id": "x", "source_id": "x", "prompt": "How many two plus one?", "expected_answer": "SECRET_GOLD"}
    batch = TransferCollator(tokenizer)([row])
    a = tokenizer.decode(batch["student_prompts"][0])
    b = tokenizer.decode(batch["teacher_prompts"][0])
    assert "Excellent" not in a and "Excellent" in b
    changed = dict(row, expected_answer="DIFFERENT_GOLD")
    assert torch.equal(batch["teacher_prompts"], TransferCollator(tokenizer)([changed])["teacher_prompts"])
    with pytest.raises(ValueError, match="too long"):
        TransferCollator(tokenizer, 2)([row])


@pytest.mark.parametrize("teacher_mode,trajectory_mode", [
    ("recursive", "on_policy"), ("frozen", "on_policy"),
    ("recursive", "off_policy"), ("frozen", "off_policy"),
])
def test_real_training_saves_updates_and_fixed_teacher(tmp_path, teacher_mode, trajectory_mode):
    torch.set_num_threads(1)
    model_path = tmp_path / "model"
    initial, tokenizer = tiny_assets(model_path)
    initial_state = {k: v.clone() for k, v in initial.state_dict().items()}
    train = tmp_path / "train.jsonl"
    rows = [{"id": f"x{i}", "source_id": f"x{i}", "split": "train", "prompt": "How many two plus one?"} for i in range(4)]
    train.write_text("".join(json.dumps(x) + "\n" for x in rows))
    config = json.loads(Path("configs/rsd/qwen35_9b_2h100.json").read_text())
    config.update(model_name_or_path=str(model_path), model_revision=None, train_file=str(train),
                  output_dir=str(tmp_path / "run"), teacher_mode=teacher_mode, trajectory_mode=trajectory_mode,
                  kl_direction="reverse" if trajectory_mode == "on_policy" else "forward",
                  max_steps=2, save_steps=1, max_completion_length=4, max_prompt_length=48, max_length=64,
                  learning_rate=1e-3, loss_chunk_size=2)
    trainer = run(config, use_cpu=True)
    assert trainer.state.global_step == 2
    assert any(not torch.equal(v, initial_state[k]) for k, v in trainer.model.state_dict().items())
    if teacher_mode == "frozen":
        assert all(torch.equal(v, initial_state[k]) for k, v in trainer.frozen_teacher.state_dict().items())
        assert all(p.grad is None for p in trainer.frozen_teacher.parameters())
    output = Path(config["output_dir"])
    assert (output / "checkpoint-1" / "optimizer.pt").exists()
    assert (output / "final" / "model.safetensors").exists()
    traces = [json.loads(x) for x in (output / "trajectory_rank0.jsonl").read_text().splitlines()]
    assert len(traces) == 2
    assert {x["trajectory_mode"] for x in traces} == {trajectory_mode}
    # Resume includes optimizer/RNG/Trainer state and continues to the same budget.
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    resumed = run(config, str(output / "checkpoint-1"), use_cpu=True)
    assert resumed.state.global_step == 2
    for k, v in resumed.model.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_resume_cannot_cross_experiment(tmp_path):
    a, b = tmp_path / "R0", tmp_path / "F0"
    prepare_contract(a, {"arm": "R0"})
    b.mkdir()
    checkpoint = b / "checkpoint-20"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="belong"):
        prepare_contract(a, {"arm": "R0"}, checkpoint)
    with pytest.raises(ValueError, match="different"):
        prepare_contract(a, {"arm": "F0"})


def test_reference_is_opt_in_and_only_in_teacher(tmp_path):
    _, tok = tiny_assets(tmp_path / "model")
    row = {"id": "x", "source_id": "x", "prompt": "How many two plus one?", "verified_solution": "3"}
    normal = TransferCollator(tok)([row])
    for placement in ("user", "assistant_prefill"):
        batch = TransferCollator(tok, context_mode="verified_reference", placement=placement)([row])
        assert torch.equal(batch["student_prompts"], normal["student_prompts"])
        assert "3" in tok.decode(batch["teacher_prompts"][0])
    with pytest.raises(ValueError, match="explicit verified_solution"):
        TransferCollator(tok, context_mode="verified_reference")([dict(row, verified_solution=None)])


def test_on_policy_forward_kl_is_independent_choice(tmp_path):
    model, tokenizer = tiny_assets(tmp_path / "model")
    from datasets import Dataset
    config = json.loads(Path("configs/rsd/qwen35_9b_2h100.json").read_text())
    config.update(kl_direction="forward", max_steps=1, max_completion_length=3, max_length=64)
    row = {"id": "x", "source_id": "x", "prompt": "How many two plus one?"}
    model.config._name_or_path = str(tmp_path / "model")
    trainer = TransferTrainer(model=model, args=make_training_args(config, tmp_path / "run", True),
                              train_dataset=Dataset.from_list([row]), processing_class=tokenizer,
                              data_collator=TransferCollator(tokenizer), kl_direction="forward",
                              trajectory_mode="on_policy", student_thinking=False, teacher_thinking=False)
    trainer.train()
    assert trainer.state.global_step == 1
    assert trainer.kl_direction == "forward"
