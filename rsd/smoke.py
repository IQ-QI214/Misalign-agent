"""Offline CPU demonstration of the full four-arm pipeline with RANDOM tiny weights.

Output proves software execution only. It measures no Qwen or safety-transfer result.
"""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/cpu_pipeline_smoke")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory; smoke does not overwrite runs")
    output.mkdir(parents=True)
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, set_seed
    from .upstream import ROOT
    from .train import load_config, run
    from .evaluation import generate_predictions, read_jsonl, score_records, write_json
    from .analysis import analyze_matrix
    torch.set_num_threads(1)
    set_seed(7)
    data = read_jsonl(ROOT / "data/rsd_pilot/train.jsonl")
    words = sorted(set(re.findall(r"\w+|[^\w\s]", json.dumps(data, ensure_ascii=False))))
    vocab = {word: index for index, word in enumerate(["[PAD]", "[EOS]", "[UNK]"] + words)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}assistant:"
    base_path = output / "random_tiny_base"
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_layer=1, n_head=2, n_embd=16,
                                     n_positions=1024, bos_token_id=1, eos_token_id=1, pad_token_id=0,
                                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    model.save_pretrained(base_path)
    tokenizer.save_pretrained(base_path)
    del model
    dataset = read_jsonl(ROOT / "data/rsd_pilot/dev.jsonl")
    summaries = {}
    for label in ("base", "F0", "R0", "F1", "R1"):
        model_path = base_path
        if label != "base":
            cfg = load_config(ROOT / "configs/rsd/qwen35_9b_2h100.json", label)
            cfg.update(model_name_or_path=str(base_path), model_revision=None,
                       output_dir=str(output / label), max_steps=2, save_steps=1,
                       max_completion_length=4, max_prompt_length=512, max_length=1024,
                       learning_rate=1e-3, loss_chunk_size=2)
            trainer = run(cfg, use_cpu=True)
            assert trainer.state.global_step == 2
            del trainer
            model_path = output / label / "final"
        pred_path = output / f"{label}-predictions.jsonl"
        generate_predictions(dataset, pred_path, model_name_or_path=str(model_path), checkpoint_id=label,
                             dtype="float32", device="cpu", max_new_tokens=4, max_prompt_tokens=512,
                             temperature=0.0, seed=0)
        summaries[label] = score_records(dataset, read_jsonl(pred_path))
        write_json(output / f"{label}-summary.json", summaries[label])
    matrix = analyze_matrix(summaries)
    write_json(output / "matrix.json", matrix)
    receipt = {"status": "cpu_pipeline_completed", "model": "random_tiny_gpt2",
               "training_arms": 4, "optimizer_steps_per_arm": 2,
               "eval_records_per_model": len(dataset), "qwen35_9b_verified": False,
               "gpu_verified": False, "safety_transfer_conclusion": None}
    write_json(output / "smoke_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
