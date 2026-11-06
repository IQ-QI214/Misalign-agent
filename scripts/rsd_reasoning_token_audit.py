#!/usr/bin/env python3
"""Read-only CPU tokenization of prepared inputs, never model/gold evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--download-manifest", required=True)
    parser.add_argument("--data-dir", default="data/rsd_reasoning_v2")
    parser.add_argument("--benchmark-dir", default="data/rsd_benchmarks")
    parser.add_argument("--output", default="validation/reasoning_token_budget_v2.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from rsd.collator import render_prompt
    from rsd.reasoning_data import TEACHER_CUE
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "configs/rsd/model.lock.json").read_text())
    manifest = json.loads(Path(args.download_manifest).read_text())
    if manifest["revision"] != lock["revision"]:
        raise ValueError("Tokenizer revision differs from model lock")
    hashes = {}
    for item in manifest["files"]:
        name = item["file"]
        if Path(name).name != name:
            raise ValueError("Tokenizer path must be local")
        data = (Path(args.tokenizer) / name).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if item["sha256"] != actual:
            raise ValueError("Tokenizer bytes changed")
        hashes[name] = actual
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    sources = [("train_student", Path(args.data_dir) / "train.prompts.jsonl", None, 1024),
               ("train_teacher", Path(args.data_dir) / "train.prompts.jsonl", TEACHER_CUE, 1024),
               ("task_dev", Path(args.data_dir) / "dev.prompts.jsonl", None, 2048),
               ("task_test", Path(args.data_dir) / "sealed/test.prompts.jsonl", None, 2048),
               ("safety_dev", Path(args.benchmark_dir) / "dev.jsonl", None, 2048),
               ("safety_test", Path(args.benchmark_dir) / "test.jsonl", None, 2048)]
    summaries = {}
    for name, path, cue, budget in sources:
        blob = path.read_bytes()
        rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
        lengths = sorted(len(tokenizer(render_prompt(tokenizer, row["prompt"], feedback=cue,
                                                    thinking=False, placement="user"),
                                       add_special_tokens=False)["input_ids"]) for row in rows)
        if not lengths:
            raise ValueError("Empty token audit cohort")
        def percentile(q):
            return lengths[math.ceil(q * len(lengths)) - 1]
        summaries[name] = {"rows": len(rows), "file_sha256": hashlib.sha256(blob).hexdigest(),
                           "min": lengths[0], "p50": percentile(.5), "p95": percentile(.95),
                           "p99": percentile(.99), "max": lengths[-1], "prompt_budget": budget,
                           "over_budget": sum(length > budget for length in lengths)}
    result = {"version": "rsd-reasoning-token-audit-v2", "execution": "CPU_TOKENIZER_ONLY_NO_MODEL_WEIGHTS",
              "model_repository": lock["repository"], "model_revision": lock["revision"],
              "tokenizer_files_sha256": hashes, "eos_token_id": tokenizer.eos_token_id,
              "thinking": False, "template": "rsd.collator.render_prompt",
              "render_code_sha256": hashlib.sha256((root / "rsd/collator.py").read_bytes()).hexdigest(),
              "summaries": summaries, "all_within_prompt_budget": all(v["over_budget"] == 0 for v in summaries.values()),
              "output_length_distribution": "UNKNOWN_NO_GENERATION",
              "gpu_capacity_for_completion256_training": "UNVERIFIED"}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
