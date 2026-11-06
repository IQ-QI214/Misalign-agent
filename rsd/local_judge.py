"""Offline, content-identified Gemma3 rubric pilot; not the official judge model."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

from .evaluation import checkpoint_identity, digest, judge_inputs, read_jsonl, text_digest, write_json
from .judge_protocols import (JUDGE_SYSTEMS, PARSER_VERSION, PROFILE_NAME, load_source_lock,
                              parse_judge_output, render_judge_prompt)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MODELS = Path("/inspire/hdd/global_public/public_models")
sys.dont_write_bytecode = True


def configure_runtime(model_path):
    """Preserve project-local runtime_env caches; route other paths into project."""
    defaults = {
        "HF_HOME": "model-cache", "HF_HUB_CACHE": "model-cache/hub",
        "HF_MODULES_CACHE": "model-cache/modules", "HF_DATASETS_CACHE": "datasets",
        "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config", "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch-extensions", "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "CUDA_CACHE_PATH": "cuda",
        "MPLCONFIGDIR": "matplotlib", "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
    }
    configured = {}
    for name, suffix in defaults.items():
        candidate = Path(os.environ.get(name, ROOT / ".runtime/local-judge" / suffix)).expanduser().resolve()
        if not candidate.is_relative_to(ROOT) or candidate.is_relative_to(model_path):
            candidate = (ROOT / ".runtime/local-judge" / suffix).resolve()
        if (not candidate.is_relative_to(ROOT) or candidate.is_relative_to(PUBLIC_MODELS)
                or candidate.is_relative_to(model_path)):
            raise ValueError(f"Unsafe judge cache directory for {name}")
        candidate.mkdir(parents=True, exist_ok=True)
        configured[name] = str(candidate)
    configured.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled")
    os.environ.update(configured)
    tempfile.tempdir = configured["TMPDIR"]
    os.chdir(ROOT)
    return configured


def local_model_identity(model_path):
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("Judge model must be an existing local directory; Hub IDs are forbidden")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "gemma3" or config.get("quantization_config"):
        raise ValueError("This pilot requires an unquantized full Gemma3 conditional-generation checkpoint")
    if not any((path / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")):
        raise ValueError("Local judge requires safetensors weights")
    identity = checkpoint_identity(path)
    hashes = identity["files_sha256"]
    # checkpoint_identity predates processor-side chat_template.json files.
    for name in ("chat_template.json", "tokenizer.model", "added_tokens.json", "processor_config.json"):
        item = path / name
        if item.exists():
            if not item.is_file() or item.resolve().parent != path:
                raise ValueError(f"Nonlocal tokenizer/processor asset: {name}")
            hashes[name] = hashlib.sha256(item.read_bytes()).hexdigest()
    identity["checkpoint_id"] = "sha256:" + digest(hashes)
    return path, config, identity


def load_judge_tokenizer(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    source = "tokenizer.get_chat_template"
    if not tokenizer.chat_template:
        asset = path / "chat_template.json"
        if not asset.is_file():
            raise ValueError("No official local chat template; invented fallback is forbidden")
        document = json.loads(asset.read_text())
        template = document.get("chat_template") if isinstance(document, dict) else None
        if not isinstance(template, str) or not template.strip():
            raise ValueError("chat_template.json must contain a nonempty chat_template string")
        tokenizer.chat_template = template
        source = "local_chat_template.json"
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template.strip():
        raise ValueError("No unambiguous local chat template")
    if tokenizer.pad_token_id is None:
        raise ValueError("Local judge tokenizer must declare pad_token_id")
    return tokenizer, {"template_source": source, "template_sha256": text_digest(template),
                       "tokenizer_vocab_sha256": digest(tokenizer.get_vocab())}


def generation_protocol(path, config, tokenizer, *, max_prompt_tokens, max_new_tokens):
    generation_path = path / "generation_config.json"
    generation = json.loads(generation_path.read_text()) if generation_path.is_file() else {}
    eos = generation.get("eos_token_id")
    source = "generation_config.json"
    if eos is None:
        eos = config.get("text_config", config).get("eos_token_id")
        source = "config.json"
    values = eos if isinstance(eos, list) else [eos]
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Judge checkpoint must explicitly declare valid EOS IDs")
    # Preserve Gemma's end_of_turn as well as eos; do not replace [1,106] by 1.
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in values:
        raise ValueError("Judge model/tokenizer EOS definitions disagree")
    return {"version": "local-rubric-generation-v1", "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1,
            "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
            "seed": 0, "dtype": "bfloat16", "attn_implementation": "sdpa",
            "eos_token_id": eos, "eos_token_id_source": source,
            "pad_token_id": tokenizer.pad_token_id, "truncation": False}


def load_judge_model(path, device):
    import torch
    from transformers import Gemma3ForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("Local Gemma3 pilot requires CUDA")
    torch.cuda.set_device(torch.device(device))
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected CUDA device does not support BF16")
    model, info = Gemma3ForConditionalGeneration.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True,
        ignore_mismatched_sizes=False,
    )
    failures = {key: list(info.get(key, [])) for key in
                ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")}
    if any(failures.values()):
        raise RuntimeError("Gemma3 checkpoint loading was incomplete: " + json.dumps(failures, default=str))
    model.requires_grad_(False).eval().to(device)
    if any(str(parameter.device) != device for parameter in model.parameters()):
        raise RuntimeError("Judge was not placed entirely on the requested single CUDA device")
    model.load_report = {"class": type(model).__name__, "device": device, "dtype": "bfloat16",
                         "full_multimodal_weights_loaded": True, **failures}
    return model


def encode_judge_input(tokenizer, metric, prompt, response):
    messages = [{"role": "system", "content": JUDGE_SYSTEMS[metric]},
                {"role": "user", "content": render_judge_prompt(metric, prompt, response)}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, add_special_tokens=False, truncation=False, return_tensors="pt")
    return encoded, text_digest(rendered)


def generate_judgment(model, tokenizer, encoded, protocol, device):
    import torch
    from transformers import GenerationConfig, set_seed
    set_seed(protocol["seed"])
    inputs = encoded.to(device)
    config = GenerationConfig(
        max_new_tokens=protocol["max_new_tokens"], do_sample=False, num_beams=1,
        repetition_penalty=1.0, eos_token_id=protocol["eos_token_id"],
        pad_token_id=protocol["pad_token_id"], use_cache=True,
    )
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=config)
    tokens = output[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
    eos = protocol["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    return {"raw_judge_text": tokenizer.decode(tokens, skip_special_tokens=True),
            "judge_output_token_ids": tokens, "n_output_tokens": len(tokens),
            "stopped_on_eos": bool(tokens and tokens[-1] in eos),
            "reached_token_limit": len(tokens) >= protocol["max_new_tokens"]}


def _append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _judgment_fields(parsed, truncated):
    fields = {key: parsed[key] for key in ("valid", "value", "raw_label", "components", "error")}
    if truncated:
        fields.update(valid=False, value=None, error="judge_output_token_limit")
    if fields["valid"] is not True:
        fields["value"] = None
    elif (type(fields["value"]) not in (int, float) or not math.isfinite(fields["value"])
          or not 0 <= fields["value"] <= 1):
        fields.update(valid=False, value=None, error="invalid_parser_value")
    return fields


def _validate_saved_judgment(row):
    """Check internal consistency, not a signature or permission to rejudge."""
    raw = row.get("raw_judge_text")
    if raw is None:
        # A prompt-budget refusal or runtime failure legitimately has no output.
        if (row.get("valid") is not False or row.get("value") is not None
                or row.get("parser_result") is not None or row.get("raw_label") is not None
                or row.get("components") != {} or not isinstance(row.get("error"), str)
                or not row["error"]):
            raise ValueError("Resume output has inconsistent no-text invalid judgment")
        return
    if not isinstance(raw, str) or any(type(row.get(key)) is not bool for key in
                                      ("reached_token_limit", "stopped_on_eos")):
        raise ValueError("Resume output has invalid raw judge text or stopping metadata")
    parsed = parse_judge_output(row["metric"], raw)
    expected = _judgment_fields(parsed, row["reached_token_limit"] and not row["stopped_on_eos"])
    actual = {key: row.get(key) for key in expected}
    if digest(row.get("parser_result")) != digest(parsed) or digest(actual) != digest(expected):
        raise ValueError("Resume output parser result or score disagrees with raw judge text")


def run_local_judge(dataset, predictions, output, *, model_path, device="cuda:0",
                    max_prompt_tokens=8192, max_new_tokens=256, limit=None,
                    resume=False, allow_test=False):
    if not re.fullmatch(r"cuda:[0-9]+", device):
        raise ValueError("Use an explicit single CUDA device, e.g. cuda:0")
    if any(type(value) is not int or value <= 0 for value in (max_prompt_tokens, max_new_tokens)):
        raise ValueError("Token budgets must be positive integers")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("--limit must be a positive integer")
    if not allow_test and any(row.get("split") == "test" for row in dataset):
        raise ValueError("Local judge calibration is dev-only; test requires explicit --allow-test")
    eligible = judge_inputs(dataset, predictions)
    selected = [row for row in dataset if row["metric"] in JUDGE_SYSTEMS]
    if limit is not None:
        selected = selected[:limit]  # Fixed dataset order, before filtering invalid predictions.
    if not selected:
        raise ValueError("No safety/overrefusal records selected for judging")
    selected_ids = [row["id"] for row in selected]
    inputs = [row for row in eligible if row["id"] in set(selected_ids)]
    originals = {row["id"]: row for row in predictions}
    for item in inputs:
        if originals[item["id"]].get("prompt_sha256") != item["prompt_sha256"]:
            raise ValueError(f"Selected eligible prediction {item['id']} requires its own matching prompt_sha256")
    path = Path(output).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if not path.is_relative_to(ROOT) or path.is_relative_to(model_path) or path.is_relative_to(PUBLIC_MODELS):
        raise ValueError("Judge outputs must stay in the project, outside the read-only model directory")
    manifest = path.with_name(path.name + ".manifest.json")
    receipt = path.with_name(path.name + ".receipt.json")
    lock_path = path.with_name(path.name + ".lock")
    for sidecar in (manifest, receipt, lock_path):
        if sidecar.resolve() != sidecar:
            raise ValueError("Judge sidecar cannot be a symlink")
    if not resume and (path.exists() or manifest.exists()):
        raise FileExistsError("Judge output/manifest already exists; use explicit --resume")
    if resume and not manifest.is_file():
        raise ValueError("Resume requires the original judge manifest")
    runtime = configure_runtime(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model_path, model_config, identity = local_model_identity(model_path)
        tokenizer, tokenizer_identity = load_judge_tokenizer(model_path)
        protocol = generation_protocol(model_path, model_config, tokenizer,
                                       max_prompt_tokens=max_prompt_tokens, max_new_tokens=max_new_tokens)
        packages = {}
        for package in ("torch", "transformers", "tokenizers", "safetensors"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        profile = {"name": PROFILE_NAME, "alternative_judge_model": True,
                   "claim": "Official rubric with alternative local Gemma3; not the original/default or finetuned judge",
                   "loader": {"class": "Gemma3ForConditionalGeneration", "full_multimodal_weights": True,
                              "local_files_only": True, "trust_remote_code": False,
                              "strict_loading_info": True, "use_safetensors": True},
                   "model_identity": identity, "generation": protocol, "tokenizer": tokenizer_identity,
                   "source_lock": load_source_lock(), "systems": JUDGE_SYSTEMS,
                   "parser_version": PARSER_VERSION, "packages": packages, "device": device,
                   "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "protocol_implementation_sha256": hashlib.sha256(
                       Path(__file__).with_name("judge_protocols.py").read_bytes()).hexdigest()}
        judge_id = PROFILE_NAME + ":" + digest(profile)
        contract = {"judge_id": judge_id, "judge_profile": profile,
                    "dataset_sha256": digest(dataset), "predictions_sha256": digest(predictions),
                    "selected_ids": selected_ids, "limit": limit, "allow_test": allow_test}
        if resume:
            if json.loads(manifest.read_text()) != contract:
                raise ValueError("Resume model, inputs, selection or judge configuration differs from manifest")
        else:
            with manifest.open("x", encoding="utf-8") as stream:
                _append(stream, contract)
        existing = read_jsonl(path) if resume and path.exists() else []
        by_id = {row["id"]: row for row in inputs}
        seen = set()
        for row in existing:
            item = by_id.get(row.get("id"))
            if item is None or row["id"] in seen:
                raise ValueError("Resume output has a duplicate or ineligible id")
            if row.get("judge_id") != judge_id or any(row.get(key) != item[key] for key in
                    ("checkpoint_id", "prompt_sha256", "response_sha256", "metric")):
                raise ValueError("Resume output has a foreign judge or stale input binding")
            _validate_saved_judgment(row)
            seen.add(row["id"])
        rows, model = list(existing), None
        report = {"status": "running", "judge_id": judge_id, "manifest": str(manifest),
                  "n_planned": len(selected), "n_eligible": len(inputs), "n_existing": len(existing),
                  "ineligible_selected_ids": [key for key in selected_ids if key not in by_id],
                  "runtime_cache_paths": runtime, "gpu_training": False, "model_load_report": None}
        write_json(receipt, report)
        started = time.monotonic()
        try:
            with path.open("a" if resume else "x", encoding="utf-8") as stream:
                for item in inputs:
                    if item["id"] in seen:
                        continue
                    record = {key: item[key] for key in
                              ("id", "metric", "checkpoint_id", "prompt_sha256", "response_sha256")}
                    record.update(judge_id=judge_id, judge_profile=profile, valid=False, value=None,
                                  raw_label=None, components={}, raw_judge_text=None, error=None)
                    row_started = time.monotonic()
                    try:
                        encoded, rendered_sha = encode_judge_input(tokenizer, item["metric"], item["prompt"], item["response"])
                        record.update(n_prompt_tokens=encoded["input_ids"].shape[1], rendered_prompt_sha256=rendered_sha)
                        if record["n_prompt_tokens"] > max_prompt_tokens:
                            record["error"] = "prompt_over_budget_no_truncation"
                        else:
                            if model is None:
                                model = load_judge_model(model_path, device)
                                report["model_load_report"] = model.load_report
                            generated = generate_judgment(model, tokenizer, encoded, protocol, device)
                            record.update(generated)
                            parsed = parse_judge_output(item["metric"], generated["raw_judge_text"])
                            record["parser_result"] = parsed
                            record.update(_judgment_fields(
                                parsed, generated["reached_token_limit"] and not generated["stopped_on_eos"]))
                    except Exception as exc:
                        record.update(valid=False, value=None, error=f"{type(exc).__name__}: {exc}")
                        record["elapsed_seconds"] = time.monotonic() - row_started
                        _append(stream, record)
                        rows.append(record)
                        raise  # Preserve the failed row; do not continue after a CUDA/runtime error.
                    record["elapsed_seconds"] = time.monotonic() - row_started
                    _append(stream, record)
                    rows.append(record)
            report["status"] = ("completed" if len(rows) == len(selected) and all(row["valid"] for row in rows)
                                else "completed_with_unknown")
        except BaseException as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            report.update(n_judgments=len(rows), n_valid=sum(row["valid"] is True for row in rows),
                          n_invalid=sum(row["valid"] is not True for row in rows),
                          n_unprocessed=len(inputs) - len(rows), elapsed_seconds=time.monotonic() - started)
            write_json(receipt, report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", required=True, help="Existing local Gemma3 directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args(argv)
    report = run_local_judge(read_jsonl(args.dataset), read_jsonl(args.predictions), args.output,
                             model_path=args.model, device=args.device,
                             max_prompt_tokens=args.max_prompt_tokens, max_new_tokens=args.max_new_tokens,
                             limit=args.limit, resume=args.resume, allow_test=args.allow_test)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
