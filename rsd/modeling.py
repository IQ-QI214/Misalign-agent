"""Strict text-model loading for the RSD experiments.

Qwen3.5's published checkpoint contains a vision encoder.  Load its supported
text-only class and let Transformers apply the registered checkpoint-key
conversion.  Saving this model produces an ordinary qwen3_5_text checkpoint,
which this loader also accepts.  Neither entry point moves a model to a GPU.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_QWEN35_TYPES = {"qwen3_5", "qwen3_5_text"}
_QWEN35_UNUSED = re.compile(r"^(?:model\.)?(?:visual\.|mtp(?:\.|_))")


def _local_checkpoint_keys(path: str | Path) -> list[str] | None:
    """Read safetensors headers or a shard index without materializing weights."""
    root = Path(path)
    if not root.is_dir():
        return None
    indexes = [root / "model.safetensors.index.json", root / "pytorch_model.bin.index.json"]
    for index in indexes:
        if index.is_file():
            data = json.loads(index.read_text(encoding="utf-8"))
            weight_map = data.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Invalid checkpoint weight map: {index}")
            return sorted(weight_map)
    weights = root / "model.safetensors"
    if weights.is_file():
        from safetensors import safe_open

        with safe_open(weights, framework="pt", device="cpu") as handle:
            return sorted(handle.keys())
    return None


def _dtype(value: Any):
    import torch

    supported = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    if isinstance(value, str) and value in supported:
        return supported[value]
    if value in supported.values():
        return value
    raise ValueError(f"dtype must be one of {tuple(supported)}, got {value!r}")


def load_text_model(
    model_name_or_path: str | Path,
    revision: str | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
):
    """Load pretrained text weights, fail on incomplete loading, stay on CPU.

    Qwen3.5 uses its explicit text class; other architectures must be registered
    in Transformers' causal-LM auto mapping.  No trust_remote_code, device_map,
    mismatched-size fallback, or random missing-weight acceptance is enabled.
    The returned model has a JSON-serializable ``load_report`` attribute.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    if not isinstance(attn_implementation, str) or not attn_implementation:
        raise ValueError("attn_implementation must be a non-empty string")
    model_name = str(model_name_or_path)
    requested_dtype = _dtype(dtype)
    identity = {"revision": revision, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(model_name, **identity)
    family = config.model_type
    local_keys = _local_checkpoint_keys(model_name)

    if family in _QWEN35_TYPES:
        try:
            from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        except ImportError as exc:
            raise RuntimeError("This Transformers version lacks Qwen3.5 text-model support") from exc
        model_class = Qwen3_5ForCausalLM
        text_config = config.get_text_config() if family == "qwen3_5" else config
        if not isinstance(text_config, Qwen3_5TextConfig):
            raise ValueError(f"Unexpected Qwen3.5 text configuration: {type(text_config).__name__}")
        config = text_config
    else:
        if family not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise ValueError(f"Unsupported causal text-model family: {family!r}")
        # Multimodal containers need an explicitly audited text-loading path.
        if getattr(config, "vision_config", None) is not None:
            raise ValueError(f"No audited text-only loading path for multimodal family: {family!r}")
        model_class = AutoModelForCausalLM

    model, info = model_class.from_pretrained(
        model_name,
        config=config,
        dtype=requested_dtype,
        attn_implementation=attn_implementation,
        output_loading_info=True,
        ignore_mismatched_sizes=False,
        **identity,
    )
    missing = list(info.get("missing_keys", []))
    mismatched = list(info.get("mismatched_keys", []))
    unexpected = list(info.get("unexpected_keys", []))
    errors = list(info.get("error_msgs", []))
    allowed = [key for key in unexpected if family in _QWEN35_TYPES and _QWEN35_UNUSED.match(key)]
    disallowed = [key for key in unexpected if key not in allowed]
    if missing or mismatched or disallowed or errors:
        raise RuntimeError(
            "Checkpoint did not load completely: "
            + json.dumps(
                {"missing_keys": missing, "mismatched_keys": mismatched,
                 "unexpected_keys": disallowed, "error_msgs": errors},
                default=str,
            )
        )
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("Model loading unexpectedly placed parameters outside CPU")
    model.config.use_cache = False
    # Transformers may already suppress the explicitly allowed vision/MTP keys.
    # A local header audit records these omissions without loading their tensors.
    omitted_local = (
        [key for key in local_keys if _QWEN35_UNUSED.match(key)]
        if local_keys is not None and family in _QWEN35_TYPES else []
    )
    model.load_report = {
        "model_name_or_path": model_name,
        "revision": revision,
        "resolved_commit": getattr(config, "_commit_hash", None),
        "source_model_type": family,
        "loaded_model_type": model.config.model_type,
        "model_class": type(model).__name__,
        "dtype": str(requested_dtype),
        "attn_implementation": attn_implementation,
        "missing_keys": missing,
        "mismatched_keys": mismatched,
        "unexpected_keys": unexpected,
        "allowed_unexpected_keys": allowed,
        "local_header_audited": local_keys is not None,
        "local_key_prefixes": sorted({key.split(".")[0] for key in local_keys}) if local_keys else None,
        "omitted_local_non_text_keys": omitted_local,
        "device": "cpu",
    }
    return model


def load_tokenizer(model_name_or_path: str | Path, revision: str | None = None):
    """Load the matching tokenizer; preserve its special-token definitions."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name_or_path), revision=revision, trust_remote_code=False
    )
    # Generation batches must left-pad; single-trajectory training is unaffected.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
