"""Dev-only task diagnostics, separate from frozen legacy evaluation results.

Rules are fixed before examining new budget-comparison responses. This module
does not turn calibration results into a confirmatory metric or rewrite scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .evaluation import _validated_inputs, digest, exact_match, read_jsonl, text_digest


DOMAIN_ANSWER_TYPES = {
    "amount_count": "integer", "units": "integer", "sets": "integer", "time": "clock",
}
PROTOCOL = {
    "version": "rsd-task-calibration-v1", "scope": "development_calibration_only",
    "domain_answer_types": DOMAIN_ANSWER_TYPES,
    "normalization": "NFKC_and_outer_whitespace_only",
    "anchors": ["whole_response_value", "explicit_FINAL_or_final_Chinese_line",
                "simple_boxed_value", "last_nonempty_line_value", "last_line_equation_right_hand_value"],
    "ambiguity": "conflicting_values_or_malformed_explicit_markers_are_unknown",
    "integer_content": "exact_rational_value_with_integer_denominator",
    "integer_format": "whole_response_signed_ASCII_digits",
    "clock_content": "H_or_HH_colon_MM_valid_24_hour_time",
    "clock_format": "whole_response_HH_colon_MM_valid_24_hour_time",
    "budget_reached": "observed_length_flag_only_not_proof_of_truncation",
    "gold_usage": "comparison_after_extraction_only",
    "legacy_metric": "legacy_exact_match_v1_unchanged",
}
_INTEGER_FORMAT = re.compile(r"[+-]?[0-9]+")
_RATIONAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?(?:/[+-]?[0-9]+)?")
_CLOCK = re.compile(r"([0-9]{1,2}):([0-9]{2})")
_FINAL = re.compile(r"(?im)^[ \t]*(?:FINAL|最终答案)[ \t]*[:：][ \t]*(.*?)[ \t]*$")
_BOX = re.compile(r"\\boxed\{([^{}]*)\}")


def _normalized(text):
    return unicodedata.normalize("NFKC", text).strip()


def parse_value(text, answer_type):
    """Canonicalize one complete value; never inspect the prompt or gold."""
    text = _normalized(text)
    if answer_type == "integer":
        if not _RATIONAL.fullmatch(text):
            return None
        try:
            value = Fraction(text)
        except (ValueError, ZeroDivisionError):
            return None
        return str(value.numerator) if value.denominator == 1 else None
    if answer_type == "clock":
        match = _CLOCK.fullmatch(text)
        if not match:
            return None
        hour, minute = map(int, match.groups())
        return f"{hour:02d}:{minute:02d}" if 0 <= hour < 24 and 0 <= minute < 60 else None
    raise ValueError(f"Unsupported answer_type: {answer_type}")


def format_ok(response, answer_type):
    text = _normalized(response)
    if answer_type == "integer":
        return _INTEGER_FORMAT.fullmatch(text) is not None
    if answer_type == "clock":
        return re.fullmatch(r"[0-9]{2}:[0-9]{2}", text) is not None and parse_value(text, answer_type) is not None
    raise ValueError(f"Unsupported answer_type: {answer_type}")


def extract_answer(response, answer_type):
    """Extract only declared answer anchors. Signature deliberately excludes gold.

    Generic prose, unfinished expressions and arbitrary last-number matching are
    unsupported. Explicit conflicting final statements remain ambiguous even if
    one of them happens to equal the expected answer.
    """
    if answer_type not in set(DOMAIN_ANSWER_TYPES.values()):
        raise ValueError(f"Unsupported answer_type: {answer_type}")
    text = _normalized(response)
    candidates, malformed = [], []

    def add(source, span, explicit=False):
        value = parse_value(span, answer_type)
        if value is not None:
            candidates.append({"source": source, "text": span, "value": value})
        elif explicit:
            malformed.append({"source": source, "text": span})

    add("whole_response", text)
    for span in _FINAL.findall(text):
        add("explicit_final", span, True)
    boxes = _BOX.findall(text)
    for span in boxes:
        add("boxed", span, True)
    if text.count("\\boxed{") != len(boxes):
        malformed.append({"source": "boxed", "text": "unclosed_or_nested_box"})
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        add("last_nonempty_line", last)
        if "=" in last:
            left, _, right = last.rpartition("=")
            # Do not reinterpret !=, <=, >= or == as an equation answer.
            if left.strip() and not left.rstrip().endswith(("!", "<", ">", "=")):
                add("last_equation_rhs", right.strip())
    values = {candidate["value"] for candidate in candidates}
    if len(values) > 1:
        status, value = "ambiguous", None
    elif malformed:
        status, value = "malformed", None
    elif len(values) == 1:
        status, value = "extracted", next(iter(values))
    else:
        status, value = "no_final_answer", None
    return {"answer_status": status, "answer_value": value,
            "candidates": candidates, "malformed_markers": malformed}


def _validate_protocol_and_prediction(row, prompt, expected_checkpoint):
    if row.get("prompt_sha256") != text_digest(prompt):
        raise ValueError(f"{row['id']}: missing or mismatching prompt hash")
    identity = row.get("checkpoint_identity")
    if identity is not None and (not isinstance(identity, dict) or identity.get("checkpoint_id") != expected_checkpoint):
        raise ValueError(f"{row['id']}: checkpoint identity disagrees with prediction")
    if "response_sha256" in row and (not isinstance(row.get("response"), str) or
                                      row["response_sha256"] != text_digest(row["response"])):
        raise ValueError(f"{row['id']}: response hash mismatch")
    protocol = row.get("generation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{row['id']}: generation protocol is required")
    budget = protocol.get("max_new_tokens")
    if type(budget) is not int or budget <= 0:
        raise ValueError(f"{row['id']}: invalid generation budget")
    n_tokens = row.get("n_tokens")
    reached = row.get("reached_token_limit")
    if n_tokens is not None:
        if type(n_tokens) is not int or not 0 <= n_tokens <= budget:
            raise ValueError(f"{row['id']}: invalid token count")
        if reached is not None and (type(reached) is not bool or reached != (n_tokens == budget)):
            raise ValueError(f"{row['id']}: inconsistent budget flag")
    elif reached is not None and type(reached) is not bool:
        raise ValueError(f"{row['id']}: budget flag must be boolean")
    stop = row.get("stop_reason", "unknown")
    if stop not in {"eos", "length", "error", "unknown"}:
        raise ValueError(f"{row['id']}: unsupported stop_reason")
    if stop == "length" and reached is not True:
        raise ValueError(f"{row['id']}: length stop without reached budget")
    return protocol


def calibrate_records(dataset, predictions):
    if not dataset or any(row.get("split") != "dev" for row in dataset):
        raise ValueError("Calibration accepts a nonempty dev split only; test evaluation is disabled")
    for row in dataset:
        if row.get("metric") != "exact_match" or row.get("domain") not in DOMAIN_ANSWER_TYPES:
            raise ValueError(f"{row.get('id')}: explicit supported task domain and exact_match are required")
        if not isinstance(row.get("task_type"), str) or not row["task_type"]:
            raise ValueError(f"{row.get('id')}: task_type is required")
        answer_type = DOMAIN_ANSWER_TYPES[row["domain"]]
        if parse_value(str(row.get("expected_answer", "")), answer_type) is None:
            raise ValueError(f"{row.get('id')}: expected answer is invalid for declared domain")
    data, pred, _, checkpoint_id = _validated_inputs(dataset, predictions, [], None)
    if checkpoint_id is None:
        raise ValueError("At least one identified prediction is required")
    protocols = {}
    for item_id, prediction in pred.items():
        protocol = _validate_protocol_and_prediction(prediction, data[item_id]["prompt"], checkpoint_id)
        protocols[digest(protocol)] = protocol
    if len(protocols) != 1:
        raise ValueError("Predictions must use exactly one generation protocol")
    generation_protocol = next(iter(protocols.values()))
    records = []
    for item_id, row in data.items():
        answer_type = DOMAIN_ANSWER_TYPES[row["domain"]]
        prediction = pred.get(item_id)
        diagnostic = {
            "id": item_id, "source_id": row["source_id"], "family": row["family"],
            "domain": row["domain"], "task_type": row["task_type"], "split": "dev",
            "answer_type": answer_type, "prompt_sha256": text_digest(row["prompt"]),
            "response_sha256": None, "legacy_exact_match_v1": None, "legacy_reason": None,
            "format_ok": None, "answer_correct": None, "answer_status": "generation_error",
            "answer_value": None, "budget_reached": None, "stop_reason": "unknown",
            "generation_status": "missing", "reason": "missing_prediction",
            "n_tokens": None, "candidates": [], "malformed_markers": [],
        }
        if prediction is not None:
            response = prediction.get("response")
            diagnostic.update(budget_reached=prediction.get("reached_token_limit"),
                              stop_reason=prediction.get("stop_reason", "unknown"),
                              n_tokens=prediction.get("n_tokens"))
            if isinstance(response, str):
                diagnostic["response_sha256"] = text_digest(response)
            error = prediction.get("error")
            # Empty decoded text with generated tokens is an observed failure to
            # deliver an answer, rather than a transport/model-loading failure.
            empty_observed = isinstance(response, str) and not response.strip() and type(prediction.get("n_tokens")) is int
            if error or not isinstance(response, str) or (prediction.get("valid") is False and not empty_observed):
                diagnostic.update(generation_status="error", reason=error or "invalid_prediction")
            else:
                diagnostic.update(extract_answer(response, answer_type))
                legacy, legacy_reason = exact_match(response, row["expected_answer"])
                diagnostic.update(generation_status="observed", reason=None,
                                  legacy_exact_match_v1=legacy, legacy_reason=legacy_reason,
                                  format_ok=format_ok(response, answer_type))
                if diagnostic["answer_status"] == "extracted":
                    gold = parse_value(str(row["expected_answer"]), answer_type)
                    diagnostic["answer_correct"] = float(diagnostic["answer_value"] == gold)
        records.append(diagnostic)
    groups = []
    for key in sorted({(r["family"], r["task_type"], r["answer_type"]) for r in records}):
        selected = [r for r in records if (r["family"], r["task_type"], r["answer_type"]) == key]
        observed = [r for r in selected if r["generation_status"] == "observed"]
        extracted = [r for r in observed if r["answer_status"] == "extracted"]
        legacy = [r["legacy_exact_match_v1"] for r in selected if r["legacy_exact_match_v1"] is not None]
        correct = sum(r["answer_correct"] == 1 for r in extracted)
        counts = Counter(r["answer_status"] for r in selected)
        groups.append({
            "family": key[0], "task_type": key[1], "answer_type": key[2], "split": "dev",
            "n_planned": len(selected), "n_sources": len({r["source_id"] for r in selected}),
            "n_observed": len(observed), "n_generation_error": sum(r["generation_status"] == "error" for r in selected),
            "n_missing": sum(r["generation_status"] == "missing" for r in selected),
            "n_correct": correct, "n_incorrect": len(extracted) - correct,
            "n_no_final_answer": counts["no_final_answer"], "n_ambiguous": counts["ambiguous"],
            "n_malformed": counts["malformed"], "n_answer_unknown": len(selected) - len(extracted),
            "n_format_ok": sum(r["format_ok"] is True for r in observed),
            "n_format_failed": sum(r["format_ok"] is False for r in observed),
            "n_budget_reached": sum(r["budget_reached"] is True for r in selected),
            "n_budget_unknown": sum(r["budget_reached"] is None for r in selected),
            "stop_reason_counts": dict(sorted(Counter(r["stop_reason"] for r in selected).items())),
            "answer_accuracy_when_extractable": correct / len(extracted) if extracted else None,
            "extraction_coverage_of_planned": len(extracted) / len(selected),
            "confirmed_correct_fraction_of_planned": correct / len(selected),
            "format_rate_when_observed": sum(r["format_ok"] is True for r in observed) / len(observed) if observed else None,
            "legacy_exact_match_v1": sum(legacy) / len(legacy) if legacy else None,
            "n_legacy_valid": len(legacy),
        })
    return {
        "schema_version": "rsd-task-calibration-v1", "scope": "development_calibration_only",
        "confirmatory_metric": False, "checkpoint_id": checkpoint_id,
        "dataset_sha256": digest(sorted(dataset, key=lambda row: row["id"])),
        "cohort_sha256": digest(sorted([{"id": r["id"], "source_id": r["source_id"],
                                          "prompt_sha256": r["prompt_sha256"]} for r in records], key=lambda row: row["id"])),
        "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generation_protocol": generation_protocol, "generation_protocol_sha256": digest(generation_protocol),
        "n_planned": len(dataset), "n_predictions": len(predictions), "groups": groups, "records": records,
        "interpretation": [
            "Dev calibration only, not a newly confirmed primary metric or safety-transfer result.",
            "Answer accuracy is conditional on extraction; always report coverage and unknown counts alongside it.",
            "Confirmed-correct fraction of planned questions is a coverage lower bound, not proof that unknown answers are wrong.",
            "Budget reached does not prove truncation; absent stop metadata remains unknown.",
            "Legacy scores, training data and historical prediction files are unchanged.",
        ],
    }


def compare_budgets(short, long):
    """Compare paired 128/512 diagnostics; only max_new_tokens may differ."""
    for key in ("schema_version", "scope", "checkpoint_id", "dataset_sha256", "cohort_sha256",
                "protocol_sha256", "implementation_sha256"):
        if not short.get(key) or short[key] != long.get(key):
            raise ValueError(f"Budget comparison identity mismatch: {key}")
    if short["scope"] != "development_calibration_only":
        raise ValueError("Budget comparison is dev-only")
    for value in (short, long):
        if value.get("confirmatory_metric") is not False:
            raise ValueError("Budget comparison requires descriptive calibration artifacts")
        if digest(value["generation_protocol"]) != value["generation_protocol_sha256"]:
            raise ValueError("Generation protocol hash mismatch")
        if digest(value["protocol"]) != value["protocol_sha256"]:
            raise ValueError("Calibration protocol hash mismatch")
        required = {"profile", "seed", "eos_token_id", "eos_token_id_source", "pad_token_id",
                    "tokenizer_vocab_sha256", "tokenizer_template_sha256", "enable_thinking",
                    "max_prompt_tokens", "temperature", "do_sample", "top_p", "top_k",
                    "num_beams", "repetition_penalty", "dtype"}
        if any(key not in value["generation_protocol"] or value["generation_protocol"][key] is None for key in required):
            raise ValueError("Budget comparison requires complete generation identity/settings")
    a, b = dict(short["generation_protocol"]), dict(long["generation_protocol"])
    if a.pop("max_new_tokens", None) != 128 or b.pop("max_new_tokens", None) != 512:
        raise ValueError("Expected ordered 128/512 budgets; equal or other budgets are not this comparison")
    if a != b:
        raise ValueError("Only max_new_tokens may differ; seed/EOS/prompt-generation settings changed")
    left = {r["id"]: r for r in short["records"]}
    right = {r["id"]: r for r in long["records"]}
    if len(left) != len(short["records"]) or len(right) != len(long["records"]) or left.keys() != right.keys():
        raise ValueError("Budget comparison record cohorts differ")
    pairs = []
    for item_id, row in left.items():
        other = right[item_id]
        if any(row[k] != other[k] for k in ("source_id", "prompt_sha256", "family", "task_type", "answer_type", "split")):
            raise ValueError("Budget comparison prompt or task identity differs")
        if row["generation_status"] != "observed" or other["generation_status"] != "observed":
            raise ValueError("Budget comparison requires both observed generations for every planned record")
        pairs.append({"id": item_id, "source_id": row["source_id"],
                      "answer_status_128": row["answer_status"], "answer_status_512": other["answer_status"],
                      "answer_correct_128": row["answer_correct"], "answer_correct_512": other["answer_correct"],
                      "format_ok_128": row["format_ok"], "format_ok_512": other["format_ok"],
                      "budget_reached_128": row["budget_reached"], "budget_reached_512": other["budget_reached"]})
    return {"schema_version": "rsd-budget-calibration-comparison-v1", "scope": "development_calibration_only",
            "checkpoint_id": short["checkpoint_id"], "dataset_sha256": short["dataset_sha256"],
            "protocol_sha256": short["protocol_sha256"], "budgets": [128, 512],
            "only_changed_generation_field": "max_new_tokens", "n_pairs": len(pairs), "records": pairs,
            "groups_128": short["groups"], "groups_512": long["groups"],
            "interpretation": "Paired budget sensitivity diagnostics only; no causal or confirmatory performance claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = calibrate_records(read_jsonl(args.dataset), read_jsonl(args.predictions))
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


if __name__ == "__main__":
    main()
