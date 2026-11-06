"""CPU-only scoring for frozen GSM8K / ARC ability cohorts.

``final-answer-v1`` uses NFKC normalization and an uppercase ASCII ``FINAL:``
marker on the last nonempty line. Only numeric tasks permit a whole-response
numeric fallback. Reasoning is never searched for a convenient answer. The
numeric grammar permits signed integers, properly grouped commas, decimals,
and integer fractions; it does not permit units, expressions, or exponents.

A missing response, malformed answer, or truncated generation is UNKNOWN.
Accuracy among valid responses and all-prompt worst/best-case bounds are
reported separately. These bounds are identification bounds, not confidence
intervals, and UNKNOWN is never converted into an observed incorrect answer.

Example::

    python -m rsd.task_evaluation --gold ability_dev_gold.jsonl \
        --responses predictions.jsonl --output ability_scores.json

Test gold is rejected by default, including through the Python API. The
``--allow-confirmation-scoring`` switch is an explicit operator acknowledgement,
NOT an authorization mechanism: the upstream experiment controller must first
verify the six-run confirmation-release gate. This standalone scorer cannot
verify that gate and does not release or select confirmation checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata


PROTOCOL_VERSION = "final-answer-v1"
VERSION = PROTOCOL_VERSION
_BENCHMARK_TYPES = {"gsm8k": "numeric", "arc_challenge": "choice"}
_INTEGER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
_DECIMAL = rf"[+-]?(?:{_INTEGER}(?:\.[0-9]*)?|\.[0-9]+)"
_FRACTION = rf"[+-]?{_INTEGER}/[+-]?{_INTEGER}"
_NUMERIC = re.compile(rf"(?:{_DECIMAL}|{_FRACTION})")
_MARKER = re.compile(r"(?<![A-Za-z0-9_])FINAL[ \t]*:")
_FINAL_LINE = re.compile(r"FINAL:[ \t]*(.*)")
_TRUNCATED = {"length", "max_tokens", "max_new_tokens", "max_output_tokens",
              "max_length", "token_limit", "truncated"}
_COMPLETED = {"stop", "eos", "eos_token", "end_turn"}


def _normalized(text):
    return unicodedata.normalize("NFKC", text).strip()


def _numeric(text):
    if not _NUMERIC.fullmatch(text):
        return None
    try:
        text = text.replace(",", "")
        if "/" in text:
            numerator, denominator = text.split("/")
            return Fraction(int(numerator), int(denominator))
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def _canonical_number(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _unknown(reason, *, format_error=True):
    return {"valid": False, "answer": None, "reason": reason,
            "format_error": format_error, "extraction": None}


def extract_answer(response, answer_type, *, choice_labels=None, finish_reason=None):
    """Extract one final answer without consulting the expected answer.

    Missing finish metadata is accepted but is explicitly recorded by the
    scorer; it does not prove that generation completed. Known truncation and
    any non-completion finish reason override even a well-formed final answer.
    """
    if answer_type not in {"numeric", "choice"}:
        raise ValueError("unsupported answer_type")
    if finish_reason is not None:
        if not isinstance(finish_reason, str):
            raise ValueError("finish_reason must be a string or null")
        if finish_reason in _TRUNCATED:
            return _unknown("truncated_response", format_error=False)
        if finish_reason not in _COMPLETED:
            return _unknown("incomplete_or_unknown_finish_reason", format_error=False)
    if not isinstance(response, str):
        return _unknown("invalid_response_type")
    text = _normalized(response)
    if not text:
        return _unknown("empty_response")
    markers = list(_MARKER.finditer(text))
    if len(markers) > 1:
        return _unknown("multiple_final_markers")
    if markers:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        final = _FINAL_LINE.fullmatch(lines[-1])
        if final is None:
            return _unknown("final_not_standalone_last_line")
        answer = final.group(1).strip()
        extraction = "final_line"
        if not answer:
            return _unknown("empty_final_answer")
    elif answer_type == "numeric" and _numeric(text) is not None:
        answer = text
        extraction = "whole_response_numeric"
    else:
        return _unknown("missing_final_marker")
    if answer_type == "numeric":
        value = _numeric(answer)
        if value is None:
            return _unknown("invalid_numeric_answer")
        answer = _canonical_number(value)
    else:
        if not choice_labels:
            raise ValueError("choice answers require choice_labels")
        if answer not in choice_labels:
            return _unknown("invalid_choice_label")
    return {"valid": True, "answer": answer, "reason": None,
            "format_error": False, "extraction": extraction}


def _index(rows, label):
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label}: every row must be an object")
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"{label}: every row needs a nonempty string id")
        if item_id in indexed:
            raise ValueError(f"{label}: duplicate id {item_id}")
        indexed[item_id] = row
    return indexed


def validate_gold(rows, *, allow_confirmation_scoring=False):
    """Validate gold identities and scalar labels; never accept solution text."""
    indexed = _index(rows, "gold")
    source_ids = set()
    for item_id, row in indexed.items():
        if row.get("split") not in {"dev", "test"}:
            raise ValueError(f"gold {item_id}: split must be dev or test")
        if row["split"] == "test" and not allow_confirmation_scoring:
            raise ValueError("confirmation test scoring is locked; explicit allow_confirmation_scoring is required")
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"gold {item_id}: missing source_id")
        if source_id in source_ids:
            raise ValueError(f"gold: duplicate source_id {source_id}")
        source_ids.add(source_id)
        if _BENCHMARK_TYPES.get(row.get("benchmark")) != row.get("answer_type") or row.get("benchmark") not in _BENCHMARK_TYPES:
            raise ValueError(f"gold {item_id}: invalid benchmark/answer_type pair")
        sha = row.get("prompt_sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(f"gold {item_id}: invalid prompt_sha256")
        expected = row.get("expected_answer")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"gold {item_id}: expected_answer must be a nonempty string")
        if row["answer_type"] == "numeric":
            if _numeric(_normalized(expected)) is None:
                raise ValueError(f"gold {item_id}: invalid scalar numeric expected_answer")
        else:
            labels = row.get("choice_labels")
            if (not isinstance(labels, list) or not 3 <= len(labels) <= 5
                    or any(not isinstance(label, str) or not re.fullmatch(r"[A-Z0-9]", label) for label in labels)
                    or len(set(labels)) != len(labels)):
                raise ValueError(f"gold {item_id}: invalid choice_labels")
            if expected not in labels:
                raise ValueError(f"gold {item_id}: expected_answer is not a choice label")
    return indexed


def _summary(records):
    planned = len(records)
    valid = sum(row["valid"] for row in records)
    missing = sum(row["reason"] == "missing_response" for row in records)
    correct = sum(row["outcome"] == "correct" for row in records)
    unknown = planned - valid
    return {"n_planned": planned, "n_valid": valid, "n_correct": correct,
            "n_incorrect": valid - correct, "n_missing": missing,
            "n_invalid": unknown - missing, "n_unknown": unknown,
            "n_format_errors": sum(row["format_error"] for row in records),
            "unknown_reasons": dict(sorted(Counter(row["reason"] for row in records if not row["valid"]).items())),
            "accuracy_valid": correct / valid if valid else None,
            "value": correct / valid if valid else None,
            "all_prompt_accuracy_lower_bound": correct / planned if planned else None,
            "all_prompt_accuracy_upper_bound": (correct + unknown) / planned if planned else None}


def score_records(gold, responses, *, allow_confirmation_scoring=False):
    """Score only the frozen cohort; extra or duplicate response IDs fail closed."""
    gold = list(gold)
    responses = list(responses)
    indexed_gold = validate_gold(gold, allow_confirmation_scoring=allow_confirmation_scoring)
    indexed_responses = _index(responses, "responses")
    extra = set(indexed_responses) - set(indexed_gold)
    if extra:
        raise ValueError(f"responses contain unknown ids: {sorted(extra)}")
    records = []
    for item_id, reference in indexed_gold.items():
        prediction = indexed_responses.get(item_id)
        record = {"id": item_id, "source_id": reference["source_id"],
                  "split": reference["split"], "benchmark": reference["benchmark"],
                  "metric": "exact_match", "protocol_version": PROTOCOL_VERSION,
                  "prompt_sha256": reference["prompt_sha256"],
                  "response_sha256": None, "finish_reason": None,
                  "finish_reason_observed": False, "prompt_hash_verified": False}
        if prediction is None:
            parsed = _unknown("missing_response", format_error=False)
        else:
            if "prompt_sha256" in prediction:
                if prediction["prompt_sha256"] != reference["prompt_sha256"]:
                    raise ValueError(f"response {item_id}: prompt hash mismatch")
                record["prompt_hash_verified"] = True
            response = prediction.get("response")
            if isinstance(response, str):
                record["response_sha256"] = hashlib.sha256(response.encode("utf-8")).hexdigest()
            record["finish_reason"] = prediction.get("finish_reason")
            record["finish_reason_observed"] = prediction.get("finish_reason") is not None
            parsed = extract_answer(response, reference["answer_type"],
                                    choice_labels=reference.get("choice_labels"),
                                    finish_reason=prediction.get("finish_reason"))
        expected = reference["expected_answer"]
        if reference["answer_type"] == "numeric":
            expected = _canonical_number(_numeric(_normalized(expected)))
        value = float(parsed["answer"] == expected) if parsed["valid"] else None
        record.update(parsed)
        record.update({"value": value, "outcome": "UNKNOWN" if value is None else "correct" if value else "incorrect"})
        records.append(record)
    groups = []
    for split, benchmark in sorted({(row["split"], row["benchmark"]) for row in records}):
        selected = [row for row in records if (row["split"], row["benchmark"]) == (split, benchmark)]
        groups.append({"split": split, "benchmark": benchmark, "metric": "exact_match",
                       "direction": "higher_is_better", **_summary(selected)})
    unknown = sum(not row["valid"] for row in records)
    return {"version": VERSION, "protocol_version": PROTOCOL_VERSION,
            "status": "complete" if records and unknown == 0 else "incomplete",
            "n_planned": len(records), "n_unknown": unknown,
            "confirmation_scoring_acknowledged": bool(allow_confirmation_scoring),
            "confirmation_release_gate_verified": False,
            "bounds_kind": "identification_bounds_not_confidence_intervals",
            "groups": groups, "records": records}


def _read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    return rows


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-confirmation-scoring", action="store_true",
                        help="acknowledge test scoring; upstream six-run release gate is still required")
    args = parser.parse_args(argv)
    # Read and validate the gold gate before even loading model responses.
    gold = _read_jsonl(args.gold)
    validate_gold(gold, allow_confirmation_scoring=args.allow_confirmation_scoring)
    result = score_records(gold, _read_jsonl(args.responses),
                           allow_confirmation_scoring=args.allow_confirmation_scoring)
    _write_json(args.output, result)
    return result


if __name__ == "__main__":
    main()
