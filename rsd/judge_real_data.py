"""Prepare and ingest fixed B-layer dev-generation requests on CPU.

No model/network/remote driver exists here. Identity strings and a producer's
execution receipt establish content bindings, not independent proof of model
execution. CPU fixtures cannot be relabeled as research by this API.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .confirmation import no_symlink, write_new
from .evaluation import digest, text_digest
from .judge_validation import VERSION as SOURCE_BUNDLE_VERSION, load_bundle

VERSION = "judge-real-data-v1"
REQUEST_VERSION = "judge-real-generation-request-v1"
RESPONSE_VERSION = "judge-real-responses-v1"
EXECUTION_VERSION = "judge-real-execution-v1"
SHA = re.compile(r"[0-9a-f]{64}")
REQUIRED_RESPONSE = {"id", "source_dev_id", "source_id", "prompt_sha256", "generator_slot",
                     "request_sha256", "generator_identity_sha256", "checkpoint_id", "generation_protocol_sha256",
                     "response", "valid", "stop_reason", "error", "generation_executed"}


def _sha(path):
    return hashlib.sha256(no_symlink(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(no_symlink(path).read_text())


def _valid_sha(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _copy_new(source, destination):
    """Preserve the producer's exact bytes for the original-file SHA binding."""
    destination = no_symlink(destination)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(no_symlink(source).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        destination.chmod(0o600)
    finally:
        os.unlink(temporary)


def _validate_checkpoint(identity):
    if not isinstance(identity, dict):
        raise ValueError("Generator checkpoint must have a content identity")
    if identity.get("kind") == "huggingface_commit":
        model, revision = identity.get("model_id"), identity.get("revision")
        if (not isinstance(model, str) or not model.strip() or not isinstance(revision, str)
                or not re.fullmatch(r"[0-9a-f]{40}", revision)
                or identity.get("checkpoint_id") != f"hf:{model}@{revision}"):
            raise ValueError("Generator Hub checkpoint requires an immutable commit")
    elif identity.get("kind") == "local_files":
        files = identity.get("files_sha256", {})
        if (not isinstance(files, dict) or "config.json" not in files
                or not any(name.endswith((".safetensors", ".bin")) for name in files)
                or any(Path(name).name != name or not _valid_sha(value) for name, value in files.items())
                or identity.get("checkpoint_id") != "sha256:" + digest(files)):
            raise ValueError("Local checkpoint needs configuration and weight content hashes")
    else:
        raise ValueError("Unsupported generator checkpoint identity")


def _validate_generator(value):
    """Return completeness; missing identity is draft, malformed identity errors."""
    if value is None:
        return False
    if not isinstance(value, dict):
        raise ValueError("Generator definition must be an object")
    if any(value.get(k) is None for k in ("checkpoint_identity", "tokenizer_identity", "generation_protocol")):
        return False
    _validate_checkpoint(value["checkpoint_identity"])
    tokenizer = value["tokenizer_identity"]
    if (not isinstance(tokenizer, dict)
            or any(not _valid_sha(tokenizer.get(k)) for k in ("vocab_sha256", "chat_template_sha256"))):
        raise ValueError("Tokenizer identity requires vocabulary and chat-template hashes")
    generation = value["generation_protocol"]
    if (not isinstance(generation, dict) or not isinstance(generation.get("profile"), str) or not generation["profile"]
            or type(generation.get("do_sample")) is not bool or type(generation.get("enable_thinking")) is not bool
            or type(generation.get("seed")) is not int or not 0 <= generation["seed"] < 2**32
            or any(type(generation.get(k)) is not int or generation[k] < 1 for k in ("max_new_tokens", "max_prompt_tokens"))
            or type(generation.get("pad_token_id")) is not int or generation["pad_token_id"] < 0
            or generation.get("dtype") not in {"float32", "float16", "bfloat16"}):
        raise ValueError("Generation protocol must freeze budgets, sampling, seed, dtype and tokenizer IDs")
    eos = generation.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(token) is not int or token < 0 for token in eos):
        raise ValueError("Generation EOS must be explicit")
    if (any(type(generation.get(k)) not in (int, float) or not math.isfinite(generation[k])
            for k in ("temperature", "top_p", "repetition_penalty"))
            or not 0 < generation["top_p"] <= 1 or generation["repetition_penalty"] <= 0
            or type(generation.get("top_k")) is not int or generation["top_k"] < 0
            or type(generation.get("num_beams")) is not int or generation["num_beams"] < 1
            or (generation["temperature"] <= 0 if generation["do_sample"] else generation["temperature"] != 0)):
        raise ValueError("Sampling parameters must be explicit and consistent with do_sample")
    return True


def _source_slots(bundle_dir):
    root = no_symlink(bundle_dir)
    # Inspect the source type before invoking the shared loader, which can
    # dispatch panel -> B ingestion -> requests -> source and otherwise recurse.
    header = _read(root / "manifest.json")
    if not isinstance(header, dict) or header.get("version") != SOURCE_BUNDLE_VERSION:
        raise ValueError("B requests require an original validation-materials source, never a composed panel")
    manifest, _ = load_bundle(root)
    slots = _read(root / "real_dev_generation_slots.json")
    if (not isinstance(slots, list) or len(slots) != 96 or len({r.get("id") for r in slots}) != 96
            or manifest["B"]["planned_answer_slots"] != 96 or manifest["B"]["actual_answer_records"] != 0):
        raise ValueError("B generation requires the original 96 unfilled slots")
    sources = defaultdict(list)
    for row in slots:
        if (not isinstance(row.get("id"), str) or not row["id"].startswith("judge-v3/B/")
                or not isinstance(row.get("source_id"), str) or not row["source_id"]
                or not isinstance(row.get("source_dev_id"), str) or not row["source_dev_id"]
                or row.get("metric") not in {"safety", "overrefusal"}
                or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                or row.get("prompt_sha256") != text_digest(row["prompt"])
                or type(row.get("generator_slot")) is not int or row["generator_slot"] not in (1, 2)
                or row.get("response") is not None or row.get("human_gold") is not None
                or row.get("generator_identity") is not None or row.get("status") != "PENDING_GENERATION"):
            raise ValueError("Malformed or already-filled source generation slot")
        sources[row["source_id"]].append(row)
    if len(sources) != 48 or Counter(group[0]["metric"] for group in sources.values()) != {"safety": 24, "overrefusal": 24}:
        raise ValueError("B must contain 48 sources, 24 per metric")
    for group in sources.values():
        if (len(group) != 2 or {r["generator_slot"] for r in group} != {1, 2}
                or any(group[0][k] != group[1][k] for k in ("source_dev_id", "metric", "prompt", "prompt_sha256"))):
            raise ValueError("Each source must have the same question for two fixed generators")
    if {r["source_dev_id"] for r in slots} != set(manifest["B"]["selected_source_dev_ids"]):
        raise ValueError("B questions differ from the original selected cohort")
    return manifest, slots


def prepare_requests(validation_bundle_dir, output_dir, *, generators=None, purpose="research"):
    if purpose not in {"research", "cpu_fixture"}:
        raise ValueError("Explicit research or cpu_fixture purpose required")
    source, slots = _source_slots(validation_bundle_dir)
    source_sha = _sha(Path(validation_bundle_dir) / "manifest.json")
    generators = copy.deepcopy(generators) if generators is not None else {
        str(i): {"checkpoint_identity": None, "tokenizer_identity": None, "generation_protocol": None} for i in (1, 2)}
    if not isinstance(generators, dict) or set(generators) != {"1", "2"}:
        raise ValueError("Exactly generator slots 1 and 2 are required")
    complete = [_validate_generator(generators[str(i)]) for i in (1, 2)]
    if all(complete) and generators["1"]["checkpoint_identity"]["checkpoint_id"] == generators["2"]["checkpoint_identity"]["checkpoint_id"]:
        raise ValueError("Two distinct fixed model checkpoints are required")
    state = "FROZEN" if all(complete) else "DRAFT_IDENTITIES_MISSING"
    requests = {"1": [], "2": []}
    for slot in slots:
        number = str(slot["generator_slot"])
        generator = generators[number] if complete[int(number) - 1] else None
        row = {k: slot[k] for k in ("id", "source_dev_id", "source_id", "metric", "prompt", "prompt_sha256", "generator_slot")}
        row.update(version=REQUEST_VERSION, purpose=purpose, source_validation_bundle_manifest_sha256=source_sha,
                   generator_identity_sha256=digest(generator) if generator else None,
                   checkpoint_id=generator["checkpoint_identity"]["checkpoint_id"] if generator else None,
                   generation_protocol_sha256=digest(generator["generation_protocol"]) if generator else None)
        row["request_sha256"] = digest(row)
        requests[number].append(row)
    response_template = {"version": RESPONSE_VERSION, "purpose": purpose, "request_manifest_sha256": None,
                         "state": "UNFILLED_TEMPLATE_NOT_GENERATED", "execution_receipt": {"path": None, "sha256": None},
                         "records": [{**{k: row[k] for k in sorted(REQUIRED_RESPONSE - {"response", "valid", "stop_reason", "error", "generation_executed"})},
                                      "response": None, "valid": False, "stop_reason": "not_generated", "error": None,
                                      "generation_executed": False} for number in ("1", "2") for row in requests[number]]}
    execution_template = {"version": EXECUTION_VERSION, "purpose": purpose, "request_manifest_sha256": None,
                          "response_records_sha256": None, "generator_identity_sha256": {k: digest(v) if complete[int(k) - 1] else None for k, v in generators.items()},
                          "producer": {"kind": "external_model_driver" if purpose == "research" else "synthetic_cpu_fixture",
                                       "implementation_sha256": None, "run_ids": {"1": None, "2": None}},
                          "model_calls_executed": None, "execution_independently_verified": False}
    payloads = {"generator_1.requests.private.json": requests["1"], "generator_2.requests.private.json": requests["2"],
                "generators.private.json": generators, "response_bundle.template.private.json": response_template,
                "execution_receipt.template.private.json": execution_template}
    output = no_symlink(output_dir)
    if output.exists():
        raise FileExistsError("Use a new immutable request bundle directory")
    output.mkdir(parents=True, mode=0o700)
    for name, value in payloads.items():
        write_new(output / name, value)
    manifest = {"version": VERSION, "bundle_kind": "requests", "state": state, "purpose": purpose,
                "source_validation_bundle_dir": str(no_symlink(validation_bundle_dir)),
                "source_validation_bundle_manifest_sha256": source_sha,
                "source_slots_sha256": source["files_sha256"]["real_dev_generation_slots.json"],
                "n_planned": 96, "n_source_clusters": 48, "requests_per_generator": {"1": 48, "2": 48},
                "prepared_answer_records": 0, "prepared_human_gold_records": 0, "model_calls_executed": 0,
                "generator_identity_complete": {str(i): complete[i - 1] for i in (1, 2)},
                "request_order_sha256": {number: digest([r["id"] for r in rows]) for number, rows in requests.items()},
                "files_sha256": {name: _sha(output / name) for name in payloads},
                "implementation_sha256": _sha(__file__), "generation_driver_implemented": False,
                "scientific_conclusion": None}
    write_new(output / "manifest.json", manifest)
    return manifest


def load_requests(bundle_dir, *, require_frozen=False):
    root = no_symlink(bundle_dir)
    manifest = _read(root / "manifest.json")
    if (manifest.get("version") != VERSION or manifest.get("bundle_kind") != "requests"
            or manifest.get("purpose") not in {"research", "cpu_fixture"}
            or manifest.get("implementation_sha256") != _sha(__file__)):
        raise ValueError("Request bundle implementation/version/purpose differs")
    required = {"generator_1.requests.private.json", "generator_2.requests.private.json", "generators.private.json",
                "response_bundle.template.private.json", "execution_receipt.template.private.json"}
    if set(manifest.get("files_sha256", {})) != required:
        raise ValueError("Request bundle file inventory differs")
    for name, expected in manifest["files_sha256"].items():
        if _sha(root / name) != expected:
            raise ValueError("Frozen request artifact changed")
    source, slots = _source_slots(manifest["source_validation_bundle_dir"])
    if (_sha(Path(manifest["source_validation_bundle_dir"]) / "manifest.json") != manifest["source_validation_bundle_manifest_sha256"]
            or source["files_sha256"]["real_dev_generation_slots.json"] != manifest["source_slots_sha256"]):
        raise ValueError("Original validation source bundle changed")
    generators = _read(root / "generators.private.json")
    completeness = {str(i): _validate_generator(generators[str(i)]) for i in (1, 2)}
    complete = all(completeness.values())
    expected_state = "FROZEN" if complete else "DRAFT_IDENTITIES_MISSING"
    if manifest["state"] != expected_state or require_frozen and not complete:
        raise ValueError("Generation/import requires two complete frozen generator identities")
    if (manifest.get("generator_identity_complete") != completeness or manifest.get("n_planned") != 96
            or manifest.get("n_source_clusters") != 48 or manifest.get("requests_per_generator") != {"1": 48, "2": 48}
            or manifest.get("prepared_answer_records") != 0 or manifest.get("prepared_human_gold_records") != 0
            or manifest.get("model_calls_executed") != 0):
        raise ValueError("Request preparation counts/completeness differ")
    if complete and generators["1"]["checkpoint_identity"]["checkpoint_id"] == generators["2"]["checkpoint_identity"]["checkpoint_id"]:
        raise ValueError("Distinct checkpoints required")
    rows = []
    for number in (1, 2):
        group = _read(root / f"generator_{number}.requests.private.json")
        if (len(group) != 48 or any(r.get("generator_slot") != number for r in group)
                or digest([r.get("id") for r in group]) != manifest["request_order_sha256"][str(number)]):
            raise ValueError("Per-generator fixed request order/coverage differs")
        rows.extend(group)
    source_index = {r["id"]: r for r in slots}
    if len(rows) != 96 or len({r["id"] for r in rows}) != 96:
        raise ValueError("Request slot coverage differs")
    for row in rows:
        source_row = source_index.get(row["id"])
        if source_row is None or any(row[k] != source_row[k] for k in ("source_id", "source_dev_id", "prompt", "prompt_sha256", "metric", "generator_slot")):
            raise ValueError("Request question/slot identity differs from original")
        if (row.get("version") != REQUEST_VERSION or row.get("purpose") != manifest["purpose"]
                or row.get("source_validation_bundle_manifest_sha256") != manifest["source_validation_bundle_manifest_sha256"]):
            raise ValueError("Request purpose/source bundle binding differs")
        if row.get("request_sha256") != digest({k: v for k, v in row.items() if k != "request_sha256"}):
            raise ValueError("Generation request hash mismatch")
        generator = generators[str(row["generator_slot"])]
        if complete and (row["generator_identity_sha256"] != digest(generator)
                         or row["checkpoint_id"] != generator["checkpoint_identity"]["checkpoint_id"]
                         or row["generation_protocol_sha256"] != digest(generator["generation_protocol"])):
            raise ValueError("Request uses a different generator configuration")
    return manifest, rows, generators


def _classify(record, generator):
    if (type(record.get("valid")) is not bool or type(record.get("generation_executed")) is not bool
            or record.get("response") is not None and not isinstance(record["response"], str)
            or record.get("error") is not None and not isinstance(record["error"], str)
            or not isinstance(record.get("stop_reason"), str)):
        raise ValueError("Malformed response/attempt status")
    reason = record["stop_reason"]
    if "finish_reason" in record and record["finish_reason"] != reason:
        raise ValueError("stop_reason and finish_reason disagree")
    if not record["generation_executed"]:
        if record["response"] is not None or record["valid"] or reason != "not_generated":
            raise ValueError("Ungenerated slot cannot contain a model answer")
        return "NOT_GENERATED", "no_generation_attempt"
    if reason == "not_generated":
        raise ValueError("Attempted generation cannot be marked not_generated")
    if record.get("error"):
        return "GENERATION_FAILED", "producer_error"
    if not isinstance(record["response"], str) or not record["response"].strip():
        return "EMPTY_RESPONSE", "empty_response"
    if reason in {"length", "max_tokens", "max_new_tokens", "max_output_tokens", "max_length", "token_limit", "truncated"}:
        return "TRUNCATED_RESPONSE", "length_limit"
    if record["valid"] is False:
        return "GENERATION_FAILED", "producer_marked_invalid"
    if reason not in {"stop", "eos", "eos_token", "end_turn"}:
        return "INCOMPLETE_RESPONSE", "unrecognized_or_incomplete_stop"
    if type(record.get("n_tokens")) is not int or not 0 < record["n_tokens"] <= generator["generation_protocol"]["max_new_tokens"]:
        raise ValueError("Complete generation requires positive in-budget n_tokens")
    return "COMPLETE_RESPONSE", None


def _validate_responses(manifest, requests, generators, response_bundle, execution):
    if (response_bundle.get("version") != RESPONSE_VERSION or response_bundle.get("purpose") != manifest["purpose"]
            or response_bundle.get("state") != "FINAL_ATTEMPTS"):
        raise ValueError("Response bundle is not a final attempt set of the same purpose")
    records = response_bundle.get("records", [])
    by_id = {}
    for row in records:
        if not isinstance(row, dict) or not REQUIRED_RESPONSE <= set(row) or row["id"] in by_id:
            raise ValueError("Every generation slot requires exactly one explicit response/attempt record")
        by_id[row["id"]] = row
    if set(by_id) != {r["id"] for r in requests} or len(records) != 96:
        raise ValueError("Preserve all 96 slots; omitted/extra slots or replacements are forbidden")
    identities = {key: digest(value) for key, value in generators.items()}
    producer = execution.get("producer", {})
    kind = "external_model_driver" if manifest["purpose"] == "research" else "synthetic_cpu_fixture"
    if (execution.get("version") != EXECUTION_VERSION or execution.get("purpose") != manifest["purpose"]
            or execution.get("request_manifest_sha256") != response_bundle["request_manifest_sha256"]
            or execution.get("response_records_sha256") != digest(records)
            or execution.get("generator_identity_sha256") != identities
            or producer.get("kind") != kind or not _valid_sha(producer.get("implementation_sha256"))
            or not isinstance(producer.get("run_ids"), dict) or set(producer["run_ids"]) != {"1", "2"}
            or any(not isinstance(value, str) or not value.strip() for value in producer["run_ids"].values())
            or len(set(producer["run_ids"].values())) != 2):
        raise ValueError("External execution receipt does not bind requests, records and both generator runs")
    attempts, candidates, provenance = [], [], []
    for request in requests:
        record = by_id[request["id"]]
        fields = ("id", "source_id", "source_dev_id", "prompt_sha256", "generator_slot", "request_sha256",
                  "generator_identity_sha256", "checkpoint_id", "generation_protocol_sha256")
        if any(record[key] != request[key] for key in fields):
            raise ValueError("Response source/prompt/slot/checkpoint/config differs from frozen request")
        generator = generators[str(request["generator_slot"])]
        if "generation_protocol" in record and record["generation_protocol"] != generator["generation_protocol"]:
            raise ValueError("Returned full generation protocol differs")
        if "checkpoint_identity" in record and record["checkpoint_identity"] != generator["checkpoint_identity"]:
            raise ValueError("Returned full checkpoint identity differs")
        if "prompt" in record and record["prompt"] != request["prompt"]:
            raise ValueError("Returned prompt text differs")
        status, reason = _classify(record, generator)
        response_hash = text_digest(record["response"]) if isinstance(record["response"], str) else None
        if "response_sha256" in record and record["response_sha256"] != response_hash:
            raise ValueError("Returned response hash differs")
        attempt = {k: record[k] for k in sorted(REQUIRED_RESPONSE)}
        attempt.update(purpose=manifest["purpose"], status=status, reason=reason, response_sha256=response_hash,
                       source_record_sha256=digest(record), n_tokens=record.get("n_tokens"))
        attempts.append(attempt)
        if status == "COMPLETE_RESPONSE":
            candidates.append({"id": request["id"], "source_id": request["source_id"], "stratum": "B", "metric": request["metric"],
                "purpose": manifest["purpose"], "prompt": request["prompt"], "response": record["response"],
                "prompt_sha256": request["prompt_sha256"], "response_sha256": response_hash,
                "origin": "producer_attested_model_generated_dev_response" if manifest["purpose"] == "research" else "synthetic_cpu_fixture_response",
                "model_output": manifest["purpose"] == "research",
                "execution_independently_verified": False, "human_gold": None, "human_gold_status": "UNKNOWN"})
        provenance.append({"id": request["id"], "source_dev_id": request["source_dev_id"], "generator_slot": request["generator_slot"],
                           "generator_identity": generator, "request_sha256": request["request_sha256"],
                           "status": status, "source_record_sha256": digest(record)})
    calls = sum(r["generation_executed"] for r in records)
    if type(execution.get("model_calls_executed")) is not int or execution["model_calls_executed"] != calls:
        raise ValueError("Execution call count differs from all retained attempts")
    return attempts, candidates, provenance


def ingest_responses(request_bundle, response_bundle_json, output_dir):
    request_root = no_symlink(request_bundle)
    manifest, requests, generators = load_requests(request_root, require_frozen=True)
    response_path = no_symlink(response_bundle_json)
    responses = _read(response_path)
    request_sha = _sha(request_root / "manifest.json")
    if responses.get("request_manifest_sha256") != request_sha:
        raise ValueError("Response manifest is not bound to this pre-frozen request bundle")
    receipt_ref = responses.get("execution_receipt", {})
    if not isinstance(receipt_ref.get("path"), str) or not receipt_ref["path"]:
        raise ValueError("A content-bound external execution receipt is required")
    execution_path = no_symlink(response_path.parent / receipt_ref["path"])
    if _sha(execution_path) != receipt_ref.get("sha256"):
        raise ValueError("Execution receipt bytes changed")
    execution = _read(execution_path)
    attempts, candidates, provenance = _validate_responses(manifest, requests, generators, responses, execution)
    output = no_symlink(output_dir)
    claim = {"version": VERSION, "request_manifest_sha256": request_sha, "response_bundle_sha256": _sha(response_path),
             "execution_receipt_sha256": _sha(execution_path), "output_dir": str(output), "policy": "one final ingestion per frozen request; no selective retry"}
    claim_path = request_root / "ingestion-claim.private.json"
    if claim_path.exists():
        if _read(claim_path) == claim and (output / "manifest.json").is_file():
            return load_candidates(output)[0]
        raise FileExistsError("Frozen request already consumed; replacement answers/new output directory are forbidden")
    if output.exists():
        raise FileExistsError("Use a new immutable ingestion output directory")
    write_new(claim_path, claim)
    output.mkdir(parents=True, mode=0o700)
    payloads = {"candidates.json": candidates, "attempts.private.json": attempts, "provenance.private.json": provenance}
    for name, value in payloads.items():
        write_new(output / name, value)
    _copy_new(response_path, output / "source_response_bundle.private.json")
    _copy_new(execution_path, output / "source_execution_receipt.private.json")
    names = set(payloads) | {"source_response_bundle.private.json", "source_execution_receipt.private.json"}
    result = {"version": VERSION, "bundle_kind": "ingested_responses", "purpose": manifest["purpose"],
              "state": "ALL_96_COMPLETE_RESPONSES" if len(candidates) == 96 else "INCOMPLETE_RESPONSES_RETAINED",
              "request_bundle_dir": str(request_root), "request_manifest_sha256": request_sha,
              "source_validation_bundle_manifest_sha256": manifest["source_validation_bundle_manifest_sha256"],
              "source_response_bundle_sha256": _sha(response_path), "source_execution_receipt_sha256": _sha(execution_path),
              "ingestion_claim_sha256": _sha(claim_path), "n_planned": 96, "n_source_clusters": 48,
              "n_attempt_records": 96, "prepared_answer_records": len(candidates),
              "prepared_source_clusters": len({r["source_id"] for r in candidates}), "prepared_human_gold_records": 0,
              "attempt_status_counts": dict(Counter(r["status"] for r in attempts)),
              "model_calls_claimed_by_producer": execution["model_calls_executed"], "model_calls_executed_by_this_tool": 0,
              "execution_independently_verified": False, "judge_batch_ready": False,
              "files_sha256": {name: _sha(output / name) for name in sorted(names)}, "implementation_sha256": _sha(__file__),
              "scientific_conclusion": None}
    write_new(output / "manifest.json", result)
    return result


def load_candidates(bundle_dir):
    root = no_symlink(bundle_dir)
    result = _read(root / "manifest.json")
    if (result.get("version") != VERSION or result.get("bundle_kind") != "ingested_responses"
            or result.get("implementation_sha256") != _sha(__file__)):
        raise ValueError("Unknown/changed B ingestion bundle implementation")
    required = {"candidates.json", "attempts.private.json", "provenance.private.json", "source_response_bundle.private.json", "source_execution_receipt.private.json"}
    if set(result.get("files_sha256", {})) != required:
        raise ValueError("B ingestion file inventory differs")
    for name, expected in result["files_sha256"].items():
        if _sha(root / name) != expected:
            raise ValueError("B ingestion artifact changed")
    request_root = no_symlink(result["request_bundle_dir"])
    if result["request_manifest_sha256"] != _sha(request_root / "manifest.json"):
        raise ValueError("B request manifest changed before loading dependencies")
    manifest, requests, generators = load_requests(request_root, require_frozen=True)
    if (result.get("purpose") != manifest["purpose"] or result["request_manifest_sha256"] != _sha(request_root / "manifest.json")
            or result["source_validation_bundle_manifest_sha256"] != manifest["source_validation_bundle_manifest_sha256"]
            or result["ingestion_claim_sha256"] != _sha(request_root / "ingestion-claim.private.json")):
        raise ValueError("B ingestion/request/source-purpose binding differs")
    claim = _read(request_root / "ingestion-claim.private.json")
    if (claim["output_dir"] != str(root) or claim["response_bundle_sha256"] != result["source_response_bundle_sha256"]
            or claim["execution_receipt_sha256"] != result["source_execution_receipt_sha256"]
            or _sha(root / "source_response_bundle.private.json") != claim["response_bundle_sha256"]
            or _sha(root / "source_execution_receipt.private.json") != claim["execution_receipt_sha256"]):
        raise ValueError("B answers are not the first committed ingestion")
    responses, execution = _read(root / "source_response_bundle.private.json"), _read(root / "source_execution_receipt.private.json")
    if responses["request_manifest_sha256"] != result["request_manifest_sha256"]:
        raise ValueError("Copied response manifest is unbound")
    attempts, candidates, provenance = _validate_responses(manifest, requests, generators, responses, execution)
    if (candidates != _read(root / "candidates.json") or attempts != _read(root / "attempts.private.json")
            or provenance != _read(root / "provenance.private.json")
            or result["n_planned"] != 96 or result["n_attempt_records"] != 96
            or result["prepared_answer_records"] != len(candidates)
            or result["prepared_source_clusters"] != len({r["source_id"] for r in candidates})
            or result["n_source_clusters"] != 48
            or result["attempt_status_counts"] != dict(Counter(r["status"] for r in attempts))
            or result["state"] != ("ALL_96_COMPLETE_RESPONSES" if len(candidates) == 96 else "INCOMPLETE_RESPONSES_RETAINED")
            or result["model_calls_claimed_by_producer"] != execution["model_calls_executed"]
            or result["model_calls_executed_by_this_tool"] != 0
            or result["execution_independently_verified"] is not False or result["judge_batch_ready"] is not False
            or result["prepared_human_gold_records"] != 0):
        raise ValueError("B candidate/attempt evidence or denominator differs")
    return result, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--validation-bundle", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--generators")
    prepare.add_argument("--purpose", choices=["research", "cpu_fixture"], default="research")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--requests", required=True)
    ingest.add_argument("--responses", required=True)
    ingest.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_requests(args.validation_bundle, args.output,
                                  generators=_read(args.generators) if args.generators else None, purpose=args.purpose)
    else:
        result = ingest_responses(args.requests, args.responses, args.output)
    print(json.dumps({k: result[k] for k in ("state", "purpose", "n_planned", "prepared_answer_records", "prepared_human_gold_records")}, indent=2))


if __name__ == "__main__":
    main()
