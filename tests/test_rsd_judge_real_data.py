"""Synthetic CPU fixtures only: neither model executions nor human gold."""
import copy
import json
from pathlib import Path

import pytest

from rsd import judge_real_data as real
from rsd.evaluation import digest
from test_rsd_judge_validation import bundle  # shared synthetic dev-source fixture


def generators():
    result = {}
    for number in (1, 2):
        model, revision = f"fixture/model-{number}", str(number) * 40
        result[str(number)] = {
            "checkpoint_identity": {"kind": "huggingface_commit", "model_id": model, "revision": revision,
                                    "checkpoint_id": f"hf:{model}@{revision}"},
            "tokenizer_identity": {"vocab_sha256": str(number) * 64, "chat_template_sha256": "a" * 64},
            "generation_protocol": {"profile": "synthetic-cpu-fixture", "do_sample": False,
                                    "enable_thinking": False, "max_new_tokens": 16, "max_prompt_tokens": 128,
                                    "seed": 0, "eos_token_id": 1, "pad_token_id": 0, "dtype": "float32",
                                    "temperature": 0.0, "top_p": 1.0, "top_k": 0, "num_beams": 1,
                                    "repetition_penalty": 1.0}}
    return result


def response_files(request_dir, target, *, mutate=None, receipt_mutate=None):
    request_manifest = real._sha(request_dir / "manifest.json")
    payload = json.loads((request_dir / "response_bundle.template.private.json").read_text())
    payload.update(request_manifest_sha256=request_manifest, state="FINAL_ATTEMPTS")
    for index, row in enumerate(payload["records"]):
        row.update(response=f"  Raw synthetic fixture response {index}.\n", valid=True,
                   stop_reason="eos", generation_executed=True, n_tokens=4)
    if mutate:
        mutate(payload)
    execution = json.loads((request_dir / "execution_receipt.template.private.json").read_text())
    execution.update(request_manifest_sha256=request_manifest, response_records_sha256=digest(payload["records"]),
                     model_calls_executed=sum(r["generation_executed"] for r in payload["records"]))
    execution["producer"].update(implementation_sha256="b" * 64, run_ids={"1": "cpu-test-run-1", "2": "cpu-test-run-2"})
    if receipt_mutate:
        receipt_mutate(execution)
    execution_path = target / "external-execution.private.json"
    execution_path.write_text(json.dumps(execution) + "\n")
    payload["execution_receipt"] = {"path": execution_path.name, "sha256": real._sha(execution_path)}
    path = target / "responses.private.json"
    path.write_text(json.dumps(payload) + "\n")
    return path


@pytest.fixture
def frozen_requests(bundle, tmp_path):
    path = tmp_path / "requests"
    manifest = real.prepare_requests(bundle[0], path, generators=generators(), purpose="cpu_fixture")
    return path, manifest


def test_current_missing_identity_exports_only_draft_requests(bundle, tmp_path):
    path = tmp_path / "draft"
    manifest = real.prepare_requests(bundle[0], path)
    assert manifest["state"] == "DRAFT_IDENTITIES_MISSING"
    assert manifest["n_planned"] == 96 and manifest["n_source_clusters"] == 48
    assert manifest["prepared_answer_records"] == manifest["prepared_human_gold_records"] == manifest["model_calls_executed"] == 0
    assert manifest["source_validation_bundle_manifest_sha256"] == real._sha(bundle[0] / "manifest.json")
    for number in (1, 2):
        records = json.loads((path / f"generator_{number}.requests.private.json").read_text())
        assert len(records) == 48
        assert all(r["generator_identity_sha256"] is None and r["checkpoint_id"] is None for r in records)
    with pytest.raises(ValueError, match="two complete frozen"):
        real.load_requests(path, require_frozen=True)
    with pytest.raises(ValueError, match="two complete frozen"):
        real.ingest_responses(path, path / "response_bundle.template.private.json", tmp_path / "invalid")


def test_partial_generator_identity_stays_draft(bundle, tmp_path):
    values = generators(); values["2"]["tokenizer_identity"] = None
    manifest = real.prepare_requests(bundle[0], tmp_path / "partial", generators=values, purpose="cpu_fixture")
    assert manifest["state"] == "DRAFT_IDENTITIES_MISSING"
    assert manifest["generator_identity_complete"] == {"1": True, "2": False}


@pytest.mark.parametrize("damage", ["mutable_revision", "missing_tokenizer_hash", "missing_generation_seed", "same_checkpoint"])
def test_bad_generator_identity_never_becomes_frozen(bundle, tmp_path, damage):
    values = generators()
    if damage == "mutable_revision": values["1"]["checkpoint_identity"]["revision"] = "main"
    elif damage == "missing_tokenizer_hash": values["1"]["tokenizer_identity"].pop("vocab_sha256")
    elif damage == "missing_generation_seed": values["1"]["generation_protocol"].pop("seed")
    else: values["2"]["checkpoint_identity"] = copy.deepcopy(values["1"]["checkpoint_identity"])
    with pytest.raises(ValueError):
        real.prepare_requests(bundle[0], tmp_path / "bad", generators=values, purpose="cpu_fixture")


def test_ingest_preserves_raw_text_private_identity_and_unknown_gold(frozen_requests, tmp_path):
    request_dir, manifest = frozen_requests
    responses = response_files(request_dir, tmp_path)
    output = tmp_path / "ingested"
    report = real.ingest_responses(request_dir, responses, output)
    loaded, rows = real.load_candidates(output)
    assert loaded == report
    assert report["prepared_answer_records"] == 96 and report["n_attempt_records"] == 96
    assert report["prepared_source_clusters"] == 48 and report["prepared_human_gold_records"] == 0
    assert report["model_calls_executed_by_this_tool"] == 0 and not report["execution_independently_verified"]
    assert report["purpose"] == "cpu_fixture"
    assert all(row["purpose"] == "cpu_fixture" and row["model_output"] is False for row in rows)
    assert rows[0]["response"] == "  Raw synthetic fixture response 0.\n"
    assert all(row["stratum"] == "B" and row["human_gold"] is None and row["human_gold_status"] == "UNKNOWN" for row in rows)
    assert all(not {"generator_identity", "generator_slot", "checkpoint_id", "author_hypothesis"} & set(row) for row in rows)
    assert real._sha(output / "source_response_bundle.private.json") == real._sha(responses)
    assert real.ingest_responses(request_dir, responses, output) == report  # exact idempotent verification only


def test_failed_empty_truncated_and_unattempted_slots_are_retained(frozen_requests, tmp_path):
    request_dir, _ = frozen_requests
    def mutate(payload):
        records = payload["records"]
        records[0].update(valid=False, error="synthetic OOM", response="Partial error text", stop_reason="error")
        records[1].update(response="", valid=False, stop_reason="eos")
        records[2].update(response="unfinished", valid=False, stop_reason="length", n_tokens=16)
        records[3].update(response=None, valid=False, stop_reason="not_generated", generation_executed=False)
        records[4].update(response="FINAL: 3", valid=False, stop_reason="eos")
        records[5].update(response="unknown ending", valid=True, stop_reason="unknown")
    responses = response_files(request_dir, tmp_path, mutate=mutate)
    output = tmp_path / "partial"
    report = real.ingest_responses(request_dir, responses, output)
    attempts = json.loads((output / "attempts.private.json").read_text())
    assert len(attempts) == 96 and report["n_planned"] == 96 and report["prepared_answer_records"] == 90
    assert report["attempt_status_counts"] == {"GENERATION_FAILED": 2, "EMPTY_RESPONSE": 1, "TRUNCATED_RESPONSE": 1,
                                              "NOT_GENERATED": 1, "INCOMPLETE_RESPONSE": 1, "COMPLETE_RESPONSE": 90}
    assert attempts[0]["response"] == "Partial error text"
    assert attempts[2]["stop_reason"] == "length"
    assert attempts[3]["generation_executed"] is False
    assert report["model_calls_claimed_by_producer"] == 95
    assert len(real.load_candidates(output)[1]) == 90


@pytest.mark.parametrize("damage", ["missing", "duplicate", "foreign_id", "wrong_prompt", "wrong_source", "wrong_checkpoint", "wrong_config", "wrong_slot_hash", "wrong_purpose", "finish_conflict", "no_execution", "wrong_request_manifest"])
def test_import_rejects_wrong_slot_identity_or_missing_denominator_before_claim(frozen_requests, tmp_path, damage):
    request_dir, _ = frozen_requests
    def mutate(payload):
        row = payload["records"][0]
        if damage == "missing": payload["records"].pop()
        elif damage == "duplicate": payload["records"][1] = copy.deepcopy(row)
        elif damage == "foreign_id": row["id"] = "unplanned-question"
        elif damage == "wrong_prompt": row["prompt_sha256"] = "0" * 64
        elif damage == "wrong_source": row["source_id"] = "different-source"
        elif damage == "wrong_checkpoint": row["checkpoint_id"] = "unfrozen-model"
        elif damage == "wrong_config": row["generation_protocol_sha256"] = "0" * 64
        elif damage == "wrong_slot_hash": row["request_sha256"] = "0" * 64
        elif damage == "wrong_purpose": payload["purpose"] = "research"
        elif damage == "finish_conflict": row["finish_reason"] = "length"
        elif damage == "no_execution": row["generation_executed"] = False
        else: payload["request_manifest_sha256"] = "0" * 64
    response = response_files(request_dir, tmp_path, mutate=mutate)
    with pytest.raises(ValueError):
        real.ingest_responses(request_dir, response, tmp_path / "bad")
    assert not (request_dir / "ingestion-claim.private.json").exists()


@pytest.mark.parametrize("damage", ["wrong_records_hash", "wrong_driver", "missing_driver_hash", "same_run", "wrong_count"])
def test_execution_receipt_is_required_and_content_bound(frozen_requests, tmp_path, damage):
    request_dir, _ = frozen_requests
    def mutate(receipt):
        if damage == "wrong_records_hash": receipt["response_records_sha256"] = "0" * 64
        elif damage == "wrong_driver": receipt["producer"]["kind"] = "external_model_driver"
        elif damage == "missing_driver_hash": receipt["producer"]["implementation_sha256"] = None
        elif damage == "same_run": receipt["producer"]["run_ids"]["2"] = receipt["producer"]["run_ids"]["1"]
        else: receipt["model_calls_executed"] = 95
    response = response_files(request_dir, tmp_path, receipt_mutate=mutate)
    with pytest.raises(ValueError):
        real.ingest_responses(request_dir, response, tmp_path / "bad")


def test_consumed_request_forbids_selective_replacement(frozen_requests, tmp_path):
    request_dir, _ = frozen_requests
    response = response_files(request_dir, tmp_path)
    real.ingest_responses(request_dir, response, tmp_path / "first")
    with pytest.raises(FileExistsError, match="already consumed"):
        real.ingest_responses(request_dir, response, tmp_path / "second")
    response = response_files(request_dir, tmp_path, mutate=lambda data: data["records"][0].update(response="a chosen replacement"))
    with pytest.raises(FileExistsError, match="already consumed"):
        real.ingest_responses(request_dir, response, tmp_path / "first")


def test_bundle_tamper_symlink_and_source_changes_fail_closed(frozen_requests, tmp_path):
    request_dir, manifest = frozen_requests
    response = response_files(request_dir, tmp_path)
    output = tmp_path / "ingested"
    real.ingest_responses(request_dir, response, output)
    target = output / "candidates.json"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact changed"):
        real.load_candidates(output)
    link = tmp_path / "linked-requests"; link.symlink_to(request_dir)
    with pytest.raises(ValueError, match="Symlink"):
        real.load_requests(link)
    source = Path(manifest["source_validation_bundle_dir"]) / "real_dev_generation_slots.json"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="binding mismatch"):
        real.load_requests(request_dir)


@pytest.mark.parametrize("field,value", [("purpose", "research"), ("prepared_answer_records", 200),
                                         ("prepared_human_gold_records", 96), ("execution_independently_verified", True),
                                         ("attempt_status_counts", {}), ("judge_batch_ready", True)])
def test_summary_metadata_cannot_upgrade_fixture_or_hide_attempts(frozen_requests, tmp_path, field, value):
    request_dir, _ = frozen_requests
    response = response_files(request_dir, tmp_path)
    output = tmp_path / "ingested"
    real.ingest_responses(request_dir, response, output)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest[field] = value
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        real.load_candidates(output)


def test_request_order_is_predeclared_not_rewritten_by_importer(frozen_requests):
    request_dir, _ = frozen_requests
    path = request_dir / "generator_1.requests.private.json"
    rows = json.loads(path.read_text()); rows.reverse(); path.write_text(json.dumps(rows))
    manifest = json.loads((request_dir / "manifest.json").read_text())
    manifest["files_sha256"][path.name] = real._sha(path)
    (request_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fixed request order"):
        real.load_requests(request_dir)


def test_composed_panel_source_rejected_before_recursive_loader_dispatch(frozen_requests, tmp_path, monkeypatch):
    request_dir, _ = frozen_requests
    panel = tmp_path / "cyclic-panel"
    panel.mkdir()
    (panel / "manifest.json").write_text(json.dumps({"version": "judge-validation-panel-v1"}))
    manifest = json.loads((request_dir / "manifest.json").read_text())
    manifest["source_validation_bundle_dir"] = str(panel)
    manifest["source_validation_bundle_manifest_sha256"] = real._sha(panel / "manifest.json")
    (request_dir / "manifest.json").write_text(json.dumps(manifest))
    calls = []
    def cyclic_dispatch(path):
        calls.append(path)
        return real.load_requests(request_dir)
    monkeypatch.setattr(real, "load_bundle", cyclic_dispatch)
    with pytest.raises(ValueError, match="original validation-materials"):
        real.load_requests(request_dir)
    assert calls == []


def test_candidate_checks_request_hash_before_loading_dependencies(frozen_requests, tmp_path, monkeypatch):
    request_dir, _ = frozen_requests
    response = response_files(request_dir, tmp_path)
    output = tmp_path / "ingested"
    real.ingest_responses(request_dir, response, output)
    path = request_dir / "manifest.json"
    path.write_text(path.read_text() + "\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("An unbound dependency must not be loaded")
    monkeypatch.setattr(real, "load_requests", forbidden)
    with pytest.raises(ValueError, match="before loading dependencies"):
        real.load_candidates(output)
