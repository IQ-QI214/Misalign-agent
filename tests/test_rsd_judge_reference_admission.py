"""Synthetic engineering reviews only; no real human labels or model calls."""
import copy
import json
from pathlib import Path

import pytest

from rsd import judge_reference_admission as admission
from rsd import judge_reference_data as reference
from rsd.judge_validation import blind_sheet


def dump(path, value):
    path.write_bytes(reference.canonical(value) + b"\n")


@pytest.fixture
def source_bundle(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    rows = []
    for i in range(40):
        prompt, response = f"Unique source question {i}?", f"Original source answer {i}."
        rows.append({"id": f"judge-v3/C/fixture-{i}", "source_id": "prompt-sha256:" + reference.sha(reference.norm(prompt).encode()),
                     "stratum": "C", "prompt": prompt, "response": response,
                     # Simulated producer schema only, never a real model response.
                     "origin": "official_public_external_model_response", "model_output": True,
                     "prompt_sha256": reference.sha(prompt.encode()), "response_sha256": reference.sha(response.encode()),
                     "metric": None, "prompt_eligibility": "UNKNOWN", "accepted_for_validation": False,
                     "human_gold": None, "human_gold_status": "UNKNOWN",
                     "provenance": {"subset": "helpful-base" if i % 2 else "harmless-base",
                                    "answer_field": "chosen" if i < 20 else "rejected"}})
    payloads = {"candidates.private.json": rows,
                "blind_key.private.json": {admission._annotation_id(row): row["id"] for row in rows}}
    for i in (1, 2):
        prompts, answers = reference.blind_reviews(rows, i)
        payloads[f"reviewer_{i}_prompt_eligibility.json"] = prompts
        payloads[f"reviewer_{i}_responses.json"] = answers
    for name, value in payloads.items():
        dump(root / name, value)
    manifest = {"version": reference.VERSION, "state": "C40_CANDIDATES_PREPARED_NOT_ACCEPTED_NOT_LABELED",
                "candidate_records": 40, "unique_source_clusters": 40, "accepted_validation_records": 0,
                "human_prompt_eligibility_labels": 0, "human_rubric_gold_records": 0,
                "files": {name: {"sha256": reference.sha((root / name).read_bytes()), "bytes": (root / name).stat().st_size}
                          for name in payloads}}
    dump(root / "manifest.json", manifest)
    return root, rows


def reviews(source_bundle, tmp_path, labels1=None, labels2=None):
    root, rows = source_bundle
    default = {row["id"]: "BENIGN" if i < 20 else "HARMFUL" for i, row in enumerate(rows)}
    labels1 = default if labels1 is None else labels1
    labels2 = labels1 if labels2 is None else labels2
    paths = []
    for slot, labels in ((1, labels1), (2, labels2)):
        document = json.loads((root / f"reviewer_{slot}_prompt_eligibility.json").read_text())
        document.update(reviewer_id=f"SYNTHETIC_TEST_ONLY_REVIEWER_{slot}", reviewer_kind="human", independent_annotation_claim=True)
        by_id = {admission._annotation_id(row): row["id"] for row in rows}
        for item in document["annotations"]:
            item["prompt_eligibility"] = labels[by_id[item["annotation_id"]]]
            item["reason"] = None if item["prompt_eligibility"] == "UNKNOWN" else "Synthetic CPU fixture, not actual human review."
        path = tmp_path / f"completed-review-{slot}.json"
        dump(path, document)
        paths.append(path)
    return paths


def rebind_output_artifact(root, name):
    manifest = json.loads((root / "manifest.json").read_text())
    blob = (root / name).read_bytes()
    manifest["files"][name] = {"sha256": reference.sha(blob), "bytes": len(blob)}
    dump(root / "manifest.json", manifest)


def test_load_reference_and_original_blank_forms_are_not_human_reviews(source_bundle, tmp_path):
    root, rows = source_bundle
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.iterdir()}
    manifest, actual = admission.load_reference(root)
    assert actual == rows and manifest["candidate_records"] == 40
    output = tmp_path / "no-admission"
    with pytest.raises(ValueError, match="Named human reviewer"):
        admission.admit(root, [root / "reviewer_1_prompt_eligibility.json", root / "reviewer_2_prompt_eligibility.json"], output)
    assert not output.exists()
    assert before == {p.relative_to(root).as_posix(): p.read_bytes() for p in root.iterdir()}


def test_admission_uses_human_eligibility_not_bucket_or_preference_and_replays(source_bundle, tmp_path):
    root, rows = source_bundle
    paths = reviews(source_bundle, tmp_path)
    output = tmp_path / "admission"
    manifest = admission.admit(root, list(reversed(paths)), output)
    loaded, decisions, candidates = admission.load_admission(output)
    assert loaded == manifest
    assert manifest["reference_bundle_manifest_sha256"] == reference.sha((root / "manifest.json").read_bytes())
    assert manifest["admitted_metric_counts"] == {"overrefusal": 20, "safety": 20}
    assert manifest["target_exactly_met"] is True and manifest["human_rubric_gold_records"] == 0
    assert len(decisions) == len(candidates) == 40
    assert json.loads((output / "eligibility_review_1.original.json").read_text())["reviewer_slot"] == 1
    assert candidates[0]["provenance"]["subset"] == "harmless-base"
    assert candidates[0]["metric"] == "overrefusal"  # No source bucket mapping.
    assert all(row["human_gold"] is None and row["human_gold_status"] == "UNKNOWN" for row in candidates)
    for i in (1, 2):
        assert json.loads((output / f"reviewer_{i}.json").read_text()) == blind_sheet(candidates, i)
    with pytest.raises(FileExistsError):
        admission.admit(root, paths, output)


def test_unknown_disagreement_unsuitable_and_coverage_gaps_preserve_all_40(source_bundle, tmp_path):
    root, rows = source_bundle
    left = {row["id"]: "BENIGN" for row in rows}
    right = left.copy()
    for i in range(5):
        left[rows[i]["id"]] = right[rows[i]["id"]] = "UNSUITABLE"
    for i in range(5, 10):
        right[rows[i]["id"]] = "HARMFUL"
    for i in range(10, 15):
        left[rows[i]["id"]] = right[rows[i]["id"]] = "UNKNOWN"
    result = admission.admit(root, reviews(source_bundle, tmp_path, left, right), tmp_path / "mixed")
    assert result["candidate_records"] == 40
    assert (result["admitted_records"], result["excluded_records"], result["unknown_records"]) == (25, 5, 10)
    assert result["coverage_gap"] == {"overrefusal": 0, "safety": 20}
    assert result["coverage_excess"] == {"overrefusal": 5, "safety": 0}
    assert result["target_exactly_met"] is False
    _, decisions, candidates = admission.load_admission(tmp_path / "mixed")
    assert len(decisions) == 40 and len(candidates) == 25  # No cap at target20.
    assert all(decision["metric"] is None for decision in decisions if decision["admission_status"] != "ADMITTED")


def test_distinct_adjudicator_can_resolve_only_disputed_or_unknown_cases(source_bundle, tmp_path):
    root, rows = source_bundle
    left = {row["id"]: "BENIGN" for row in rows}
    right = left.copy()
    right[rows[0]["id"]] = "HARMFUL"
    left[rows[1]["id"]] = right[rows[1]["id"]] = "UNKNOWN"
    paths = reviews(source_bundle, tmp_path, left, right)
    template = admission.adjudication_template(root, paths)
    assert len(template["annotations"]) == 2
    assert all(item["prompt_eligibility"] == "UNKNOWN" for item in template["annotations"])
    template.update(reviewer_id="SYNTHETIC_TEST_ONLY_ADJUDICATOR", reviewer_kind="human", independent_annotation_claim=True)
    template["annotations"][0].update(prompt_eligibility="HARMFUL", reason="Synthetic dispute resolution.")
    path = tmp_path / "adjudication.json"
    dump(path, template)
    result = admission.admit(root, paths, tmp_path / "adjudicated", adjudication_path=path)
    assert result["admitted_records"] == 39 and result["unknown_records"] == 1
    _, decisions, _ = admission.load_admission(tmp_path / "adjudicated")
    assert decisions[0]["metric"] == "safety" and decisions[0]["resolution"] == "HUMAN_ADJUDICATION"
    template["reviewer_id"] = "synthetic_test_only_reviewer_1"  # Casefold identity duplicate.
    dump(path, template)
    with pytest.raises(ValueError, match="third distinct"):
        admission.admit(root, paths, tmp_path / "bad-adjudicator", adjudication_path=path)


def test_adjudication_cannot_override_primary_known_agreement(source_bundle, tmp_path):
    root, _ = source_bundle
    paths = reviews(source_bundle, tmp_path)
    template = admission.adjudication_template(root, paths)
    assert template["annotations"] == []
    template.update(reviewer_id="SYNTHETIC_TEST_ONLY_ADJUDICATOR", reviewer_kind="human", independent_annotation_claim=True)
    template["annotations"] = [json.loads(paths[0].read_text())["annotations"][0]]
    adjudication = tmp_path / "unnecessary-adjudication.json"
    dump(adjudication, template)
    with pytest.raises(ValueError, match="only unresolved"):
        admission.admit(root, paths, tmp_path / "overridden", adjudication_path=adjudication)


def test_two_declared_human_unknown_reviews_do_not_become_zero_or_accepted(source_bundle, tmp_path):
    root, rows = source_bundle
    unknown = {row["id"]: "UNKNOWN" for row in rows}
    output = tmp_path / "all-unknown"
    manifest = admission.admit(root, reviews(source_bundle, tmp_path, unknown), output)
    assert (manifest["admitted_records"], manifest["excluded_records"], manifest["unknown_records"]) == (0, 0, 40)
    assert manifest["coverage_gap"] == {"overrefusal": 20, "safety": 20}
    _, decisions, candidates = admission.load_admission(output)
    assert candidates == [] and all(row["metric"] is None for row in decisions)
    assert json.loads((output / "reviewer_1.json").read_text())["annotations"] == []


@pytest.mark.parametrize("mutation", ["boolean_label", "integer_label", "missing_case", "duplicate_case", "foreign_id",
                                     "changed_prompt", "wrong_hash", "answer_label", "same_identity", "boolean_slot",
                                     "claimed_human_integer", "same_slot", "empty_rationale"])
def test_invalid_primary_reviews_fail_before_output(source_bundle, tmp_path, mutation):
    root, _ = source_bundle
    paths = reviews(source_bundle, tmp_path)
    document = json.loads(paths[1].read_text())
    item = document["annotations"][0]
    if mutation == "boolean_label": item["prompt_eligibility"] = True
    elif mutation == "integer_label": item["prompt_eligibility"] = 1
    elif mutation == "missing_case": document["annotations"].pop()
    elif mutation == "duplicate_case": document["annotations"].append(copy.deepcopy(item))
    elif mutation == "foreign_id": item["annotation_id"] = "unknown"
    elif mutation == "changed_prompt": item["prompt"] = "Different question keeping the old hash."
    elif mutation == "wrong_hash": item["prompt_sha256"] = "0" * 64
    elif mutation == "answer_label": item["raw_label"] = "1_full_compliance"
    elif mutation == "same_identity": document["reviewer_id"] = "SYNTHETIC_TEST_ONLY_REVIEWER_1"
    elif mutation == "boolean_slot": document["reviewer_slot"] = True
    elif mutation == "claimed_human_integer": document["independent_annotation_claim"] = 1
    elif mutation == "same_slot": document["reviewer_slot"] = 1
    elif mutation == "empty_rationale": item["reason"] = " "
    dump(paths[1], document)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        admission.admit(root, paths, output)
    assert not output.exists()


def test_same_review_file_and_symlink_inputs_and_outputs_are_rejected(source_bundle, tmp_path):
    root, _ = source_bundle
    paths = reviews(source_bundle, tmp_path)
    with pytest.raises(ValueError, match="separate files"):
        admission.admit(root, [paths[0], paths[0]], tmp_path / "same")
    link = tmp_path / "review-link.json"
    link.symlink_to(paths[0])
    with pytest.raises(ValueError, match="Symlink"):
        admission.admit(root, [link, paths[1]], tmp_path / "linked")
    directory_link = tmp_path / "source-link"
    directory_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        admission.load_reference(directory_link)
    destination = tmp_path / "output-link"
    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="Symlink"):
        admission.admit(root, paths, destination)


@pytest.mark.parametrize("mutation", ["reported_count", "reported_boolean_count", "decision", "candidate_metric", "candidate_boolean", "blind_sheet", "review_identity"])
def test_load_recomputes_decisions_even_when_changed_artifact_hash_is_rebound(source_bundle, tmp_path, mutation):
    root, _ = source_bundle
    output = tmp_path / "admitted"
    admission.admit(root, reviews(source_bundle, tmp_path), output)
    if mutation in {"reported_count", "reported_boolean_count"}:
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["admitted_records"] = 39 if mutation == "reported_count" else True
        dump(output / "manifest.json", manifest)
    elif mutation == "review_identity":
        name = "eligibility_review_1.original.json"
        document = json.loads((output / name).read_text())
        document["reviewer_id"] = "A different identity"
        dump(output / name, document)
        rebind_output_artifact(output, name)
    else:
        name = "decisions.json" if mutation == "decision" else "reviewer_1.json" if mutation == "blind_sheet" else "admitted_candidates.json"
        value = json.loads((output / name).read_text())
        if mutation == "decision": value[0]["metric"] = "safety"
        elif mutation == "candidate_metric": value[0]["metric"] = "safety"
        elif mutation == "candidate_boolean": value[0]["accepted_for_validation"] = 1
        else: value["annotations"][0]["raw_label"] = "1_full_compliance"
        dump(output / name, value)
        rebind_output_artifact(output, name)
    with pytest.raises(ValueError):
        admission.load_admission(output)


def test_original_reference_artifact_and_snapshot_symlinks_are_checked(source_bundle, tmp_path):
    root, _ = source_bundle
    output = tmp_path / "admitted"
    admission.admit(root, reviews(source_bundle, tmp_path), output)
    path = output / "reference/candidates.private.json"
    blob = path.read_bytes()
    path.unlink()
    target = tmp_path / "copy.json"
    target.write_bytes(blob)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        admission.load_admission(output)


def test_reference_manifest_cannot_redirect_paths_or_preassign_gold(source_bundle):
    root, _ = source_bundle
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["../external.json"] = manifest["files"].pop("blind_key.private.json")
    dump(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="inventory"):
        admission.load_reference(root)


@pytest.mark.parametrize("case", ["preassigned_metric", "preassigned_gold", "missing_gold", "duplicated_source"])
def test_reference_candidate_semantics_checked_after_valid_byte_hash(source_bundle, case):
    root, rows = source_bundle
    rows = copy.deepcopy(rows)
    if case == "preassigned_metric": rows[0]["metric"] = "safety"
    elif case == "preassigned_gold": rows[0]["human_gold"] = {"value": 0}
    elif case == "missing_gold": del rows[0]["human_gold"]
    else: rows[1]["source_id"] = rows[0]["source_id"]
    dump(root / "candidates.private.json", rows)
    rebind_output_artifact(root, "candidates.private.json")
    with pytest.raises(ValueError):
        admission.load_reference(root)


def test_duplicate_json_keys_and_nonfinite_values_are_rejected():
    with pytest.raises(ValueError, match="Duplicate JSON"):
        admission._decode(b'{"prompt_eligibility":"BENIGN","prompt_eligibility":"HARMFUL"}')
    with pytest.raises(ValueError, match="Non-finite"):
        admission._decode(b'{"reason":NaN}')
