"""Admit external C candidates using separately declared human prompt reviews.

No source bucket, preference, model answer label, or judge score decides prompt
eligibility. Original evidence is copied byte-for-byte into an immutable bundle;
loading replays every decision from that evidence. Identity/independence are
human claims, not identities authenticated by this program. No model is run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import re

from .judge_reference_data import VERSION as REFERENCE_VERSION, canonical, norm, sha
from .judge_validation import blind_sheet

VERSION = "judge-reference-prompt-admission-v1"
ELIGIBILITY = ("BENIGN", "HARMFUL", "UNSUITABLE", "UNKNOWN")
METRIC = {"BENIGN": "overrefusal", "HARMFUL": "safety"}
TARGETS = {"overrefusal": 20, "safety": 20}
REFERENCE_FILES = {"candidates.private.json", "blind_key.private.json",
                   "reviewer_1_prompt_eligibility.json", "reviewer_2_prompt_eligibility.json",
                   "reviewer_1_responses.json", "reviewer_2_responses.json"}


def _no_symlink(path):
    path = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlink paths are not accepted for admission evidence")
    return path


def _read(path):
    path = _no_symlink(path)
    if not path.is_file():
        raise ValueError("Admission evidence must be a regular file")
    return path.read_bytes()


def _decode(blob):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Non-finite JSON value is not accepted")
    return json.loads(blob, object_pairs_hook=unique, parse_constant=invalid)


def _sha_value(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _member(root, name):
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or any(part in (".", "..") for part in name.split("/")) or "\\" in name):
        raise ValueError("Nonlocal admission artifact path")
    result = _no_symlink(root / name)
    if not result.is_relative_to(root):
        raise ValueError("Admission artifact escapes bundle")
    return result


def _verify_files(root, files, expected_names):
    if not isinstance(files, dict) or set(files) != set(expected_names):
        raise ValueError("Evidence inventory differs from the required files")
    for name, expected in files.items():
        blob = _read(_member(root, name))
        if (not isinstance(expected, dict) or not _sha_value(expected.get("sha256"))
                or type(expected.get("bytes")) is not int or expected["bytes"] < 0
                or sha(blob) != expected["sha256"] or len(blob) != expected["bytes"]):
            raise ValueError("Evidence artifact SHA/byte binding mismatch")


def load_reference(reference_bundle):
    """Verify the immutable C40 source bundle and return (manifest, candidates)."""
    root = _no_symlink(reference_bundle)
    manifest = _decode(_read(root / "manifest.json"))
    if (manifest.get("version") != REFERENCE_VERSION
            or manifest.get("state") != "C40_CANDIDATES_PREPARED_NOT_ACCEPTED_NOT_LABELED"
            or type(manifest.get("candidate_records")) is not int or manifest["candidate_records"] != 40
            or type(manifest.get("unique_source_clusters")) is not int or manifest["unique_source_clusters"] != 40
            or any(type(manifest.get(key)) is not int or manifest[key] != 0 for key in
                   ("accepted_validation_records", "human_prompt_eligibility_labels", "human_rubric_gold_records"))):
        raise ValueError("Reference bundle must be the unaccepted and unlabeled C40 preparation")
    _verify_files(root, manifest.get("files"), REFERENCE_FILES)
    rows = _decode(_read(root / "candidates.private.json"))
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("Reference candidate denominator must remain 40")
    ids, sources = set(), set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not row["id"].startswith("judge-v3/C/") or row["id"] in ids
                or not isinstance(row.get("source_id"), str) or row["source_id"] in sources
                or any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("prompt", "response"))):
            raise ValueError("Reference candidate identities/text are missing or duplicated")
        ids.add(row["id"])
        sources.add(row["source_id"])
        if (row["source_id"] != "prompt-sha256:" + sha(norm(row["prompt"]).encode())
                or row.get("prompt_sha256") != sha(row["prompt"].encode())
                or row.get("response_sha256") != sha(row["response"].encode())):
            raise ValueError("Reference candidate prompt/response/source binding mismatch")
        if (row.get("stratum") != "C" or "metric" not in row or row.get("metric") is not None
                or row.get("prompt_eligibility") != "UNKNOWN" or row.get("accepted_for_validation") is not False
                or "human_gold" not in row or row.get("human_gold") is not None or row.get("human_gold_status") != "UNKNOWN"):
            raise ValueError("Reference candidates must not carry preassigned eligibility, metrics, or gold")
    return manifest, rows


def _annotation_id(row):
    return sha((REFERENCE_VERSION + "\n" + row["id"]).encode())


def _reviewer_identity(document):
    identity = document.get("reviewer_id")
    if (not isinstance(identity, str) or not identity.strip() or identity != identity.strip()
            or document.get("reviewer_kind") != "human"
            or document.get("independent_annotation_claim") is not True):
        raise ValueError("Named human reviewer and independent-annotation declaration required")
    return identity


def _review(document, rows, *, adjudication=False, allowed_ids=None):
    """Whitelist prompt-only labels; response/rubric fields are forbidden."""
    allowed_header = {"version", "reviewer_slot", "reviewer_id", "reviewer_kind", "independent_annotation_claim",
                      "scope", "allowed_prompt_eligibility", "annotations"}
    if not isinstance(document, dict) or set(document) - allowed_header:
        raise ValueError("Review contains fields outside the prompt-only schema")
    expected_scope = "prompt_only_eligibility_adjudication" if adjudication else "prompt_only_eligibility"
    if (document.get("version") != REFERENCE_VERSION or document.get("scope") != expected_scope
            or document.get("allowed_prompt_eligibility") != list(ELIGIBILITY)):
        raise ValueError("Review version/scope/eligibility taxonomy differs")
    reviewer = _reviewer_identity(document)
    slot = document.get("reviewer_slot")
    if not adjudication and (type(slot) is not int or slot not in (1, 2)):
        raise ValueError("Primary review requires slot 1 or 2")
    if adjudication and slot is not None:
        raise ValueError("Adjudicator must not impersonate a primary reviewer slot")
    original = {_annotation_id(row): row for row in rows}
    annotations = document.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Review annotations must be a list")
    index = {}
    fields = {"annotation_id", "prompt", "prompt_sha256", "prompt_eligibility", "reason"}
    for item in annotations:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("Prompt review cannot carry answer labels, source preferences, or extra fields")
        key = item["annotation_id"]
        if not isinstance(key, str) or key not in original or key in index:
            raise ValueError("Unknown or duplicate prompt annotation")
        source = original[key]
        if item["prompt"] != source["prompt"] or item["prompt_sha256"] != source["prompt_sha256"]:
            raise ValueError("Prompt review text/hash binding mismatch")
        label, reason = item["prompt_eligibility"], item["reason"]
        if not isinstance(label, str) or label not in ELIGIBILITY:
            raise ValueError("Prompt eligibility must use an explicit string category, never a boolean")
        if (reason is not None and (not isinstance(reason, str) or not reason.strip())) or (label != "UNKNOWN" and reason is None):
            raise ValueError("Known prompt eligibility requires a nonempty textual rationale")
        if adjudication and key not in allowed_ids:
            raise ValueError("Adjudication may address only unresolved primary-review cases")
        index[key] = item
    if not adjudication and set(index) != set(original):
        raise ValueError("Each primary review must retain every one of the 40 candidates")
    return {"reviewer_id": reviewer, "reviewer_slot": slot, "annotations": index}


def _derive(rows, review_documents, adjudication_document=None):
    if len(review_documents) != 2:
        raise ValueError("Exactly two complete primary reviews are required")
    reviews = [_review(document, rows) for document in review_documents]
    if {review["reviewer_slot"] for review in reviews} != {1, 2}:
        raise ValueError("Primary review slots must be distinct")
    reviews.sort(key=lambda review: review["reviewer_slot"])
    reviewer_ids = [review["reviewer_id"] for review in reviews]
    if len({identity.casefold() for identity in reviewer_ids}) != 2:
        raise ValueError("Primary human reviewer identities must be distinct")
    unresolved = set()
    for row in rows:
        key = _annotation_id(row)
        labels = [review["annotations"][key]["prompt_eligibility"] for review in reviews]
        if labels[0] != labels[1] or labels[0] == "UNKNOWN":
            unresolved.add(key)
    adjudicator = None
    if adjudication_document is not None:
        adjudicator = _review(adjudication_document, rows, adjudication=True, allowed_ids=unresolved)
        if adjudicator["reviewer_id"].casefold() in {identity.casefold() for identity in reviewer_ids}:
            raise ValueError("A third distinct human adjudicator is required")
    decisions, admitted = [], []
    for row in rows:
        key = _annotation_id(row)
        original = [review["annotations"][key] for review in reviews]
        labels = [item["prompt_eligibility"] for item in original]
        known_agreement = key not in unresolved
        adjudication = adjudicator["annotations"].get(key) if adjudicator else None
        resolved = labels[0] if known_agreement else (adjudication["prompt_eligibility"] if adjudication else "UNKNOWN")
        metric = METRIC.get(resolved)
        status = "ADMITTED" if metric else "EXCLUDED" if resolved == "UNSUITABLE" else "UNKNOWN"
        decisions.append({"id": row["id"], "source_id": row["source_id"], "stratum": "C",
                          "eligibility_annotation_id": key, "prompt_sha256": row["prompt_sha256"],
                          "response_sha256": row["response_sha256"],
                          "original_review_labels": {"reviewer_1": labels[0], "reviewer_2": labels[1]},
                          "original_review_reasons": {"reviewer_1": original[0]["reason"], "reviewer_2": original[1]["reason"]},
                          "original_labels_equal": labels[0] == labels[1], "original_known_agreement": known_agreement,
                          "adjudicator_label": adjudication["prompt_eligibility"] if adjudication else None,
                          "adjudicator_reason": adjudication["reason"] if adjudication else None,
                          "resolution": "TWO_REVIEWERS_AGREE" if known_agreement else "HUMAN_ADJUDICATION" if adjudication else "UNRESOLVED",
                          "resolved_prompt_eligibility": resolved, "admission_status": status, "metric": metric,
                          "human_rubric_gold_status": "UNKNOWN", "human_rubric_gold": None})
        if metric:
            candidate = copy.deepcopy(row)
            candidate.update(metric=metric, prompt_eligibility=resolved, accepted_for_validation=True,
                             admission_stage="human_prompt_eligibility_only",
                             metric_origin="human_prompt_eligibility_not_source_bucket_or_preference",
                             human_gold=None, human_gold_status="UNKNOWN")
            admitted.append(candidate)
    counts = Counter(decision["admission_status"] for decision in decisions)
    metric_counts = {metric: sum(row["metric"] == metric for row in admitted) for metric in TARGETS}
    summary = {"candidate_records": len(rows), "admitted_records": counts["ADMITTED"],
               "excluded_records": counts["EXCLUDED"], "unknown_records": counts["UNKNOWN"],
               "target_counts": TARGETS, "admitted_metric_counts": metric_counts,
               "coverage_gap": {metric: max(0, TARGETS[metric] - count) for metric, count in metric_counts.items()},
               "coverage_excess": {metric: max(0, count - TARGETS[metric]) for metric, count in metric_counts.items()},
               "target_exactly_met": metric_counts == TARGETS,
               "human_rubric_gold_records": 0, "reviewer_ids": reviewer_ids,
               "adjudicator_id": adjudicator["reviewer_id"] if adjudicator else None}
    return decisions, admitted, summary


def adjudication_template(reference_bundle, review_paths):
    """Produce an unfilled prompt-only form for unresolved cases, never labels."""
    _, rows = load_reference(reference_bundle)
    reviews, _, _ = _read_reviews(review_paths, None)
    decisions, _, _ = _derive(rows, reviews)
    ids = {row["id"] for row in decisions if not row["original_known_agreement"]}
    return {"version": REFERENCE_VERSION, "scope": "prompt_only_eligibility_adjudication", "reviewer_slot": None,
            "reviewer_id": None, "reviewer_kind": None, "independent_annotation_claim": None,
            "allowed_prompt_eligibility": list(ELIGIBILITY),
            "annotations": [{"annotation_id": _annotation_id(row), "prompt": row["prompt"],
                             "prompt_sha256": row["prompt_sha256"], "prompt_eligibility": "UNKNOWN", "reason": None}
                            for row in rows if row["id"] in ids]}


def _read_reviews(review_paths, adjudication_path):
    if not isinstance(review_paths, (list, tuple)) or len(review_paths) != 2:
        raise ValueError("Exactly two review paths are required")
    paths = [_no_symlink(path) for path in review_paths]
    if len(set(paths)) != 2:
        raise ValueError("Primary reviews must be separate files")
    blobs = [_read(path) for path in paths]
    adjudication_blob = _read(adjudication_path) if adjudication_path is not None else None
    return [_decode(blob) for blob in blobs], blobs, adjudication_blob


def _write_bytes(path, blob):
    path = _no_symlink(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    return {"sha256": sha(blob), "bytes": len(blob)}


def admit(reference_bundle, review_paths, output_dir, *, adjudication_path=None):
    """Write an immutable admission bundle after all inputs pass validation."""
    reference_root = _no_symlink(reference_bundle)
    reference_manifest, rows = load_reference(reference_root)
    review_documents, review_blobs, adjudication_blob = _read_reviews(review_paths, adjudication_path)
    decisions, admitted, summary = _derive(rows, review_documents, _decode(adjudication_blob) if adjudication_blob else None)
    # File names refer to reviewer slots, irrespective of caller argument order.
    review_blobs = [blob for document, blob in sorted(zip(review_documents, review_blobs),
                                                    key=lambda pair: pair[0]["reviewer_slot"])]
    output = _no_symlink(output_dir)
    if output.exists():
        raise FileExistsError("Admission output must be a new directory; evidence is immutable")
    # Snapshot only already-validated bytes; none of the source evidence changes.
    payloads = {"reference/manifest.json": _read(reference_root / "manifest.json")}
    for name in REFERENCE_FILES:
        payloads["reference/" + name] = _read(reference_root / name)
    for index, blob in enumerate(review_blobs, 1):
        payloads[f"eligibility_review_{index}.original.json"] = blob
    if adjudication_blob is not None:
        payloads["eligibility_adjudication.original.json"] = adjudication_blob
    payloads.update({"decisions.json": canonical(decisions) + b"\n",
                     "admitted_candidates.json": canonical(admitted) + b"\n",
                     "reviewer_1.json": canonical(blind_sheet(admitted, 1)) + b"\n",
                     "reviewer_2.json": canonical(blind_sheet(admitted, 2)) + b"\n"})
    output.mkdir(parents=True, mode=0o700)
    files = {name: _write_bytes(_member(output, name), blob) for name, blob in payloads.items()}
    manifest = {"version": VERSION, "state": "HUMAN_PROMPT_ADMISSION_RECORDED_RUBRIC_GOLD_PENDING",
                "reference_bundle_manifest_sha256": sha(payloads["reference/manifest.json"]),
                "reference_candidate_file_sha256": reference_manifest["files"]["candidates.private.json"]["sha256"],
                "review_file_sha256": [sha(blob) for blob in review_blobs],
                "adjudication_file_sha256": sha(adjudication_blob) if adjudication_blob is not None else None,
                **summary, "selection_policy": "Retain all 40 decisions and every admitted candidate; do not cap, replace or select using answer labels.",
                "independence_status": "Declared human identities and independent-review claims; not authenticated by this program.",
                "new_model_calls": 0, "full_panel_ready": False, "scientific_conclusion": None,
                "implementation_sha256": sha(Path(__file__).read_bytes()), "files": files}
    _write_bytes(output / "manifest.json", canonical(manifest) + b"\n")
    # Re-read and replay the finished snapshot before reporting success.
    load_admission(output)
    return manifest


def load_admission(output_dir):
    """Verify files and recompute eligibility, metrics, coverage and blind sheets."""
    root = _no_symlink(output_dir)
    manifest = _decode(_read(root / "manifest.json"))
    if (manifest.get("version") != VERSION
            or manifest.get("state") != "HUMAN_PROMPT_ADMISSION_RECORDED_RUBRIC_GOLD_PENDING"
            or manifest.get("full_panel_ready") is not False or manifest.get("scientific_conclusion") is not None
            or type(manifest.get("new_model_calls")) is not int or manifest["new_model_calls"] != 0):
        raise ValueError("Unknown or invalid admission manifest")
    names = {"reference/manifest.json", "eligibility_review_1.original.json", "eligibility_review_2.original.json",
             "decisions.json", "admitted_candidates.json", "reviewer_1.json", "reviewer_2.json"}
    names.update("reference/" + name for name in REFERENCE_FILES)
    if manifest.get("adjudication_file_sha256") is not None:
        names.add("eligibility_adjudication.original.json")
    _verify_files(root, manifest.get("files"), names)
    reference_manifest, rows = load_reference(root / "reference")
    if (manifest.get("reference_bundle_manifest_sha256") != sha(_read(root / "reference/manifest.json"))
            or manifest.get("reference_candidate_file_sha256") != reference_manifest["files"]["candidates.private.json"]["sha256"]):
        raise ValueError("Admission source-bundle identity mismatch")
    review_blobs = [_read(root / f"eligibility_review_{i}.original.json") for i in (1, 2)]
    if manifest.get("review_file_sha256") != [sha(blob) for blob in review_blobs]:
        raise ValueError("Admission reviewer-file identity mismatch")
    adjudication_blob = (_read(root / "eligibility_adjudication.original.json")
                         if "eligibility_adjudication.original.json" in names else None)
    if adjudication_blob is not None and manifest["adjudication_file_sha256"] != sha(adjudication_blob):
        raise ValueError("Admission adjudicator-file identity mismatch")
    decisions, admitted, summary = _derive(rows, [_decode(blob) for blob in review_blobs],
                                         _decode(adjudication_blob) if adjudication_blob is not None else None)
    if any(canonical(manifest.get(key)) != canonical(value) for key, value in summary.items()):
        raise ValueError("Admission reported counts/coverage differ from recomputed decisions")
    if (canonical(_decode(_read(root / "decisions.json"))) != canonical(decisions)
            or canonical(_decode(_read(root / "admitted_candidates.json"))) != canonical(admitted)):
        raise ValueError("Admission decisions/candidates differ from original human-review evidence")
    for i in (1, 2):
        if canonical(_decode(_read(root / f"reviewer_{i}.json"))) != canonical(blind_sheet(admitted, i)):
            raise ValueError("Admission blind sheet differs from admitted candidate projection")
    return manifest, decisions, admitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bundle", required=True)
    parser.add_argument("--reviews", nargs=2, required=True)
    parser.add_argument("--adjudications")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = admit(args.reference_bundle, args.reviews, args.output_dir, adjudication_path=args.adjudications)
    print(json.dumps({key: result[key] for key in ("candidate_records", "admitted_records", "excluded_records",
                                                   "unknown_records", "coverage_gap", "human_rubric_gold_records")}))


if __name__ == "__main__":
    main()
