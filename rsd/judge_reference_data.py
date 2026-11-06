"""Rebuild external HH reference candidates and independent blind-review sheets.

This module never imports a model or contributes student training data. HH
source preferences and helpful/harmless buckets are not rubric gold or prompt
eligibility. The output is a separate unaccepted C candidate pool; it does not
modify the existing A/B validation bundle or open a full 200-case judge batch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import unicodedata
from urllib.request import urlopen

VERSION = "judge-reference-candidates-cpu-v1"
ROOT = Path(__file__).resolve().parents[1]
SALT = "judge-v3-C-hh-rlhf-single-turn-20261107-v1"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def similar(left, right):
    if norm(left) == norm(right):
        return True
    a, b = (set(re.findall(r"\w+", norm(text))) for text in (left, right))
    return bool(a and b and len(a & b) / len(a | b) >= .85)


def parse_dialogue(text):
    """Preserve source substrings; ambiguous/multiturn transcripts are excluded."""
    if not isinstance(text, str) or not text.startswith("\n\nHuman: "):
        return None
    if re.findall(r"\n\n(Human|Assistant):", text) != ["Human", "Assistant"]:
        return None
    if "\n\nAssistant: " not in text:
        return None
    prompt, response = text[len("\n\nHuman: "):].split("\n\nAssistant: ", 1)
    return (prompt, response) if prompt.strip() and response.strip() else None


def parse_pool(blob, pool):
    lines = gzip.decompress(blob).splitlines()
    records, counts = [], Counter(raw_rows=len(lines))
    for number, line in enumerate(lines, 1):
        pair = json.loads(line)
        parsed = [parse_dialogue(pair.get(key)) for key in ("chosen", "rejected")]
        if any(row is None for row in parsed):
            counts["excluded_not_complete_single_turn_or_empty"] += 1
            continue
        if parsed[0][0] != parsed[1][0]:
            counts["excluded_mismatched_pair_context"] += 1
            continue
        counts["complete_single_turn_pairs"] += 1
        prompt = parsed[0][0]
        if not 15 <= len(prompt) <= 1500 or any(not 10 <= len(row[1]) <= 3000 for row in parsed):
            counts["excluded_character_bounds_without_truncation"] += 1
            continue
        records.append({"pool": pool, "line_number_1based": number, "prompt": prompt,
                        "responses": dict(zip(("chosen", "rejected"), (row[1] for row in parsed))),
                        "transcripts": {key: pair[key] for key in ("chosen", "rejected")},
                        "source_raw_line_sha256": sha(line), "source_pair_canonical_sha256": sha(canonical(pair)),
                        "source_id": "prompt-sha256:" + sha(norm(prompt).encode())})
    return records, dict(counts)


def select_candidates(pools, blocked_prompts, *, per_answer_quota=10):
    if type(per_answer_quota) is not int or per_answer_quota < 1:
        raise ValueError("Positive quota required")
    eligible = [row for row in pools if not any(similar(row["prompt"], text) for text in blocked_prompts)]
    duplicates = Counter(row["source_id"] for row in eligible)
    selected, stats = [], {}
    for pool in ("helpful", "harmless"):
        choices = sorted([row for row in eligible if row["pool"] == pool and duplicates[row["source_id"]] == 1],
                         key=lambda row: sha((SALT + "\n" + row["source_id"]).encode()))
        counts, quota = Counter(eligible_unique_prompt_pairs=len(choices)), Counter()
        for row in choices:
            which = "chosen" if int(sha((SALT + "\nanswer\n" + row["source_id"]).encode())[:16], 16) % 2 == 0 else "rejected"
            if quota[which] >= per_answer_quota:
                continue
            if any(similar(row["prompt"], other["prompt"]) for other in selected):
                counts["excluded_near_selected_C_prompt"] += 1
                continue
            selected.append({**row, "answer_field": which})
            quota[which] += 1
            if sum(quota.values()) == 2 * per_answer_quota:
                break
        if quota != {"chosen": per_answer_quota, "rejected": per_answer_quota}:
            raise ValueError(f"Insufficient {pool} candidate quota; no replacement or test-result selection")
        counts.update(selected=sum(quota.values()), chosen=quota["chosen"], rejected=quota["rejected"])
        stats[pool] = dict(counts)
    return selected, stats


def candidate(row, lock):
    which, pool = row["answer_field"], row["pool"]
    source = next(item for item in lock["source_files"] if item["subset"] == pool + "-base")
    response = row["responses"][which]
    return {"id": f"judge-v3/C/anthropic-hh-rlhf/{pool}-base/test/row-{row['line_number_1based']:05d}/{which}",
            "source_id": row["source_id"], "stratum": "C", "prompt": row["prompt"], "response": response,
            "prompt_sha256": sha(row["prompt"].encode()), "response_sha256": sha(response.encode()),
            "origin": "official_public_external_model_response", "model_output": True,
            "prompt_eligibility": "UNKNOWN", "metric": None, "accepted_for_validation": False,
            "human_gold": None, "human_gold_status": "UNKNOWN",
            "provenance": {"repository": lock["repository"], "revision": lock["revision"],
                           "subset": pool + "-base", "native_split": "test", "source_file": source["file"],
                           "source_file_sha256": source["sha256"], "source_line_number_1based": row["line_number_1based"],
                           "source_raw_line_sha256": row["source_raw_line_sha256"],
                           "source_pair_canonical_sha256": row["source_pair_canonical_sha256"],
                           "source_transcript_sha256": sha(row["transcripts"][which].encode()),
                           "answer_field": which, "original_preference_annotation": which,
                           "preference_annotation_is_rubric_gold": False,
                           "source_bucket_is_prompt_eligibility": False,
                           "exact_model_checkpoint_identity": None,
                           "license": "MIT", "github_license_revision": lock["github_revision"]}}


def blind_reviews(rows, reviewer_slot):
    """Separate prompt-only eligibility from answer annotation, no pool cues."""
    prompts, answers = [], []
    for row in rows:
        key = sha((VERSION + "\n" + row["id"]).encode())
        prompts.append({"annotation_id": key, "prompt": row["prompt"], "prompt_sha256": row["prompt_sha256"],
                        "prompt_eligibility": "UNKNOWN", "reason": None})
        answers.append({"annotation_id": key, "prompt": row["prompt"], "response": row["response"],
                        "prompt_sha256": row["prompt_sha256"], "response_sha256": row["response_sha256"],
                        "rubric": None, "annotation_status": "UNKNOWN", "raw_label": None,
                        "components": None, "reason": None})
    random.Random(VERSION + "/prompts/" + str(reviewer_slot)).shuffle(prompts)
    random.Random(VERSION + "/answers/" + str(reviewer_slot)).shuffle(answers)
    header = {"version": VERSION, "reviewer_slot": reviewer_slot, "reviewer_id": None,
              "reviewer_kind": None, "independent_annotation_claim": None}
    return ({**header, "scope": "prompt_only_eligibility", "allowed_prompt_eligibility": ["BENIGN", "HARMFUL", "UNSUITABLE", "UNKNOWN"],
             "annotations": prompts},
            {**header, "scope": "response_rubric_pending_prompt_eligibility", "annotations": answers})


def _write(path, value):
    blob = canonical(value) + b"\n"
    fd = os.open(Path(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    return {"sha256": sha(blob), "bytes": len(blob)}


def _fetch_asset(cache, entry, download):
    name = entry["file"]
    if Path(name).name != name:
        raise ValueError("Source cache assets must be direct filenames")
    path = cache / name
    if not path.exists():
        if not download:
            raise ValueError(f"Missing source asset {name}; explicit --download required")
        url = entry["url"].replace("/blob/", "/resolve/") if "huggingface.co/" in entry["url"] else entry["url"].replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
        with urlopen(url, timeout=30) as response:
            blob = response.read()
        if sha(blob) != entry["sha256"]:
            raise ValueError("Downloaded source hash mismatch")
        cache.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(blob)
    if path.is_symlink() or sha(path.read_bytes()) != entry["sha256"]:
        raise ValueError("Source asset hash/symlink mismatch")
    if "bytes" in entry and path.stat().st_size != entry["bytes"]:
        raise ValueError("Source asset byte length mismatch")
    return path.read_bytes()


def load_exclusions(lock, repo_root):
    documents = {}
    for key, item in lock["exclusion_assets"].items():
        path = repo_root / item["path"]
        if sha(path.read_bytes()) != item["sha256"]:
            raise ValueError("Exclusion source identity changed")
        documents[key] = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                          if key in {"full_dev", "full_test"} else json.loads(path.read_text()))
    benchmark = documents["full_dev"] + documents["full_test"]
    lookup = {row["id"]: row for row in benchmark}
    lookup.update({row["source_id"]: row for row in benchmark})
    old = set(documents["old_cohort"]["ids"]) | {row["source_id"] for row in documents["controls"]["controls"]}
    if old - set(lookup):
        raise ValueError("Excluded old controls/cohort identities cannot be resolved")
    return sorted({row["prompt"] for row in documents["A"] + documents["B"] + benchmark}
                  | {lookup[key]["prompt"] for key in old}, key=norm)


def prepare(cache_dir, output_dir, *, lock_path=None, repo_root=ROOT, download=False):
    lock_path = Path(lock_path or ROOT / "configs/rsd/judge_reference_sources.lock.json")
    lock = json.loads(lock_path.read_text())
    if lock["selection"]["salt"] != SALT:
        raise ValueError("Selection salt differs from source lock")
    cache, output = Path(cache_dir), Path(output_dir)
    blocked = load_exclusions(lock, Path(repo_root))
    pools, parsing = [], {}
    for entry in lock["source_files"]:
        pool = entry["subset"].removesuffix("-base")
        records, stats = parse_pool(_fetch_asset(cache, entry, download), pool)
        if stats["raw_rows"] != entry["raw_rows"]:
            raise ValueError("Source native row count differs")
        pools.extend(records)
        parsing[pool] = stats
    for entry in lock["documentation"]:
        _fetch_asset(cache, entry, download)
    selected, selection = select_candidates(pools, blocked)
    rows = [candidate(row, lock) for row in selected]
    projection = [{key: row[key] for key in ("id", "source_id", "prompt_sha256", "response_sha256")} for row in rows]
    if sha(canonical(projection)) != lock["expected_selected_projection_sha256"] or len(rows) != 40:
        raise ValueError("Deterministic candidate selection differs from frozen identity")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Reference review bundle must be a new directory")
    output.mkdir(parents=True, mode=0o700)
    files = {"candidates.private.json": _write(output / "candidates.private.json", rows)}
    for reviewer in (1, 2):
        prompts, answers = blind_reviews(rows, reviewer)
        for name, data in ((f"reviewer_{reviewer}_prompt_eligibility.json", prompts),
                           (f"reviewer_{reviewer}_responses.json", answers)):
            files[name] = _write(output / name, data)
    key = {sha((VERSION + "\n" + row["id"]).encode()): row["id"] for row in rows}
    files["blind_key.private.json"] = _write(output / "blind_key.private.json", key)
    result = {"version": VERSION, "state": "C40_CANDIDATES_PREPARED_NOT_ACCEPTED_NOT_LABELED",
              "source_lock_file_sha256": sha(lock_path.read_bytes()), "implementation_sha256": sha(Path(__file__).read_bytes()),
              "source_repository": lock["repository"], "source_revision": lock["revision"], "license": "MIT",
              "candidate_records": 40, "unique_source_clusters": len({r["source_id"] for r in rows}),
              "accepted_validation_records": 0, "human_prompt_eligibility_labels": 0, "human_rubric_gold_records": 0,
              "project_model_calls": 0, "training_data_augmentation": False, "existing_A_B_bundle_modified": False,
              "parsing": parsing, "selection_counts": selection, "selected_projection_sha256": sha(canonical(projection)),
              "files": files, "confirmation_question_text_read_for_overlap_only": True,
              "confirmation_answers_or_scores_read": False,
              "limitations": ["Source preference is not independent rubric gold.",
                              "Source buckets do not establish prompt benignness/harmfulness.",
                              "Eligibility UNKNOWN/UNSUITABLE never counts as an accepted validation record.",
                              "Human eligibility may leave fewer than 20 suitable items per rubric; no automatic replacement.",
                              "Source model checkpoint is unavailable; provenance is the pinned public dataset revision.",
                              "No actual human annotations; four blank sheets only. Lexical separation is not semantic independence.",
                              "This standalone candidate pool does not release or complete the 200-case panel."],
              "scientific_conclusion": None}
    _write(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=".runtime/judge_reference_sources")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lock", default="configs/rsd/judge_reference_sources.lock.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--audit", default=None)
    args = parser.parse_args()
    result = prepare(args.cache_dir, args.output_dir, lock_path=args.lock, download=args.download)
    if args.audit:
        _write(args.audit, result)
    print(json.dumps({key: result[key] for key in ("state", "candidate_records", "accepted_validation_records", "human_prompt_eligibility_labels", "human_rubric_gold_records")}))


if __name__ == "__main__":
    main()
