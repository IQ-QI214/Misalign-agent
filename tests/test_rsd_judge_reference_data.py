"""External-reference preparation tests; synthetic transcripts, no model calls."""
from collections import Counter
import copy
import gzip
import json

import pytest

from rsd import judge_reference_data as ref


def dialogue(prompt, response):
    return "\n\nHuman: " + prompt + "\n\nAssistant: " + response


def pool_blob(pool, count=90):
    rows = [{"chosen": dialogue(f"Unique-{pool}-{i} item-{i} question-{i}?", "Complete preferred response " + str(i)),
             "rejected": dialogue(f"Unique-{pool}-{i} item-{i} question-{i}?", "Complete alternative response " + str(i))}
            for i in range(count)]
    return gzip.compress(b"\n".join(ref.canonical(row) for row in rows))


def test_exact_single_turn_extraction_without_trimming_or_context_loss():
    prompt, answer = "  A question with whitespace? ", " A full answer.\n"
    assert ref.parse_dialogue(dialogue(prompt, answer)) == (prompt, answer)
    for bad in (None, "Human: Question\nAssistant: Answer", dialogue(" ", "answer"),
                dialogue("Question", " "), dialogue("Question", "answer") + dialogue("Second", "reply")):
        assert ref.parse_dialogue(bad) is None


def test_mismatched_pairs_and_length_limits_are_excluded_not_rewritten():
    rows = [
        {"chosen": dialogue("Sufficient prompt length?", "Original response long enough."),
         "rejected": dialogue("A different prompt length?", "Original other response long enough.")},
        {"chosen": dialogue("Sufficient prompt length?", "x" * 3001),
         "rejected": dialogue("Sufficient prompt length?", "Original other response long enough.")},
        {"chosen": dialogue("Sufficient prompt length?", "Original response long enough."),
         "rejected": dialogue("Sufficient prompt length?", "Original other response long enough.")},
    ]
    records, counts = ref.parse_pool(gzip.compress(b"\n".join(ref.canonical(row) for row in rows)), "helpful")
    assert len(records) == 1 and records[0]["line_number_1based"] == 3
    assert records[0]["responses"]["chosen"] == "Original response long enough."
    assert counts["excluded_mismatched_pair_context"] == 1
    assert counts["excluded_character_bounds_without_truncation"] == 1


def test_deterministic_selection_is_disjoint_balanced_and_independent_of_answer_text():
    pools = sum([ref.parse_pool(pool_blob(pool), pool)[0] for pool in ("helpful", "harmless")], [])
    selected, counts = ref.select_candidates(pools, [])
    reordered, _ = ref.select_candidates(list(reversed(pools)), [])
    assert [(r["source_id"], r["answer_field"]) for r in selected] == [(r["source_id"], r["answer_field"]) for r in reordered]
    assert len(selected) == len({r["source_id"] for r in selected}) == 40
    assert Counter((r["pool"], r["answer_field"]) for r in selected) == {
        ("helpful", "chosen"): 10, ("helpful", "rejected"): 10,
        ("harmless", "chosen"): 10, ("harmless", "rejected"): 10}
    changed = copy.deepcopy(pools)
    for row in changed:
        row["responses"] = {"chosen": "Completely different", "rejected": "No rubric labels used"}
    selected_changed, _ = ref.select_candidates(changed, [])
    assert [(r["source_id"], r["answer_field"]) for r in selected_changed] == [(r["source_id"], r["answer_field"]) for r in selected]


def test_exact_duplicates_and_excluded_question_near_matches_cannot_be_selected():
    pools = sum([ref.parse_pool(pool_blob(pool), pool)[0] for pool in ("helpful", "harmless")], [])
    duplicate = copy.deepcopy(pools[0])
    duplicate["pool"] = "harmless"
    pools.append(duplicate)
    blocked = pools[1]["prompt"]
    selected, _ = ref.select_candidates(pools, [blocked])
    assert pools[0]["source_id"] not in {r["source_id"] for r in selected}
    assert all(not ref.similar(r["prompt"], blocked) for r in selected)
    with pytest.raises(ValueError, match="Insufficient"):
        ref.select_candidates([], [])


def test_source_hash_mismatch_or_missing_cache_fails_before_writes(tmp_path):
    item = {"file": "source.gz", "sha256": "0" * 64, "bytes": 10, "url": "https://example.invalid/never-called"}
    with pytest.raises(ValueError, match="explicit --download"):
        ref._fetch_asset(tmp_path, item, False)
    (tmp_path / "source.gz").write_bytes(b"wrong data")
    with pytest.raises(ValueError, match="hash/symlink"):
        ref._fetch_asset(tmp_path, item, False)
    with pytest.raises(ValueError, match="direct filenames"):
        ref._fetch_asset(tmp_path, {**item, "file": "../source.gz"}, False)


def candidate_fixture():
    row = ref.parse_pool(pool_blob("helpful", 1), "helpful")[0][0]
    row["answer_field"] = "chosen"
    lock = {"repository": "Anthropic/hh-rlhf", "revision": "a" * 40, "github_revision": "b" * 40,
            "source_files": [{"subset": "helpful-base", "file": "helpful.gz", "sha256": "c" * 64}]}
    return ref.candidate(row, lock)


def test_preference_and_source_bucket_never_become_eligibility_or_rubric_gold():
    row = candidate_fixture()
    assert row["provenance"]["original_preference_annotation"] == "chosen"
    assert row["metric"] is None and row["prompt_eligibility"] == "UNKNOWN"
    assert row["human_gold"] is None and row["accepted_for_validation"] is False
    assert row["provenance"]["preference_annotation_is_rubric_gold"] is False
    assert row["provenance"]["exact_model_checkpoint_identity"] is None


def test_prompt_eligibility_blind_sheet_is_separate_and_has_no_answer_or_bucket():
    row = candidate_fixture()
    prompts, answers = ref.blind_reviews([row], 1)
    prompt, answer = prompts["annotations"][0], answers["annotations"][0]
    assert set(prompt) == {"annotation_id", "prompt", "prompt_sha256", "prompt_eligibility", "reason"}
    assert set(answer) == {"annotation_id", "prompt", "response", "prompt_sha256", "response_sha256",
                           "rubric", "annotation_status", "raw_label", "components", "reason"}
    assert prompt["annotation_id"] == answer["annotation_id"]
    assert prompt["prompt_eligibility"] == "UNKNOWN" and answer["rubric"] is None
    assert prompts["reviewer_id"] is None and answers["reviewer_id"] is None
    for forbidden in ("provenance", "chosen", "subset", "metric", "model", "source_id", "preference"):
        assert forbidden not in prompt and forbidden not in answer


def test_output_write_is_immutable(tmp_path):
    path = tmp_path / "review.json"
    original = {"human_gold": None}
    receipt = ref._write(path, original)
    assert receipt["sha256"] == ref.sha(path.read_bytes())
    with pytest.raises(FileExistsError):
        ref._write(path, {"human_gold": "changed"})
    assert json.loads(path.read_text()) == original
