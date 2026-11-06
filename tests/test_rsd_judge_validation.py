"""Preparation/gold-binding tests with synthetic reviewer stubs, not human gold."""
import copy
import hashlib
import json
from collections import Counter

import pytest

from rsd import judge_validation as jv
from rsd.evaluation import judge_inputs, text_digest


@pytest.fixture
def bundle(tmp_path):
    cohort = json.loads((jv.ROOT / "configs/rsd/milestone_dev_cohort_v1.json").read_text())["ids"]
    ids = cohort + [f"xstest_safe/fixture-{i}" for i in range(32)] + [f"strongreject/fixture-{i}" for i in range(32)]
    rows = [{"id": key, "source_id": "source-" + key, "family": "fixture", "split": "dev",
             "prompt": f"Fixture-{i} category-{i % 7}?", "benchmark": "xstest_safe" if key.startswith("xstest") else "strongreject",
             "metric": "overrefusal" if key.startswith("xstest") else "safety"} for i, key in enumerate(ids)]
    source = tmp_path / "dev.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "bundle"
    manifest = jv.build(source, output, expected_dev_sha=jv._file_sha(source))
    return output, manifest, source


def review_files(bundle, tmp_path, *, unknown=False, disagree=False):
    output, _, _ = bundle
    paths = []
    for number in (1, 2):
        review = json.loads((output / f"reviewer_{number}.json").read_text())
        review.update(reviewer_id=f"synthetic-test-reviewer-{number}", reviewer_kind="human", independent_annotation_claim=True)
        if not unknown:
            for row in review["annotations"]:
                row["annotation_status"] = "LABELED"
                row["reason"] = "CPU test fixture only; no actual human annotation"
                if row["metric"] == "overrefusal":
                    row["raw_label"] = "2_full_refusal" if disagree and number == 2 else "1_full_compliance"
                else:
                    row["components"] = {"refusal": 1, "convincingness": 2, "specificity": 2}
        path = tmp_path / f"review{number}.json"
        path.write_text(json.dumps(review))
        paths.append(path)
    return paths


def test_authored_panel_has_64_answers_32_sources_and_no_human_gold():
    rows = jv.constructed_panel()
    assert len(rows) == 64 and len({r["id"] for r in rows}) == 64
    assert len({r["source_id"] for r in rows}) == 32
    assert Counter(r["metric"] for r in rows) == {"overrefusal": 48, "safety": 16}
    assert Counter(r["author_hypothesis"].get("raw_label") for r in rows if r["variant"] == "original" and r["metric"] == "overrefusal") == {
        "1_full_compliance": 8, "2_full_refusal": 8, "3_partial_refusal": 8}
    assert all(r["human_gold"] is None and not r["model_output"] for r in rows)
    for r in rows:
        assert r["prompt_sha256"] == text_digest(r["prompt"])
        assert r["response_sha256"] == text_digest(r["response"])


def test_blind_projection_does_not_leak_author_or_generator_metadata():
    rows = jv.constructed_panel()
    for row in rows:
        row.update(generator="secret generator", arm="R0", checkpoint="step512", profile="candidate", extra_secret="secret")
    sheet = jv.blind_sheet(rows, 1)
    assert sheet["reviewer_id"] is None and sheet["reviewer_kind"] is None
    allowed = {"annotation_id", "metric", "prompt", "response", "prompt_sha256", "response_sha256",
               "annotation_status", "raw_label", "components", "reason"}
    assert all(set(r) == allowed and r["raw_label"] is None for r in sheet["annotations"])
    assert [r["annotation_id"] for r in sheet["annotations"]] != [r["annotation_id"] for r in jv.blind_sheet(rows, 2)["annotations"]]


def test_bundle_counts_slots_are_not_answers_and_excludes_old_cohort(bundle):
    root, manifest, _ = bundle
    assert manifest["prepared_answer_records"] == 64 and manifest["prepared_human_gold_records"] == 0
    assert manifest["B"]["actual_answer_records"] == 0 and manifest["C"]["actual_answer_records"] == 0
    slots = json.loads((root / "real_dev_generation_slots.json").read_text())
    assert len(slots) == 96 and len({r["source_id"] for r in slots}) == 48
    assert all(r["response"] is None and r["generator_identity"] is None for r in slots)
    assert not set(manifest["B"]["selected_source_dev_ids"]) & set(manifest["excluded_historical_ids"])
    assert jv.load_bundle(root)[0] == manifest


def test_source_hash_existing_output_and_test_data_fail_closed(bundle, tmp_path):
    root, _, source = bundle
    with pytest.raises(ValueError, match="pinned SHA256"):
        jv.build(source, tmp_path / "bad")
    with pytest.raises(FileExistsError):
        jv.build(source, root, expected_dev_sha=jv._file_sha(source))
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="dev-only"):
        jv.select_real_dev_questions(rows, set(), jv.constructed_panel())


def test_near_duplicate_old_source_cannot_supply_missing_quota():
    rows = [{"id": "old", "source_id": "old", "family": "fixture", "split": "dev", "metric": "overrefusal",
             "benchmark": "xstest_safe", "prompt": "Which material is the most effective conductor of electric current among these options?"}]
    rows.append({**rows[0], "id": "new", "source_id": "new", "prompt": rows[0]["prompt"].replace("the most", "a most")})
    with pytest.raises(ValueError, match="no test fallback"):
        jv.select_real_dev_questions(rows, {"old"}, [], per_metric=1)


def test_blank_review_templates_cannot_be_frozen_as_human_gold(bundle, tmp_path):
    root, _, _ = bundle
    with pytest.raises(ValueError, match="human reviewer"):
        jv.freeze_human_gold(root, [root / "reviewer_1.json", root / "reviewer_2.json"], tmp_path / "gold.json")
    assert not (tmp_path / "gold.json").exists()


def test_human_gold_is_not_copied_from_author_hypothesis_and_partial_gate_stays_closed(bundle, tmp_path):
    root, _, _ = bundle
    reviews = review_files(bundle, tmp_path)
    gold_path = tmp_path / "gold.json"
    gold = jv.freeze_human_gold(root, reviews, gold_path)
    assert gold["n_human_labeled"] == 64 and gold["n_unknown"] == 0
    assert gold["planned_full_panel_complete"] is False
    assert all(r["value"] == 0 for r in gold["records"])
    with pytest.raises(ValueError, match="200-case"):
        jv.bound_judge_inputs(root, gold_path)
    partial = jv.bound_judge_inputs(root, gold_path, allow_partial_panel=True)
    assert partial["contract"]["scope"] == "partial_panel_diagnostic_only"
    assert len(judge_inputs(partial["dataset"], partial["predictions"])) == 64
    assert all("author_hypothesis" not in r for r in partial["dataset"] + partial["predictions"])
    with pytest.raises(FileExistsError):
        jv.freeze_human_gold(root, reviews, gold_path)


def test_disagreement_and_unknown_preserve_original_denominator(bundle, tmp_path):
    root, _, _ = bundle
    reviews = review_files(bundle, tmp_path, disagree=True)
    result = jv.freeze_human_gold(root, reviews, tmp_path / "disagreement.json")
    assert result["n_prepared"] == 64 and result["n_human_labeled"] == 16 and result["n_unknown"] == 48
    assert all(r["value"] is None for r in result["records"] if r["metric"] == "overrefusal")
    reviews = review_files(bundle, tmp_path, unknown=True)
    result = jv.freeze_human_gold(root, reviews, tmp_path / "unknown.json")
    assert result["n_unknown"] == 64 and result["n_human_labeled"] == 0


def test_only_distinct_human_adjudication_resolves_disagreement(bundle, tmp_path):
    root, _, _ = bundle
    paths = review_files(bundle, tmp_path, disagree=True)
    second = json.loads(paths[1].read_text())
    adjudication = {"reviewer_id": "synthetic-test-adjudicator", "reviewer_kind": "human",
                    "annotations": [r for r in second["annotations"] if r["metric"] == "overrefusal"]}
    adjudicated = tmp_path / "adjudication.json"
    adjudicated.write_text(json.dumps(adjudication))
    result = jv.freeze_human_gold(root, paths, tmp_path / "resolved.json", adjudication_path=adjudicated)
    assert result["n_human_labeled"] == 64
    assert sum(r["adjudicated"] for r in result["records"]) == 48
    assert all(r["value"] == 1 for r in result["records"] if r["metric"] == "overrefusal")
    adjudication["reviewer_id"] = "synthetic-test-reviewer-1"
    adjudicated.write_text(json.dumps(adjudication))
    with pytest.raises(ValueError, match="distinct human adjudicator"):
        jv.freeze_human_gold(root, paths, tmp_path / "invalid-adjudicated.json", adjudication_path=adjudicated)


@pytest.mark.parametrize("mutation", ["same_reviewer", "wrong_hash", "changed_displayed_text", "missing_case", "duplicate_case", "bad_label"])
def test_bad_human_review_rejected(bundle, tmp_path, mutation):
    root, _, _ = bundle
    paths = review_files(bundle, tmp_path)
    review = json.loads(paths[1].read_text())
    if mutation == "same_reviewer":
        review["reviewer_id"] = "synthetic-test-reviewer-1"
    elif mutation == "wrong_hash":
        review["annotations"][0]["response_sha256"] = "0" * 64
    elif mutation == "changed_displayed_text":
        review["annotations"][0]["response"] = "Different text with an unchanged original hash"
    elif mutation == "missing_case":
        review["annotations"].pop()
    elif mutation == "duplicate_case":
        review["annotations"].append(review["annotations"][0])
    else:
        next(r for r in review["annotations"] if r["metric"] == "overrefusal")["raw_label"] = "safe"
    paths[1].write_text(json.dumps(review))
    with pytest.raises(ValueError):
        jv.freeze_human_gold(root, paths, tmp_path / "bad-gold.json")


def test_mutated_bundle_and_frozen_input_binding_are_rejected(bundle, tmp_path):
    root, _, _ = bundle
    paths = review_files(bundle, tmp_path)
    gold_path = tmp_path / "gold.json"
    gold = jv.freeze_human_gold(root, paths, gold_path)
    gold["records"][0]["response_sha256"] = "0" * 64
    gold_path.write_text(json.dumps(gold))
    with pytest.raises(ValueError, match="input binding"):
        jv.bound_judge_inputs(root, gold_path, allow_partial_panel=True)
    (root / "authored_candidates.json").write_text("[]")
    with pytest.raises(ValueError, match="artifact binding"):
        jv.load_bundle(root)
