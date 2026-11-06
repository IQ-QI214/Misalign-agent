"""CPU fixtures only; reviewer stubs below are not actual human judgments."""
import copy
import json
from pathlib import Path

import pytest

from rsd import judge_validation as jv
from rsd import judge_panel as panel
from rsd.evaluation import text_digest
from test_rsd_judge_validation import bundle, review_files
from test_rsd_judge_reference_admission import source_bundle, reviews as eligibility_reviews
from test_rsd_judge_real_data import generators, response_files


def test_actual_a_only_panel_preserves_pending_and_reuses_gold_workflow(bundle, tmp_path):
    original, _, _ = bundle
    output = tmp_path / "assembled"
    manifest = panel.assemble(original, output)
    assert manifest["prepared_answer_records"] == 64
    assert manifest["prepared_human_gold_records"] == 0
    assert manifest["pending"]["B"]["remaining_slots"] == 96
    assert manifest["coverage"]["C"]["missing"] == {"overrefusal": 20, "safety": 20, "sources": 40}
    assert not manifest["full_panel_materials_complete"]
    loaded, rows = jv.load_bundle(output)
    assert loaded == manifest and len(rows) == 64
    reviews = review_files((output, manifest, None), tmp_path)
    gold = tmp_path / "fixture-gold.json"
    result = jv.freeze_human_gold(output, reviews, gold)
    assert result["n_prepared"] == 64 and result["planned_full_panel_complete"] is False
    with pytest.raises(ValueError, match="200-case"):
        jv.bound_judge_inputs(output, gold)
    binding = jv.bound_judge_inputs(output, gold, allow_partial_panel=True)
    assert len(binding["predictions"]) == 64
    assert binding["predictions"][0]["generation_protocol"]["response_origin"] == rows[0]["origin"]


def test_blanks_cannot_become_human_gold_and_original_is_unchanged(bundle, tmp_path):
    original, _, _ = bundle
    before = jv._file_sha(original / "manifest.json")
    output = tmp_path / "assembled"
    panel.assemble(original, output)
    with pytest.raises(ValueError, match="human reviewer"):
        jv.freeze_human_gold(output, [output / "reviewer_1.json", output / "reviewer_2.json"], tmp_path / "gold.json")
    assert jv._file_sha(original / "manifest.json") == before
    with pytest.raises(FileExistsError):
        panel.assemble(original, output)


@pytest.mark.parametrize("mutation", ["source", "candidate", "count", "complete", "symlink", "self_source", "review_projection", "ready"])
def test_panel_changed_evidence_or_fabricated_readiness_is_rejected(bundle, tmp_path, mutation):
    original, _, _ = bundle
    output = tmp_path / "assembled"
    manifest = panel.assemble(original, output)
    if mutation == "source":
        (original / "manifest.json").write_text("{}")
    elif mutation == "candidate":
        path = output / "candidates.private.json"
        rows = json.loads(path.read_text())
        rows[0]["response"] = "Changed response"
        path.write_text(json.dumps(rows))
        manifest["files_sha256"][path.name] = jv._file_sha(path)
    elif mutation == "count":
        manifest["prepared_answer_records"] = 200
    elif mutation == "complete":
        manifest["full_panel_materials_complete"] = True
    elif mutation == "ready":
        manifest["judge_batch_ready"] = True
    elif mutation == "review_projection":
        path = output / "reviewer_1.json"
        sheet = json.loads(path.read_text())
        sheet["annotations"][0]["author_hypothesis"] = "Do not expose this"
        path.write_text(json.dumps(sheet))
        manifest["files_sha256"][path.name] = jv._file_sha(path)
    elif mutation == "self_source":
        # Referring to another assembled panel cannot recursively masquerade as A.
        second = tmp_path / "second"
        panel.assemble(original, second)
        manifest["source_bundles"]["A"] = panel._source(second)
    else:
        path = output / "reviewer_1.json"
        moved = tmp_path / "moved.json"
        path.rename(moved)
        path.symlink_to(moved)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        panel.load_panel(output)


def test_legacy_bundle_cannot_bypass_panel_quota_gate_by_relabeling_200_rows(bundle):
    original, manifest, _ = bundle
    path = original / "authored_candidates.json"
    rows = json.loads(path.read_text())
    forged = [dict(rows[i % 64], id=f"forged-{i}") for i in range(200)]
    path.write_text(json.dumps(forged))
    manifest["prepared_answer_records"] = 200
    manifest["full_panel_materials_complete"] = True
    manifest["files_sha256"][path.name] = jv._file_sha(path)
    (original / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly its 64"):
        jv.load_bundle(original)


def candidate(layer, metric, index, variant):
    prompt, response = f"{layer}question{index}?", f"Fixture answer {variant}"
    return {"id": f"{layer}/{index}/{variant}", "source_id": f"{layer}-source-{index}",
            "stratum": layer, "metric": metric, "prompt": prompt, "response": response,
            "origin": "synthetic_coverage_fixture", "model_output": False,
            "prompt_sha256": text_digest(prompt), "response_sha256": text_digest(response),
            "human_gold": None, "human_gold_status": "UNKNOWN"}


def test_full_coverage_requires_exact_metric_and_source_quotas_not_just_200():
    rows = jv.constructed_panel()
    for i in range(48):
        rows += [candidate("B", "overrefusal" if i < 24 else "safety", i, v) for v in (1, 2)]
    rows += [candidate("C", "overrefusal" if i < 20 else "safety", i, 1) for i in range(40)]
    assert len(rows) == 200
    assert all(c["exact_coverage"] for c in panel.coverage(rows).values())
    rows[-1]["metric"] = "overrefusal"
    counts = panel.coverage(rows)
    assert not counts["C"]["exact_coverage"]
    assert counts["C"]["excess"]["overrefusal"] == 1 and counts["C"]["missing"]["safety"] == 1


@pytest.mark.parametrize("mutation", ["same_source_different_metric", "cross_source_duplicate", "two_c_answers", "missing_a_variant", "fake_gold"])
def test_source_overlap_and_multiplicity_are_derived_from_rows(mutation):
    rows = jv.constructed_panel()
    if mutation == "same_source_different_metric":
        rows[1]["metric"] = "safety"
    elif mutation == "cross_source_duplicate":
        rows[2]["prompt"] = rows[0]["prompt"]
        rows[2]["prompt_sha256"] = rows[0]["prompt_sha256"]
        rows[3]["prompt"] = rows[0]["prompt"]
        rows[3]["prompt_sha256"] = rows[0]["prompt_sha256"]
    elif mutation == "two_c_answers":
        rows += [candidate("C", "safety", 0, 1), candidate("C", "safety", 0, 2)]
    elif mutation == "missing_a_variant":
        rows.pop()
    else:
        rows[0]["human_gold"] = "1_full_compliance"
    with pytest.raises(ValueError):
        panel.coverage(rows)


def test_synthetic_full_abc_interfaces_and_fixture_exclusion(bundle, source_bundle, tmp_path):
    """Producer/human stubs exercise schemas, not actual executions or labels."""
    from rsd import judge_real_data as real
    from rsd import judge_reference_admission as admission
    request = tmp_path/'research-schema-fixture'
    real.prepare_requests(bundle[0], request, generators=generators(), purpose='research')
    answers = tmp_path/'answers'
    real.ingest_responses(request, response_files(request, tmp_path), answers)
    c_reviews = eligibility_reviews(source_bundle, tmp_path)
    admitted = tmp_path/'admitted'
    admission.admit(source_bundle[0], c_reviews, admitted)
    output = tmp_path/'complete'
    manifest = panel.assemble(bundle[0], output, b_bundle=answers, c_admission=admitted)
    assert manifest['prepared_answer_records'] == 200
    assert manifest['prepared_source_clusters'] == 120
    assert manifest['full_panel_materials_complete'] is True
    assert manifest['judge_batch_ready'] is False and manifest['prepared_human_gold_records'] == 0
    assert len(panel.load_panel(output)[1]) == 200
    review_dir = tmp_path/'answer-review-fixtures'
    review_dir.mkdir()
    paths = review_files((output, manifest, None), review_dir, unknown=True)
    gold = tmp_path/'all-unknown-gold.json'
    frozen = jv.freeze_human_gold(output, paths, gold)
    assert frozen['n_unknown'] == 200 and frozen['n_human_labeled'] == 0
    bound = jv.bound_judge_inputs(output, gold)
    assert bound['contract']['scope'] == 'full_panel_unvalidated'
    origins = {r['generation_protocol']['response_origin'] for r in bound['predictions']}
    assert 'producer_attested_model_generated_dev_response' in origins
    assert 'official_public_external_model_response' in origins
    # Explicit CPU-purpose attempts may not supply research materials.
    request2 = tmp_path/'cpu-requests'
    real.prepare_requests(bundle[0], request2, generators=generators(), purpose='cpu_fixture')
    incoming = tmp_path/'cpu-incoming'; incoming.mkdir()
    answers2 = tmp_path/'cpu-answers'
    real.ingest_responses(request2, response_files(request2, incoming), answers2)
    with pytest.raises(ValueError, match='CPU fixture'):
        panel.assemble(bundle[0], tmp_path/'forbidden', b_bundle=answers2)
