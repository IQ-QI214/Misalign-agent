"""Synthetic annotations/judge outputs only; no actual human gold or model calls."""
import copy
import json
from pathlib import Path

import pytest

from rsd import judge_validation as jv
from rsd import judge_validation_analysis as ja
from rsd import local_judge
from rsd.evaluation import digest, text_digest
from rsd.judge_protocols import (PARSER_VERSION, PROFILE_NAME, V3_PROFILE_NAME,
                                 judge_systems, load_source_lock, parse_judge_output)


def empty_sessions():
    return [{"profile_name": p, "round": r, "binding": None, "manifest": None, "judgments": []}
            for p, r in ja.SESSION_ORDER]


def synthetic_profile(name):
    hashes = {"config.json": text_digest("synthetic config; no model"),
              "model.safetensors": text_digest("synthetic no-weights fixture")}
    return {"name": name, "alternative_judge_model": True,
            "claim": "CPU synthetic test fixture; no actual judge or human gold",
            "loader": {"class": "Gemma3ForConditionalGeneration", "full_multimodal_weights": True,
                       "local_files_only": True, "trust_remote_code": False,
                       "strict_loading_info": True, "use_safetensors": True},
            "model_identity": {"checkpoint_id": "sha256:" + digest(hashes), "files_sha256": hashes},
            "generation": {"version": "local-rubric-generation-v1", "max_prompt_tokens": 8192,
                           "max_new_tokens": 256, "do_sample": False, "num_beams": 1,
                           "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                           "seed": 0, "dtype": "bfloat16", "attn_implementation": "sdpa",
                           "eos_token_id": [1, 106], "eos_token_id_source": "generation_config.json",
                           "pad_token_id": 0, "truncation": False},
            "tokenizer": {"template_source": "synthetic", "template_sha256": text_digest("template"),
                          "tokenizer_vocab_sha256": text_digest("vocab")},
            "source_lock": load_source_lock(name), "systems": judge_systems(name),
            "parser_version": PARSER_VERSION,
            "packages": {k: "synthetic-no-model" for k in ("torch", "transformers", "tokenizers", "safetensors")},
            "device": "cuda:0", "driver_sha256": ja._sha(local_judge.__file__),
            "protocol_implementation_sha256": ja._sha(Path(ja.__file__).with_name("judge_protocols.py"))}


def binding(rows, gold, name, manifest_sha, gold_sha):
    contract = {"version": jv.VERSION, "bundle_manifest_sha256": manifest_sha,
                "human_gold_file_sha256": gold_sha, "profile_name": name,
                "source_lock_sha256": digest(load_source_lock(name)),
                "protocol_implementation_sha256": ja._sha(Path(ja.__file__).with_name("judge_protocols.py")),
                "scope": "partial_panel_diagnostic_only", "n_prepared": len(rows),
                "n_human_labeled": gold["n_human_labeled"], "scientific_conclusion": None}
    checkpoint = "validation-materials:sha256:" + digest(contract)
    dataset = [{"id": r["id"], "source_id": r["source_id"], "family": "judge-v3/" + r["metric"],
                "split": "dev", "metric": r["metric"], "prompt": r["prompt"],
                "benchmark": "independent_judge_validation_candidates", "evaluation_only": True,
                "not_subject_model_performance": True} for r in rows]
    predictions = [{"id": r["id"], "checkpoint_id": checkpoint, "valid": True,
                    "prompt_sha256": r["prompt_sha256"], "response_sha256": r["response_sha256"],
                    "response": r["response"], "generation_protocol": {"model_inference": False,
                    "response_origin": r["origin"], "original_model_output": r["model_output"], "contract": contract}} for r in rows]
    return {"contract": contract, "dataset": dataset, "predictions": predictions}


def raw(metric, label=None, components=None):
    if metric == "overrefusal":
        return label
    return "\n".join(f"{i}.b {components[k]}" for i, k in enumerate(ja.COMPONENTS, 1))


def set_raw(record, text, truncated=False):
    parsed = parse_judge_output(record["metric"], text)
    record.update(raw_judge_text=text, parser_result=parsed, reached_token_limit=truncated,
                  stopped_on_eos=not truncated, **local_judge._judgment_fields(parsed, truncated))


def make_fixture(rows=None, unknown_ids=()):
    rows = copy.deepcopy(rows if rows is not None else jv.constructed_panel()[:4] + jv.constructed_panel()[48:52])
    manifest_sha = text_digest("synthetic bundle manifest")
    gold = {"version": jv.VERSION, "bundle_manifest_sha256": manifest_sha,
            "review_files_sha256": [text_digest("synthetic review 1"), text_digest("synthetic review 2")],
            "reviewer_ids": ["synthetic-stub-1", "synthetic-stub-2"], "adjudication_sha256": None,
            "adjudicator_id": None, "n_prepared": len(rows), "records": [],
            "planned_full_panel_complete": False, "scientific_conclusion": None}
    for r in rows:
        unknown = r["id"] in unknown_ids
        item = {"metric": r["metric"], "annotation_status": "UNKNOWN" if unknown else "LABELED",
                "raw_label": "1_full_compliance" if r["metric"] == "overrefusal" and not unknown else None,
                "components": {"refusal": 0, "convincingness": 2, "specificity": 4}
                if r["metric"] == "safety" and not unknown else None}
        gold["records"].append({**{k: r[k] for k in ("id", "source_id", "metric", "prompt_sha256", "response_sha256")},
                                "annotation_id": text_digest(jv.VERSION + "\n" + r["id"]),
                                "human_gold_status": item["annotation_status"], "raw_label": item["raw_label"],
                                "components": item["components"], "value": jv._annotation_value(item) if not unknown else None})
    gold["n_unknown"] = len(unknown_ids)
    gold["n_human_labeled"] = len(rows) - len(unknown_ids)
    gold_sha = digest(gold)
    sessions = []
    for name, number in ja.SESSION_ORDER:
        bound = binding(rows, gold, name, manifest_sha, gold_sha)
        profile = synthetic_profile(name)
        judge_id = name + ":" + digest(profile)
        manifest = {"judge_id": judge_id, "judge_profile": profile,
                    "dataset_sha256": digest(bound["dataset"]), "predictions_sha256": digest(bound["predictions"]),
                    "selected_ids": [r["id"] for r in rows], "limit": None, "allow_test": False}
        records = []
        for r, pred in zip(rows, bound["predictions"]):
            record = {**{k: r[k] for k in ("id", "metric", "prompt_sha256", "response_sha256")},
                      "checkpoint_id": pred["checkpoint_id"], "judge_id": judge_id,
                      "judge_profile": copy.deepcopy(profile)}
            set_raw(record, raw(r["metric"], "1_full_compliance", {"refusal": 0, "convincingness": 2, "specificity": 4}))
            records.append(record)
        sessions.append({"profile_name": name, "round": number, "binding": bound, "manifest": manifest, "judgments": records})
    return rows, gold, sessions, {"bundle_manifest_sha256": manifest_sha,
                                 "human_gold_file_sha256": gold_sha, "bootstrap_draws": 100}


def add_original_annotations(gold):
    for record in gold["records"]:
        item = {"annotation_status": record["human_gold_status"], "raw_label": record["raw_label"],
                "components": record["components"]}
        record.update(original_annotations=[copy.deepcopy(item), copy.deepcopy(item)],
                      original_reviews_agree=item["annotation_status"] == "LABELED", adjudicated=False)


def sync_binding_after_synthetic_gold_edit(rows, gold, sessions, kwargs):
    gold["n_human_labeled"] = sum(r["human_gold_status"] == "LABELED" for r in gold["records"])
    gold["n_unknown"] = len(rows) - gold["n_human_labeled"]
    kwargs["human_gold_file_sha256"] = digest(gold)
    for session in sessions:
        session["binding"] = binding(rows, gold, session["profile_name"], kwargs["bundle_manifest_sha256"], kwargs["human_gold_file_sha256"])
        bound = session["binding"]
        session["manifest"].update(dataset_sha256=digest(bound["dataset"]), predictions_sha256=digest(bound["predictions"]))
        for record, pred in zip(session["judgments"], bound["predictions"]):
            record["checkpoint_id"] = pred["checkpoint_id"]


def test_no_gold_no_judge_retains_denominators_and_null_without_reliability():
    rows = jv.constructed_panel()
    result = ja.analyze_validation(rows, None, empty_sessions(), bundle_manifest_sha256=text_digest("manifest"))
    assert result["n_prepared_answer_records"] == 64
    assert result["n_independent_source_clusters"] == 32
    assert result["n_recorded_model_output_answers"] == result["n_human_labeled"] == 0
    assert result["n_human_unknown"] == 64
    assert result["validated_for_exploration"] is False and result["scientific_conclusion"] is None
    assert result["reliability"] == "UNKNOWN"
    group = result["groups"]["A/overrefusal"]
    assert group["rounds"][0]["n_judgments_missing"] == 48
    assert group["rounds"][0]["three_class"]["confusion_gold_rows_judge_columns"] is None
    assert group["rounds"][0]["binary_overrefusal"]["mae"] is None
    assert group["paired_profile_differences"]["n_complete_sources"] == 0
    assert group["human_review"]["original_annotation_evidence"] == "UNKNOWN"
    assert group["human_review"]["n_adjudicated"] is None
    assert result["groups"]["B/safety"]["rounds"][0]["continuous_score"]["mae"] is None


def test_confusion_binary_bias_and_component_errors_are_separate():
    rows, gold, sessions, kwargs = make_fixture()
    set_raw(sessions[0]["judgments"][0], "3_partial_refusal")
    set_raw(sessions[0]["judgments"][4], raw("safety", components={"refusal": 1, "convincingness": 5, "specificity": 1}))
    result = ja.analyze_validation(rows, gold, sessions, **kwargs)
    x = result["groups"]["A/overrefusal"]["rounds"][0]
    assert x["three_class"]["confusion_gold_rows_judge_columns"]["1_full_compliance"]["3_partial_refusal"] == 1
    assert x["three_class"]["accuracy"] == .75
    assert x["binary_overrefusal"]["signed_bias"] == .25
    assert x["binary_overrefusal"]["false_positive_count"] == 1
    s = result["groups"]["A/safety"]["rounds"][0]
    assert s["components"]["refusal"]["mae"] == .25
    assert s["components"]["convincingness"]["signed_bias"] == .75
    assert s["components"]["specificity"]["signed_bias"] == -.75
    assert s["continuous_score"]["mae"] == .125
    assert s["continuous_score"]["signed_bias"] == -.125


def test_unknown_invalid_missing_are_distinct_and_partial_source_excluded():
    rows = jv.constructed_panel()[:4]
    rows, gold, sessions, kwargs = make_fixture(rows, unknown_ids=(rows[0]["id"],))
    set_raw(sessions[0]["judgments"][1], "invalid prose")
    sessions[0]["judgments"].pop()
    result = ja.analyze_validation(rows, gold, sessions, **kwargs)
    r = result["groups"]["A/overrefusal"]["rounds"][0]
    assert (r["n_prepared"], r["n_gold_unknown"], r["n_judgments_valid"], r["n_judgments_invalid"], r["n_judgments_missing"]) == (4, 1, 2, 1, 1)
    assert r["n_paired_gold_valid"] == 1
    assert result["groups"]["A/overrefusal"]["paired_profile_differences"]["n_complete_sources"] == 0


def test_two_rounds_and_two_variants_do_not_increase_independent_n():
    rows, gold, sessions, kwargs = make_fixture(jv.constructed_panel()[:4])
    for session in sessions:
        if session["profile_name"] == PROFILE_NAME:
            for record in session["judgments"][:2]:
                set_raw(record, "2_full_refusal")
    result = ja.analyze_validation(rows, gold, sessions, **kwargs)
    contrast = result["groups"]["A/overrefusal"]["paired_profile_differences"]
    estimate = contrast["fields"]["value"]["mae"]
    assert contrast["n_complete_sources"] == 2 and contrast["n_paired_answer_records"] == 4
    assert estimate["n_source_clusters"] == 2 and estimate["estimate"] == -.5
    assert estimate["bootstrap_draws"] == 100 and estimate["ci"] == [-1.0, 0.0]
    assert result["n_valid_judgments_across_rounds_not_independent_n"] == 16


def test_source_weighting_is_equal_even_with_different_answer_multiplicity():
    rows = copy.deepcopy(jv.constructed_panel()[:3])
    for row in rows:
        row["stratum"] = "B"
    rows, gold, sessions, kwargs = make_fixture(rows)
    for session in sessions:
        if session["profile_name"] == PROFILE_NAME:
            for record in session["judgments"][:2]:
                set_raw(record, "2_full_refusal")
    result = ja.analyze_validation(rows, gold, sessions, **kwargs)
    contrast = result["groups"]["B/overrefusal"]["paired_profile_differences"]
    assert contrast["fields"]["value"]["mae"]["estimate"] == -.5  # not -2/3


def test_three_class_profile_error_detects_full_vs_partial_hidden_by_binary():
    rows, gold, sessions, kwargs = make_fixture(jv.constructed_panel()[:4])
    for record in gold["records"]:
        record.update(raw_label="2_full_refusal", value=1.)
    sync_binding_after_synthetic_gold_edit(rows, gold, sessions, kwargs)
    for session in sessions:
        for index, record in enumerate(session["judgments"]):
            label = "3_partial_refusal" if session["profile_name"] == PROFILE_NAME and index < 2 else "2_full_refusal"
            set_raw(record, label)
    result = ja.analyze_validation(rows, gold, sessions, **kwargs)
    fields = result["groups"]["A/overrefusal"]["paired_profile_differences"]["fields"]
    assert fields["value"]["mae"]["estimate"] == 0
    assert fields["value"]["signed_bias"]["estimate"] == 0
    error = fields["three_class_error"]["error_rate"]
    assert error["n_source_clusters"] == 2 and error["estimate"] == -.5
    assert error["ci"] == [-1., 0.] and error["bootstrap_draws"] == 100


def test_original_human_disagreement_unknowns_confusion_and_adjudication_are_separate():
    rows, gold, sessions, kwargs = make_fixture()
    add_original_annotations(gold)
    gold.update(adjudication_sha256=text_digest("synthetic adjudication only"), adjudicator_id="synthetic-stub-3")
    # Original reviewers differ. The synthetic adjudicator retains final gold.
    gold["records"][1]["original_annotations"][1]["raw_label"] = "2_full_refusal"
    gold["records"][1].update(original_reviews_agree=False, adjudicated=True)
    gold["records"][2]["original_annotations"][0] = {"annotation_status": "UNKNOWN", "raw_label": None, "components": None}
    gold["records"][2].update(original_reviews_agree=False, adjudicated=True)
    gold["records"][4]["original_annotations"][1]["components"] = {"refusal": 1, "convincingness": 5, "specificity": 1}
    gold["records"][4].update(original_reviews_agree=False, adjudicated=True)
    sync_binding_after_synthetic_gold_edit(rows, gold, sessions, kwargs)
    result = ja.analyze_validation(rows, gold, sessions, **kwargs)
    human = result["groups"]["A/overrefusal"]["human_review"]
    assert human["original_annotation_evidence"] == "COMPLETE"
    assert human["reviewer_order"] == gold["reviewer_ids"]
    assert human["n_original_both_labeled"] == 3 and human["n_original_at_least_one_unknown"] == 1
    assert human["original_exact_agreement_rate"] == pytest.approx(2 / 3)
    assert human["original_annotation_status_reviewer1_rows_reviewer2_columns"]["UNKNOWN"]["LABELED"] == 1
    assert human["three_class_confusion_reviewer1_rows_reviewer2_columns"]["1_full_compliance"]["2_full_refusal"] == 1
    assert human["n_adjudicated"] == 2 and human["n_adjudication_status_known"] == 4
    assert human["adjudication_rate_among_known_status"] == .5
    strong = result["groups"]["A/safety"]["human_review"]
    assert strong["original_exact_agreement_rate"] == .75
    assert strong["components"]["refusal"]["mae"] == .25
    assert strong["components"]["refusal"]["exact_agreement_rate"] == .75
    assert strong["components"]["convincingness"]["signed_bias"] == .75
    assert strong["components"]["specificity"]["signed_bias"] == -.75
    assert strong["continuous_score"]["signed_bias"] == -.125
    assert strong["continuous_score"]["exact_agreement_rate"] == .75
    assert strong["n_adjudicated"] == 1


def test_legacy_gold_does_not_infer_original_agreement_from_final_labels():
    rows, gold, sessions, kwargs = make_fixture()
    for record in gold["records"]:
        record.update(original_reviews_agree=True, adjudicated=False)
    sync_binding_after_synthetic_gold_edit(rows, gold, sessions, kwargs)
    result = ja.analyze_validation(rows, gold, sessions, **kwargs)
    human = result["groups"]["A/overrefusal"]["human_review"]
    assert human["original_annotation_evidence"] == "UNKNOWN"
    assert human["n_original_annotation_records_missing"] == 4
    assert human["original_exact_agreement_rate"] is None
    assert human["three_class_confusion_reviewer1_rows_reviewer2_columns"] is None
    # Existing explicit adjudication status remains usable, never inferred.
    assert human["n_adjudication_status_known"] == 4 and human["n_adjudicated"] == 0


@pytest.mark.parametrize("mutation,pattern", [
    (lambda g: g["records"][0].update(original_annotations=[]), "two ordered"),
    (lambda g: g["records"][0]["original_annotations"][0].update(extra="field"), "annotation schema"),
    (lambda g: g["records"][0]["original_annotations"][0].update(annotation_status="UNKNOWN"), "null labels"),
    (lambda g: g["records"][0].update(original_reviews_agree=False), "agreement/status"),
    (lambda g: g["records"][0].update(adjudicated=True), "matching frozen receipt"),
    (lambda g: g["records"][0].update(raw_label="2_full_refusal", value=1.), "differ from frozen gold"),
])
def test_original_annotations_are_validated_not_merely_reported(mutation, pattern):
    rows, gold, sessions, kwargs = make_fixture()
    add_original_annotations(gold)
    mutation(gold)
    sync_binding_after_synthetic_gold_edit(rows, gold, sessions, kwargs)
    with pytest.raises(ValueError, match=pattern):
        ja.analyze_validation(rows, gold, sessions, **kwargs)


def test_original_disagreement_without_adjudication_cannot_become_gold():
    rows, gold, sessions, kwargs = make_fixture()
    add_original_annotations(gold)
    gold["records"][0]["original_annotations"][1]["raw_label"] = "2_full_refusal"
    gold["records"][0]["original_reviews_agree"] = False
    sync_binding_after_synthetic_gold_edit(rows, gold, sessions, kwargs)
    with pytest.raises(ValueError, match="need adjudication"):
        ja.analyze_validation(rows, gold, sessions, **kwargs)


def test_degenerate_bootstrap_retains_estimate_without_zero_uncertainty_claim():
    rows, gold, sessions, kwargs = make_fixture()
    result = ja.analyze_validation(rows, gold, sessions, **kwargs)
    stat = result["groups"]["A/overrefusal"]["paired_profile_differences"]["fields"]["value"]["mae"]
    assert stat["estimate"] == 0 and stat["ci"] is None
    assert stat["uncertainty_status"].startswith("degenerate_empirical")


def test_repeat_and_injection_changes_are_not_substituted_for_accuracy():
    rows, gold, sessions, kwargs = make_fixture(jv.constructed_panel()[:4])
    set_raw(sessions[1]["judgments"][1], "2_full_refusal")
    result = ja.analyze_validation(rows, gold, sessions, **kwargs)
    group = result["groups"]["A/overrefusal"]
    assert group["repeat_stability"][V3_PROFILE_NAME]["three_class_change_rate"] == .25
    assert group["repeat_stability"][V3_PROFILE_NAME]["score_signed_change_round2_minus_round1"] == -.25
    pair = group["injection_pairs"]["sessions"][1]
    assert pair["n_valid_pairs"] == 2 and pair["score_signed_change_injected_minus_original"] == .5
    assert pair["gold_score_change"] == 0 and pair["score_error_change"] == .5


@pytest.mark.parametrize("mutation,pattern", [
    (lambda r, g, s, k: r[0].update(response="changed"), "text/hash"),
    (lambda r, g, s, k: r[2].update(source_id=r[0]["source_id"]), "Source cluster"),
    (lambda r, g, s, k: g["records"][0].update(value=1.), "gold value"),
    (lambda r, g, s, k: g.update(n_unknown=1), "denominators"),
    (lambda r, g, s, k: s.reverse(), "session order"),
    (lambda r, g, s, k: s[0]["binding"]["predictions"][0].update(response="changed"), "dataset/prediction"),
    (lambda r, g, s, k: s[0]["binding"]["predictions"][0]["generation_protocol"].update(response_origin="foreign"), "dataset/prediction"),
    (lambda r, g, s, k: s[0]["binding"]["predictions"][0]["generation_protocol"].update(original_model_output=True), "dataset/prediction"),
    (lambda r, g, s, k: s[0]["manifest"].update(selected_ids=[]), "selected inputs"),
    (lambda r, g, s, k: s[0]["judgments"].append(copy.deepcopy(s[0]["judgments"][0])), "foreign/duplicate"),
    (lambda r, g, s, k: s[0]["judgments"][0].update(raw_judge_text="2_full_refusal"), "raw judge text"),
    (lambda r, g, s, k: s[0]["judgments"][0].update(value=1.), "raw judge text"),
    (lambda r, g, s, k: s[0]["judgments"][0].update(parser_result=None), "raw judge text"),
    (lambda r, g, s, k: s[0]["judgments"][0].update(prompt_sha256=text_digest("foreign")), "foreign/duplicate"),
    (lambda r, g, s, k: s[0]["manifest"]["judge_profile"].update(driver_sha256=text_digest("old driver")), "provenance"),
    (lambda r, g, s, k: s[0]["manifest"]["judge_profile"]["source_lock"].update(extra="tampered"), "provenance"),
    (lambda r, g, s, k: s[0]["manifest"]["judge_profile"].pop("model_identity"), "complete local_judge"),
    (lambda r, g, s, k: s[0].update(binding=None), "requires frozen input"),
])
def test_tampering_fails_closed(mutation, pattern):
    rows, gold, sessions, kwargs = make_fixture()
    mutation(rows, gold, sessions, kwargs)
    with pytest.raises(ValueError, match=pattern):
        ja.analyze_validation(rows, gold, sessions, **kwargs)


def test_profile_comparison_cannot_change_generation_identity():
    rows, gold, sessions, kwargs = make_fixture()
    for session in sessions:
        if session["profile_name"] != V3_PROFILE_NAME:
            continue
        profile = session["manifest"]["judge_profile"]
        profile["generation"]["max_new_tokens"] = 512
        new_id = V3_PROFILE_NAME + ":" + digest(profile)
        session["manifest"]["judge_id"] = new_id
        for record in session["judgments"]:
            record.update(judge_profile=copy.deepcopy(profile), judge_id=new_id)
    with pytest.raises(ValueError, match="changed model, generation"):
        ja.analyze_validation(rows, gold, sessions, **kwargs)


def test_truncated_and_no_text_failures_are_invalid_not_zero():
    rows, gold, sessions, kwargs = make_fixture()
    set_raw(sessions[0]["judgments"][0], "1_full_compliance", truncated=True)
    record = sessions[0]["judgments"][1]
    record.update(raw_judge_text=None, valid=False, value=None, parser_result=None,
                  raw_label=None, components={}, error="runtime_failure")
    result = ja.analyze_validation(rows, gold, sessions, **kwargs)
    r = result["groups"]["A/overrefusal"]["rounds"][0]
    assert r["n_judgments_invalid"] == 2 and r["n_paired_gold_valid"] == 2
    assert r["invalid_reasons"] == {"judge_output_token_limit": 1, "runtime_failure": 1}


def test_real_material_loader_freezer_and_bound_export_compatibility(tmp_path):
    historical = json.loads((jv.ROOT / "configs/rsd/milestone_dev_cohort_v1.json").read_text())["ids"]
    ids = historical + [f"xstest_safe/fixture-{i}" for i in range(32)] + [f"strongreject/fixture-{i}" for i in range(32)]
    dev = [{"id": key, "source_id": "source-" + key, "family": "fixture", "split": "dev",
            "prompt": f"Fixture-{i} category-{i % 7}?", "benchmark": "xstest_safe" if key.startswith("xstest") else "strongreject",
            "metric": "overrefusal" if key.startswith("xstest") else "safety"} for i, key in enumerate(ids)]
    source = tmp_path / "dev.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in dev))
    bundle = tmp_path / "materials"
    jv.build(source, bundle, expected_dev_sha=ja._sha(source))
    review_paths = []
    for n in (1, 2):
        p = bundle / f"reviewer_{n}.json"
        sheet = json.loads(p.read_text())
        sheet.update(reviewer_id=f"synthetic-stub-{n}", reviewer_kind="human", independent_annotation_claim=True)
        # All labels UNKNOWN. This receipt is not actual human validation.
        p = tmp_path / f"synthetic-review-{n}.json"
        p.write_text(json.dumps(sheet))
        review_paths.append(p)
    gold_path = tmp_path / "synthetic-frozen-gold.json"
    gold = jv.freeze_human_gold(bundle, review_paths, gold_path)
    _, rows = jv.load_bundle(bundle)
    sessions = empty_sessions()
    for session in sessions:
        session["binding"] = jv.bound_judge_inputs(bundle, gold_path, profile_name=session["profile_name"], allow_partial_panel=True)
    result = ja.analyze_validation(rows, gold, sessions, bundle_manifest_sha256=ja._sha(bundle / "manifest.json"),
                                   human_gold_file_sha256=ja._sha(gold_path))
    assert result["n_human_unknown"] == 64 and result["reliability"] == "UNKNOWN"
    human = result["groups"]["A/overrefusal"]["human_review"]
    assert human["original_annotation_evidence"] == "COMPLETE"
    assert human["n_original_at_least_one_unknown"] == 48
    assert human["original_exact_agreement_rate"] is None
    assert human["n_adjudicated"] == 0
    # File-based path executes the actual verified loader and exact artifact hashes.
    plan = {"bundle_dir": str(bundle), "sessions": [{"profile_name": p, "round": r} for p, r in ja.SESSION_ORDER]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    report_path = tmp_path / "report.json"
    report = ja.analyze_plan(plan_path, report_path)
    assert report["input_file_sha256"][str(bundle / "manifest.json")] == ja._sha(bundle / "manifest.json")
    assert report["analysis_code_sha256"] == ja._sha(ja.__file__)
    assert json.loads(report_path.read_text())["n_prepared_answer_records"] == 64
    with pytest.raises(FileExistsError):
        ja.analyze_plan(plan_path, report_path)
