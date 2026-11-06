"""CPU acceptance: pair real evaluator summaries without inventing independent samples."""
import copy
import hashlib
import json

import pytest

from rsd.evaluation import digest, score_records, text_digest
from rsd.paired_analysis import analyze_paired, main


def fixture_data(split="dev"):
    return [
        {"id": name, "source_id": source, "split": split, "family": family,
         "benchmark": benchmark, "metric": metric, "prompt": "Fixture request " + name}
        for name, source, family, benchmark, metric in [
            ("s1a", "s1", "strongreject/category1", "strongreject", "safety"),
            ("s1b", "s1", "strongreject/category1", "strongreject", "safety"),
            ("s2", "s2", "strongreject/category1", "strongreject", "safety"),
            ("o1", "o1", "xstest/category1", "xstest", "overrefusal"),
            ("o2", "o2", "xstest/category1", "xstest", "overrefusal"),
        ]
    ]


def fixture_run(data, arm, seed, values=None, step=512):
    values = values or {}
    checkpoint = f"fixture-{arm}-{seed}-{step}"
    preds = [{"id": r["id"], "response": "CPU fixture response", "checkpoint_id": checkpoint,
              "generation_protocol": {"temperature": 0, "max_new_tokens": 256}} for r in data]
    judgments = [{"id": p["id"], "checkpoint_id": checkpoint,
                  "response_sha256": text_digest(p["response"]),
                  "judge_id": "frozen-fixture-judge", "value": values.get(p["id"], 0),
                  "valid": values.get(p["id"], 0) is not None} for p in preds]
    return {"arm": arm, "seed": seed, "step": step,
            "run_contract_sha256": digest({"arm": arm, "seed": seed, "step": step}),
            "summary": score_records(data, preds, judgments, checkpoint_id=checkpoint)}


def fixture_runs(data, right=None, seeds=(0, 1, 2), step=512):
    return [fixture_run(data, arm, seed, right if arm == "R0" else {}, step)
            for seed in seeds for arm in ("F0", "R0")]


def analyze(data, runs, **kwargs):
    return analyze_paired(data, runs, bootstrap_draws=100, **kwargs)


def metric_group(report, metric="safety"):
    return next(g for g in report["groups"] if g["metric"] == metric)


def test_real_legacy_score_records_shape_and_source_weighting():
    data = fixture_data()
    runs = fixture_runs(data, {"s1a": .8, "s1b": .2, "s2": .1, "o1": 1})
    # This is the existing rsd-eval-v1 output: no benchmark per record and no run seed/step.
    assert runs[0]["summary"]["schema_version"] == "rsd-eval-v1"
    assert "seed" not in runs[0]["summary"]
    assert "step" not in runs[0]["summary"]
    assert "benchmark" not in runs[0]["summary"]["records"][0]
    assert "n_valid" not in runs[0]["summary"]
    # Shape read from the archived step-0/1/2 benchmark-summary.json files.
    assert set(runs[0]["summary"]) == {
        "schema_version", "checkpoint_id", "dataset_sha256", "protocol_sha256", "protocol",
        "n_planned", "n_predictions", "n_judgments", "groups", "records", "status", "interpretation"}
    assert set(runs[0]["summary"]["records"][0]) == {
        "id", "source_id", "split", "metric", "family", "status", "value", "reason"}
    report = analyze(data, runs)
    safety = metric_group(report)
    assert safety["per_seed"][0]["source_statistics"]["mean_delta_complete_sources"] == pytest.approx(.3)
    assert safety["per_seed"][0]["source_statistics"]["descriptive_direction"] == "R0_higher"
    assert safety["per_seed"][0]["n_records_planned"] == 3
    across = safety["across_prespecified_seeds"]
    assert across["n_sources_planned"] == across["n_sources_valid"] == 2  # not 3 x 3 records
    assert across["n_training_seeds"] == 3
    assert across["mean_delta_complete_sources"] == pytest.approx(.3)
    assert metric_group(report, "overrefusal")["across_prespecified_seeds"]["mean_delta_complete_sources"] == .5
    assert "mean_delta_complete_sources" not in report
    assert report["scientific_conclusion"] is None and report["significance_claim"] is None
    assert len({r["run_contract_sha256"] for r in report["run_provenance"]}) == 6
    assert all(len(r["summary_canonical_sha256"]) == 64 for r in report["run_provenance"])


def test_unknown_variant_removes_cluster_not_denominator_and_not_zero():
    data = fixture_data()
    runs = fixture_runs(data, {"s1a": .8, "s1b": None, "s2": .1})
    report = analyze(data, runs)
    assert report["status"] == "complete_runs_with_unknown"
    seed = metric_group(report)["per_seed"][0]
    assert (seed["n_records_planned"], seed["n_record_pairs_valid"], seed["n_record_pairs_unknown"]) == (3, 2, 1)
    stats = seed["source_statistics"]
    assert (stats["n_sources_planned"], stats["n_sources_valid"], stats["n_sources_unknown"]) == (2, 1, 1)
    assert stats["mean_delta_complete_sources"] == .1
    assert stats["planned_source_mean_bounds_from_unknown"] == [-.45, .55]
    assert seed["sources"][0]["delta"] is None
    assert seed["sources"][0]["n_variants_valid"] == 1
    assert seed["arm_status_counts"]["R0"]["invalid"] == 1


def test_complete_cases_are_intersected_across_all_prespecified_seeds():
    data = fixture_data()
    runs = fixture_runs(data, {"s1a": .8, "s1b": .2, "s2": .1})
    runs[-1] = fixture_run(data, "R0", 2, {"s1a": .8, "s1b": None, "s2": .1})
    report = analyze(data, runs)
    group = metric_group(report)
    assert group["per_seed"][0]["source_statistics"]["n_sources_valid"] == 2
    assert group["across_prespecified_seeds"]["n_sources_valid"] == 1
    assert group["across_prespecified_seeds"]["mean_delta_complete_sources"] == pytest.approx(.1)
    assert group["across_prespecified_seeds"]["sources"][0]["delta"] is None


def test_missing_seed_does_not_silently_change_estimand():
    data = fixture_data()
    report = analyze(data, fixture_runs(data, seeds=(0, 1)))
    assert report["status"] == "incomplete_runs"
    assert report["missing_runs"] == [{"training_seed": 2, "arm": "F0"}, {"training_seed": 2, "arm": "R0"}]
    group = metric_group(report)
    assert group["across_prespecified_seeds"] is None
    absent = group["per_seed"][2]
    assert absent["source_statistics"]["mean_delta_complete_sources"] is None
    assert absent["source_statistics"]["descriptive_direction"] == "UNKNOWN"
    assert absent["source_statistics"]["planned_source_mean_bounds_from_unknown"] == [-1, 1]
    assert absent["n_record_pairs_unknown"] == absent["n_records_planned"] == 3


def test_zero_and_constant_observations_have_no_fake_zero_width_population_ci():
    data = fixture_data()
    report = analyze(data, fixture_runs(data))
    stats = metric_group(report)["across_prespecified_seeds"]
    assert stats["mean_delta_complete_sources"] == 0
    assert stats["bootstrap_interval"] is None
    assert "not_zero_population_uncertainty" in stats["bootstrap_unavailable_reason"]
    assert stats["bounded_sensitivity_interval"][0] < 0 < stats["bounded_sensitivity_interval"][1]
    assert "Independent source" in stats["bounded_sensitivity_assumptions"]
    assert stats["planned_source_mean_bounds_from_unknown"] == [0, 0]  # finite observed cohort only


def test_bootstrap_is_deterministic_bounded_and_conditional():
    data = fixture_data()
    runs = fixture_runs(data, {"s1a": .8, "s1b": .2, "s2": .1})
    a = analyze(data, runs)
    b = analyze(data, runs)
    assert a == b
    stats = metric_group(a)["across_prespecified_seeds"]
    assert .1 <= stats["bootstrap_interval"][0] < stats["bootstrap_interval"][1] <= .5
    assert stats["sparse_change_warning"] is True
    assert "training seeds" in stats["interval_scope"]


@pytest.mark.parametrize("field,value,match", [
    ("dataset_sha256", "f" * 64, "dataset hash"),
    ("protocol_sha256", "f" * 64, "protocol body/hash"),
    ("n_planned", 4, "planned denominator"),
    ("checkpoint_id", None, "checkpoint identity"),
    ("status", "incomplete", "completion status"),
])
def test_summary_identity_and_denominator_mismatch_rejected(field, value, match):
    data = fixture_data()
    runs = fixture_runs(data)
    runs[0]["summary"][field] = value
    with pytest.raises(ValueError, match=match):
        analyze(data, runs)


def test_equal_hash_with_changed_protocol_body_is_rejected_and_distinct_protocols_rejected():
    data = fixture_data()
    runs = fixture_runs(data)
    protocol = runs[0]["summary"]["protocol"]
    protocol["generation_protocol_hash"] = "f" * 64
    with pytest.raises(ValueError, match="protocol body/hash"):
        analyze(data, runs)
    runs[0]["summary"]["protocol_sha256"] = digest(protocol)
    with pytest.raises(ValueError, match="protocols differ"):
        analyze(data, runs)


@pytest.mark.parametrize("case", ["source", "id", "duplicate", "partial", "unknown_value"])
def test_corrupt_record_identity_or_unknown_value_rejected(case):
    data = fixture_data()
    runs = fixture_runs(data)
    records = runs[0]["summary"]["records"]
    if case == "source":
        records[0]["source_id"] = "another-source"
    elif case == "id":
        records[0]["id"] = "foreign"
    elif case == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif case == "partial":
        records.pop()
    else:
        records[0]["status"] = "invalid"
        records[0]["value"] = 0
    with pytest.raises(ValueError):
        analyze(data, runs)


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf"), True, "0"])
def test_values_must_be_finite_numeric_bounded(value):
    data = fixture_data()
    runs = fixture_runs(data)
    runs[0]["summary"]["records"][0]["value"] = value
    with pytest.raises(ValueError, match="finite numeric"):
        analyze(data, runs)


def test_overrefusal_is_binary_without_reinterpreting_raw_labels():
    data = fixture_data()
    runs = fixture_runs(data, {"o1": .5})
    with pytest.raises(ValueError, match="frozen binary"):
        analyze(data, runs)
    runs = fixture_runs(data, {"o1": 1})
    for run in runs:
        for record in run["summary"]["records"]:
            record["raw_label"] = "compliance"  # deliberately contradictory; not a semantic input
    report = analyze(data, runs)
    assert metric_group(report, "overrefusal")["per_seed"][0]["source_statistics"]["mean_delta_complete_sources"] == .5


@pytest.mark.parametrize("field,value", [("seed", 99), ("seed", True), ("step", 256), ("arm", "other"), ("run_contract_sha256", "bad")])
def test_run_envelope_mismatch_rejected(field, value):
    data = fixture_data()
    runs = fixture_runs(data)
    runs[0][field] = value
    with pytest.raises(ValueError):
        analyze(data, runs)


def test_duplicate_runs_and_changed_matched_recipe_rejected():
    data = fixture_data()
    runs = fixture_runs(data)
    with pytest.raises(ValueError, match="Duplicate run"):
        analyze(data, runs + [runs[0]])
    for run in runs:
        run["matched_recipe_sha256"] = "a" * 64
    assert analyze(data, runs)["matched_recipe_sha256"] == "a" * 64
    runs[0]["matched_recipe_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="recipe hashes differ"):
        analyze(data, runs)


def test_test_release_optin_and_fixed_primary_endpoint_role():
    data = fixture_data("test")
    with pytest.raises(ValueError, match="allow_test"):
        analyze(data, fixture_runs(data))
    report = analyze(data, fixture_runs(data), allow_test=True)
    assert metric_group(report)["analysis_role"] == "primary_endpoint_candidate"
    assert metric_group(report, "overrefusal")["analysis_role"] == "secondary_descriptive"
    earlier = analyze(data, fixture_runs(data, step=256), step=256, allow_test=True)
    assert metric_group(earlier)["analysis_role"] == "secondary_descriptive"
    assert earlier["primary_optimizer_step"] == 512


def test_same_source_can_be_evaluated_under_two_metrics_without_pooling():
    data = fixture_data()
    data[3]["source_id"] = "s1"
    report = analyze(data, fixture_runs(data, {"s1a": 1, "s1b": 1, "o1": 1}))
    assert len(report["groups"]) == 2
    assert metric_group(report)["across_prespecified_seeds"]["n_sources_planned"] == 2
    assert metric_group(report, "overrefusal")["across_prespecified_seeds"]["n_sources_planned"] == 2


def test_cli_records_original_file_hashes_and_refuses_overwrite(tmp_path):
    data = fixture_data()
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in data) + "\n")
    entries = []
    for index, run in enumerate(fixture_runs(data)):
        name = f"summary-{index}.json"
        (tmp_path / name).write_text(json.dumps(run["summary"]))
        entries.append({**run, "summary": name})
    plan = {"dataset": "data.jsonl", "runs": entries, "expected_seeds": [0, 1, 2], "step": 512, "bootstrap_draws": 100}
    plan_path, output = tmp_path / "plan.json", tmp_path / "analysis.json"
    plan_path.write_text(json.dumps(plan))
    args = ["--plan", str(plan_path), "--output", str(output)]
    main(args)
    report = json.loads(output.read_text())
    assert len(report["input_file_sha256"]) == 8
    for path, sha in report["input_file_sha256"].items():
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == sha
    with pytest.raises(FileExistsError):
        main(args)


def test_hoeffding_sensitivity_uses_source_count_and_known_bounded_radius():
    from math import log, sqrt
    from rsd.paired_analysis import _statistics
    sources = [{"source_id": str(i), "family": "one-stratum", "delta": 0} for i in range(100)]
    stats = _statistics(sources, alpha=.05, draws=100, rng_seed=0)
    radius = sqrt(2 * log(40) / 100)
    assert stats["bounded_sensitivity_interval"] == pytest.approx([-radius, radius])
    assert stats["bootstrap_interval"] is None
    assert radius > 0


def test_direction_is_signed_without_turning_it_into_a_significance_claim():
    data = fixture_data()
    runs = [fixture_run(data, arm, seed, {"s1a": .8, "s1b": .2, "s2": .1} if arm == "F0" else {})
            for seed in (0, 1, 2) for arm in ("F0", "R0")]
    report = analyze(data, runs)
    group = metric_group(report)
    assert group["direction_of_metric"] == "lower_is_better"
    assert group["per_seed"][0]["source_statistics"]["descriptive_direction"] == "R0_lower"
    assert group["across_prespecified_seeds"]["mean_delta_complete_sources"] == pytest.approx(-.3)
    assert report["significance_claim"] is None
