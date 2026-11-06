"""CPU-only paired F0/R0 analysis; no inference, pooled-seed pseudo-replication or p-value search.

This consumes frozen score_records summaries plus explicit run envelopes. It
validates content identity, not the truthfulness of externally supplied run
metadata. Benchmark validity / judge reliability remain separate requirements.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

from .evaluation import METRICS, VERSION as SCORE_VERSION, digest, read_jsonl, validate_dataset

VERSION = "rsd-paired-analysis-v2"
ARMS = ("F0", "R0")
PRIMARY_STEP = 512
SHA = re.compile(r"[0-9a-f]{64}")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _benchmark(row):
    value = row.get("benchmark") or row.get("domain") or "unspecified_task"
    if not isinstance(value, str) or not value:
        raise ValueError("Benchmark/domain must be a nonempty string")
    if row["metric"] in {"safety", "overrefusal"} and value == "unspecified_task":
        raise ValueError("Safety/overrefusal rows require benchmark identity")
    return value


def _group(row):
    return row["metric"], _benchmark(row), row["split"]


def _index_summary(summary, data, dataset_sha):
    if summary.get("schema_version") != SCORE_VERSION or summary.get("dataset_sha256") != dataset_sha:
        raise ValueError("Summary schema/dataset hash differs from frozen dataset")
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict) or summary.get("protocol_sha256") != digest(protocol):
        raise ValueError("Summary protocol body/hash mismatch")
    if not isinstance(protocol.get("generation_protocol_hash"), str) or not SHA.fullmatch(protocol["generation_protocol_hash"]):
        raise ValueError("Explicit generation protocol identity is required")
    if any(row["metric"] != "exact_match" for row in data.values()) and not protocol.get("judge_ids"):
        raise ValueError("Judge identity is required for safety/overrefusal")
    checkpoint = summary.get("checkpoint_id")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("Summary requires checkpoint identity")
    records = summary.get("records")
    if not isinstance(records, list) or summary.get("n_planned") != len(data):
        raise ValueError("Summary planned denominator differs")
    indexed = {}
    for record in records:
        key = record.get("id")
        if key not in data or key in indexed:
            raise ValueError("Summary contains unknown or duplicate record ids")
        original = data[key]
        if any(record.get(k) != original[k] for k in ("source_id", "metric", "family", "split")):
            raise ValueError("Summary record identity differs from dataset")
        status, value = record.get("status"), record.get("value")
        if status not in {"valid", "invalid", "missing"}:
            raise ValueError("Unknown score record status")
        if status == "valid":
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Valid scores must be finite numeric values in [0,1]")
            if original["metric"] in {"exact_match", "overrefusal"} and value not in (0, 1):
                raise ValueError("Exact-match and overrefusal require frozen binary values")
        elif value is not None:
            raise ValueError("Invalid/missing scores must remain null")
        indexed[key] = record
    if set(indexed) != set(data):
        raise ValueError("Summary must preserve all planned records, including missing/invalid")
    complete = all(r["status"] == "valid" for r in records)
    if summary.get("status") != ("complete" if complete else "incomplete"):
        raise ValueError("Summary completion status contradicts records")
    # Derive analysis from records, never trust potentially stale summary groups.
    return indexed


def _quantile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def _statistics(source_rows, *, alpha, draws, rng_seed):
    """Equal source weights; each complete source is one statistical unit."""
    valid = [r for r in source_rows if r["delta"] is not None]
    values = [r["delta"] for r in valid]
    n, total = len(values), len(source_rows)
    unknown = total - n
    mean = sum(values) / n if n else None
    bounds = [(sum(values) - unknown) / total, (sum(values) + unknown) / total] if total else None
    result = {"n_sources_planned": total, "n_sources_valid": n, "n_sources_unknown": unknown,
              "mean_delta_complete_sources": mean,
              "planned_source_mean_bounds_from_unknown": bounds,
              "n_sources_changed": sum(value != 0 for value in values),
              "bootstrap_interval": None, "bootstrap_unavailable_reason": None,
              "bounded_sensitivity_interval": None,
              "interval_scope": "Conditional on observed complete sources and these trained runs; not uncertainty across training seeds or judge errors."}
    if not n:
        result["bootstrap_unavailable_reason"] = "no_complete_source_pairs"
        return result
    # Hoeffding for independent bounded source differences in [-1,1]. This is a
    # sensitivity analysis: source independence and transportability are not verified.
    radius = math.sqrt(2 * math.log(2 / alpha) / n)
    result["bounded_sensitivity_interval"] = [max(-1.0, mean - radius), min(1.0, mean + radius)]
    result["bounded_sensitivity_assumptions"] = "Independent source differences in [-1,1]; fixed source/stratum composition; non-informative missingness required to generalize complete cases. Not a verified population CI."
    if n < 2 or len(set(values)) == 1:
        result["bootstrap_unavailable_reason"] = "degenerate_empirical_distribution_not_zero_population_uncertainty"
        return result
    strata = defaultdict(list)
    for row in valid:
        strata[row["family"]].append(row["delta"])
    rng = random.Random(rng_seed)
    sampled = []
    for _ in range(draws):
        sampled.append(sum(sum(rng.choice(v) for _ in range(len(v))) for _, v in sorted(strata.items())) / n)
    sampled.sort()
    low, high = _quantile(sampled, alpha / 2), _quantile(sampled, 1 - alpha / 2)
    if low == high:
        result["bootstrap_unavailable_reason"] = "degenerate_stratified_bootstrap_not_zero_population_uncertainty"
    else:
        result["bootstrap_interval"] = [low, high]
    result["bootstrap_method"] = "Percentile resampling of source clusters within fixed family strata; equal source weights."
    result["sparse_change_warning"] = sum(value != 0 for value in values) < 5
    return result


def analyze_paired(dataset, runs, *, expected_seeds=(0, 1, 2), step=PRIMARY_STEP,
                   bootstrap_draws=2000, random_seed=0, alpha=.05, allow_test=False):
    """Analyze envelopes {arm, seed, step, run_contract_sha256, summary}.

    Absent seeds/arms are explicitly incomplete; they never produce a cross-seed
    estimate. A source is complete only if all its variants are valid in both
    arms. Across seeds, use only sources complete in every prespecified seed.
    """
    if type(step) is not int or step < 0:
        raise ValueError("Step must be a nonnegative optimizer step")
    if (not expected_seeds or any(type(s) is not int or s < 0 for s in expected_seeds)
            or len(set(expected_seeds)) != len(expected_seeds)):
        raise ValueError("Expected training seeds must be explicit unique nonnegative integers")
    if type(bootstrap_draws) is not int or bootstrap_draws < 100 or type(random_seed) is not int:
        raise ValueError("Bootstrap requires >=100 draws and an integer analysis seed")
    if type(alpha) not in (int, float) or not 0 < alpha < 1 or type(allow_test) is not bool:
        raise ValueError("Invalid alpha or allow_test")
    data = validate_dataset(dataset)
    if not data or any(r["split"] not in {"dev", "test"} for r in dataset):
        raise ValueError("Analysis requires nonempty dev/test evaluation records")
    if not allow_test and any(r["split"] == "test" for r in dataset):
        raise ValueError("Test analysis requires explicit allow_test after external release authorization")
    dataset_sha = digest(sorted(dataset, key=lambda r: r["id"]))
    grouped = defaultdict(list)
    source_identity = {}
    for row in dataset:
        group = _group(row)
        identity = group + (row["family"],)
        source_key = group + (row["source_id"],)
        if source_key in source_identity and source_identity[source_key] != identity:
            raise ValueError("One source within a metric/benchmark/split cannot cross family strata")
        source_identity[source_key] = identity
        grouped[group].append(row)
    by_run, provenance, protocols, recipes = {}, [], set(), set()
    for envelope in runs:
        if (envelope.get("arm") not in ARMS or type(envelope.get("seed")) is not int
                or envelope["seed"] not in expected_seeds or type(envelope.get("step")) is not int
                or envelope["step"] != step):
            raise ValueError("Run arm, training seed or optimizer step differs from plan")
        key = envelope["seed"], envelope["arm"]
        if key in by_run:
            raise ValueError("Duplicate run arm/seed")
        if not isinstance(envelope.get("run_contract_sha256"), str) or not SHA.fullmatch(envelope["run_contract_sha256"]):
            raise ValueError("Run envelope requires a SHA256 contract identity")
        recipe = envelope.get("matched_recipe_sha256")
        if recipe is not None and (not isinstance(recipe, str) or not SHA.fullmatch(recipe)):
            raise ValueError("Matched recipe identity must be SHA256")
        recipes.add(recipe)
        summary = envelope["summary"]
        by_run[key] = _index_summary(summary, data, dataset_sha)
        protocols.add(summary["protocol_sha256"])
        provenance.append({"arm": key[1], "training_seed": key[0], "step": step,
                           "checkpoint_id": summary["checkpoint_id"],
                           "run_contract_sha256": envelope["run_contract_sha256"], "matched_recipe_sha256": recipe,
                           "summary_canonical_sha256": digest(summary),
                           "envelope_canonical_sha256": digest(envelope)})
    if len(protocols) > 1:
        raise ValueError("Run generation/judge/scoring protocols differ")
    if len(recipes) > 1:
        raise ValueError("Matched training recipe hashes differ or are only partially supplied")
    missing = [{"training_seed": seed, "arm": arm} for seed in expected_seeds for arm in ARMS
               if (seed, arm) not in by_run]
    groups = []
    for key, cohort in sorted(grouped.items()):
        metric, benchmark, split = key
        sources = defaultdict(list)
        for row in cohort:
            sources[row["source_id"]].append(row)
        per_seed = []
        for seed in expected_seeds:
            left, right = by_run.get((seed, "F0")), by_run.get((seed, "R0"))
            pairs = []
            for row in cohort:
                a = left[row["id"]] if left is not None else None
                b = right[row["id"]] if right is not None else None
                valid = a is not None and b is not None and a["status"] == b["status"] == "valid"
                pairs.append({"id": row["id"], "source_id": row["source_id"],
                    "F0_status": a["status"] if a else "missing_run", "R0_status": b["status"] if b else "missing_run",
                    "F0": a["value"] if a else None, "R0": b["value"] if b else None,
                    "delta": b["value"] - a["value"] if valid else None})
            pair_index = {r["id"]: r for r in pairs}
            source_rows = []
            for source, variants in sorted(sources.items()):
                values = [pair_index[r["id"]]["delta"] for r in variants]
                complete = all(v is not None for v in values)
                source_rows.append({"source_id": source, "family": variants[0]["family"],
                    "n_variants_planned": len(values), "n_variants_valid": sum(v is not None for v in values),
                    "delta": sum(values) / len(values) if complete else None})
            stats = _statistics(source_rows, alpha=alpha, draws=bootstrap_draws,
                                rng_seed=int(digest([random_seed, list(key), seed]), 16))
            mean = stats["mean_delta_complete_sources"]
            stats["descriptive_direction"] = ("UNKNOWN" if mean is None else "equal_on_observed_sources" if mean == 0
                else "R0_higher" if mean > 0 else "R0_lower")
            counts = {arm: dict(Counter(r[arm + "_status"] for r in pairs)) for arm in ARMS}
            per_seed.append({"training_seed": seed, "both_runs_present": left is not None and right is not None,
                "n_records_planned": len(cohort), "n_record_pairs_valid": sum(r["delta"] is not None for r in pairs),
                "n_record_pairs_unknown": sum(r["delta"] is None for r in pairs),
                "arm_status_counts": counts, "source_statistics": stats, "sources": source_rows, "records": pairs})
        across = None
        if not missing:
            aligned = {s["training_seed"]: {r["source_id"]: r for r in s["sources"]} for s in per_seed}
            rows = []
            for source, variants in sorted(sources.items()):
                values = [aligned[seed][source]["delta"] for seed in expected_seeds]
                rows.append({"source_id": source, "family": variants[0]["family"],
                             "delta": sum(values) / len(values) if all(v is not None for v in values) else None})
            across = _statistics(rows, alpha=alpha, draws=bootstrap_draws,
                                 rng_seed=int(digest([random_seed, list(key), "across"]), 16))
            across.update(n_training_seeds=len(expected_seeds),
                estimator="Average paired delta over all prespecified seeds within each complete source, then equal-weight sources.",
                independent_unit="source, not source multiplied by seed", sources=rows)
        groups.append({"metric": metric, "benchmark": benchmark, "split": split,
            "direction_of_metric": METRICS[metric], "delta_definition": "R0_minus_F0",
            "analysis_role": "primary_endpoint_candidate" if step == PRIMARY_STEP and metric == "safety" and benchmark == "strongreject" and split == "test" else "secondary_descriptive",
            "per_seed": per_seed, "across_prespecified_seeds": across})
    complete_pairs = not missing and all(s["n_record_pairs_unknown"] == 0 for g in groups for s in g["per_seed"])
    return {"schema_version": VERSION,
        "status": "complete" if complete_pairs else "incomplete_runs" if missing else "complete_runs_with_unknown",
        "dataset_canonical_sha256": dataset_sha, "protocol_sha256": next(iter(protocols), None),
        "analysis_implementation_sha256": _sha(__file__),
        "matched_recipe_sha256": next(iter(recipes), None), "expected_training_seeds": list(expected_seeds),
        "optimizer_step": step, "primary_optimizer_step": PRIMARY_STEP, "missing_runs": missing,
        "run_provenance": provenance, "groups": groups,
        "analysis_settings": {"alpha": alpha, "bootstrap_draws": bootstrap_draws, "analysis_random_seed": random_seed},
        "scientific_conclusion": None, "significance_claim": None, "judge_validity_established": False,
        "limitations": ["Run-envelope seed/step/arm metadata must be authenticated by the external controller; labels are not inferred from filenames.",
            "Intervals condition on these trained runs and observed complete sources; three seeds do not characterize population training variability.",
            "Incomplete sources remain unknown; complete-case estimates may be biased by missingness.",
            "Source independence and semantic deduplication are assumptions, not proven by unique source IDs.",
            "Absent matched_recipe_sha256 leaves shared training-data/cue/configuration identity unverified; full arm-specific contract hashes are retained, not equated.",
            "The legacy summary identifies judge profiles globally, not their per-metric assignment; raw prediction/judgment bindings require external receipt verification.",
            "No pooled safety/utility metric, no automatic significance decision, no correction for multiple exploratory comparisons.",
            "allow_test is an explicit analysis opt-in, not an access-control or sealed-test release mechanism."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON: dataset, runs[{arm,seed,step,run_contract_sha256,summary}], expected_seeds, step")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args(argv)
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    base = plan_path.parent
    dataset_path = (base / plan["dataset"]).resolve()
    runs, files = [], {str(dataset_path): _sha(dataset_path), str(plan_path): _sha(plan_path)}
    for entry in plan["runs"]:
        summary_path = (base / entry["summary"]).resolve()
        files[str(summary_path)] = _sha(summary_path)
        runs.append({**entry, "summary": json.loads(summary_path.read_text())})
    report = analyze_paired(read_jsonl(dataset_path), runs,
        expected_seeds=plan["expected_seeds"], step=plan["step"],
        bootstrap_draws=plan.get("bootstrap_draws", 2000), random_seed=plan.get("analysis_random_seed", 0),
        alpha=plan.get("alpha", .05), allow_test=args.allow_test)
    report["input_file_sha256"] = files
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "scientific_conclusion": None}))


if __name__ == "__main__":
    main()
