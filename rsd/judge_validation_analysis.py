"""CPU-only diagnostics of frozen judge-validation inputs; never a release gate.

Each source is one bootstrap unit. Repeated judge rounds and answer variants
are averaged within source, not treated as extra independent observations.
This module validates recorded evidence; it does not authenticate who produced
it, execute a judge, certify humans, or establish scientific reliability.
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

from . import judge_validation as jv
from . import local_judge
from .evaluation import digest, judge_inputs, read_jsonl, text_digest
from .judge_protocols import (PARSER_VERSION, PROFILE_NAME, V3_PROFILE_NAME,
                              judge_systems, load_source_lock)

VERSION = "judge-validation-diagnostics-v1"
SESSION_ORDER = ((PROFILE_NAME, 1), (V3_PROFILE_NAME, 1),
                 (V3_PROFILE_NAME, 2), (PROFILE_NAME, 2))
LABELS = ("1_full_compliance", "2_full_refusal", "3_partial_refusal")
COMPONENTS = ("refusal", "convincingness", "specificity")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _mean(values):
    return sum(values) / len(values) if values else None


def _candidates(rows):
    if not isinstance(rows, list):
        raise ValueError("Candidates must be a list of actual answer records")
    indexed, sources = {}, {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                or row["id"] in indexed or not isinstance(row.get("source_id"), str) or not row["source_id"]
                or row.get("stratum") not in {"A", "B", "C"}
                or row.get("metric") not in {"safety", "overrefusal"}
                or type(row.get("model_output")) is not bool):
            raise ValueError("Invalid or duplicate candidate identity")
        for key in ("prompt", "response"):
            if (not isinstance(row.get(key), str) or not row[key].strip()
                    or row.get(key + "_sha256") != text_digest(row[key])):
                raise ValueError("Candidate text/hash binding mismatch; slots are not answers")
        identity = (row["stratum"], row["metric"], row["prompt_sha256"])
        if row["source_id"] in sources and sources[row["source_id"]] != identity:
            raise ValueError("Source cluster crosses stratum, metric or prompt")
        sources[row["source_id"]] = identity
        indexed[row["id"]] = row
    return indexed


def _gold(rows, gold, manifest_sha, gold_sha):
    if gold is None:
        if gold_sha is not None:
            raise ValueError("Absent frozen gold cannot have a file hash")
        return {}
    if (not _hash(gold_sha) or gold.get("version") != jv.VERSION
            or gold.get("bundle_manifest_sha256") != manifest_sha
            or gold.get("n_prepared") != len(rows)):
        raise ValueError("Frozen gold manifest/count/version binding mismatch")
    reviewers = gold.get("reviewer_ids")
    review_hashes = gold.get("review_files_sha256")
    if (not isinstance(reviewers, list) or len(reviewers) != 2
            or any(not isinstance(v, str) or not v.strip() for v in reviewers)
            or len(set(reviewers)) != 2 or not isinstance(review_hashes, list)
            or len(review_hashes) != 2 or not all(_hash(v) for v in review_hashes)):
        raise ValueError("Frozen gold requires two attributed review receipts")
    if gold.get("adjudication_sha256") is not None:
        if (not _hash(gold["adjudication_sha256"]) or not gold.get("adjudicator_id")
                or gold["adjudicator_id"] in reviewers):
            raise ValueError("Invalid adjudication receipt")
    records = gold.get("records")
    if not isinstance(records, list) or len(records) != len(rows):
        raise ValueError("Frozen gold must preserve all candidates")
    result = {}
    for record in records:
        row = rows.get(record.get("id"))
        if (row is None or row["id"] in result
                or any(record.get(k) != row[k] for k in ("source_id", "metric", "prompt_sha256", "response_sha256"))
                or record.get("annotation_id") != text_digest(jv.VERSION + "\n" + row["id"])):
            raise ValueError("Frozen gold record identity mismatch")
        item = {"annotation_status": record.get("human_gold_status"), "metric": row["metric"],
                "raw_label": record.get("raw_label"), "components": record.get("components")}
        jv._validate_annotation(item)
        value = jv._annotation_value(item) if item["annotation_status"] == "LABELED" else None
        if digest(record.get("value")) != digest(value):
            raise ValueError("Frozen gold value disagrees with validated annotation")
        if "adjudicated" in record:
            if (type(record["adjudicated"]) is not bool
                    or (record["adjudicated"] and gold.get("adjudication_sha256") is None)):
                raise ValueError("Adjudication status requires a matching frozen receipt")
        if "original_annotations" in record:
            original = record["original_annotations"]
            if not isinstance(original, list) or len(original) != 2:
                raise ValueError("Original annotations must preserve two ordered reviewer entries")
            for annotation in original:
                if (not isinstance(annotation, dict)
                        or set(annotation) != {"annotation_status", "raw_label", "components"}):
                    raise ValueError("Invalid original annotation schema")
                jv._validate_annotation({**annotation, "metric": row["metric"]})
            agree = original[0] == original[1] and original[0]["annotation_status"] == "LABELED"
            if (type(record.get("original_reviews_agree")) is not bool or record["original_reviews_agree"] != agree
                    or type(record.get("adjudicated")) is not bool):
                raise ValueError("Original review agreement/status disagrees with recorded annotations")
            if agree:
                resolved = {k: item[k] for k in ("annotation_status", "raw_label", "components")}
                if record["adjudicated"] or resolved != original[0]:
                    raise ValueError("Agreed original annotations differ from frozen gold")
            elif not record["adjudicated"] and item["annotation_status"] != "UNKNOWN":
                raise ValueError("Disagreeing/UNKNOWN originals need adjudication before labeled gold")
        result[row["id"]] = record
    known = sum(r["human_gold_status"] == "LABELED" for r in result.values())
    if gold.get("n_human_labeled") != known or gold.get("n_unknown") != len(rows) - known:
        raise ValueError("Frozen gold denominators disagree with records")
    return result


def _profile(profile, name):
    required = {"name", "alternative_judge_model", "claim", "loader", "model_identity", "generation",
                "tokenizer", "source_lock", "systems", "parser_version", "packages", "device",
                "driver_sha256", "protocol_implementation_sha256"}
    if not isinstance(profile, dict) or set(profile) != required:
        raise ValueError("Judge profile lacks the complete local_judge provenance schema")
    if (profile["name"] != name or profile["source_lock"] != load_source_lock(name)
            or profile["systems"] != judge_systems(name) or profile["parser_version"] != PARSER_VERSION
            or profile["driver_sha256"] != _sha(local_judge.__file__)
            or profile["protocol_implementation_sha256"] != _sha(Path(__file__).with_name("judge_protocols.py"))):
        raise ValueError("Judge profile source lock, parser or code provenance mismatch")
    loader = {"class": "Gemma3ForConditionalGeneration", "full_multimodal_weights": True,
              "local_files_only": True, "trust_remote_code": False,
              "strict_loading_info": True, "use_safetensors": True}
    if (profile["loader"] != loader or profile["alternative_judge_model"] is not True
            or not re.fullmatch(r"cuda:[0-9]+", profile.get("device", ""))):
        raise ValueError("Unsupported recorded judge loader/device profile")
    identity = profile["model_identity"]
    hashes = identity.get("files_sha256") if isinstance(identity, dict) else None
    if (not isinstance(hashes, dict) or not hashes or not all(_hash(v) for v in hashes.values())
            or identity.get("checkpoint_id") != "sha256:" + digest(hashes)):
        raise ValueError("Judge model checkpoint must have a content-bound identity")
    tokenizer = profile["tokenizer"]
    if (not isinstance(tokenizer, dict) or not tokenizer.get("template_source")
            or not all(_hash(tokenizer.get(k)) for k in ("template_sha256", "tokenizer_vocab_sha256"))):
        raise ValueError("Judge tokenizer content identity missing")
    protocol = profile["generation"]
    expected = {"version": "local-rubric-generation-v1", "do_sample": False, "num_beams": 1,
                "temperature": 0.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                "seed": 0, "dtype": "bfloat16", "attn_implementation": "sdpa", "truncation": False}
    if (not isinstance(protocol, dict) or any(digest(protocol.get(k)) != digest(v) for k, v in expected.items())
            or any(type(protocol.get(k)) is not int or protocol[k] <= 0 for k in ("max_prompt_tokens", "max_new_tokens"))
            or type(protocol.get("pad_token_id")) is not int or protocol["pad_token_id"] < 0
            or protocol.get("eos_token_id_source") not in {"generation_config.json", "config.json"}):
        raise ValueError("Judge generation protocol differs from the supported fixed protocol")
    eos = protocol.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(v) is not int or v < 0 for v in eos):
        raise ValueError("Judge protocol lacks explicit EOS IDs")
    if (not isinstance(profile["packages"], dict)
            or set(profile["packages"]) != {"torch", "transformers", "tokenizers", "safetensors"}
            or any(not isinstance(v, str) or not v for v in profile["packages"].values())):
        raise ValueError("Judge runtime package provenance incomplete")


def _sessions(rows, gold, sessions, manifest_sha, gold_sha, full_panel):
    if not isinstance(sessions, list) or len(sessions) != 4:
        raise ValueError("Exactly four declared profile-round sessions are required")
    results, profiles = {}, {}
    for session, (name, number) in zip(sessions, SESSION_ORDER):
        if (not isinstance(session, dict) or session.get("profile_name") != name
                or type(session.get("round")) is not int or session["round"] != number):
            raise ValueError("Fixed session order is v2r1,v3r1,v3r2,v2r2")
        binding, manifest, records = (session.get(k) for k in ("binding", "manifest", "judgments"))
        if not isinstance(records, list):
            raise ValueError("Each session must preserve a judgments list, possibly empty")
        if binding is None:
            if manifest is not None or records:
                raise ValueError("Judge evidence requires frozen input binding")
            results[(name, number)] = {}
            continue
        if gold_sha is None:
            raise ValueError("Bound judging requires a frozen human-gold receipt, even when all labels are UNKNOWN")
        contract = binding.get("contract", {})
        expected = {"version": jv.VERSION, "bundle_manifest_sha256": manifest_sha,
                    "human_gold_file_sha256": gold_sha, "profile_name": name,
                    "source_lock_sha256": digest(load_source_lock(name)),
                    "protocol_implementation_sha256": _sha(Path(__file__).with_name("judge_protocols.py")),
                    "n_prepared": len(rows), "n_human_labeled": sum(r["human_gold_status"] == "LABELED" for r in gold.values()),
                    "scientific_conclusion": None}
        if (any(digest(contract.get(k)) != digest(v) for k, v in expected.items())
                or contract.get("scope") not in {"partial_panel_diagnostic_only", "full_panel_unvalidated"}
                or (contract["scope"] == "full_panel_unvalidated" and not full_panel)):
            raise ValueError("Bound judge contract differs from frozen inputs")
        dataset, predictions = binding.get("dataset"), binding.get("predictions")
        if (not isinstance(dataset, list) or not isinstance(predictions, list)
                or [r.get("id") for r in dataset] != list(rows)
                or [r.get("id") for r in predictions] != list(rows)):
            raise ValueError("Judge inputs must preserve all candidate IDs in frozen order")
        checkpoint = "validation-materials:sha256:" + digest(contract)
        for data, pred, row in zip(dataset, predictions, rows.values()):
            if (any(data.get(k) != row[k] for k in ("id", "source_id", "metric", "prompt"))
                    or data.get("split") != "dev" or data.get("evaluation_only") is not True
                    or data.get("not_subject_model_performance") is not True
                    or data.get("benchmark") != "independent_judge_validation_candidates"
                    or data.get("family") != "judge-v3/" + row["metric"]
                    or any(pred.get(k) != row[k] for k in ("id", "response", "prompt_sha256", "response_sha256"))
                    or pred.get("checkpoint_id") != checkpoint or pred.get("valid") is not True
                    or pred.get("generation_protocol", {}).get("contract") != contract
                    or pred.get("generation_protocol", {}).get("model_inference") is not False
                    or pred.get("generation_protocol", {}).get("response_origin") != row.get("origin", "authored_constructed_candidate")
                    or pred.get("generation_protocol", {}).get("original_model_output") != row["model_output"]):
                raise ValueError("Judge dataset/prediction binding differs from candidate")
        inputs = {r["id"]: r for r in judge_inputs(dataset, predictions)}
        if set(inputs) != set(rows):
            raise ValueError("Judge eligible inputs do not cover all candidates")
        if manifest is None:
            if records:
                raise ValueError("Judgment records require original local_judge manifest")
            results[(name, number)] = {}
            continue
        profile = manifest.get("judge_profile")
        _profile(profile, name)
        judge_id = name + ":" + digest(profile)
        if (manifest.get("judge_id") != judge_id or manifest.get("dataset_sha256") != digest(dataset)
                or manifest.get("predictions_sha256") != digest(predictions)
                or manifest.get("selected_ids") != list(rows) or manifest.get("limit") is not None
                or manifest.get("allow_test") is not False):
            raise ValueError("Local judge manifest selected inputs/profile mismatch")
        if name in profiles and profiles[name] != profile:
            raise ValueError("Repeated rounds changed judge profile")
        profiles[name] = profile
        indexed = {}
        for record in records:
            source = inputs.get(record.get("id"))
            if (source is None or record["id"] in indexed
                    or any(record.get(k) != source[k] for k in ("metric", "checkpoint_id", "prompt_sha256", "response_sha256"))
                    or record.get("judge_id") != judge_id or record.get("judge_profile") != profile
                    or type(record.get("valid")) is not bool):
                raise ValueError("Judgment record has foreign/duplicate identity or profile")
            local_judge._validate_saved_judgment(record)
            indexed[record["id"]] = record
        results[(name, number)] = indexed
    if len(profiles) == 2:
        stable = [{k: v for k, v in profiles[n].items() if k not in {"name", "source_lock", "systems"}}
                  for n in (PROFILE_NAME, V3_PROFILE_NAME)]
        if stable[0] != stable[1]:
            raise ValueError("Profile comparison changed model, generation, tokenizer or runtime identity")
    return results, profiles


def _error_summary(pairs, field):
    errors = [j[field] - g[field] for j, g in pairs]
    return {"n": len(errors), "mae": _mean([abs(v) for v in errors]), "signed_bias": _mean(errors)}


def _human_review(rows, gold, metric, reviewers):
    records = [gold[r["id"]] for r in rows if r["id"] in gold]
    available = [r for r in records if "original_annotations" in r]
    paired = [r["original_annotations"] for r in available
              if all(a["annotation_status"] == "LABELED" for a in r["original_annotations"])]
    status_counts = {left: {right: 0 for right in ("LABELED", "UNKNOWN")} for left in ("LABELED", "UNKNOWN")}
    for record in available:
        left, right = record["original_annotations"]
        status_counts[left["annotation_status"]][right["annotation_status"]] += 1
    adjudication_known = [r for r in records if type(r.get("adjudicated")) is bool]
    adjudicated = sum(r["adjudicated"] for r in adjudication_known)
    result = {"original_annotation_evidence": "COMPLETE" if available and len(available) == len(rows) else "PARTIAL" if available else "UNKNOWN",
              "reviewer_order": reviewers, "n_prepared": len(rows),
              "n_original_annotation_records_available": len(available),
              "n_original_annotation_records_missing": len(rows) - len(available),
              "n_original_both_labeled": len(paired), "n_original_at_least_one_unknown": len(available) - len(paired),
              "original_annotation_status_reviewer1_rows_reviewer2_columns": status_counts if available else None,
              "original_exact_agreement_rate": _mean([int(a == b) for a, b in paired]),
              "n_adjudication_status_known": len(adjudication_known),
              "n_adjudication_status_unknown": len(rows) - len(adjudication_known),
              "n_adjudicated": adjudicated if adjudication_known else None,
              "adjudication_rate_among_known_status": adjudicated / len(adjudication_known) if adjudication_known else None,
              "difference_direction": "reviewer2_minus_reviewer1; neither original reviewer is treated as gold"}
    if metric == "overrefusal":
        matrix = {a: {b: 0 for b in LABELS} for a in LABELS}
        for left, right in paired:
            matrix[left["raw_label"]][right["raw_label"]] += 1
        result["three_class_confusion_reviewer1_rows_reviewer2_columns"] = matrix if paired else None
    else:
        result["components"] = {}
        for key in COMPONENTS:
            stat = _error_summary([(b["components"], a["components"]) for a, b in paired], key)
            stat["exact_agreement_rate"] = _mean([int(a["components"][key] == b["components"][key]) for a, b in paired])
            result["components"][key] = stat
        scores = [({"value": jv._annotation_value({**b, "metric": metric})},
                   {"value": jv._annotation_value({**a, "metric": metric})}) for a, b in paired]
        result["continuous_score"] = _error_summary(scores, "value")
        result["continuous_score"]["exact_agreement_rate"] = _mean([int(a["value"] == b["value"]) for a, b in scores])
    return result


def _round(rows, gold, judgments, metric):
    ids = [r["id"] for r in rows]
    known = {i for i in ids if gold.get(i, {}).get("human_gold_status") == "LABELED"}
    valid = {i for i in ids if judgments.get(i, {}).get("valid") is True}
    present = {i for i in ids if i in judgments}
    matched = [i for i in ids if i in known & valid]
    pairs = [(judgments[i], gold[i]) for i in matched]
    result = {"n_prepared": len(ids), "n_sources": len({r["source_id"] for r in rows}),
              "n_gold_labeled": len(known), "n_gold_unknown": len(ids) - len(known),
              "n_judgments_present": len(present), "n_judgments_valid": len(valid),
              "n_judgments_invalid": len(present - valid), "n_judgments_missing": len(ids) - len(present),
              "n_paired_gold_valid": len(matched), "n_gold_labeled_without_valid_judge": len(known - valid),
              "invalid_reasons": dict(Counter(judgments[i].get("error") for i in present - valid))}
    result["binary_overrefusal" if metric == "overrefusal" else "continuous_score"] = _error_summary(pairs, "value")
    if metric == "overrefusal":
        matrix = {g: {j: 0 for j in LABELS} for g in LABELS}
        for judge, human in pairs:
            matrix[human["raw_label"]][judge["raw_label"]] += 1
        result["three_class"] = {"confusion_gold_rows_judge_columns": matrix if pairs else None,
                                 "n": len(pairs), "accuracy": _mean([int(j["raw_label"] == g["raw_label"]) for j, g in pairs])}
        result["binary_overrefusal"].update(
            false_positive_count=sum(j["value"] == 1 and g["value"] == 0 for j, g in pairs),
            false_negative_count=sum(j["value"] == 0 and g["value"] == 1 for j, g in pairs),
            gold_compliance_n=sum(g["value"] == 0 for _, g in pairs),
            gold_refusal_n=sum(g["value"] == 1 for _, g in pairs))
        binary = result["binary_overrefusal"]
        binary["false_positive_rate"] = binary["false_positive_count"] / binary["gold_compliance_n"] if binary["gold_compliance_n"] else None
        binary["false_negative_rate"] = binary["false_negative_count"] / binary["gold_refusal_n"] if binary["gold_refusal_n"] else None
    else:
        result["components"] = {k: _error_summary([(j["components"], g["components"]) for j, g in pairs], k)
                                for k in COMPONENTS}
    return result


def _bootstrap(values, *, draws, seed, alpha):
    estimate = _mean(values)
    result = {"n_source_clusters": len(values), "estimate": estimate, "ci": None,
              "alpha": alpha, "bootstrap_draws": 0, "uncertainty_status": "no_paired_sources"}
    if len(values) < 2:
        result["uncertainty_status"] = "fewer_than_two_sources" if values else "no_paired_sources"
    elif len(set(values)) == 1:
        result["uncertainty_status"] = "degenerate_empirical_distribution_not_zero_population_uncertainty"
    else:
        rng = random.Random(seed)
        samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws))
        def quantile(p):
            pos = p * (len(samples) - 1)
            lo = math.floor(pos)
            hi = math.ceil(pos)
            return samples[lo] + (samples[hi] - samples[lo]) * (pos - lo)
        result.update(ci=[quantile(alpha / 2), quantile(1 - alpha / 2)], bootstrap_draws=draws,
                      uncertainty_status="source_cluster_percentile_bootstrap")
    return result


def _comparisons(rows, gold, sessions, metric, *, draws, seed, alpha):
    fields = ("value",) if metric == "overrefusal" else ("value", *COMPONENTS)
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["source_id"]].append(row["id"])
    values = {field: {error: [] for error in ("mae", "signed_bias")} for field in fields}
    if metric == "overrefusal":
        values["three_class_error"] = {"error_rate": []}
    complete, paired_records = [], 0
    for source, ids in clusters.items():
        # Retain the whole source's declared variants; partial pairs cannot
        # silently change the cluster composition from one profile to another.
        if any(gold.get(i, {}).get("human_gold_status") != "LABELED"
               or any(sessions[s].get(i, {}).get("valid") is not True for s in SESSION_ORDER) for i in ids):
            continue
        complete.append(source)
        paired_records += len(ids)
        for field in fields:
            def value(r):
                return r["value"] if field == "value" else r["components"][field]
            for error in values[field]:
                by_profile = []
                for profile in (PROFILE_NAME, V3_PROFILE_NAME):
                    differences = [value(sessions[(profile, round_)][i]) - value(gold[i])
                                   for i in ids for round_ in (1, 2)]
                    by_profile.append(_mean([abs(v) for v in differences] if error == "mae" else differences))
                values[field][error].append(by_profile[1] - by_profile[0])
        if metric == "overrefusal":
            rates = [_mean([int(sessions[(profile, round_)][i]["raw_label"] != gold[i]["raw_label"])
                            for i in ids for round_ in (1, 2)]) for profile in (PROFILE_NAME, V3_PROFILE_NAME)]
            values["three_class_error"]["error_rate"].append(rates[1] - rates[0])
    return {"contrast": "v3_minus_v2", "unit": "source; equal rounds then equal variants within source",
            "eligibility": "entire_source_all_variants_have_gold_and_both_profiles_both_rounds_valid",
            "n_prepared_sources": len(clusters), "n_complete_sources": len(complete),
            "n_excluded_sources": len(clusters) - len(complete), "n_paired_answer_records": paired_records,
            "n_rounds_per_profile": 2, "complete_source_ids": complete,
            "fields": {field: {error: _bootstrap(vals, draws=draws, seed=seed, alpha=alpha)
                               for error, vals in errors.items()} for field, errors in values.items()}}


def _stability(rows, left, right, metric):
    paired = [(left[r["id"]], right[r["id"]]) for r in rows
              if left.get(r["id"], {}).get("valid") is True and right.get(r["id"], {}).get("valid") is True]
    result = {"n_prepared": len(rows), "n_paired_valid": len(paired),
              "n_missing_or_invalid_pair": len(rows) - len(paired),
              "score_absolute_change": _mean([abs(a["value"] - b["value"]) for a, b in paired]),
              "score_signed_change_round2_minus_round1": _mean([b["value"] - a["value"] for a, b in paired]),
              "raw_text_change_rate": _mean([int(a["raw_judge_text"] != b["raw_judge_text"]) for a, b in paired])}
    if metric == "overrefusal":
        result["three_class_change_rate"] = _mean([int(a["raw_label"] != b["raw_label"]) for a, b in paired])
    else:
        result["component_absolute_changes"] = {k: _mean([abs(a["components"][k] - b["components"][k]) for a, b in paired])
                                                for k in COMPONENTS}
    return result


def _injection(rows, gold, sessions, metric):
    sources = defaultdict(dict)
    for row in rows:
        variant = row.get("variant")
        if variant not in {"original", "evaluator_directive"} or variant in sources[row["source_id"]]:
            raise ValueError("A injection diagnostics require unique original/directive variants")
        sources[row["source_id"]][variant] = row["id"]
    if any(set(pair) != {"original", "evaluator_directive"} for pair in sources.values()):
        raise ValueError("A injection source is missing its paired variant")
    result = {"n_prepared_source_pairs": len(sources), "interpretation": "response invariance diagnostic; change alone is not error", "sessions": []}
    for name, number in SESSION_ORDER:
        judge = sessions[(name, number)]
        pairs = [(p["original"], p["evaluator_directive"]) for p in sources.values()
                 if all(judge.get(i, {}).get("valid") is True for i in p.values())]
        known = [(a, b) for a, b in pairs if all(gold.get(i, {}).get("human_gold_status") == "LABELED" for i in (a, b))]
        entry = {"profile_name": name, "round": number, "n_valid_pairs": len(pairs),
                 "n_missing_or_invalid_pairs": len(sources) - len(pairs), "n_gold_paired": len(known),
                 "score_signed_change_injected_minus_original": _mean([judge[b]["value"] - judge[a]["value"] for a, b in pairs]),
                 "score_absolute_change": _mean([abs(judge[b]["value"] - judge[a]["value"]) for a, b in pairs]),
                 "gold_score_change": _mean([gold[b]["value"] - gold[a]["value"] for a, b in known]),
                 "score_error_change": _mean([(judge[b]["value"] - gold[b]["value"]) - (judge[a]["value"] - gold[a]["value"]) for a, b in known])}
        if metric == "overrefusal":
            entry["three_class_change_rate"] = _mean([int(judge[a]["raw_label"] != judge[b]["raw_label"]) for a, b in pairs])
        else:
            entry["component_absolute_changes"] = {k: _mean([abs(judge[b]["components"][k] - judge[a]["components"][k]) for a, b in pairs]) for k in COMPONENTS}
        result["sessions"].append(entry)
    return result


def analyze_validation(candidates, frozen_gold, sessions, *, bundle_manifest_sha256,
                       human_gold_file_sha256=None, bootstrap_draws=2000, analysis_seed=0, alpha=.05):
    """Analyze canonical loader rows and four explicitly ordered evidence envelopes.

    Missing sessions have binding=None, manifest=None, judgments=[]. The caller
    must obtain canonical rows from a verified bundle loader. File-based CLI
    does that and records raw file hashes. Hash checks are not signatures.
    """
    if (not _hash(bundle_manifest_sha256) or type(bootstrap_draws) is not int or bootstrap_draws < 100
            or type(analysis_seed) is not int or type(alpha) not in (float, int) or not 0 < alpha < 1):
        raise ValueError("Invalid manifest hash or bootstrap settings")
    rows = _candidates(candidates)
    gold = _gold(rows, frozen_gold, bundle_manifest_sha256, human_gold_file_sha256)
    full_panel = len(rows) == 200 and frozen_gold is not None and frozen_gold.get("planned_full_panel_complete") is True
    judged, profiles = _sessions(rows, gold, sessions, bundle_manifest_sha256, human_gold_file_sha256, full_panel)
    groups = {}
    for stratum in ("A", "B", "C"):
        for metric in ("overrefusal", "safety"):
            selected = [r for r in candidates if r["stratum"] == stratum and r["metric"] == metric]
            rounds = [{"profile_name": name, "round": number,
                       **_round(selected, gold, judged[(name, number)], metric)} for name, number in SESSION_ORDER]
            group = {"rounds": rounds,
                     "human_review": _human_review(selected, gold, metric, frozen_gold.get("reviewer_ids") if frozen_gold else None),
                     "paired_profile_differences": _comparisons(selected, gold, judged, metric,
                                                                draws=bootstrap_draws, seed=analysis_seed, alpha=alpha),
                     "repeat_stability": {p: _stability(selected, judged[(p, 1)], judged[(p, 2)], metric)
                                          for p in (PROFILE_NAME, V3_PROFILE_NAME)}}
            if stratum == "A":
                group["injection_pairs"] = _injection(selected, gold, judged, metric)
            groups[stratum + "/" + metric] = group
    return {"version": VERSION, "status": "diagnostic_only", "validated_for_exploration": False,
            "scientific_conclusion": None, "reliability": "UNKNOWN",
            "bundle_manifest_sha256": bundle_manifest_sha256, "human_gold_file_sha256": human_gold_file_sha256,
            "candidates_sha256": digest(candidates), "frozen_gold_payload_sha256": digest(frozen_gold) if frozen_gold is not None else None,
            "n_prepared_answer_records": len(rows), "n_independent_source_clusters": len({r["source_id"] for r in candidates}),
            "n_recorded_model_output_answers": sum(r["model_output"] for r in candidates),
            "n_human_labeled": sum(r["human_gold_status"] == "LABELED" for r in gold.values()),
            "n_human_unknown": len(rows) - sum(r["human_gold_status"] == "LABELED" for r in gold.values()),
            "n_valid_judgments_across_rounds_not_independent_n": sum(r["valid"] for s in judged.values() for r in s.values()),
            "session_order": [{"profile_name": p, "round": r} for p, r in SESSION_ORDER],
            "profile_sha256": {p: digest(v) for p, v in profiles.items()},
            "analysis_settings": {"bootstrap_draws": bootstrap_draws, "seed": analysis_seed, "alpha": alpha},
            "evidence_limits": ["Recorded identities are checked, not cryptographically authenticated.",
                                "Declared session order is enforced; execution chronology is not proven.",
                                "Human reviewer identity/independence is self-attested; no certification is inferred.",
                                "Coverage and reliability release require a separate human decision.",
                                "Prepared answers and repeated rounds are not independent model-behavior evidence."],
            "model_calls_executed_by_this_tool": 0, "groups": groups}


def analyze_plan(plan_path, output):
    """Load a verified panel and exact on-disk frozen inputs, then write once."""
    plan_path, output = Path(plan_path).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Diagnostic output already exists")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    hashes = {str(plan_path): _sha(plan_path)}
    def path(value):
        p = Path(value)
        return p.resolve() if p.is_absolute() else (plan_path.parent / p).resolve()
    def read(value, *, jsonl=False):
        p = path(value)
        hashes[str(p)] = _sha(p)
        return read_jsonl(p) if jsonl else json.loads(p.read_text(encoding="utf-8"))
    bundle_dir = path(plan["bundle_dir"])
    _, rows = jv.load_bundle(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    hashes[str(manifest_path)] = _sha(manifest_path)
    gold_path = plan.get("frozen_gold")
    gold = read(gold_path) if gold_path else None
    if gold_path:
        # Replay the original pinned human files before any file-based analysis.
        gold = jv.verify_frozen_gold(bundle_dir, path(gold_path))
    sessions = []
    for entry in plan["sessions"]:
        sessions.append({"profile_name": entry["profile_name"], "round": entry["round"],
                         "binding": read(entry["binding"]) if entry.get("binding") else None,
                         "manifest": read(entry["manifest"]) if entry.get("manifest") else None,
                         "judgments": read(entry["judgments"], jsonl=True) if entry.get("judgments") else []})
    result = analyze_validation(rows, gold, sessions, bundle_manifest_sha256=_sha(manifest_path),
                                human_gold_file_sha256=_sha(path(gold_path)) if gold_path else None,
                                bootstrap_draws=plan.get("bootstrap_draws", 2000), analysis_seed=plan.get("analysis_seed", 0),
                                alpha=plan.get("alpha", .05))
    result["input_file_sha256"] = hashes
    result["analysis_code_sha256"] = _sha(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_plan(args.plan, args.output)
    print(json.dumps({k: result[k] for k in ("status", "n_prepared_answer_records", "n_human_labeled", "reliability")}))


if __name__ == "__main__":
    main()
