"""Four-cell teacher-update x feedback analysis, with explicit evidence boundaries.

This reports separate descriptive contrasts, never a combined safety/utility score
or a statistical/causal conclusion from a single training seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .evaluation import METRICS, compare_summaries, digest, write_json


ARMS = ("F0", "R0", "F1", "R1")
ALL_KEYS = ("base",) + ARMS
_CONFIG_VARIATIONS = {"teacher_mode", "feedback_mode", "run_name", "output_dir", "train_file", "deepspeed"}
_REQUIRED_CONFIG = {
    "model_name_or_path", "model_revision", "training_seed", "trajectory_mode", "kl_direction",
    "max_steps", "learning_rate", "max_grad_norm", "max_prompt_length", "max_completion_length",
    "max_length", "generation_temperature", "loss_chunk_size", "per_device_train_batch_size",
    "gradient_accumulation_steps", "expected_world_size",
}
_REQUIRED_CONTRACT = {
    "config", "train_sha256", "upstream", "world_size", "effective_global_batch",
    "profile", "template", "deepspeed_sha256",
}


def _model_identity(contract):
    config = contract.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model_name_or_path"), str):
        raise ValueError("Contract lacks model configuration")
    revision = config.get("model_revision")
    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
        return {"model_name_or_path": config["model_name_or_path"], "model_revision": revision}
    identity = contract.get("model_identity")
    if isinstance(identity, dict) and isinstance(identity.get("checkpoint_id"), str) and identity["checkpoint_id"].startswith("sha256:"):
        return {"checkpoint_id": identity["checkpoint_id"]}
    raise ValueError("Contract requires an immutable Hub revision or local model content identity")


def validate_contracts(contracts):
    """Verify the declared factorial design; does not prove runtime execution."""
    if set(contracts) != set(ALL_KEYS):
        raise ValueError("Supply all five contracts: base, F0, R0, F1, R1")
    model = _model_identity(contracts["base"])
    normalized = {}
    for arm in ARMS:
        contract = contracts[arm]
        if not isinstance(contract, dict) or not _REQUIRED_CONTRACT <= contract.keys():
            raise ValueError(f"{arm}: incomplete run contract")
        config = contract["config"]
        if not isinstance(config, dict) or not _REQUIRED_CONFIG <= config.keys():
            raise ValueError(f"{arm}: incomplete training configuration")
        if _model_identity(contract) != model:
            raise ValueError(f"{arm}: initial model identity differs from base")
        if config.get("teacher_mode") != ("frozen" if arm.startswith("F") else "recursive"):
            raise ValueError(f"{arm}: teacher_mode contradicts factorial arm")
        if not isinstance(config.get("feedback_mode"), str) or not config["feedback_mode"]:
            raise ValueError(f"{arm}: missing feedback_mode")
        for field in ("train_sha256", "deepspeed_sha256"):
            if not isinstance(contract[field], str) or not re.fullmatch(r"[0-9a-f]{64}", contract[field]):
                raise ValueError(f"{arm}: {field} must identify content")
        shared = dict(contract)
        shared["config"] = {key: value for key, value in config.items() if key not in _CONFIG_VARIATIONS}
        # Optional execution identifiers may differ without changing the design.
        for field in ("run_id", "attempt_id", "output_dir", "created_at"):
            shared.pop(field, None)
        normalized[arm] = shared
    neutral = contracts["F0"]["config"]["feedback_mode"]
    structured = contracts["F1"]["config"]["feedback_mode"]
    if neutral == structured:
        raise ValueError("Feedback factor has no contrast")
    if contracts["R0"]["config"]["feedback_mode"] != neutral or contracts["R1"]["config"]["feedback_mode"] != structured:
        raise ValueError("Feedback settings are not paired across teacher modes")
    reference = normalized["F0"]
    for arm in ARMS[1:]:
        if normalized[arm] != reference:
            config_diff = sorted(k for k in set(reference["config"]) | set(normalized[arm]["config"])
                                 if reference["config"].get(k) != normalized[arm]["config"].get(k)
                                 or (k in reference["config"]) != (k in normalized[arm]["config"]))
            outer_diff = sorted(k for k in set(reference) | set(normalized[arm]) if k != "config"
                                and (reference.get(k) != normalized[arm].get(k)
                                     or (k in reference) != (k in normalized[arm])))
            raise ValueError(f"{arm}: non-factor contract differences: config={config_diff}, contract={outer_diff}")
    return {"status": "declared_design_verified", "shared_model_identity": model,
            "shared_training_seed": reference["config"]["training_seed"],
            "shared_design_sha256": digest(reference),
            "feedback_levels": {"0": neutral, "1": structured},
            "runtime_execution_verified": False}


def _groups(summary):
    return {(row["metric"], row["family"], row["split"]): row for row in summary["groups"]}


def _summary_binding(summaries, contracts):
    """Optional linkage supplied by a future runner; absent hashes stay unknown."""
    states = {}
    for key in ALL_KEYS:
        value = summaries[key].get("run_contract_sha256")
        if value is None:
            states[key] = "unverified"
        elif value != digest(contracts[key]):
            raise ValueError(f"{key}: summary is bound to a different run contract")
        else:
            states[key] = "verified"
    return states


def analyze_matrix(summaries, contracts=None):
    if set(summaries) != set(ALL_KEYS):
        raise ValueError("Supply all five summaries: base, F0, R0, F1, R1")
    baseline = summaries["base"]
    # compare_summaries checks schema, dataset, protocol and each group's valid IDs.
    comparisons = {arm: compare_summaries(baseline, summaries[arm]) for arm in ARMS}
    comparison_groups = {arm: _groups(comparisons[arm]) for arm in ARMS}
    score_groups = {arm: _groups(summary) for arm, summary in summaries.items()}
    verified = validate_contracts(contracts) if contracts is not None else {
        "status": "not_supplied", "runtime_execution_verified": False,
    }
    bindings = _summary_binding(summaries, contracts) if contracts is not None else {
        key: "unverified" for key in ALL_KEYS
    }
    step_values = {arm: summaries[arm].get("checkpoint_step") for arm in ARMS}
    if any(value is not None for value in step_values.values()):
        if any(value is None for value in step_values.values()) or len(set(step_values.values())) != 1:
            raise ValueError("Training arms have different or partially specified evaluated checkpoint steps")
    groups = []
    for key, base_group in score_groups["base"].items():
        rows = {arm: comparison_groups[arm][key] for arm in ARMS}
        same_valid = all(row["same_valid_ids"] for row in rows.values())
        arm_values = {arm: score_groups[arm][key]["value"] for arm in ARMS}
        interaction = None
        if same_valid:
            interaction = (arm_values["R1"] - arm_values["F1"]) - (arm_values["R0"] - arm_values["F0"])
        groups.append({
            "metric": key[0], "family": key[1], "split": key[2], "direction": METRICS[key[0]],
            "baseline": base_group["value"], "arm_values": arm_values,
            "baseline_deltas": {arm: rows[arm]["delta"] for arm in ARMS},
            "baseline_changes": {arm: rows[arm]["descriptive_change"] for arm in ARMS},
            "interaction": interaction, "same_valid_ids_all_five": same_valid,
            "interaction_status": "descriptive" if interaction is not None else "UNKNOWN",
            "counts": {name: {field: score_groups[name][key][field]
                              for field in ("n_planned", "n_sources", "n_valid", "n_invalid", "n_missing")}
                       for name in ALL_KEYS},
        })
    checkpoint_ids = {key: summaries[key]["checkpoint_id"] for key in ALL_KEYS}
    duplicate_ids = sorted({value for value in checkpoint_ids.values()
                            if value is not None and list(checkpoint_ids.values()).count(value) > 1})
    return {
        "schema_version": "rsd-factorial-analysis-v1", "analysis_kind": "descriptive",
        "dataset_sha256": baseline["dataset_sha256"], "protocol_sha256": baseline["protocol_sha256"],
        "checkpoints": checkpoint_ids, "repeated_checkpoint_identities": duplicate_ids,
        "evaluated_step": next(iter(step_values.values())) if all(v is not None for v in step_values.values()) else None,
        "contrast_definition": "(R1-F1)-(R0-F0)", "contract_validation": verified,
        "summary_contract_binding": bindings, "groups": groups,
        "limitations": [
            "Single-seed contrasts have no statistical significance or uncertainty estimate.",
            "A verified declared design does not verify runtime execution or establish a causal effect.",
            "Interaction is a difference of recursion effects, not an absolute safety benefit.",
            "Missing or unequal valid ID sets produce UNKNOWN, never a measured zero.",
            "Safety, utility and over-refusal are separate outcomes; no combined transfer score is computed.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ALL_KEYS:
        parser.add_argument("--" + key.lower(), required=True, dest=key,
                            help=f"{key} score summary JSON")
        parser.add_argument("--" + key.lower() + "-contract", dest=key + "_contract",
                            help="Optional run contract JSON; provide all five or none")
    parser.add_argument("--output", required=True)
    args = vars(parser.parse_args(argv))
    summaries = {key: json.loads(Path(args[key]).read_text()) for key in ALL_KEYS}
    contract_paths = {key: args[key + "_contract"] for key in ALL_KEYS}
    if any(contract_paths.values()) and not all(contract_paths.values()):
        parser.error("provide all five --*-contract arguments or none")
    contracts = {key: json.loads(Path(path).read_text()) for key, path in contract_paths.items()} if all(contract_paths.values()) else None
    write_json(args["output"], analyze_matrix(summaries, contracts))


if __name__ == "__main__":
    main()
