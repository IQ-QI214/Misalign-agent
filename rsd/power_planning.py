"""Deterministic CPU planning scenarios, never measured power or research results.

The binary illustration uses one binary pair per independent source and exact
two-sided McNemar rejection. It is not the project's three-seed-averaged,
continuous StrongREJECT primary estimator. Continuous calculations assume a
known hypothetical SD of the already seed-averaged source differences.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

VERSION = "rsd-detectability-scenarios-v1"
SOURCES = {
    "mcnemar": "https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html",
    "mcnemar_implementation": "https://www.statsmodels.org/stable/_modules/statsmodels/stats/contingency_tables.html#mcnemar",
    "normal_mean_planning": "https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm",
}


def _number(value, label, *, low=None, high=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, not a boolean")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{label} is outside its permitted range")
    return float(value)


def _n(value, *, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError("n must be an integer source count")


def _alpha(value):
    value = _number(value, "alpha", low=0, high=1)
    if value in (0, 1):
        raise ValueError("alpha must be strictly between zero and one")
    return value


def binomial_probabilities(n, probability):
    """Stable normalized PMF, expanded away from a mode using only stdlib."""
    _n(n, allow_zero=True)
    p = _number(probability, "probability", low=0, high=1)
    values = [0.0] * (n + 1)
    if p in (0, 1):
        values[n if p == 1 else 0] = 1.0
        return values
    mode = min(n, math.floor((n + 1) * p))
    values[mode] = 1.0
    for k in range(mode, n):
        values[k + 1] = values[k] * (n - k) / (k + 1) * p / (1 - p)
    for k in range(mode, 0, -1):
        values[k - 1] = values[k] * k / (n - k + 1) * (1 - p) / p
    total = math.fsum(values)
    return [value / total for value in values]


@lru_cache(maxsize=4096)
def _rejection_cutoff(discordant, alpha):
    """Reject when B<=c or B>=M-c; -1 means no possible rejection."""
    # Under the null every binomial probability has denominator 2**M. Integer
    # comparison avoids rounding a tail exactly on alpha across the threshold.
    numerator, denominator = alpha.as_integer_ratio()
    cumulative, coefficient = 0, 1
    cutoff = -1
    for k in range(discordant // 2 + 1):
        cumulative += coefficient
        if 2 * cumulative * denominator <= numerator * (1 << discordant):
            cutoff = k
        else:
            break
        coefficient = coefficient * (discordant - k) // (k + 1)
    return cutoff


def mcnemar_exact_power(n, discordance, net_difference, *, alpha=.05):
    """Sum exact rejection probabilities over M~Bin(n,q), B|M~Bin(M,p).

    P(R=1,F=0)=(q+delta)/2; P(R=0,F=1)=(q-delta)/2.
    Agreement probability is 1-q. Only these three categories matter.
    Exact means finite-sample test/power summation, with floating-point arithmetic.
    """
    _n(n)
    q = _number(discordance, "discordance", low=0, high=1)
    delta = _number(net_difference, "net difference", low=-1, high=1)
    alpha = _alpha(alpha)
    if abs(delta) > q:
        raise ValueError("Absolute net difference cannot exceed paired discordance")
    if q == 0:
        return 0.0
    directional_p = (q + delta) / (2 * q)
    terms = []
    for m, probability_m in enumerate(binomial_probabilities(n, q)):
        c = _rejection_cutoff(m, alpha)
        if c < 0 or probability_m == 0:
            continue
        alternative = binomial_probabilities(m, directional_p)
        rejected = math.fsum(alternative[:c + 1]) + math.fsum(alternative[m - c:])
        terms.append(probability_m * rejected)
    return min(1.0, max(0.0, math.fsum(terms)))


def continuous_normal_planning(n, source_difference_sd, *, target_power=.8, alpha=.05, target_shift=.05):
    """Known-SD normal planning approximation for mean source differences."""
    _n(n)
    sd = _number(source_difference_sd, "source difference SD", low=0, high=1)
    power = _number(target_power, "target power", low=0, high=1)
    shift = _number(target_shift, "target shift", low=0, high=1)
    alpha = _alpha(alpha)
    if sd == 0 or not .5 < power < 1 or shift == 0:
        raise ValueError("Positive hypothetical SD/shift and target power in (0.5,1) required")
    normal = NormalDist()
    z_alpha, z_power = normal.inv_cdf(1-alpha/2), normal.inv_cdf(power)
    se = sd / math.sqrt(n)
    approximate_shift = (z_alpha + z_power) * se
    noncentrality = target_shift / se
    # P(Z>z_alpha) + P(Z<-z_alpha) for a normal mean shifted by delta/SE.
    normal_power_at_target = normal.cdf(noncentrality - z_alpha) + normal.cdf(-noncentrality - z_alpha)
    return {"n_independent_sources": n, "hypothetical_sd_of_seed_averaged_source_differences": sd,
            "target_power": power, "alpha": alpha,
            "normal_approx_detectable_shift": approximate_shift,
            "normal_approx_sampling_half_width": z_alpha * se,
            "illustrative_target_shift": shift,
            "normal_model_power_at_target_shift": normal_power_at_target,
            "normal_approx_required_sources_at_target_shift": math.ceil(((z_alpha + z_power) * sd / shift)**2),
            "empirical_variance_measured": False,
            "scope": "known hypothetical source-level SD; normal model; not calibrated power of the actual bootstrap procedure"}


def measurement_error_envelope(observed_interval, mae_f0=None, mae_r0=None, *, applicable_to_target=False):
    """Triangle-inequality sensitivity, not an automatically valid confidence CI.

    Each arm's MAE upper bound must cover its actual target response population.
    Repeated judge agreement or errors on constructed controls do not establish it.
    """
    if (not isinstance(observed_interval, (list, tuple)) or len(observed_interval) != 2
            or type(applicable_to_target) is not bool):
        raise ValueError("Explicit two-endpoint interval and applicability boolean required")
    low, high = [_number(x, "contrast interval endpoint", low=-1, high=1) for x in observed_interval]
    if low > high:
        raise ValueError("Observed interval endpoints are reversed")
    errors = [None if x is None else _number(x, "per-arm MAE upper bound", low=0, high=1)
              for x in (mae_f0, mae_r0)]
    result = {"observed_interval": [low, high], "per_arm_mae_upper_bounds": {"F0": errors[0], "R0": errors[1]},
              "applicable_to_target_population_assumed": applicable_to_target,
              "combined_error_bound": None, "sensitivity_envelope": None,
              "conditional_direction": "UNKNOWN", "scope": "sensitivity envelope, not a calibrated confidence interval",
              "scientific_conclusion": None}
    if not applicable_to_target or any(x is None for x in errors):
        result["status"] = "UNKNOWN_MISSING_OR_INAPPLICABLE_ERROR_BOUND"
        return result
    bound = math.fsum(errors)
    adjusted = [max(-1.0, low - bound), min(1.0, high + bound)]
    direction = "positive_only" if adjusted[0] > 0 else "negative_only" if adjusted[1] < 0 else "includes_zero"
    result.update(status="CONDITIONAL_ILLUSTRATION_ONLY", combined_error_bound=bound,
                  sensitivity_envelope=adjusted, conditional_direction=direction)
    return result


def build_scenarios(plan_path):
    """Read declared counts only; no test answers, labels, models or run results."""
    plan_path = Path(plan_path)
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    if plan.get("schema_version") != "rsd-exploration-v2-plan" or plan.get("training_enabled") is not False:
        raise ValueError("Expected the disabled exploration-v2 planning document")
    safety_n, binary_n = (plan["datasets"]["safety_test"][k] for k in ("strongreject", "xstest_safe"))
    _n(safety_n); _n(binary_n)
    alpha = _alpha(plan["primary_estimand"]["alpha"])
    binary = []
    for q in (.05, .10, .20, .40):
        for delta in (.05, .10):
            feasible = delta <= q
            binary.append({"n_source_pairs": binary_n, "hypothetical_discordance": q,
                           "hypothetical_net_difference": delta, "alpha": alpha,
                           "status": "SCENARIO" if feasible else "INFEASIBLE_DELTA_EXCEEDS_DISCORDANCE",
                           "exact_test_rejection_probability": mcnemar_exact_power(binary_n, q, delta, alpha=alpha) if feasible else None})
    historical = [{"n_source_pairs": n, "hypothetical_discordance": .2, "hypothetical_net_difference": delta,
                   "exact_test_rejection_probability": mcnemar_exact_power(n, .2, delta, alpha=.05)}
                  for n, delta in ((205, .05), (205, .1), (658, .05))]
    return {"schema_version": VERSION, "status": "CPU_PLANNING_SCENARIOS_NOT_OBSERVED_RESULTS",
            "source_plan_sha256": hashlib.sha256(raw).hexdigest(),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "planned_counts": {"strongreject_sources": safety_n, "xstest_safe_sources": binary_n,
                               "fixed_training_seeds": plan["training_seeds"], "training_seeds_do_not_multiply_source_n": True},
            "actual_primary": plan["primary_estimand"], "formal_mde_selected_by_this_tool": None,
            "scientific_target_effect_selected_by_this_tool": None,
            "binary_illustration_scope": "one binary pair per independent source, one prespecified two-sided test; not the three-seed averaged actual estimator",
            "binary_scenarios": binary, "historical_document_scenario_reproduction": historical,
            "continuous_primary_planning": [continuous_normal_planning(safety_n, sd, alpha=alpha) for sd in (.1, .2, .3, .5)],
            "measurement_error_examples": [measurement_error_envelope([.03, .07], .02, .02, applicable_to_target=True),
                                           measurement_error_envelope([.03, .07], .005, .005, applicable_to_target=True),
                                           measurement_error_envelope([.03, .07], None, None, applicable_to_target=False)],
            "assumptions_not_verified": ["Source independence and representativeness", "Actual paired discordance or source-level SD",
                                         "Validity/transportability of per-arm judge error bounds", "Power of the actual clustered bootstrap and UNKNOWN handling",
                                         "Generalization across training randomness beyond the three fixed seeds"],
            "baseline_variance_boundary": "One untrained-model dev baseline cannot estimate the future F0/R0 paired difference SD or discordance; separate development-pilot evidence is needed before confirmation unsealing.",
            "model_calls": 0, "gpu_calls": 0, "confirmation_answers_or_scores_read": False,
            "research_plan_modified": False, "scientific_result": None, "sources": SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/rsd/exploration_v2.plan.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_scenarios(args.plan)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "planned_counts": result["planned_counts"], "formal_mde": None}))


if __name__ == "__main__":
    main()
