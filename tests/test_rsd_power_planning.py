"""Independent finite-sample checks; no model results or empirical variances."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import pytest

from rsd import power_planning as pp


def enumerate_trinomial_power(n, q, delta, alpha=.05):
    """Independent enumeration over positive/negative/agree category counts."""
    positive, negative, agreement = (q+delta)/2, (q-delta)/2, 1-q
    result = 0.0
    for b in range(n+1):
        for c in range(n-b+1):
            m = b+c
            pvalue = min(Fraction(1), Fraction(2*sum(math.comb(m, i) for i in range(min(b,c)+1)), 1 << m))
            if pvalue <= Fraction(str(alpha)):
                result += math.comb(n,b)*math.comb(n-b,c)*positive**b*negative**c*agreement**(n-b-c)
    return result


@pytest.mark.parametrize('n,q,delta,alpha', [(6,1,1,.05),(8,.5,.2,.05),(12,.8,-.4,.05),
                                         (10,.6,.1,.125),(10,1,0,.125),(7,.2,0,.05)])
def test_exact_power_matches_independent_count_enumeration(n,q,delta,alpha):
    assert pp.mcnemar_exact_power(n,q,delta,alpha=alpha) == pytest.approx(enumerate_trinomial_power(n,q,delta,alpha), abs=2e-13)


@pytest.mark.parametrize('n', [6,20,205])
@pytest.mark.parametrize('q', [.01,.2,.8,1])
def test_null_rejection_probability_does_not_exceed_alpha(n,q):
    assert 0 <= pp.mcnemar_exact_power(n,q,0) <= .05+2e-13


def test_zero_discordance_and_small_sample_discreteness():
    assert pp.mcnemar_exact_power(205,0,0) == 0
    assert pp.mcnemar_exact_power(5,1,1) == 0  # 2/2**5 > .05 even if all pairs differ one way.
    assert pp.mcnemar_exact_power(6,1,1) == 1
    assert pp.mcnemar_exact_power(205,.2,.05) == pytest.approx(pp.mcnemar_exact_power(205,.2,-.05))


@pytest.mark.parametrize('n,delta,expected', [(205,.05,.30661392640854573),(205,.1,.8844051285352766),(658,.05,.8006133243696928)])
def test_reproduces_existing_document_scenarios(n,delta,expected):
    assert pp.mcnemar_exact_power(n,.2,delta) == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize('n,q,delta,alpha', [(True,.2,.05,.05),(0,.2,.05,.05),(10,.2,.3,.05),
                                         (10,.2,-.3,.05),(10,-.1,0,.05),(10,1.1,0,.05),
                                         (10,.2,float('nan'),.05),(10,.2,.1,0),(10,.2,.1,1),
                                         (10,.2,.1,True),(10,True,.1,.05)])
def test_invalid_or_impossible_probability_scenarios_are_rejected(n,q,delta,alpha):
    with pytest.raises(ValueError):
        pp.mcnemar_exact_power(n,q,delta,alpha=alpha)


def test_centered_binomial_distribution_is_stable_and_has_known_mean():
    for n,p in [(1000,.001),(1000,.999),(1000,.5),(0,.5),(10,0),(10,1)]:
        values = pp.binomial_probabilities(n,p)
        assert math.fsum(values) == pytest.approx(1, abs=2e-14)
        assert math.fsum(i*v for i,v in enumerate(values)) == pytest.approx(n*p, abs=2e-10)
        assert all(0 <= v <= 1 for v in values)


def test_continuous_planning_uses_source_sd_and_n_not_training_seed_multiplier():
    result = pp.continuous_normal_planning(248,.3)
    assert result['normal_approx_detectable_shift'] == pytest.approx(.053370251775330495)
    assert result['normal_approx_required_sources_at_target_shift'] == 283
    assert result['normal_model_power_at_target_shift'] == pytest.approx(.7468827744661811)
    assert result['empirical_variance_measured'] is False
    assert pp.continuous_normal_planning(248*4,.3)['normal_approx_detectable_shift'] == pytest.approx(result['normal_approx_detectable_shift']/2)
    with pytest.raises(ValueError):
        pp.continuous_normal_planning(248,0)


def test_measurement_error_can_cross_zero_despite_positive_observed_interval():
    result = pp.measurement_error_envelope([.03,.07],.02,.02,applicable_to_target=True)
    assert result['combined_error_bound'] == pytest.approx(.04)
    assert result['sensitivity_envelope'] == pytest.approx([-.01,.11])
    assert result['conditional_direction'] == 'includes_zero'
    assert result['scientific_conclusion'] is None
    tighter = pp.measurement_error_envelope([.03,.07],.005,.005,applicable_to_target=True)
    assert tighter['sensitivity_envelope'] == pytest.approx([.02,.08])
    assert tighter['conditional_direction'] == 'positive_only'


@pytest.mark.parametrize('left,right,applicable', [(None,None,False),(.02,None,True),(.02,.02,False)])
def test_missing_or_inapplicable_error_is_unknown_not_zero(left,right,applicable):
    result = pp.measurement_error_envelope([.03,.07],left,right,applicable_to_target=applicable)
    assert result['combined_error_bound'] is None and result['sensitivity_envelope'] is None
    assert result['conditional_direction'] == 'UNKNOWN'


def test_error_envelope_validation_and_bounded_contrast_range():
    for interval,left,right,applies in [([.1,-.1],0,0,True),([-.1,.1],True,0,True),
                                        ([-.1,.1],.1,float('inf'),True),([-.1,.1],.1,.1,1)]:
        with pytest.raises(ValueError):
            pp.measurement_error_envelope(interval,left,right,applicable_to_target=applies)
    assert pp.measurement_error_envelope([-.9,.9],1,1,applicable_to_target=True)['sensitivity_envelope']==[-1,1]


def test_report_reads_only_disabled_plan_and_keeps_scientific_targets_unselected(tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'configs/rsd/exploration_v2.plan.json'
    before=path.read_bytes()
    report=pp.build_scenarios(path)
    assert path.read_bytes()==before
    assert report['source_plan_sha256']==hashlib.sha256(before).hexdigest()
    assert report['planned_counts']['strongreject_sources']==248
    assert report['planned_counts']['xstest_safe_sources']==205
    assert report['formal_mde_selected_by_this_tool'] is None
    assert report['scientific_target_effect_selected_by_this_tool'] is None
    assert report['scientific_result'] is None and report['model_calls']==0
    assert report['actual_primary']['delta_mde'] is None
    assert report['confirmation_answers_or_scores_read'] is False
    assert sum(x['status'].startswith('INFEASIBLE') for x in report['binary_scenarios'])==1
    changed=json.loads(before); changed['training_enabled']=True
    invalid=tmp_path/'enabled.json';invalid.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        pp.build_scenarios(invalid)
