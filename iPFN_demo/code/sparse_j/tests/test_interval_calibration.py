import numpy as np
import pytest

from pfn_social.e12_interval_calibration import conformal_factor, interval_scores, apply_factors


def test_finite_sample_rank_uses_96th_of_100_without_interpolation():
    result=conformal_factor(np.arange(1,101)/10)
    assert result['order_statistic']==96
    assert result['raw_quantile']==9.6 and result['factor']==9.6


def test_small_calibration_set_requires_infinity_and_never_shrinks():
    assert np.isinf(conformal_factor(np.ones(8))['factor'])
    assert conformal_factor(np.full(100,.3))['factor']==1.
    assert conformal_factor(np.arange(19))['order_statistic']==19


@pytest.mark.parametrize('scores',[[],[-1.],[np.nan],[np.inf],[[1.,2.]]])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):
        conformal_factor(scores)


def test_score_uses_interval_midpoint_not_point_estimate():
    np.testing.assert_allclose(interval_scores([0.,8.],[-1.,1.],[3.,5.]),[.5,2.5])


def test_expansion_preserves_unit_factor_endpoints_and_contains_original():
    lo=np.array([-2.,1e-14,-5.]);hi=np.array([1.,2e-14,-1.])
    newlo,newhi=apply_factors(lo,hi,[2.,1.,1.5])
    np.testing.assert_array_equal(newlo[1:2],lo[1:2])
    np.testing.assert_array_equal(newhi[1:2],hi[1:2])
    assert np.all(newlo<=lo) and np.all(newhi>=hi)
    np.testing.assert_allclose(newhi-newlo,(hi-lo)*[2.,1.,1.5])


@pytest.mark.parametrize('factors',[[.9,1,1],[np.inf,1,1],[1,1]])
def test_application_rejects_invalid_frozen_factors(factors):
    with pytest.raises(ValueError):
        apply_factors([-1,-1,-1],[1,1,1],factors)


def test_freeze_requires_registered_calibration_mechanisms_and_rejects_test_ids(tmp_path,monkeypatch):
    import json
    import run_e12_calibration as runner
    monkeypatch.setattr(runner,'FAMILY_BASE_IDS',{103:11})
    monkeypatch.setattr(runner,'BUDGETS',(50,))
    cal=tmp_path/'calibration';cal.mkdir()
    for name in ('truth','predictions'):
        (cal/name).mkdir()
    receipts=[]
    for i,mid in enumerate(runner.CALIBRATION_IDS,1):
        context=f'f103_m{mid}_r0_observed50'
        np.savez(cal/'truth'/f'f103_m{mid}.npz',arm_means=np.array([[0,0,i/10,2*i/10]]))
        np.savez(cal/'predictions'/f'{context}__j.npz',ate_lower=-np.ones(3),ate_upper=np.ones(3))
        receipts.append(dict(context_id=context,method='j',failure=None,interval_failure=None))
    runner.dump(cal/'execution.json',dict(receipts=receipts))
    artifact=runner.freeze_calibration(cal,tmp_path,'registered-hash')
    frozen=json.loads(artifact.read_text())
    assert [r['factor'] for r in frozen['factors']]==[9.6,9.6,19.2]
    first_hash=runner.digest(artifact)
    assert runner.freeze_calibration(cal,tmp_path,'registered-hash')==artifact
    assert runner.digest(artifact)==first_hash
    receipts[0]['context_id']='f103_m3000_r0_observed50'
    runner.dump(cal/'execution.json',dict(receipts=receipts))
    with pytest.raises(ValueError,match='complete registered'):
        runner.freeze_calibration(cal,tmp_path,'registered-hash')
