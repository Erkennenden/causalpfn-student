import numpy as np
import pytest
from scipy.special import ndtri

from audit_e12_evidence import task_summaries
from diagnose_e12_oracle import oracle_posterior


def test_oracle_matches_scalar_conjugate_posterior_and_shared_node_uncertainty():
    # beta~N(0,1), y=beta+N(0,1), observed y=2 -> beta|y~N(1,1/2).
    # Two identical query nodes share beta; averaging them must NOT halve Var.
    query=np.tile(np.array([0.,2.,1.,3.])[None,:,None],(2,1,1))
    result=oracle_posterior(np.ones((1,1)),query,np.array([2.]),
                            np.zeros(1),np.eye(1),1.)
    np.testing.assert_allclose(result['arm_means'],np.tile([0.,2.,1.,3.],(2,1)))
    np.testing.assert_allclose(result['ate_effects'],[1.,2.,3.])
    np.testing.assert_allclose(result['effect_covariance'],.5*np.outer([1.,2.,3.],[1.,2.,3.]))
    np.testing.assert_allclose(result['ate_upper']-result['ate_effects'],
                               ndtri(.975)*np.sqrt(.5)*np.array([1.,2.,3.]))


@pytest.mark.parametrize('noise',[-1.,0.,np.nan])
def test_oracle_rejects_invalid_noise(noise):
    with pytest.raises(ValueError):
        oracle_posterior(np.ones((1,1)),np.ones((2,4,1)),np.ones(1),np.zeros(1),np.eye(1),noise)


def test_task_sd_is_not_pooled_rmse_or_within_task_variance():
    rows=[]
    for mechanism,errors in enumerate(([0.,2.],[2.,4.])):
        for error in errors:
            rows.append(dict(family=100,mechanism=mechanism,method='j',m_observed=50,
                             estimand='ATE',effect='direct',valid=True,mse=error**2,bias=error))
    _,summary=task_summaries(rows)
    row=summary[0]
    task_rmse=np.sqrt([2.,10.])
    assert row['mean_task_rmse']==pytest.approx(task_rmse.mean())
    assert row['sd_task_rmse']==pytest.approx(task_rmse.std(ddof=1))
    assert row['pooled_rmse']==pytest.approx(np.sqrt(6))
    assert row['sd_task_bias']==pytest.approx(np.sqrt(2))
    rows[-1]['valid']=False
    _,summary=task_summaries(rows)
    assert summary[0]['tasks_complete']==1 and not summary[0]['complete_support']
    assert summary[0]['pooled_rmse']==pytest.approx(np.sqrt(2))
