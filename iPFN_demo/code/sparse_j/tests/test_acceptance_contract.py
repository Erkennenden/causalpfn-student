import numpy as np
import pytest
from e12_acceptance_contract import support_diagnostics,estimate_for_request,ordinary_plugin_values


def test_direct_support_is_per_unit_state_and_not_global_arm_presence():
    a=np.ones((4,4))-np.eye(4);t=np.array([0,0,1,1])
    result=support_diagnostics(a,t,np.array([0,2]))
    assert result['direct_observation_counts'].shape==(4,2,2)
    assert result['direct_observation_counts'].sum()==2
    assert result['direct_observation_mask'][0,0,1]
    assert result['direct_observation_mask'][2,1,0]
    assert not result['direct_observation_mask'][1,0,1]


def test_not_identified_never_calls_predictor_and_preserves_request():
    def forbidden(**kwargs):raise AssertionError('Predictor must not be called')
    result=estimate_for_request(forbidden,dict(x=np.zeros((2,20))),target_id='upstream-id',
        status='NOT_IDENTIFIED',node_ids=['B','A'],empirical_support_mask=None,population_support_mask=None)
    assert result['mu'] is None and result['target_id']=='upstream-id'
    assert list(result['node_ids'])==['B','A']
    with pytest.raises(ValueError,match='No point'):
        ordinary_plugin_values(result,[(0,0,0)])


def test_axis_mapping_and_support_guard_do_not_relabel_identification():
    mu=np.arange(8).reshape(2,4);emp=np.ones((2,2,2),dtype=bool);emp[:,1,1]=False
    result=estimate_for_request(lambda **kw:dict(arm_means=mu),dict(x=np.zeros((2,20))),
        target_id='external-target',status='POINT',node_ids=[7,3],empirical_support_mask=emp,
        population_support_mask=np.ones_like(emp))
    assert result['mu'][1,1,0]==6 and result['status']=='POINT'
    assert ordinary_plugin_values(result,[(1,1,0)])[0]==6
    with pytest.raises(ValueError,match='extrapolation'):
        ordinary_plugin_values(result,[(1,1,1)])


def test_empty_arm_does_not_change_population_support_or_target():
    emp=np.zeros((2,2,2),dtype=bool);pop=np.ones_like(emp)
    result=estimate_for_request(lambda **kw:dict(arm_means=np.ones((2,4))),dict(x=np.zeros((2,20))),
        target_id='fixed-target',status='POINT',node_ids=[0,1],empirical_support_mask=emp,population_support_mask=pop)
    assert result['population_support_mask'].all() and not result['empirical_support_mask'].any()
    assert result['target_id']=='fixed-target' and result['status']=='POINT'
