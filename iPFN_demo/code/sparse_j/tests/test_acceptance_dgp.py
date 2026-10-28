from dataclasses import replace
import numpy as np
import pytest
from diagnose_e12_oracle import enumerate_node_truth
from pfn_social.e12_configuration import configuration
from pfn_social.phase3_hierarchical import _components
from pfn_social.phase5_large_network import generate_mechanism
from scripts.phase5_strong_controls import design_from_h


@pytest.mark.parametrize('indices',[list(range(7,11)),list(range(11,15))+list(range(19,22))])
def test_shifted_truth_matches_independent_neighbor_assignment_enumeration(indices):
    m=generate_mechanism(replace(configuration(100),n=50),4000)
    beta=m.coefficients.copy();beta[indices]*=2
    changed=replace(m,coefficients=beta,components=_components(m.adjacency,m.h,beta))
    _,query=design_from_h(m.h,m.adjacency,np.zeros(50))
    np.testing.assert_allclose(query[0]@beta,enumerate_node_truth(changed),atol=1e-10)
