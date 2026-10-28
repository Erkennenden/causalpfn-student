import unittest
import numpy as np
from analyze_e14_followup import bootstrap_difference

class BootstrapTests(unittest.TestCase):
    def test_identical_errors(self):
        r=bootstrap_difference([[[1,2,1,2],[27,3,27,3]]],replicates=200)
        self.assertEqual((r['difference'],r['lower'],r['upper']),(0,0,0))
    def test_strata_sizes(self):
        r=bootstrap_difference([[[4,1,1,1]]*2,[[16,1,1,1]]*3],replicates=200)
        np.testing.assert_allclose([r[k] for k in ('difference','lower','upper')],np.sqrt(11.2)-1)
    def test_missing_support(self):
        r=bootstrap_difference([[[4,1,1,1],[0,0,0,0]]],replicates=1000)
        self.assertTrue(600<r['valid_bootstraps']<850)
        self.assertEqual(r['difference'],1)
        self.assertEqual(bootstrap_difference([[[0,0,0,0]]])['valid_bootstraps'],0)
    def test_coverage_denominator(self):
        r=bootstrap_difference([[[8,10,0,10]]*5],root=False,replicates=100)
        np.testing.assert_allclose([r[k] for k in ('difference','lower','upper')],.8)

if __name__=='__main__':unittest.main()
