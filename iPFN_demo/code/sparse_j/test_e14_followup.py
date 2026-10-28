import itertools
import unittest
import numpy as np
from pfn_social.e14_rgg import SCENARIOS,generate_rgg,radius_graph
from pfn_social.e14_followup import REPLICATION,MATCHED,generate_matched,matched_radius_graph
from pfn_social.phase3_hierarchical import structural_outcome
from pfn_social.phase5_large_network import make_repeat

class FollowupTests(unittest.TestCase):
    def test_independent_replication(self):
        a,_,_=generate_rgg(SCENARIOS[2],0); b,_,_=generate_rgg(REPLICATION[0],0)
        self.assertFalse(np.array_equal(a.parent.x,b.parent.x))
        self.assertFalse(np.array_equal(a.parent.coefficients,b.parent.coefficients))

    def test_exact_radius_and_degree(self):
        for cfg in MATCHED:
            m,pos,info=generate_matched(cfg,0)
            self.assertEqual(info['edges'],1800)
            self.assertTrue(np.all(m.parent.adjacency.sum(1)>0))
            np.testing.assert_array_equal(m.parent.adjacency,radius_graph(pos,info['radius']))
        with self.assertRaises(ValueError):matched_radius_graph(np.zeros((200,2)))

    def test_pairing(self):
        a,_,_=generate_matched(MATCHED[0],0); b,_,_=generate_matched(MATCHED[1],0)
        for key in ('x','h','coefficients'):
            np.testing.assert_array_equal(getattr(a.parent,key),getattr(b.parent,key))
        np.testing.assert_array_equal(a.observation_permutation,b.observation_permutation)
        for rep in (0,1):
            x,y=make_repeat(a.parent,rep),make_repeat(b.parent,rep)
            np.testing.assert_array_equal(x['observed_treatment'],y['observed_treatment'])
            ex=x['y_obs'].numpy()-structural_outcome(a.parent,x['observed_treatment'].numpy())
            ey=y['y_obs'].numpy()-structural_outcome(b.parent,y['observed_treatment'].numpy())
            np.testing.assert_allclose(ex,ey,atol=1e-6)

    def test_truth_enumeration(self):
        for cfg in MATCHED:
            m,_,_=generate_matched(cfg,0); m=m.parent
            i=int(np.argmin(m.adjacency.sum(1))); neighbors=np.flatnonzero(m.adjacency[i])
            self.assertLessEqual(len(neighbors),12)
            for own,arm in itertools.product((0,1),repeat=2):
                ys=[]
                for bits in itertools.product((0,1),repeat=len(neighbors)):
                    if int(sum(bits)>len(neighbors)//2)!=arm:continue
                    t=np.zeros(200);t[neighbors]=bits;t[i]=own
                    ys.append(structural_outcome(m,t)[i])
                self.assertAlmostEqual(np.mean(ys),m.truth[i,2*own+arm],places=11)

if __name__=='__main__':unittest.main()
