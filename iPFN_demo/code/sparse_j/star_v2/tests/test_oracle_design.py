"""Stage 5.1 deterministic references; no RNG, simulator or model calls.

Fixed before first execution: atol=1e-12, rtol=0. Prior equality is exact.
Expected design rows and outcomes use independent scalar arithmetic.
"""
import hashlib
import itertools
import math
import unittest

import numpy as np
from star_v2 import oracle_design as oracle

ATOL = 1e-12
RTOL = 0
EVIDENCE = {"prior": [], "factual": [], "query": [], "exposure_enumeration": []}


def star(degree):
    a = np.zeros((degree+1, degree+1), dtype=np.float64)
    a[0,1:] = 1.
    a[1:,0] = 1.
    return a


def hidden(n):
    return np.array([[(i-2)/8, (2*i+1)/16, (3-i)/32]
                     for i in range(n)], dtype=np.float64)


def scalar_row(a,h,t,i,*,own=None,conditional_e=None):
    neighbors = [j for j in range(len(a)) if a[i,j] == 1]
    degree = len(neighbors)
    hb = [sum(float(h[j,k]) for j in neighbors)/degree for k in range(3)]
    e = (sum(float(t[j]) for j in neighbors)/degree
         if conditional_e is None else conditional_e)
    joint = ([sum(float(t[j])*float(h[j,k]) for j in neighbors)/degree
              for k in range(3)] if conditional_e is None
             else [e*value for value in hb])
    ti = float(t[i]) if own is None else float(own)
    row = [1.,float(h[i,0]),float(h[i,1]),float(h[i,2]),hb[0],hb[1],hb[2],
           ti,ti*h[i,0],ti*h[i,1],ti*h[i,2],
           e,e*h[i,0],e*h[i,1],e*h[i,2],
           ti*e,ti*e*h[i,0],ti*e*h[i,1],ti*e*h[i,2],
           joint[0],joint[1],joint[2],ti*joint[0],ti*joint[1],ti*joint[2]]
    return row


def scalar_outcome(a,h,t,beta,i):
    neighbors = [j for j in range(len(a)) if a[i,j] == 1]
    degree = len(neighbors)
    hb = [sum(float(h[j,k]) for j in neighbors)/degree for k in range(3)]
    e = sum(float(t[j]) for j in neighbors)/degree
    joint = [sum(float(t[j])*float(h[j,k]) for j in neighbors)/degree for k in range(3)]
    own = float(t[i])
    baseline = beta[0]+sum(h[i,k]*beta[1+k]+hb[k]*beta[4+k] for k in range(3))
    direct = beta[7]+sum(h[i,k]*beta[8+k] for k in range(3))
    spill = beta[11]+sum(h[i,k]*beta[12+k] for k in range(3))
    interaction = beta[15]+sum(h[i,k]*beta[16+k] for k in range(3))
    return (baseline+own*direct+(spill+own*interaction)*e
            +sum(joint[k]*beta[19+k] for k in range(3))
            +own*sum(joint[k]*beta[22+k] for k in range(3)))


def enumerated_exposure(degree,s):
    values = []
    for assignment in itertools.product((0,1),repeat=degree):
        fraction = sum(assignment)/degree
        if int(fraction > .5) == s:
            values.append(fraction)
    return sum(values)/len(values)


def exact_count_mean(degree,s):
    counts = [k for k in range(degree+1) if int(k/degree > .5) == s]
    numerator = sum(k*math.comb(degree,k) for k in counts)
    denominator = degree*sum(math.comb(degree,k) for k in counts)
    return numerator/denominator


def digest(array):
    return (array.dtype.str,array.shape,hashlib.sha256(array.tobytes()).hexdigest())


class OracleDesignTests(unittest.TestCase):
    def compare(self,group,actual,expected):
        np.testing.assert_allclose(actual,expected,atol=ATOL,rtol=RTOL)
        EVIDENCE[group].append(float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)))))

    def test_prior_each_element_independent_reference(self):
        mean,covariance = oracle.coefficient_prior()
        expected_mean = np.array([.6 if i in (7,11) else 0. for i in range(25)])
        scales = [.5]+[.8/math.sqrt(3)]*3+[.35/math.sqrt(3)]*3
        scales += [.5]+[.7/math.sqrt(3)]*3
        scales += [.5]+[.7/math.sqrt(3)]*3
        scales += [.25]+[.35/math.sqrt(3)]*3
        scales += [.4/math.sqrt(3)]*3+[.2/math.sqrt(3)]*3
        expected_covariance = np.zeros((25,25))
        for i in range(25):
            for j in range(25):
                if i == j:
                    expected_covariance[i,j] = scales[i]*scales[i]
                elif (7 <= i <= 10 and j == i+4) or (7 <= j <= 10 and i == j+4):
                    expected_covariance[i,j] = .35*scales[i]*scales[j]
        np.testing.assert_array_equal(mean,expected_mean)
        np.testing.assert_array_equal(covariance,expected_covariance)
        self.compare("prior",mean,expected_mean)
        self.compare("prior",covariance,expected_covariance)

    def test_prior_correlations_and_all_other_offdiagonals(self):
        _,covariance = oracle.coefficient_prior()
        for i in range(25):
            for j in range(25):
                if i != j and abs(i-j) == 4 and min(i,j) in range(7,11):
                    self.assertAlmostEqual(covariance[i,j]/math.sqrt(
                        covariance[i,i]*covariance[j,j]),.35,places=15)
                elif i != j:
                    self.assertEqual(covariance[i,j],0.)

    def test_factual_every_column_against_scalar_reference(self):
        a,h = star(5),hidden(6)
        t = np.array([1.,0.,1.,0.,1.,1.])
        actual = oracle.factual_design(true_h=h,adjacency=a,treatment=t)
        expected = np.array([scalar_row(a,h,t,i) for i in range(6)])
        self.assertEqual(actual.shape,(6,25))
        for column in range(25):
            self.compare("factual",actual[:,column],expected[:,column])

    def test_factual_artificial_beta_against_expanded_scalar_equation(self):
        a,h = star(7),hidden(8)
        t = np.array([0.,1.,0.,1.,1.,0.,0.,1.])
        beta = np.array([(k-9)/32 for k in range(25)])
        actual = oracle.factual_design(true_h=h,adjacency=a,treatment=t)@beta
        expected = np.array([scalar_outcome(a,h,t,beta,i) for i in range(8)])
        self.compare("factual",actual,expected)

    def test_factual_joint_feature_is_not_mean_exposure_times_hbar(self):
        a,h = star(3),hidden(4)
        t = np.array([0.,1.,0.,0.])
        rows = oracle.factual_design(true_h=h,adjacency=a,treatment=t)
        self.compare("factual",rows[0,19:22],h[1]/3)
        self.assertFalse(np.allclose(rows[0,19:22],rows[0,11]*rows[0,4:7],atol=ATOL,rtol=RTOL))

    def test_query_every_state_and_column_scalar_reference(self):
        a,h = star(4),hidden(5)
        query = oracle.query_design(true_h=h,adjacency=a)
        self.assertEqual(query.shape,(5,2,2,25))
        for i in range(5):
            for t,s in itertools.product((0,1),repeat=2):
                e = enumerated_exposure(int(a[i].sum()),s)
                expected = scalar_row(a,h,np.zeros(5),i,own=t,conditional_e=e)
                self.compare("query",query[i,t,s],expected)
        for column,(t,s) in enumerate(((0,0),(0,1),(1,0),(1,1))):
            np.testing.assert_array_equal(query.reshape(5,4,25)[:,column],query[:,t,s])

    def test_query_rows_match_complete_iid_neighbor_assignment_enumeration(self):
        for degree in range(1,8):
            a,h = star(degree),hidden(degree+1)
            query = oracle.query_design(true_h=h,adjacency=a)
            for own,s in itertools.product((0,1),repeat=2):
                rows = []
                for leaves in itertools.product((0,1),repeat=degree):
                    if int(sum(leaves)/degree > .5) == s:
                        treatment = np.array((own,)+leaves,dtype=np.float64)
                        rows.append(scalar_row(a,h,treatment,0))
                expected = [sum(row[k] for row in rows)/len(rows) for k in range(25)]
                self.compare("query",query[0,own,s],expected)
                self.compare("exposure_enumeration",query[0,own,s,11],
                             enumerated_exposure(degree,s))

    def test_star_degree_199_center_and_degree_one_leaf_boundaries(self):
        a,h = star(199),hidden(200)
        query = oracle.query_design(true_h=h,adjacency=a)
        self.assertEqual(query.shape,(200,2,2,25))
        for own,s in itertools.product((0,1),repeat=2):
            center_e = exact_count_mean(199,s)
            self.compare("query",query[0,own,s],
                         scalar_row(a,h,np.zeros(200),0,own=own,conditional_e=center_e))
            self.assertLess(query[0,own,0,11],.5)
            self.assertGreater(query[0,own,1,11],.5)
            self.assertNotEqual(query[0,own,s,11],float(s))
            for i in range(1,200):
                self.assertEqual(query[i,own,s,11],float(s))
                self.compare("query",query[i,own,s],
                             scalar_row(a,h,np.zeros(200),i,own=own,conditional_e=float(s)))
        EVIDENCE["center_exposure"] = query[0,0,:,11].tolist()
        EVIDENCE["leaf_exposure"] = query[1,0,:,11].tolist()

    def test_factual_strict_majority_tie_and_star_center_threshold(self):
        a = star(4)
        e,s = oracle.factual_exposure(adjacency=a,treatment=np.array([0.,1.,1.,0.,0.]))
        self.assertEqual(e[0],.5)
        self.assertEqual(s[0],0)
        a = star(199)
        for treated,expected in ((0,0),(99,0),(100,1),(199,1)):
            t = np.zeros(200);t[1:1+treated] = 1
            e,s = oracle.factual_exposure(adjacency=a,treatment=t)
            self.assertEqual(e[0],treated/199)
            self.assertEqual(s[0],expected)
            np.testing.assert_array_equal(e[1:],np.zeros(199))

    def test_column_relations_between_own_treatment_states(self):
        q = oracle.query_design(true_h=hidden(6),adjacency=star(5))
        for s in (0,1):
            np.testing.assert_array_equal(q[:,0,s,:7],q[:,1,s,:7])
            np.testing.assert_array_equal(q[:,0,s,11:15],q[:,1,s,11:15])
            np.testing.assert_array_equal(q[:,0,s,19:22],q[:,1,s,19:22])
            for start,end in ((7,11),(15,19),(22,25)):
                np.testing.assert_array_equal(q[:,0,s,start:end],np.zeros((6,end-start)))
            np.testing.assert_array_equal(q[:,1,s,7],np.ones(6))
            np.testing.assert_array_equal(q[:,1,s,8:11],hidden(6))
            np.testing.assert_array_equal(q[:,1,s,15:19],q[:,1,s,11:15])
            np.testing.assert_array_equal(q[:,1,s,22:25],q[:,1,s,19:22])

    def test_float64_finite_immutable_and_no_shared_storage(self):
        a,h,t = star(3),hidden(4).astype(np.float32),np.array([0,1,0,1],dtype=np.int64)
        before = tuple(digest(x) for x in (a,h,t))
        factual = oracle.factual_design(true_h=h,adjacency=a,treatment=t)
        query = oracle.query_design(true_h=h,adjacency=a)
        mean,cov = oracle.coefficient_prior()
        self.assertEqual(before,tuple(digest(x) for x in (a,h,t)))
        for output in (factual,query,mean,cov):
            self.assertEqual(output.dtype,np.dtype("float64"))
            self.assertTrue(np.isfinite(output).all())
            self.assertFalse(output.flags.writeable)
            with self.assertRaises(ValueError):
                output.setflags(write=True)
            for value in (a,h,t):
                self.assertFalse(np.shares_memory(output,value))
        copy = factual.copy()
        h[:] = 10
        a[:] = 0
        t[:] = 1
        np.testing.assert_array_equal(factual,copy)

    def test_invalid_graph_hidden_and_treatment_explicitly_rejected(self):
        a,h,t = star(3),hidden(4),np.zeros(4)
        for bad in (np.zeros((4,4)),np.ones((4,4)),a[:3],a+np.eye(4),
                    np.full((4,4),np.nan)):
            with self.subTest(graph=bad.shape),self.assertRaises(ValueError):
                oracle.factual_design(true_h=h,adjacency=bad,treatment=t)
        asymmetric = a.copy();asymmetric[0,1] = 0
        with self.assertRaises(ValueError):
            oracle.query_design(true_h=h,adjacency=asymmetric)
        for bad in (h[:3],np.zeros((4,2)),np.full((4,3),np.inf)):
            with self.assertRaises(ValueError):
                oracle.query_design(true_h=bad,adjacency=a)
        for bad in (np.zeros(3),np.full(4,.5),np.full(4,np.nan)):
            with self.assertRaises(ValueError):
                oracle.factual_design(true_h=h,adjacency=a,treatment=bad)

    def test_non_numeric_and_privileged_extra_inputs_rejected(self):
        a,h,t = star(3),hidden(4),np.zeros(4)
        for bad in (h.tolist(),h.astype(complex),h.astype(object)):
            with self.assertRaises(TypeError):
                oracle.query_design(true_h=bad,adjacency=a)
        for name in ("beta","truth","full_y","noise"):
            with self.subTest(extra=name),self.assertRaises(TypeError):
                oracle.factual_design(true_h=h,adjacency=a,treatment=t,**{name:object()})

    def test_repeatability_without_rng_or_input_mutation(self):
        a,h,t = star(5),hidden(6),np.array([1.,0.,1.,0.,1.,0.])
        before = tuple(digest(x) for x in (a,h,t))
        f1 = oracle.factual_design(true_h=h,adjacency=a,treatment=t)
        f2 = oracle.factual_design(true_h=h,adjacency=a,treatment=t)
        q1 = oracle.query_design(true_h=h,adjacency=a)
        q2 = oracle.query_design(true_h=h,adjacency=a)
        np.testing.assert_array_equal(f1,f2)
        np.testing.assert_array_equal(q1,q2)
        self.assertEqual(before,tuple(digest(x) for x in (a,h,t)))


if __name__ == "__main__":
    unittest.main()
