"""Synthetic deterministic fixtures only; no formal mechanism RNG is consumed."""
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from star_v2 import dgp
from star_v2.config import CONFIG

ATOL = 1e-12
ERRORS = {}
TEST_NAMESPACE = 987654321  # Test-only: never a production argument or registry entry.


def fixed_arrays(n=200):
    a = dgp.star_adjacency(n)
    h = np.linspace(-0.7, 0.9, n * 3, dtype=np.float64).reshape(n, 3)
    beta = np.linspace(-0.35, 0.55, 25, dtype=np.float64)
    return a, h, beta


def reference_rows(a, h, beta, treatment):
    """Independent scalar neighbor sums and explicit 25-column DGP design."""
    n = len(a)
    components, fractions, js, outcomes = [], [], [], []
    for i in range(n):
        neighbors = [k for k in range(n) if a[i, k] == 1]
        hb = [sum(float(h[k, q]) for k in neighbors) / len(neighbors) for q in range(3)]
        e = sum(float(treatment[k]) for k in neighbors) / len(neighbors)
        j = [sum(float(treatment[k]) * float(h[k, q]) for k in neighbors)
             / len(neighbors) for q in range(3)]
        own = float(treatment[i])
        hetero = [1.0, *[float(v) for v in h[i]]]
        base_row = [*hetero, *hb]
        parts = [
            sum(v * float(b) for v, b in zip(base_row, beta[:7])),
            sum(v * float(b) for v, b in zip(hetero, beta[7:11])),
            sum(v * float(b) for v, b in zip(hetero, beta[11:15])),
            sum(v * float(b) for v, b in zip(hetero, beta[15:19])),
        ]
        row = [*base_row, *[own*v for v in hetero], *[e*v for v in hetero],
               *[own*e*v for v in hetero], *j, *[own*v for v in j]]
        assert len(row) == 25
        components.append(parts)
        fractions.append(e)
        js.append(j)
        outcomes.append(sum(v * float(b) for v, b in zip(row, beta)))
    return tuple(np.asarray(x, dtype=np.float64)
                 for x in (components, fractions, js, outcomes))


def binomial_mean(degree, state):
    counts = [k for k in range(degree+1) if (k/degree > 0.5) == bool(state)]
    denominator = sum(math.comb(degree, k) for k in counts)
    numerator = sum(k * math.comb(degree, k) for k in counts)
    return numerator / (degree * denominator)


def reference_truth(a, h, beta):
    result = np.empty((len(a), 2, 2), dtype=np.float64)
    for i in range(len(a)):
        neighbors = np.flatnonzero(a[i]).tolist()
        hb = [sum(float(h[k, q]) for k in neighbors)/len(neighbors) for q in range(3)]
        hi = [1.0, *map(float, h[i])]
        b = sum(v*float(c) for v,c in zip([*hi,*hb], beta[:7]))
        tau = sum(v*float(c) for v,c in zip(hi,beta[7:11]))
        gamma = sum(v*float(c) for v,c in zip(hi,beta[11:15]))
        eta = sum(v*float(c) for v,c in zip(hi,beta[15:19]))
        for t,s in itertools.product((0,1),repeat=2):
            j_mean = sum(hb[q]*(float(beta[19+q])+t*float(beta[22+q]))
                         for q in range(3))
            result[i,t,s] = b+t*tau+binomial_mean(len(neighbors),s)*(gamma+t*eta+j_mean)
    return result


@contextmanager
def mocked_mechanism_rng():
    """Valid registered identities, but values come ONLY from separate test RNG."""
    seen = []
    def factory(identity):
        code = identity.entropy[4]
        if code not in (1, 2, 3):
            raise AssertionError("Pair/global RNG must not be consumed in S2.3")
        seen.append(identity)
        return np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([TEST_NAMESPACE, code])))
    with patch.object(dgp, "make_generator", side_effect=factory):
        yield seen


def fixture_mechanism():
    with mocked_mechanism_rng():
        return dgp.generate_star_mechanism(cohort_code=1, mechanism_id=0)


class DGPTests(unittest.TestCase):
    def check_error(self, label, actual, expected):
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        ERRORS[label] = max(ERRORS.get(label, 0.0), error)
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)

    def test_T1_star_topology_and_no_rng(self):
        with patch.object(np.random, "default_rng", side_effect=AssertionError("RNG")), \
             patch.object(np.random, "Generator", side_effect=AssertionError("RNG")):
            a = dgp.star_adjacency()
        self.assertEqual(a.shape,(200,200))
        self.assertEqual(a.dtype,np.float64)
        np.testing.assert_array_equal(a,a.T)
        np.testing.assert_array_equal(np.diag(a),np.zeros(200))
        np.testing.assert_array_equal(a.sum(1),[199]+[1]*199)
        self.assertEqual(a.sum()/2,199)
        np.testing.assert_array_equal(a[1:,1:],0)
        np.testing.assert_array_equal(a[0,1:],1)
        for value in (True,200.0,1,-1):
            with self.subTest(value=value):
                with self.assertRaises((TypeError,ValueError)):
                    dgp.star_adjacency(value)

    def test_T2_exposure_extremes_majority_and_center(self):
        a = dgp.star_adjacency()
        for treated_leaves in (0,99,100,199):
            for center in (0,1):
                t = np.zeros(200)
                t[0] = center
                t[1:1+treated_leaves] = 1
                e,s = dgp.exposure(a,t)
                self.assertEqual(e[0],treated_leaves/199)
                self.assertEqual(s[0],int(treated_leaves>99.5))
                np.testing.assert_array_equal(e[1:],center)
                np.testing.assert_array_equal(s[1:],center)

    def test_T2_exact_half_maps_to_zero(self):
        a = dgp.star_adjacency(3)
        e,s = dgp.exposure(a,np.array([1,1,0]))
        self.assertEqual(e[0],0.5)
        self.assertEqual(s[0],0)

    def test_T2_reject_invalid_graph_and_treatment(self):
        a = dgp.star_adjacency(4)
        bad_graphs = [np.zeros((4,4)),np.eye(4),np.ones((4,3)),a*0.5]
        asym = a.copy(); asym[0,1]=0
        bad_graphs += [asym,np.full((4,4),np.nan)]
        for graph in bad_graphs:
            with self.subTest(graph=repr(graph)):
                with self.assertRaises((TypeError,ValueError)):
                    dgp.exposure(graph,np.zeros(4))
        for t in (np.zeros((4,1)),np.zeros(3),[0,0,2,0],[0,0,np.nan,0],["0"]*4):
            with self.subTest(t=t):
                with self.assertRaises((TypeError,ValueError)):
                    dgp.exposure(a,t)

    def test_T3_mocked_generation_dtypes_channels_and_formula(self):
        with mocked_mechanism_rng() as seen:
            m = dgp.generate_star_mechanism(cohort_code=1,mechanism_id=0)
        self.assertEqual([i.entropy for i in seen],[
            (20261030,2026103001,1,0,1,0),
            (20261030,2026103001,1,0,2,0),
            (20261030,2026103001,1,0,3,0)])
        self.assertEqual(m.population_mean.shape,(20,))
        self.assertEqual(m.x.shape,(200,20))
        self.assertEqual(m.h.shape,(200,3))
        self.assertEqual(m.coefficients.shape,(25,))
        self.assertEqual(m.x.dtype,np.float32)
        for field in ("population_mean","h","coefficients","components","mu","ite","ate"):
            self.assertEqual(getattr(m,field).dtype,np.float64)
        pop_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([TEST_NAMESPACE,1])))
        x_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([TEST_NAMESPACE,2])))
        population = pop_rng.normal(0,0.8,20)
        expected_x = (population+0.6*x_rng.normal(size=(200,20))).astype(np.float32)
        np.testing.assert_array_equal(m.population_mean,population)
        np.testing.assert_array_equal(m.x,expected_x)
        z = expected_x.astype(np.float64) @ m.projection
        expected_h = np.column_stack((np.tanh(1.5*z[:,0]),np.sin(1.5*z[:,1]),
                                      np.tanh(z[:,0]*z[:,1])))
        np.testing.assert_array_equal(m.h,expected_h)
        mean,cov = dgp.coefficient_prior()
        beta_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([TEST_NAMESPACE,3])))
        np.testing.assert_array_equal(m.coefficients,beta_rng.multivariate_normal(mean,cov))
        self.assertEqual(m.mu.shape,(200,2,2))
        np.testing.assert_array_equal(m.node_ids,np.arange(200))

    def test_T3_prior_25_columns_and_correlations(self):
        mean,cov = dgp.coefficient_prior()
        expected_mean = np.zeros(25); expected_mean[7]=expected_mean[11]=0.6
        scales = [0.5]+[0.8/math.sqrt(3)]*3+[0.35/math.sqrt(3)]*3
        scales += [0.5]+[0.7/math.sqrt(3)]*3
        scales += [0.5]+[0.7/math.sqrt(3)]*3
        scales += [0.25]+[0.35/math.sqrt(3)]*3
        scales += [0.4/math.sqrt(3)]*3+[0.2/math.sqrt(3)]*3
        expected_cov = np.diag(np.square(scales))
        for k in range(4):
            expected_cov[7+k,11+k]=expected_cov[11+k,7+k]=0.35*scales[7+k]*scales[11+k]
        np.testing.assert_array_equal(mean,expected_mean)
        self.check_error("prior_covariance",cov,expected_cov)
        self.assertGreater(cov[7,11],0)
        self.assertFalse(np.all(np.diag(cov)==cov[0,0]))

    def test_T3_projection_fixed_cached_and_hash(self):
        p = dgp.common_projection()
        reference = np.linalg.qr(np.random.Generator(np.random.PCG64(2026101604))
                                     .normal(size=(20,2)))[0]
        np.testing.assert_array_equal(p,reference)
        with patch.object(np.linalg,"qr",side_effect=AssertionError("projection redrawn")):
            first,second = fixture_mechanism(),fixture_mechanism()
        np.testing.assert_array_equal(first.projection,second.projection)
        np.testing.assert_array_equal(first.projection,p)
        self.assertEqual(dgp.array_sha256(first.projection),dgp.array_sha256(reference))
        self.assertEqual(first.metadata()["representation_seed"],2026101604)

    def test_T4_scalar_reference_components_exposure_j_and_outcome(self):
        a,h,beta = fixed_arrays(4)
        t = np.array([1,0,1,0],dtype=np.float32)
        parts,e,j,y = reference_rows(a,h,beta,t)
        self.check_error("components",dgp.structural_components(a,h,beta),parts)
        self.check_error("exposure",dgp.exposure(a,t)[0],e)
        self.check_error("treated_neighbors",dgp.treated_neighbor_features(a,h,t),j)
        self.check_error("structural_mean",dgp.structural_mean(a,h,beta,t),y)

    def test_T4_float64_before_common_structural_operations(self):
        a,h,beta = fixed_arrays(4)
        h = h.astype(np.float32)
        t = np.array([1,0,1,0],dtype=np.float32)
        reference = reference_rows(a,h,beta,t)[3]
        original = dgp.structural_outcome
        observed = []
        def checked(world,treatment):
            observed.append(True)
            for value in (world.adjacency,world.h,world.coefficients,world.components,treatment):
                self.assertEqual(value.dtype,np.float64)
            return original(world,treatment)
        with patch.object(dgp,"structural_outcome",side_effect=checked):
            actual = dgp.structural_mean(a.astype(np.float32),h,beta,t)
        self.assertTrue(observed)
        self.assertEqual(actual.dtype,np.float64)
        self.check_error("float32_input_float64_arithmetic",actual,reference)
        self.assertGreater(float(np.max(np.abs(actual.astype(np.float32).astype(np.float64)-reference))),
                           ATOL)

    def test_T4_reject_invalid_structure(self):
        a,h,beta = fixed_arrays(4)
        for hidden,coeff in ((h[:,:2],beta),(h,beta[:24]),
                             (np.full_like(h,np.inf),beta),(h,np.full(25,np.nan))):
            with self.subTest(shape=(hidden.shape,coeff.shape)):
                with self.assertRaises((TypeError,ValueError)):
                    dgp.structural_mean(a,hidden,coeff,np.zeros(4))

    def test_T5_conditional_exposure_independent_binomial_sum(self):
        for degree in (1,3,199):
            for s in (0,1):
                actual = dgp.arm_mean_exposure(degree,0.5,s)
                self.check_error("conditional_exposure",actual,binomial_mean(degree,s))
        self.assertEqual(dgp.arm_mean_exposure(1,0.5,0),0)
        self.assertEqual(dgp.arm_mean_exposure(1,0.5,1),1)

    def test_T5_full_truth_axes_finite_and_formula(self):
        a,h,beta = fixed_arrays()
        actual = dgp.four_state_truth(a,h,beta)
        self.assertEqual(actual.shape,(200,2,2))
        self.assertEqual(actual.dtype,np.float64)
        self.assertTrue(np.isfinite(actual).all())
        self.check_error("full_truth_formula",actual,reference_truth(a,h,beta))
        # No epsilon or treatment realization enters the analytic truth API.
        np.testing.assert_array_equal(actual,dgp.four_state_truth(a,h,beta))

    def test_T6_four_node_conditional_assignment_enumeration(self):
        a,h,beta = fixed_arrays(4)
        analytic = dgp.four_state_truth(a,h,beta)
        enumerated = np.empty((4,2,2))
        for i,t in itertools.product(range(4),(0,1)):
            groups = {0:[],1:[]}
            others = [k for k in range(4) if k!=i]
            for bits in itertools.product((0,1),repeat=3):
                treatment = np.zeros(4); treatment[i]=t
                treatment[others]=bits
                # Independent exposure and raw structural scalar computation.
                e = sum(treatment[k] for k in range(4) if a[i,k])/sum(a[i])
                groups[int(e>0.5)].append(reference_rows(a,h,beta,treatment)[3][i])
            for s in (0,1):
                enumerated[i,t,s] = sum(groups[s])/len(groups[s])
        self.check_error("four_node_enumeration",analytic,enumerated)

    def test_T7_effect_order_identity_and_all_node_group_means(self):
        mu = np.zeros((200,2,2))
        mu[:,0,0]=np.arange(200)/10
        mu[:,0,1]=-999  # Unused 01 must not enter any registered effect.
        mu[:,1,0]=mu[:,0,0]+np.arange(200)+1
        mu[:,1,1]=mu[:,1,0]+2*np.arange(200)+3
        ite,ate = dgp.effects_from_mu(mu)
        self.assertEqual(ite.shape,(200,3))
        self.assertEqual(ate.shape,(3,))
        self.check_error("effect_total_identity",ite[:,2],ite[:,0]+ite[:,1])
        self.check_error("direct_order",ite[:,0],np.arange(200)+1)
        self.check_error("spillover_order",ite[:,1],2*np.arange(200)+3)
        groups = dgp.group_effects(ite)
        np.testing.assert_array_equal(groups["center"],ite[0])
        np.testing.assert_array_equal(groups["leaf"],ite[1:].mean(0))
        np.testing.assert_array_equal(groups["all"],ate)
        self.assertFalse(np.array_equal(ate,ite[:50].mean(0)))
        with self.assertRaises(ValueError):
            dgp.effects_from_mu(mu.reshape(200,4))

    def test_T8_simulator_private_and_readonly(self):
        m = fixture_mechanism()
        with self.assertRaises(TypeError):
            dict(m)
        self.assertFalse(hasattr(m,"observed_indices"))
        self.assertFalse(hasattr(m,"outcomes_observed"))
        self.assertFalse(hasattr(m,"predict"))
        self.assertEqual(m.metadata()["access"],"simulator_evaluator_only")
        for name in ("x","h","coefficients","mu"):
            with self.subTest(field=name):
                with self.assertRaises(ValueError):
                    getattr(m,name).setflags(write=True)
        with self.assertRaises(TypeError):
            dgp.generate_star_mechanism(cohort_code=1,mechanism_id=0,seed=123)

    def test_T8_no_legacy_or_model_execution_path(self):
        import pfn_social.phase5_large_network as legacy
        import torch
        with patch.object(legacy,"generate_mechanism",side_effect=AssertionError("legacy dev")), \
             patch.object(legacy,"rng_for",side_effect=AssertionError("legacy RNG")), \
             patch.object(torch,"load",side_effect=AssertionError("checkpoint")):
            m = fixture_mechanism()
            m.structural_mean(np.zeros(200))
        self.assertEqual(m.rng_identities[0].channel,"population")

    def test_T8_no_global_rng_or_input_mutation(self):
        a,h,beta = fixed_arrays(4)
        t = np.array([0,1,0,1])
        before = [dgp.array_sha256(v) for v in (a,h,beta,t)]
        rng_before = np.random.get_state()
        dgp.structural_mean(a,h,beta,t)
        dgp.four_state_truth(a,h,beta)
        fixture_mechanism()
        rng_after = np.random.get_state()
        self.assertEqual(before,[dgp.array_sha256(v) for v in (a,h,beta,t)])
        self.assertEqual(rng_before[0],rng_after[0])
        np.testing.assert_array_equal(rng_before[1],rng_after[1])
        self.assertEqual(rng_before[2:],rng_after[2:])

    def test_T9_fixture_reproducibility_hash_metadata(self):
        first,second = fixture_mechanism(),fixture_mechanism()
        self.assertEqual(first.metadata(),second.metadata())
        self.assertEqual(json.loads(json.dumps(first.metadata())),first.metadata())
        self.assertEqual(first.metadata()["array_dtypes"]["x"],"<f4")
        self.assertEqual(first.metadata()["array_dtypes"]["mu"],"<f8")
        for name in ("population_mean","x","h","coefficients","mu","ite","ate"):
            np.testing.assert_array_equal(getattr(first,name),getattr(second,name))

    def test_T9_frozen_inputs_and_no_generated_artifacts(self):
        directory = Path(__file__).resolve().parents[1]
        expected = {
            "protocol/E1_5_Star_v2_Candidate_v0.3.2_FROZEN.md":"fa99e65d3082d90a8675354fc4a478051b0fe3b2980d5bf78c59f29ab279209b",
            "protocol/freeze_record.json":"9690c343fb6adca9aecb5c702dc4965513eeb66fc5dd48d372b89e5bebd71dfe",
            "protocol/rng_registry.json":"d0ea7726b03824036d0a6a74519c6480c37a29a23734f587177a59c94faecdd3",
        }
        for rel,digest in expected.items():
            self.assertEqual(hashlib.sha256((directory/rel).read_bytes()).hexdigest(),digest)
        self.assertFalse(list(directory.rglob("__pycache__")))
        self.assertFalse((directory.parents[2]/"results/e1_5_star_v2").exists())


def tearDownModule():
    print("S2.3_REFERENCE_MAX_ABS_ERRORS",json.dumps(ERRORS,sort_keys=True))


if __name__ == "__main__":
    unittest.main()
