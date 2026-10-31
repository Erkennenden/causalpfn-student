"""Stage 5.2 deterministic in-memory acceptance, no RNG or simulator calls.

Reference uses observation-space Gaussian conditioning, not prior whitening.
ATOL=1e-12, RTOL=0 fixed before first execution. Covariance thresholds are the
user-approved 10*25*eps64 rules, never test-adjusted.
"""
import itertools
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from star_v2 import oracle_numerics as num, oracle_context as ctx
from star_v2.oracle_design import coefficient_prior
from test_oracle_design import star, hidden, scalar_row

ATOL = 1e-12
RTOL = 0
EVIDENCE = {key:[] for key in ("posterior_mean","posterior_covariance","mu","ite","ate","intervals")}


def prior_reference():
    mean = np.array([.6 if i in (7,11) else 0. for i in range(25)])
    sd = ([.5]+[.8/math.sqrt(3)]*3+[.35/math.sqrt(3)]*3
          +[.5]+[.7/math.sqrt(3)]*3+[.5]+[.7/math.sqrt(3)]*3
          +[.25]+[.35/math.sqrt(3)]*3+[.4/math.sqrt(3)]*3+[.2/math.sqrt(3)]*3)
    covariance = np.zeros((25,25))
    for i in range(25):
        for j in range(25):
            if i == j:
                covariance[i,j] = sd[i]*sd[j]
            elif (7 <= i <= 10 and j == i+4) or (7 <= j <= 10 and i == j+4):
                covariance[i,j] = .35*sd[i]*sd[j]
    return mean,covariance


def posterior_reference(f,y,m,c,noise):
    # Different dimensional solve: observation covariance, no prior whitening.
    innovation = f@c@f.T+noise**2*np.eye(len(y))
    gain = np.linalg.solve(innovation,f@c).T
    return m+gain@(y-f@m),c-gain@f@c


def query_reference(a,h):
    result = np.empty((len(a),2,2,25))
    for i in range(len(a)):
        neighbors = [j for j in range(len(a)) if a[i,j] == 1]
        for own,s in itertools.product((0,1),repeat=2):
            rows = []
            for bits in itertools.product((0,1),repeat=len(neighbors)):
                if int(sum(bits)/len(neighbors) > .5) != s:
                    continue
                t = np.zeros(len(a));t[i] = own
                for j,bit in zip(neighbors,bits): t[j] = bit
                rows.append(scalar_row(a,h,t,i))
            for k in range(25):
                result[i,own,s,k] = sum(row[k] for row in rows)/len(rows)
    return result


def fixture():
    a,h = star(5),hidden(6)
    t = np.array([1.,0.,1.,1.,0.,1.])
    x = np.arange(120,dtype=np.float32).reshape(6,20)/64
    indices = np.array([4,0,2],dtype=np.int64)
    full_design = np.array([scalar_row(a,h,t,i) for i in range(6)])
    beta = np.array([(i-9)/32 for i in range(25)])  # Artificial test coefficients only.
    y = full_design[indices]@beta+np.array([.03,-.02,.01])
    inputs = dict(true_h=h,noise_sd=.7,x=x,adjacency=a,treatment=t,
                  observed_indices=indices,outcomes_observed=y)
    return inputs,full_design[indices],query_reference(a,h)


def expected_outputs():
    inputs,f,q = fixture()
    m,c = prior_reference()
    mean,cov = posterior_reference(f,inputs["outcomes_observed"],m,c,inputs["noise_sd"])
    mu = np.empty((6,2,2))
    for i,t,s in itertools.product(range(6),(0,1),(0,1)):
        mu[i,t,s] = sum(q[i,t,s,k]*mean[k] for k in range(25))
    ite = np.array([[mu[i,1,0]-mu[i,0,0],mu[i,1,1]-mu[i,1,0],
                     mu[i,1,1]-mu[i,0,0]] for i in range(6)])
    ate = np.array([sum(ite[i,k] for i in range(6))/6 for k in range(3)])
    contrasts = np.array([
        [sum(q[i,tp,sp,j]-q[i,tn,sn,j] for i in range(6))/6 for j in range(25)]
        for tp,sp,tn,sn in ((1,0,0,0),(1,1,1,0),(1,1,0,0))])
    limits = []
    for row in contrasts:
        center = sum(row[j]*mean[j] for j in range(25))
        variance = sum(row[i]*cov[i,j]*row[j] for i in range(25) for j in range(25))
        radius = 1.959963984540054*math.sqrt(variance)
        limits.append((center-radius,center+radius))
    return mean,cov,mu,ite,ate,np.array(limits)


def snapshot(inputs):
    return {k:(id(v),v.dtype.str,v.shape,v.tobytes()) for k,v in inputs.items()
            if type(v) is np.ndarray}


class OracleContextTests(unittest.TestCase):
    def compare(self,key,actual,expected):
        np.testing.assert_allclose(actual,expected,atol=ATOL,rtol=RTOL)
        EVIDENCE[key].append(float(np.max(np.abs(actual-expected))))

    def test_prior_matches_independent_reference_and_cholesky(self):
        mean,cov = coefficient_prior()
        m,c = prior_reference()
        np.testing.assert_array_equal(mean,m)
        np.testing.assert_array_equal(cov,c)
        checked = num.validate_covariance(cov)
        self.assertEqual(checked.status,"OK")
        lower = np.linalg.cholesky(checked.covariance)
        np.testing.assert_allclose(lower@lower.T,c,atol=ATOL,rtol=RTOL)
        EVIDENCE["prior_diagnostics"] = dict(checked.diagnostics)

    def test_gaussian_posterior_independent_observation_space_reference(self):
        m,c = prior_reference()
        f = np.array([[(i+1)*(j-7)/128+((i+j)%3)/16 for j in range(25)]
                      for i in range(7)],dtype=np.float64)
        y = np.array([-.2,.1,.7,-.4,.8,.3,-.1])
        expected_mean,expected_cov = posterior_reference(f,y,m,c,1.2)
        result = num.gaussian_linear_posterior(
            design=f,outcomes=y,prior_mean=m,prior_covariance=c,noise_sd=1.2)
        self.assertEqual(result.mean_status,"OK")
        self.assertEqual(result.covariance_status,"OK")
        self.compare("posterior_mean",result.mean,expected_mean)
        self.compare("posterior_covariance",result.covariance,expected_cov)

    def test_context_mu_ite_ate_and_intervals_independent_scalar_reference(self):
        inputs,_,_ = fixture()
        result = ctx.estimate_oracle_context(**inputs)
        mean,cov,mu,ite,ate,limits = expected_outputs()
        self.assertEqual(result.point_status,"OK")
        self.assertEqual(result.posterior_covariance_status,"OK")
        for key,actual,expected in (
            ("posterior_mean",result.posterior_mean,mean),
            ("posterior_covariance",result.posterior_covariance,cov),
            ("mu",result.mu,mu),("ite",result.ite,ite),("ate",result.ate,ate)):
            self.compare(key,actual,expected)
        self.assertEqual(result.mu.shape,(6,2,2))
        self.assertEqual(result.ite.shape,(6,3))
        self.assertEqual(result.ate.shape,(3,))
        self.compare("intervals",np.array([(r.lower,r.upper) for r in result.intervals]),limits)
        np.testing.assert_allclose(result.ite[:,2],result.ite[:,0]+result.ite[:,1],atol=ATOL,rtol=RTOL)
        np.testing.assert_allclose(result.ate[2],result.ate[0]+result.ate[1],atol=ATOL,rtol=RTOL)
        self.assertFalse(np.allclose(result.ate,result.ite[inputs["observed_indices"]].mean(0),
                                     atol=ATOL,rtol=RTOL))
        EVIDENCE["context_covariance_diagnostics"] = {
            k:dict(v) if hasattr(v,"items") else v
            for k,v in result.covariance_diagnostics.items()}

    def test_canonical_state_axis_and_all_query_values(self):
        inputs,_,q = fixture()
        result = ctx.estimate_oracle_context(**inputs)
        mean = expected_outputs()[0]
        for i,t,s in itertools.product(range(6),(0,1),(0,1)):
            expected = sum(q[i,t,s,j]*mean[j] for j in range(25))
            np.testing.assert_allclose(result.mu[i,t,s],expected,atol=ATOL,rtol=RTOL)
        self.assertFalse(np.allclose(result.mu[:,0,1],result.mu[:,1,0],atol=ATOL,rtol=RTOL))

    def test_intervals_only_three_ate_marginals_and_frozen_level(self):
        result = ctx.estimate_oracle_context(**fixture()[0])
        self.assertEqual(len(result.intervals),3)
        self.assertEqual(tuple(r.effect for r in result.intervals),("Direct","Spillover","Total"))
        for record in result.intervals:
            self.assertEqual(record.status,"OK")
            self.assertEqual(record.level,.95)
            self.assertEqual(record.source,"model_posterior")
            self.assertEqual(record.statistical_unit,"context")
            self.assertGreaterEqual(record.variance,0.)
        self.assertFalse(hasattr(result,"node_intervals"))
        self.assertFalse(hasattr(result,"pair_interval"))

    def test_covariance_positive_definite_and_exact_diagnostics(self):
        c = np.eye(25)*2
        result = num.validate_covariance(c)
        d = result.diagnostics
        tau = 10*25*np.finfo(np.float64).eps*2
        self.assertEqual(result.status,"OK")
        self.assertEqual(d["symmetry_residual"],0.)
        self.assertEqual(d["tau_sym"],tau)
        self.assertEqual(d["tau_psd"],tau)
        self.assertEqual(d["minimum_eigenvalue"],2.)

    def test_small_asymmetry_accepted_only_then_symmetrized(self):
        c = np.eye(25)
        c[0,1] = 10*25*np.finfo(np.float64).eps/4
        original = c.copy()
        result = num.validate_covariance(c)
        self.assertEqual(result.status,"OK")
        self.assertEqual(result.diagnostics["symmetry_residual"],c[0,1])
        self.assertEqual(result.covariance[0,1],c[0,1]/2)
        np.testing.assert_array_equal(c,original)

    def test_excess_asymmetry_rejected_before_eigenvalue_check(self):
        c = np.eye(25);c[0,1] = 1e-8
        with patch.object(num.np.linalg,"eigvalsh",
                          side_effect=AssertionError("must not symmetrize/eigendecompose")) as eigen:
            result = num.validate_covariance(c)
        eigen.assert_not_called()
        self.assertEqual(result.status,"FAILED")
        self.assertIsNone(result.covariance)
        self.assertIn("asymmetry",result.failure_reason)
        self.assertIsNone(result.diagnostics["minimum_eigenvalue"])

    def test_psd_tolerance_boundary_not_clipped_or_projected(self):
        tau = 10*25*np.finfo(np.float64).eps
        c = np.eye(25);c[0,0] = -tau/2
        result = num.validate_covariance(c)
        self.assertEqual(result.status,"OK")
        self.assertEqual(result.diagnostics["minimum_eigenvalue"],-tau/2)
        self.assertEqual(result.covariance[0,0],-tau/2)
        self.assertEqual(result.diagnostics["tau_psd"],tau)

    def test_psd_failure_keeps_residual_threshold_and_reason(self):
        c = np.eye(25);c[0,0] = -1e-8
        result = num.validate_covariance(c)
        self.assertEqual(result.status,"FAILED")
        self.assertEqual(result.diagnostics["symmetry_residual"],0.)
        self.assertEqual(result.diagnostics["minimum_eigenvalue"],-1e-8)
        self.assertLess(result.diagnostics["minimum_eigenvalue"],-result.diagnostics["tau_psd"])
        self.assertIn("-tau_psd",result.failure_reason)
        self.assertIsNone(result.covariance)

    def test_covariance_bad_shape_dtype_and_finiteness_rejected(self):
        for c in (np.eye(24),np.eye(25,dtype=np.float32),np.full((25,25),np.nan)):
            with self.subTest(shape=c.shape,dtype=c.dtype):
                result = num.validate_covariance(c)
                self.assertEqual(result.status,"FAILED")
                self.assertTrue(result.failure_reason)
                self.assertIsNone(result.covariance)

    def test_q_and_posterior_checked_and_no_explicit_inverse(self):
        with patch.object(np.linalg,"inv",side_effect=AssertionError("inverse forbidden")):
            result = ctx.estimate_oracle_context(**fixture()[0])
        for key in ("prior_covariance","Q","posterior_covariance"):
            self.assertEqual(result.covariance_diagnostics[key]["status"],"OK")
        self.assertEqual(result.covariance_diagnostics["Q_cholesky"],"OK")

    def test_negative_variance_fails_only_necessary_interval_without_clipping(self):
        contrasts = np.zeros((3,25));contrasts[0,0] = contrasts[1,1] = contrasts[2,2] = 1
        covariance = np.eye(25);covariance[1,1] = -1e-16
        self.assertEqual(num.validate_covariance(covariance).status,"OK")
        records = ctx.gaussian_ate_intervals(
            contrasts=contrasts,posterior_mean=np.zeros(25),
            posterior_covariance=covariance)
        self.assertEqual(tuple(r.status for r in records),("OK","FAILED","OK"))
        self.assertIsNone(records[1].lower)
        self.assertIsNone(records[1].upper)
        self.assertIn("Invalid posterior variance",records[1].failure_reason)
        self.assertEqual(covariance[1,1],-1e-16)

    def test_interval_single_effect_exception_preserves_points(self):
        original = ctx._effect_interval
        def fail(effect,*args):
            if effect == "Spillover": raise FloatingPointError("scripted variance failure")
            return original(effect,*args)
        with patch.object(ctx,"_effect_interval",side_effect=fail):
            result = ctx.estimate_oracle_context(**fixture()[0])
        self.assertEqual(result.point_status,"OK")
        self.assertEqual(tuple(r.status for r in result.intervals),("OK","FAILED","OK"))
        self.compare("ate",result.ate,expected_outputs()[4])
        self.assertIn("scripted variance failure",result.failure_reasons["intervals"]["Spillover"])

    def test_whole_interval_exception_preserves_mu_ite_ate(self):
        with patch.object(ctx,"gaussian_ate_intervals",
                          side_effect=FloatingPointError("scripted interval failure")):
            result = ctx.estimate_oracle_context(**fixture()[0])
        self.assertEqual(result.point_status,"OK")
        for key,actual,expected in zip(("mu","ite","ate"),
                (result.mu,result.ite,result.ate),expected_outputs()[2:5]):
            self.compare(key,actual,expected)
        for record in result.intervals:
            self.assertEqual(record.status,"FAILED")
            self.assertIsNone(record.lower)
            self.assertIn("scripted interval failure",record.failure_reason)

    def test_posterior_covariance_failure_preserves_mean_and_blocks_intervals(self):
        # Deliberate fault injection, not a real tolerance violation.
        real = num.validate_covariance
        calls = []
        def scripted(cov):
            result = real(cov);calls.append(result)
            if len(calls) == 3:
                return num.CovarianceCheck("FAILED",None,result.diagnostics,"scripted covariance failure")
            return result
        with patch.object(num,"validate_covariance",side_effect=scripted):
            result = ctx.estimate_oracle_context(**fixture()[0])
        self.assertEqual(result.posterior_covariance_status,"FAILED")
        self.assertIsNone(result.posterior_covariance)
        self.assertEqual(result.point_status,"OK")
        self.assertEqual(tuple(r.status for r in result.intervals),("FAILED",)*3)
        self.compare("ate",result.ate,expected_outputs()[4])
        self.assertIn("scripted covariance failure",result.failure_reasons["posterior"])

    def test_prior_failure_no_fabricated_points_or_intervals(self):
        # Expected negative fixture; the actual fixed prior is tested separately.
        mean,cov = prior_reference();cov[0,1] = .1
        with patch.object(ctx,"coefficient_prior",return_value=(mean,cov)):
            result = ctx.estimate_oracle_context(**fixture()[0])
        self.assertEqual(result.point_status,"FAILED")
        self.assertEqual(result.posterior_covariance_status,"FAILED")
        for value in (result.posterior_mean,result.posterior_covariance,result.mu,result.ite,result.ate):
            self.assertIsNone(value)
        self.assertTrue(all(r.status == "FAILED" and r.lower is None for r in result.intervals))

    def test_observation_order_preserved_not_sorted(self):
        inputs,_,_ = fixture()
        first = ctx.estimate_oracle_context(**inputs)
        permutation = np.array([2,0,1])
        changed = dict(inputs,observed_indices=inputs["observed_indices"][permutation],
                       outcomes_observed=inputs["outcomes_observed"][permutation])
        other = ctx.estimate_oracle_context(**changed)
        np.testing.assert_allclose(first.posterior_mean,other.posterior_mean,atol=ATOL,rtol=RTOL)
        self.compare("ate",other.ate,expected_outputs()[4])

    def test_input_objects_bytes_dtype_and_storage_isolation(self):
        inputs,_,_ = fixture();before = snapshot(inputs)
        result = ctx.estimate_oracle_context(**inputs)
        self.assertEqual(snapshot(inputs),before)
        self.assertEqual(inputs["outcomes_observed"].dtype,np.dtype("float64"))
        for output in (result.posterior_mean,result.posterior_covariance,result.mu,result.ite,result.ate):
            self.assertEqual(output.dtype,np.dtype("float64"))
            self.assertFalse(output.flags.writeable)
            with self.assertRaises(ValueError): output.setflags(write=True)
            for value in inputs.values():
                if type(value) is np.ndarray:
                    self.assertFalse(np.shares_memory(output,value))
        old = result.mu.copy()
        inputs["true_h"][:] = 99
        inputs["outcomes_observed"][:] = -99
        np.testing.assert_array_equal(result.mu,old)

    def test_invalid_indices_and_outcome_dtype_or_shape_rejected(self):
        inputs,_,_ = fixture()
        for indices in (np.array([1,1],dtype=np.int64),np.array([-1,0],dtype=np.int64),
                        np.array([6],dtype=np.int64),np.array([],dtype=np.int64),
                        np.array([0.,1.]),np.array([[0,1]],dtype=np.int64)):
            with self.subTest(indices=indices.tolist()),self.assertRaises(ValueError):
                ctx.estimate_oracle_context(**dict(inputs,observed_indices=indices))
        for y in (np.zeros(2),np.full(3,np.nan),np.zeros(3,dtype=np.float32)):
            with self.assertRaises((ValueError,TypeError)):
                ctx.estimate_oracle_context(**dict(inputs,outcomes_observed=y))

    def test_invalid_noise_and_nonfinite_inputs_rejected(self):
        inputs,_,_ = fixture()
        for noise in (0.,-1.,np.inf,np.nan,True,"1"):
            with self.subTest(noise=noise),self.assertRaises(ValueError):
                ctx.estimate_oracle_context(**dict(inputs,noise_sd=noise))
        for name in ("true_h","x","adjacency","treatment"):
            bad = inputs[name].copy();bad.flat[0] = np.nan
            with self.subTest(field=name),self.assertRaises(ValueError):
                ctx.estimate_oracle_context(**dict(inputs,**{name:bad}))

    def test_forbidden_fields_and_simulator_objects_rejected(self):
        inputs,_,_ = fixture()
        for name in ("beta","truth","truth_mu","truth_ite","truth_ate","full_y",
                     "noise","private_pair","evaluation_results"):
            with self.subTest(field=name),self.assertRaises(TypeError):
                ctx.estimate_oracle_context(**dict(inputs,**{name:object()}))
        with self.assertRaises(TypeError):
            ctx.estimate_oracle_context(object())

    def test_identity_optional_validated_and_immutable(self):
        inputs,_,_ = fixture()
        identity = {"context_id":"synthetic-context","pair_id":"synthetic-pair",
                    "request_id":"synthetic-request","protocol_hash":ctx.CONFIG.protocol_sha256}
        result = ctx.estimate_oracle_context(**inputs,context_identity=identity)
        self.assertEqual(dict(result.context_identity),identity)
        identity["context_id"] = "changed"
        self.assertEqual(result.context_identity["context_id"],"synthetic-context")
        with self.assertRaises(TypeError): result.context_identity["context_id"] = "changed"
        for bad in ({"context_id":""},{"context_id":"c","beta":"hidden"},
                    {"context_id":"c","protocol_hash":"wrong"}):
            with self.assertRaises(ValueError):
                ctx.estimate_oracle_context(**inputs,context_identity=bad)

    def test_repeatability_and_reference_only_identity(self):
        inputs,_,_ = fixture()
        first = ctx.estimate_oracle_context(**inputs)
        second = ctx.estimate_oracle_context(**inputs)
        np.testing.assert_array_equal(first.posterior_mean,second.posterior_mean)
        np.testing.assert_array_equal(first.mu,second.mu)
        self.assertEqual(first.intervals,second.intervals)
        self.assertEqual(first.method_id,"oracle-context")
        self.assertTrue(first.privileged and first.reference_only)

    def test_formal_size_without_formal_mechanism_or_rng(self):
        a,h = star(199),hidden(200)/16
        t = (np.arange(200)%3 == 0).astype(np.float64)
        indices = np.arange(0,200,4,dtype=np.int64)
        inputs = dict(true_h=h,noise_sd=1.,x=np.zeros((200,20),dtype=np.float32),
                      adjacency=a,treatment=t,observed_indices=indices,
                      outcomes_observed=np.arange(50,dtype=np.float64)/128)
        result = ctx.estimate_oracle_context(**inputs)
        self.assertEqual(result.point_status,"OK")
        self.assertEqual(result.posterior_covariance_status,"OK")
        self.assertEqual(result.mu.shape,(200,2,2))
        self.assertEqual(result.ite.shape,(200,3))
        np.testing.assert_allclose(result.ate,result.ite.mean(0),atol=ATOL,rtol=RTOL)


if __name__ == "__main__":
    unittest.main()
