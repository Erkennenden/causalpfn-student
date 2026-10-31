"""Stage 5.3 deterministic in-memory references, no RNG/generator/model.

ATOL=1e-10, RTOL=0 fixed before first execution for independent posteriors;
DGP row/difference alignment ATOL=1e-12. Frozen rank/residual gates unchanged.
"""
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from scipy.linalg import qr

from star_v2 import oracle_pair as pair
from star_v2 import oracle_context as context
from star_v2 import oracle_numerics as num
from star_v2.dgp import structural_mean
from test_oracle_context import prior_reference
from test_oracle_design import scalar_row, exact_count_mean, star

ATOL = 1e-10
RTOL = 0
EVIDENCE = {k:[] for k in ("posterior_mean","posterior_covariance","mu","ite","ate",
                          "variance","intervals","DGP_difference","DGP_alignment")}


def condition_reference(f0,f1,y0,y1,sigma,rows):
    # Exact Gaussian conditioning in beta coordinates, then observation space.
    # Rows are independently known synthetic constraints or selected using QR.
    mean,cov = prior_reference()
    df,dy = f1-f0,y1-y0
    if len(rows):
        d,delta = df[rows],dy[rows]
        gain = np.linalg.solve(d@cov@d.T,d@cov).T
        constrained_mean = mean+gain@(delta-d@mean)
        constrained_cov = cov-gain@d@cov
    else:
        constrained_mean,constrained_cov = mean,cov
    avg,yavg = (f0+f1)/2,(y0+y1)/2
    gain = np.linalg.solve(avg@constrained_cov@avg.T+sigma**2*np.eye(len(yavg)),
                           avg@constrained_cov).T
    return (constrained_mean+gain@(yavg-avg@constrained_mean),
            constrained_cov-gain@avg@constrained_cov)


def synthetic():
    avg = np.array([[((i+2)*(j%5-2))/32+((i+j)%4)/64
                     for j in range(25)] for i in range(6)],dtype=np.float64)
    df = np.zeros((6,25))
    for i,(column,value) in enumerate(((0,.8),(7,1.1),(11,-1.2),(19,.6))):
        df[i,column] = value
    f0,f1 = avg-df/2,avg+df/2
    beta = np.array([(i-8)/16 for i in range(25)])
    noise = np.array([.03,-.04,.01,.06,-.02,.05])
    return f0,f1,f0@beta+noise,f1@beta+noise


def star_fixture(*,center_observed=True):
    n = 200
    a = star(199)
    index = np.arange(n,dtype=np.float64)
    h = np.column_stack((np.sin(index*.17),np.cos(index*.31),(index%11-5)/7))
    t0 = (np.arange(n)%3 == 0).astype(np.float64);t0[0] = 0.
    t1 = t0.copy();t1[0] = 1.
    indices = np.arange(50,dtype=np.int64) if center_observed else np.arange(1,51,dtype=np.int64)
    beta = np.array([(j-9)/32 for j in range(25)])
    epsilon = np.cos(index*.23)/5
    # Actual fixed DGP float64 path. No random mechanism or PrivatePair.
    mean0,mean1 = structural_mean(a,h,beta,t0),structural_mean(a,h,beta,t1)
    y0,y1 = (mean0+epsilon)[indices],(mean1+epsilon)[indices]
    f0 = np.array([scalar_row(a,h,t0,i) for i in indices])
    f1 = np.array([scalar_row(a,h,t1,i) for i in indices])
    inputs = dict(true_h=h,noise_sd=1.,adjacency=a,treatment0=t0,treatment1=t1,
        observed_indices0=indices.copy(),observed_indices1=indices.copy(),
        outcomes_observed0=y0,outcomes_observed1=y1)
    return inputs,f0,f1,beta,mean0[indices],mean1[indices]


def independent_query(inputs):
    a,h = inputs["adjacency"],inputs["true_h"]
    q = np.empty((200,2,2,25))
    for i in range(200):
        for t in (0,1):
            for s in (0,1):
                e = exact_count_mean(int(a[i].sum()),s)
                q[i,t,s] = scalar_row(a,h,np.zeros(200),i,own=t,conditional_e=e)
    return q


def independent_rows(df):
    _,r,pivots = qr(df.T,mode="economic",pivoting=True)
    diag = np.abs(np.diag(r))
    threshold = max(df.shape)*np.finfo(np.float64).eps*max(diag)
    rank = int(np.count_nonzero(diag > threshold))
    return pivots[:rank]


def snapshot(inputs):
    return {k:(id(v),v.dtype.str,v.shape,v.tobytes()) for k,v in inputs.items()
            if type(v) is np.ndarray}


class OraclePairTests(unittest.TestCase):
    def compare(self,key,actual,expected,atol=ATOL):
        np.testing.assert_allclose(actual,expected,atol=atol,rtol=RTOL)
        EVIDENCE[key].append(float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)))))

    def test_known_orthogonal_constraint_rank_particular_and_nullspace(self):
        a = np.zeros((4,25));a[0,0] = 3;a[1,1] = 2;a[2,2] = 1;a[3,0] = 6
        z = np.arange(25,dtype=np.float64)/16
        result = pair.solve_constraints(a,a@z)
        self.assertEqual(result.status,"OK")
        self.assertEqual(result.diagnostics["rank"],3)
        np.testing.assert_allclose(result.singular_values,[math.sqrt(45),2,1,0],atol=ATOL,rtol=0)
        self.assertEqual(result.diagnostics["tol_rank"],25*num.EPS64*result.singular_values[0])
        expected_z = np.r_[z[:3],np.zeros(22)]
        np.testing.assert_allclose(result.particular,expected_z,atol=ATOL,rtol=0)
        projector = np.diag(np.r_[np.zeros(3),np.ones(22)])
        np.testing.assert_allclose(result.nullspace@result.nullspace.T,projector,atol=ATOL,rtol=0)
        self.assertEqual(result.nullspace.shape,(25,22))
        np.testing.assert_allclose(a@result.particular,a@z,atol=ATOL,rtol=0)

    def test_rank_zero_and_frozen_numerical_rank_boundary(self):
        result = pair.solve_constraints(np.zeros((3,25)),np.zeros(3))
        self.assertEqual(result.status,"OK")
        self.assertEqual(result.diagnostics["rank"],0)
        self.assertEqual(result.diagnostics["tol_rank"],0.)
        self.assertEqual(result.nullspace.shape,(25,25))
        a = np.eye(25);a[-1,-1] = 25*num.EPS64/2
        result = pair.solve_constraints(a,np.zeros(25))
        self.assertEqual(result.diagnostics["rank"],24)

    def test_inconsistent_constraints_rejected_without_posterior_update(self):
        f0,f1,y0,y1 = synthetic()
        y1 = y1.copy();y1[-1] += .01  # zero DeltaF row, impossible DeltaY
        with patch.object(np.linalg,"solve",side_effect=AssertionError("posterior must not run")):
            result = pair.shared_linear_posterior(
                design0=f0,design1=f1,outcomes0=y0,outcomes1=y1,noise_sd=.8)
        self.assertEqual(result.status,"FAILED")
        self.assertEqual(result.constraint.status,"FAILED")
        self.assertGreater(result.constraint.diagnostics["projection_residual"],
                           result.constraint.diagnostics["constraint_tolerance"])
        self.assertIsNone(result.mean)
        self.assertIsNone(result.covariance)
        self.assertNotIn("nullspace_update",result.diagnostics)

    def test_svd_failure_retains_diagnostics(self):
        with patch.object(np.linalg,"svd",side_effect=np.linalg.LinAlgError("scripted SVD failure")):
            result = pair.solve_constraints(np.zeros((3,25)),np.zeros(3))
        self.assertEqual(result.status,"FAILED")
        self.assertIn("scripted SVD failure",result.failure_reason)
        self.assertEqual(result.diagnostics["shape"],(3,25))
        self.assertIsNone(result.particular)

    def test_independent_constrained_posterior_and_sigma_squared(self):
        f0,f1,y0,y1 = synthetic()
        expected_mean,expected_cov = condition_reference(f0,f1,y0,y1,.8,np.arange(4))
        with patch.object(np.linalg,"inv",side_effect=AssertionError("inverse forbidden")):
            result = pair.shared_linear_posterior(
                design0=f0,design1=f1,outcomes0=y0,outcomes1=y1,noise_sd=.8)
        self.assertEqual(result.status,"OK")
        self.compare("posterior_mean",result.mean,expected_mean)
        self.compare("posterior_covariance",result.covariance,expected_cov)
        self.assertEqual(result.diagnostics["average_noise_variance"],.8**2)
        wrong_mean,wrong_cov = condition_reference(f0,f1,y0,y1,.8/math.sqrt(2),np.arange(4))
        self.assertGreater(np.max(np.abs(result.covariance-wrong_cov)),1e-4)
        self.assertGreater(np.max(np.abs(result.mean-wrong_mean)),1e-5)

    def test_zero_rank_average_update_matches_unconstrained_gaussian(self):
        f0,_,y0,_ = synthetic()
        expected = condition_reference(f0,f0,y0,y0,.8,np.array([],dtype=int))
        result = pair.shared_linear_posterior(
            design0=f0,design1=f0,outcomes0=y0,outcomes1=y0,noise_sd=.8)
        self.assertEqual(result.constraint.diagnostics["rank"],0)
        self.compare("posterior_mean",result.mean,expected[0])
        self.compare("posterior_covariance",result.covariance,expected[1])

    def test_full_rank_zero_nullspace_and_degenerate_intervals(self):
        beta = np.arange(25,dtype=np.float64)/32
        f0,f1 = -np.eye(25)/2,np.eye(25)/2
        y0,y1 = f0@beta+np.ones(25)*3,f1@beta+np.ones(25)*3
        with patch.object(np.linalg,"solve",side_effect=AssertionError("no nullspace solve")):
            result = pair.shared_linear_posterior(
                design0=f0,design1=f1,outcomes0=y0,outcomes1=y1,noise_sd=.8)
        self.assertEqual(result.status,"OK")
        self.assertEqual(result.constraint.diagnostics["rank"],25)
        self.assertEqual(result.constraint.diagnostics["nullspace_dimension"],0)
        self.assertEqual(result.diagnostics["nullspace_update"],"ZERO_DIMENSION")
        self.assertGreater(result.diagnostics["average_residual_norm"],0.)
        self.compare("posterior_mean",result.mean,beta)
        np.testing.assert_array_equal(result.covariance,np.zeros((25,25)))
        contrasts = np.eye(25)[:3]
        records = pair.gaussian_ate_intervals(contrasts=contrasts,
            posterior_mean=result.mean,posterior_covariance=result.covariance)
        for k,record in enumerate(records):
            self.assertEqual(record.status,"OK")
            self.assertEqual(record.variance,0.)
            self.assertEqual(record.lower,record.upper)
            self.assertAlmostEqual(record.lower,beta[k],places=12)

    def test_common_dgp_float64_alignment_and_real_star_rank_diagnostics(self):
        for observed in (True,False):
            inputs,f0,f1,beta,mean0,mean1 = star_fixture(center_observed=observed)
            result = pair.estimate_oracle_pair(**inputs)
            self.assertEqual(result.posterior_status,"OK",str(result.failure_reasons))
            self.assertEqual(result.constraint.status,"OK")
            self.compare("DGP_alignment",f0@beta,mean0,atol=1e-12)
            self.compare("DGP_alignment",f1@beta,mean1,atol=1e-12)
            delta = inputs["outcomes_observed1"]-inputs["outcomes_observed0"]
            self.compare("DGP_difference",delta,(f1-f0)@beta,atol=1e-12)
            diagnostics = result.constraint.diagnostics
            self.assertLessEqual(diagnostics["projection_residual"],diagnostics["constraint_tolerance"])
            self.assertEqual(diagnostics["rank"],len(independent_rows(f1-f0)))
            EVIDENCE["star_center_observed_"+str(observed)] = {
                **dict(diagnostics),"singular_values":result.constraint.singular_values.tolist(),
                "covariance":dict(result.diagnostics["posterior_covariance"]),
                "interval_statuses":[r.status for r in result.intervals]}

    def test_star_posterior_query_variance_and_intervals_independent_reference(self):
        inputs,f0,f1,_,_,_ = star_fixture()
        expected_mean,expected_cov = condition_reference(
            f0,f1,inputs["outcomes_observed0"],inputs["outcomes_observed1"],
            inputs["noise_sd"],independent_rows(f1-f0))
        result = pair.estimate_oracle_pair(**inputs)
        self.assertEqual(result.status,"OK",str(result.failure_reasons))
        self.compare("posterior_mean",result.posterior_mean,expected_mean)
        self.compare("posterior_covariance",result.posterior_covariance,expected_cov)
        q = independent_query(inputs)
        mu = q@expected_mean
        ite = np.array([[mu[i,1,0]-mu[i,0,0],mu[i,1,1]-mu[i,1,0],
                         mu[i,1,1]-mu[i,0,0]] for i in range(200)])
        ate = np.array([sum(ite[:,k])/200 for k in range(3)])
        self.compare("mu",result.mu,mu)
        self.compare("ite",result.ite,ite)
        self.compare("ate",result.ate,ate)
        contrasts = np.array([
            [sum(q[i,tp,sp,j]-q[i,tn,sn,j] for i in range(200))/200 for j in range(25)]
            for tp,sp,tn,sn in ((1,0,0,0),(1,1,1,0),(1,1,0,0))])
        expected_variance,expected_limits = [],[]
        for c in contrasts:
            variance = sum(c[i]*expected_cov[i,j]*c[j] for i in range(25) for j in range(25))
            center = sum(c[j]*expected_mean[j] for j in range(25))
            radius = 1.959963984540054*math.sqrt(variance)
            expected_variance.append(variance)
            expected_limits.append([center-radius,center+radius])
        self.compare("variance",np.array([r.variance for r in result.intervals]),expected_variance)
        self.compare("intervals",np.array([[r.lower,r.upper] for r in result.intervals]),expected_limits)
        np.testing.assert_allclose(result.ate[2],result.ate[0]+result.ate[1],atol=ATOL,rtol=0)
        for record in result.intervals:
            self.assertEqual(record.source,"joint_model_posterior")
            self.assertEqual(record.statistical_unit,"pair")
            self.assertEqual(record.level,.95)

    def test_inconsistent_pair_has_no_fake_points_intervals_and_keeps_identity(self):
        inputs,_,_,_,_,_ = star_fixture()
        inputs["outcomes_observed1"] = inputs["outcomes_observed1"].copy()
        inputs["outcomes_observed1"][7] += .1
        identity = {"pair_id":"synthetic-pair","context_ids":("arm0","arm1")}
        result = pair.estimate_oracle_pair(**inputs,pair_identity=identity)
        self.assertEqual(result.status,"FAILED")
        self.assertEqual(result.constraint.status,"FAILED")
        self.assertEqual(dict(result.pair_identity),identity)
        for value in (result.posterior_mean,result.posterior_covariance,result.mu,result.ite,result.ate):
            self.assertIsNone(value)
        self.assertTrue(all(r.status == "FAILED" and r.lower is None for r in result.intervals))

    def test_pair_svd_failure_keeps_identity_and_failure_reason(self):
        inputs,_,_,_,_,_ = star_fixture()
        with patch.object(pair,"solve_constraints",side_effect=np.linalg.LinAlgError("SVD fault")):
            result = pair.estimate_oracle_pair(**inputs,pair_identity={"pair_id":"p"})
        self.assertEqual(result.status,"FAILED")
        self.assertEqual(result.pair_identity["pair_id"],"p")
        self.assertIn("SVD fault",result.failure_reasons["posterior"])

    def test_single_effect_interval_failure_preserves_all_points(self):
        inputs,_,_,_,_,_ = star_fixture()
        original = pair.gaussian_ate_intervals
        def fail(**kwargs):
            records = list(original(**kwargs))
            records[1] = context.OracleInterval("Spillover","FAILED",failure_reason="scripted negative variance")
            return tuple(records)
        with patch.object(pair,"gaussian_ate_intervals",side_effect=fail):
            result = pair.estimate_oracle_pair(**inputs)
        self.assertEqual(result.point_status,"OK")
        self.assertEqual(tuple(r.status for r in result.intervals),("OK","FAILED","OK"))
        self.assertTrue(np.isfinite(result.ate).all())
        self.assertIn("negative variance",result.failure_reasons["intervals"]["Spillover"])

    def test_whole_interval_failure_preserves_points(self):
        inputs,_,_,_,_,_ = star_fixture()
        with patch.object(pair,"gaussian_ate_intervals",side_effect=FloatingPointError("interval fault")):
            result = pair.estimate_oracle_pair(**inputs)
        self.assertEqual(result.point_status,"OK")
        self.assertEqual(tuple(r.status for r in result.intervals),("FAILED",)*3)
        self.assertTrue(np.isfinite(result.mu).all())
        self.assertTrue(all("interval fault" in r.failure_reason for r in result.intervals))

    def test_covariance_failure_retains_mean_but_no_covariance_or_valid_interval(self):
        inputs,_,_,_,_,_ = star_fixture()
        real = pair.validate_covariance
        count = [0]
        def fail(c):
            count[0] += 1
            if count[0] == 2:
                bad = np.eye(25);bad[0,0] = -1e-8
                return real(bad)
            return real(c)
        with patch.object(pair,"validate_covariance",side_effect=fail):
            result = pair.estimate_oracle_pair(**inputs)
        self.assertEqual(result.covariance_status,"FAILED")
        self.assertIsNone(result.posterior_covariance)
        self.assertEqual(result.point_status,"OK")
        self.assertEqual(tuple(r.status for r in result.intervals),("FAILED",)*3)
        self.assertEqual(result.diagnostics["posterior_covariance"]["minimum_eigenvalue"],-1e-8)

    def test_input_dtype_bytes_storage_and_repeated_determinism(self):
        inputs,_,_,_,_,_ = star_fixture();before = snapshot(inputs)
        first = pair.estimate_oracle_pair(**inputs)
        second = pair.estimate_oracle_pair(**inputs)
        self.assertEqual(snapshot(inputs),before)
        np.testing.assert_array_equal(first.posterior_mean,second.posterior_mean)
        self.assertEqual(first.intervals,second.intervals)
        for output in (first.posterior_mean,first.posterior_covariance,first.mu,first.ite,first.ate,
                       first.constraint.particular,first.constraint.nullspace,first.constraint.singular_values):
            self.assertEqual(output.dtype,np.dtype("float64"))
            self.assertFalse(output.flags.writeable)
            with self.assertRaises(ValueError): output.setflags(write=True)
            for value in inputs.values():
                if type(value) is np.ndarray: self.assertFalse(np.shares_memory(output,value))
        self.assertEqual(inputs["outcomes_observed0"].dtype,np.dtype("float64"))

    def test_observation_order_never_sorted_or_matched(self):
        inputs,_,_,_,_,_ = star_fixture()
        original = pair.estimate_oracle_pair(**inputs)
        order = np.arange(49,-1,-1)
        permuted = dict(inputs)
        for key in ("observed_indices0","observed_indices1","outcomes_observed0","outcomes_observed1"):
            permuted[key] = inputs[key][order]
        other = pair.estimate_oracle_pair(**permuted)
        np.testing.assert_allclose(other.ate,original.ate,atol=ATOL,rtol=0)
        wrong = dict(inputs,observed_indices1=inputs["observed_indices1"][order])
        with self.assertRaises(ValueError): pair.estimate_oracle_pair(**wrong)

    def test_illegal_indices_dtype_nonfinite_noise_and_treatment_rejected(self):
        inputs,_,_,_,_,_ = star_fixture()
        for indices in (np.array([1,1],dtype=np.int64),np.array([-1],dtype=np.int64),
                        np.array([200],dtype=np.int64),np.array([1.,2.])):
            with self.assertRaises(ValueError):
                pair.estimate_oracle_pair(**dict(inputs,observed_indices0=indices,observed_indices1=indices))
        for y in (np.zeros(50,dtype=np.float32),np.full(50,np.nan),np.zeros(49)):
            with self.assertRaises((TypeError,ValueError)):
                pair.estimate_oracle_pair(**dict(inputs,outcomes_observed0=y))
        for sigma in (0.,-1.,np.nan,np.inf,True):
            with self.assertRaises(ValueError): pair.estimate_oracle_pair(**dict(inputs,noise_sd=sigma))
        t = inputs["treatment1"].copy();t[4] = 1-t[4]
        with self.assertRaises(ValueError): pair.estimate_oracle_pair(**dict(inputs,treatment1=t))
        h = inputs["true_h"].copy();h[0,0] = np.nan
        with self.assertRaises(ValueError): pair.estimate_oracle_pair(**dict(inputs,true_h=h))

    def test_forbidden_inputs_simulator_and_identity_rejected(self):
        inputs,_,_,_,_,_ = star_fixture()
        for key in ("beta","truth","full_y","noise","private_pair","evaluation_results"):
            with self.assertRaises(TypeError): pair.estimate_oracle_pair(**dict(inputs,**{key:object()}))
        with self.assertRaises(TypeError): pair.estimate_oracle_pair(object())
        for identity in ({"pair_id":""},{"pair_id":"p","beta":"hidden"},
                         {"pair_id":"p","protocol_hash":"wrong"}):
            with self.assertRaises(ValueError):
                pair.estimate_oracle_pair(**inputs,pair_identity=identity)


if __name__ == "__main__":
    unittest.main()
