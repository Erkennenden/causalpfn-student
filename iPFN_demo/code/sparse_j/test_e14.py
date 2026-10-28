"""Numerical/contract checks for the RGG extension (python -m unittest -v test_e14)."""
from dataclasses import replace
import itertools
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from benchmark import configuration, load_methods, predict_method
from evaluate import evaluate_context
from pfn_social.e14_rgg import SCENARIOS, generate_rgg, radius_graph
from pfn_social.e14_oracle import design, predict_oracle
from pfn_social.intervals import predict_intervals
from pfn_social.phase3_hierarchical import structural_outcome
from pfn_social.phase6_limited_outcomes import make_observed_draw, validate_factual


class RGGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.limit = threadpool_limits(1)
        cls.model = load_methods(['j'], configuration())['j']

    @classmethod
    def tearDownClass(cls):
        cls.limit.restore_original_limits()

    def test_geometric_boundary_and_no_repair(self):
        positions = np.array([[0, 0], [.3, .4], [1, 1]])
        a = radius_graph(positions, .5)
        np.testing.assert_array_equal(a, [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        with self.assertRaises(RuntimeError):
            generate_rgg(replace(SCENARIOS[0], radius=1e-12, max_attempts=1), 0)

    def test_all_scenarios_reproducible_and_exact_radius(self):
        for config in SCENARIOS:
            first, pos, metadata = generate_rgg(config, 0)
            second, pos2, metadata2 = generate_rgg(config, 0)
            np.testing.assert_array_equal(pos, pos2)
            np.testing.assert_array_equal(first.parent.truth, second.parent.truth)
            np.testing.assert_array_equal(first.parent.adjacency, radius_graph(pos, config.radius))
            self.assertEqual(metadata, metadata2)
            self.assertGreater(metadata['min_degree'], 0)

    def test_truth_by_enumerating_neighbor_assignments(self):
        for config in (SCENARIOS[0], SCENARIOS[3]):
            mechanism, _, _ = generate_rgg(config, 0)
            m = mechanism.parent
            for i in np.argsort(m.adjacency.sum(1))[:2]:
                neighbors = np.flatnonzero(m.adjacency[i])
                self.assertLessEqual(len(neighbors), 12)
                for own, arm in itertools.product((0, 1), repeat=2):
                    responses = []
                    for bits in itertools.product((0, 1), repeat=len(neighbors)):
                        if int(sum(bits) > len(neighbors)//2) != arm:
                            continue
                        treatment = np.zeros(200)
                        treatment[neighbors] = bits
                        treatment[i] = own
                        responses.append(structural_outcome(m, treatment)[i])
                    self.assertAlmostEqual(np.mean(responses), m.truth[i, 2*own+arm], places=11)

    def test_mask_nesting_and_factual_storage(self):
        m, _, _ = generate_rgg(SCENARIOS[0], 0)
        rows, _ = make_observed_draw(m, 0)
        repeat, _ = make_observed_draw(m, 1)
        for row, other in zip(rows, repeat):
            self.assertTrue(validate_factual(row['batch']))
            np.testing.assert_array_equal(row['batch']['observed_indices'], other['batch']['observed_indices'])
            self.assertEqual(row['batch']['adjacency'].shape, (200, 200))
        np.testing.assert_array_equal(rows[0]['batch']['observed_indices'], rows[1]['batch']['observed_indices'][:20])
        self.assertFalse(torch.equal(rows[0]['batch']['treatment'], repeat[0]['batch']['treatment']))

    def test_oracle_design_matches_structural_dgp(self):
        m, _, _ = generate_rgg(SCENARIOS[3], 0)
        m = m.parent
        a, h, beta = m.adjacency, m.h, m.coefficients
        t = np.random.default_rng(5).binomial(1, .5, 200)
        degree = a.sum(1)
        f = design(h, a@h/degree[:, None], t, a@t/degree, a@(t[:, None]*h)/degree[:, None])
        np.testing.assert_allclose(f@beta, structural_outcome(m, t), atol=1e-12)

    def test_predictor_effects_and_truth_separation(self):
        m, _, _ = generate_rgg(SCENARIOS[0], 2)
        rows, _ = make_observed_draw(m, 1)
        inputs = {k: v.numpy().copy() for k, v in rows[1]['batch'].items()}
        original_truth = m.parent.truth.copy()
        p1 = predict_method('j', self.model, inputs, 1)
        m.parent.truth[:] = 1e8
        p2 = predict_method('j', self.model, inputs, 1)
        np.testing.assert_array_equal(p1['arm_means'], p2['arm_means'])
        m.parent.truth[:] = original_truth
        np.testing.assert_allclose(p1['ite_effects'][:, 2], p1['ite_effects'][:, 0]+p1['ite_effects'][:, 1], atol=1e-12)
        np.testing.assert_allclose(p1['ate_effects'], p1['ite_effects'].mean(0), atol=1e-12)
        ci = predict_intervals(self.model, **inputs)
        self.assertTrue(np.all(ci['ate_upper'] > ci['ate_lower']))
        oracle = predict_oracle(true_h=m.parent.h, **inputs)
        self.assertTrue(np.all(oracle['ate_upper'] > oracle['ate_lower']))

    def test_failed_intervals_stay_in_denominator(self):
        truth = np.tile([0., 1., 2., 3.], (200, 1))
        receipt = dict(context_id='f7_m0_r0_observed50', method='j', ate_supported=[True]*3,
                       ate_interval_supported=[True]*3, interval_level=.95,
                       interval_failure={'message': 'intentional test failure'})
        rows = evaluate_context(truth, {'ate_effects': [2., 1., 3.],
            'ate_lower': [-100.]*3, 'ate_upper': [100.]*3}, receipt)
        for row in rows:
            self.assertEqual(row['interval_requested'], 1)
            self.assertEqual(row['interval_valid'], 0)
            self.assertEqual(row['coverage_including_failures'], 0)


if __name__ == '__main__':
    unittest.main()
