"""ATE protocol tests; no real model inference in this unit suite."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr, ndtri

from evaluate import evaluate_context, summarize
from make_figures import build_figure
from make_table import ate_table, read_results
from pfn_social.intervals import mixture_interval, component_effect_posterior
from pfn_social.phase6_sparse_classical import pack
from scripts import e1_5_star_ate as adapter


def fake_predictor(name, model, inputs, seed):
    # Fixture with deliberately simple means; not an experimental PFN result.
    mu = np.tile([0., 2., 1., 3.], (adapter.N, 1))
    return pack(mu, n=adapter.N)


def fake_interval(model, **kwargs):
    return dict(ate_lower=np.array([-10., -10., -10.]), ate_upper=np.array([10., 10., 10.]),
                interval_level=np.array(.95), interval_kind=np.array('frozen_J_marginal_posterior_credible'))


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.world, cls.truth = adapter.build_world(900)
        cls.inputs = adapter.make_pair(cls.base, cls.world, 900, 0, adapter.DEFAULT_SEED)[0][0]

    def predict(self, name='j', **kwargs):
        return adapter.predict_record(name, None, self.inputs, adapter.context_id(900, 0, 0), 900, 0, 0,
                                      predictor=kwargs.get('predictor', fake_predictor),
                                      interval_predictor=kwargs.get('interval_predictor', fake_interval))

    def test_all_node_ate_truth(self):
        mu = np.zeros((200, 4))
        mu[0] = [0., 2., 4., 10.]
        np.testing.assert_allclose(adapter.ate_truth(mu), [.02, .03, .05])
        with self.assertRaises(ValueError):
            adapter.ate_truth(mu[:50])

    def test_frozen_scales_and_unique_pair_ids(self):
        for cohort, count in [('smoke', 2), ('pilot', 40), ('pilot_fresh', 40), ('formal', 400)]:
            plan = adapter.frozen_plan(cohort)
            ids = {adapter.context_id(m, r, c) for m in plan['mechanism_ids'] for r in plan['pair_repeat_ids'] for c in (0, 1)}
            self.assertEqual(len(ids), count)
            self.assertEqual(plan['seed'], 20261024)
        self.assertEqual(adapter.frozen_plan('formal')['mechanism_ids'], list(range(1000, 1020)))

    def test_pilot_fresh_mapping_disjoint_and_legacy_unchanged(self):
        fresh = adapter.frozen_plan('pilot_fresh')
        self.assertEqual(fresh['mechanism_ids'], [904, 905, 906, 907])
        self.assertEqual(fresh['pair_repeat_ids'], [0, 1, 2, 3, 4])
        self.assertEqual(fresh['center_order'], [0, 1])
        self.assertEqual(fresh['methods'], ['j', 'kernel-old', 'kernel-new'])
        legacy = {'smoke': ([900], [0]), 'pilot': (list(range(900, 904)), list(range(5))),
                  'formal': (list(range(1000, 1020)), list(range(10)))}
        for name, (mids, rids) in legacy.items():
            plan = adapter.frozen_plan(name)
            self.assertEqual((plan['mechanism_ids'], plan['pair_repeat_ids']), (mids, rids))
            self.assertTrue(set(fresh['mechanism_ids']).isdisjoint(mids))
        contexts = [adapter.context_id(m, r, c) for m in fresh['mechanism_ids']
                    for r in fresh['pair_repeat_ids'] for c in fresh['center_order']]
        self.assertEqual(len(set(contexts)), 40)
        self.assertEqual(len(contexts)*len(fresh['methods']), 120)
        self.assertEqual(len(contexts)*3, 120)  # Three J marginal effect intervals per context.

    def test_fresh_ids_change_both_mechanism_and_pair_random_streams(self):
        from pfn_social.phase5_large_network import seed_entropy
        from pfn_social.phase6_limited_outcomes import configuration
        from scripts.e1_5_star import pair_rng
        config = configuration(0)
        self.assertEqual((config.split, config.split_seed), ('dev', 331029))
        for mid in range(904, 908):
            self.assertNotEqual(seed_entropy(config, mid, 'coefficient'), seed_entropy(config, 900, 'coefficient'))
            first, entropy = pair_rng(20261024, mid, 0)
            self.assertEqual(entropy, 20261024+mid)
            old, _ = pair_rng(20261024, 900, 0)
            self.assertFalse(np.array_equal(first.integers(0, 2**31, 100), old.integers(0, 2**31, 100)))
            for rid in range(1, 5):
                _, entropy = pair_rng(20261024, mid, rid)
                self.assertEqual(entropy, [20261024, 2026102501, mid, rid])

    def test_shared_seed_pair_and_method_input_copies(self):
        first, entropy = adapter.make_pair(self.base, self.world, 900, 1, adapter.DEFAULT_SEED)
        second, again = adapter.make_pair(self.base, self.world, 900, 1, adapter.DEFAULT_SEED)
        self.assertEqual(entropy, [20261024, 2026102501, 900, 1])
        self.assertEqual(entropy, again)
        np.testing.assert_array_equal(first[0]['observed_indices'], first[1]['observed_indices'])
        np.testing.assert_array_equal(first[0]['treatment'][1:], first[1]['treatment'][1:])
        for a, b in zip(first, second):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])

    def test_interval_fields(self):
        point, receipt = self.predict()
        self.assertEqual(point['ate_lower'].shape, (3,))
        self.assertEqual(point['ate_interval_supported'].tolist(), [True]*3)
        self.assertIsNone(receipt['failure'])
        self.assertIsNone(receipt['interval_failure'])
        self.assertEqual(receipt['interval_level'], .95)

    def test_baseline_unsupported_is_na_not_zero(self):
        for name in ('kernel-old', 'kernel-new'):
            point, receipt = self.predict(name)
            rows = evaluate_context(self.truth, point, receipt)
            self.assertTrue(np.isnan(point['ate_lower']).all())
            self.assertTrue(all(r['interval_requested'] == 0 and np.isnan(r['coverage_including_failures']) for r in rows))
            self.assertTrue(all(r['interval_status'] == 'unsupported' for r in rows))

    def test_interval_failure_remains_in_denominator(self):
        def fail(*args, **kwargs):
            raise FloatingPointError('Deliberate fixture failure')
        good, rg = self.predict()
        bad, rb = self.predict(interval_predictor=fail)
        rb['context_id'] = adapter.context_id(900, 0, 1)
        rows = evaluate_context(self.truth, good, rg)+evaluate_context(self.truth, bad, rb)
        for row in summarize(rows):
            self.assertEqual(row['interval_requested'], 2)
            self.assertEqual(row['interval_valid'], 1)
            self.assertEqual(row['covered'], 1)
            self.assertEqual(row['coverage'], 1.)
            self.assertEqual(row['coverage_including_failures'], .5)
            self.assertEqual(row['interval_failure_contexts'], 1)

    def test_missing_bounds_and_point_failure_not_silently_unsupported(self):
        point, receipt = self.predict()
        del point['ate_lower']
        rows = evaluate_context(self.truth, point, receipt)
        self.assertTrue(all(r['interval_requested'] == 1 and r['interval_valid'] == 0 for r in rows))
        def fail(*args, **kwargs):
            raise RuntimeError('Point fixture failure')
        point, receipt = self.predict(predictor=fail)
        rows = evaluate_context(self.truth, point, receipt)
        self.assertTrue(all(r['point_requested'] == 1 and r['point_valid'] == 0 for r in rows))
        self.assertTrue(all(r['interval_requested'] == 1 for r in rows))
        self.assertTrue(np.isnan(point['ate_effects']).all())

    def test_input_mutation_stops_instead_of_becoming_model_failure(self):
        def mutate(name, model, inputs, seed):
            inputs['x'][0, 0] += 1
            return fake_predictor(name, model, inputs, seed)
        original = self.inputs['x'].copy()
        with self.assertRaises(adapter.InputMutationError):
            self.predict(predictor=mutate)
        np.testing.assert_array_equal(self.inputs['x'], original)

    def test_report_formula_and_partial_support(self):
        rows = []
        truth = np.zeros((200, 4))
        for mid, errors in [(900, [1., 3.]), (901, [-2., 0.])]:
            for rid, error in enumerate(errors):
                receipt = dict(context_id=adapter.context_id(mid, rid, 0), method='j', interval_level=.95,
                               ate_supported=[True]*3, ate_interval_supported=[True]*3, failure=None, interval_failure=None)
                point = dict(ate_effects=np.repeat(error, 3), ate_lower=np.repeat(-10., 3), ate_upper=np.repeat(10., 3))
                rows.extend(evaluate_context(truth, point, receipt))
        for row in ate_table(rows, 50):
            self.assertEqual(row['absolute_conditional_bias'], 1.5)
            self.assertAlmostEqual(row['variance_unbiased'], 2.)
            self.assertAlmostEqual(row['rmse'], np.sqrt(3.5))
            self.assertTrue(row['ranked'])
        rows[0].update(point_valid=0, prediction_failure=True)
        row = next(r for r in ate_table(rows, 50) if r['effect'] == 'direct')
        self.assertFalse(row['ranked'])
        self.assertEqual(row['mechanisms_used'], 1)


class RepeatMetadataTests(unittest.TestCase):
    def prediction(self, center, method='j', *, fail_point=False, fail_interval=False):
        truth = np.tile([0., 2., 1., 3.], (adapter.N, 1))
        inputs = dict(x=np.zeros((adapter.N, 20)))  # Fake predictors do not inspect a DGP.
        def fail(*args, **kwargs):
            raise RuntimeError('Synthetic metadata fixture failure')
        point, receipt = adapter.predict_record(
            method, None, inputs, adapter.context_id(900, 2, center), 900, 2, center,
            predictor=fail if fail_point else fake_predictor,
            interval_predictor=fail if fail_interval else fake_interval)
        return truth, point, receipt

    def assert_rows_same(self, before, after, ignored=()):
        self.assertEqual(len(before), len(after))
        for old, new in zip(before, after):
            self.assertEqual(set(old)-set(ignored), set(new)-set(ignored))
            for field in set(old)-set(ignored):
                if isinstance(old[field], (float, np.floating)):
                    np.testing.assert_equal(old[field], new[field], err_msg=field)
                else:
                    self.assertEqual(old[field], new[field], field)

    def test_receipt_repeat_two_preserves_encoded_context_ids(self):
        for center in (0, 1):
            truth, point, receipt = self.prediction(center)
            original = dict(receipt)
            rows = evaluate_context(truth, point, receipt)
            self.assertTrue(all(r['repeat'] == 4+center for r in rows))
            metadata = adapter.receipt_metadata(receipt)
            self.assertEqual(metadata, dict(repeat=2, pair_repeat_id=2, center_t=center))
            self.assertEqual(receipt, original)
            self.assertEqual(receipt['context_id'], f'f0_m900_r{4+center}_observed50')

    def test_metadata_only_changes_leave_metrics_statuses_and_summaries_unchanged(self):
        before, after = [], []
        for center in (0, 1):
            for method in adapter.METHODS:
                for bad_point, bad_interval in ((False, False), (True, False), (False, True)):
                    truth, point, receipt = self.prediction(
                        center, method, fail_point=bad_point, fail_interval=bad_interval)
                    rows = evaluate_context(truth, point, receipt)
                    corrected = [dict(r, **adapter.receipt_metadata(receipt)) for r in rows]
                    self.assert_rows_same(rows, corrected, ('repeat', 'pair_repeat_id', 'center_t'))
                    before.extend(rows);after.extend(corrected)
        for name in ('summary', 'summary_by_family', 'summary_by_mechanism'):
            self.assert_rows_same(adapter.evaluation_tables(before)[name],
                                  adapter.evaluation_tables(after)[name])
        self.assert_rows_same(ate_table(before, 50), ate_table(after, 50))

    def test_missing_invalid_and_conflicting_receipt_metadata_rejected(self):
        _, _, good = self.prediction(0)
        for field in ('mechanism_id', 'pair_repeat_id', 'center_t'):
            for value in (None, -1, True, 2.0, '2'):
                with self.subTest(field=field, value=value):
                    bad = dict(good);bad[field] = value
                    with self.assertRaisesRegex(ValueError, 'receipt|Receipt'):
                        adapter.receipt_metadata(bad)
            bad = dict(good);del bad[field]
            with self.assertRaisesRegex(ValueError, field):
                adapter.receipt_metadata(bad)
        for changes in ({'center_t': 2}, {'center_t': 1}, {'pair_repeat_id': 3},
                        {'mechanism_id': 901}, {'context_id': 'f0_m900_r5_observed50'},
                        {'context_id': None}, {'method': 'unknown'}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, 'receipt|Receipt'):
                    adapter.receipt_metadata(dict(good, **changes))
        with tempfile.TemporaryDirectory(prefix='star_repeat_rejected_') as temp:
            bad = dict(good);del bad['pair_repeat_id']
            with self.assertRaisesRegex(ValueError, 'pair_repeat_id'):
                adapter.score_saved(Path(temp), [bad])  # Reject before opening arrays.

    def test_actual_overall_and_center_csv_export_matches_execution_receipts(self):
        with tempfile.TemporaryDirectory(prefix='star_repeat_export_') as temp:
            output = Path(temp)
            for child in ('truth', 'predictions'):
                (output/child).mkdir()
            receipts, generic = [], []
            for center in (0, 1):
                for method in adapter.METHODS:
                    truth, point, receipt = self.prediction(center, method)
                    np.savez_compressed(output/'truth'/'f0_m900.npz', arm_means=truth)
                    np.savez_compressed(output/'predictions'/
                                        (receipt['context_id']+'__'+method+'.npz'), **point)
                    receipts.append(receipt)
                    generic.extend(evaluate_context(truth, point, receipt))
            adapter.write_json(output/'execution.json', dict(receipts=receipts))
            adapter.write_views(output, dict(receipts=receipts))
            index = {(r['context_id'], r['method']): r for r in
                     json.loads((output/'execution.json').read_text())['receipts']}
            for center in (None, 0, 1):
                directory = output if center is None else output/'report_views'/f'center_t{center}'
                with (directory/'evaluation'/'context_metrics.csv').open(encoding='utf-8', newline='') as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 18 if center is None else 9)
                for row in rows:
                    receipt = index[row['context_id'], row['method']]
                    self.assertEqual(int(row['repeat']), receipt['pair_repeat_id'])
                    self.assertEqual(int(row['pair_repeat_id']), 2)
                    self.assertEqual(int(row['center_t']), receipt['center_t'])
                    self.assertEqual(row['context_id'], adapter.context_id(900, 2, receipt['center_t']))
                    if center is not None:
                        self.assertEqual(int(row['center_t']), center)
                original = generic if center is None else [r for r in generic if index[r['context_id'], r['method']]['center_t'] == center]
                # Exercise strict CSV comparison for every exported numeric/status field.
                for name, wanted in adapter.evaluation_tables(original).items():
                    if name == 'context_metrics':
                        wanted = [dict(r, **adapter.receipt_metadata(index[r['context_id'], r['method']])) for r in wanted]
                    adapter._check_csv(directory/'evaluation'/(name+'.csv'), wanted)


class IntervalMathTests(unittest.TestCase):
    def test_single_gaussian_quantiles_and_mixture_cdf(self):
        lo, hi = mixture_interval(np.array([[1., 2., 3.]]), np.array([[4., 4., 4.]]), np.array([1.]))
        np.testing.assert_allclose(lo, np.array([1., 2., 3.])+2*ndtri(.025))
        np.testing.assert_allclose(hi, np.array([1., 2., 3.])+2*ndtri(.975))
        means, var, weights = np.array([[-2.], [3.]]), np.array([[1.], [4.]]), np.array([.3, .7])
        lo, hi = mixture_interval(means, var, weights)
        for endpoint, probability in [(lo, .025), (hi, .975)]:
            self.assertAlmostEqual(float((weights[:, None]*ndtr((endpoint-means)/np.sqrt(var))).sum()), probability)

    def test_ate_variance_preserves_shared_node_covariance(self):
        f, q = np.ones((1, 1)), np.zeros((2, 4, 1))
        q[:, 2, 0] = 1.
        q[:, 3, 0] = 2.
        means, variance, _ = component_effect_posterior(f, q, np.array([0.]), np.array([0.]), np.eye(1), np.array([1.]))
        # Both nodes share one coefficient: averaging does NOT divide variance by 2.
        np.testing.assert_allclose(means, 0.)
        np.testing.assert_allclose(variance[0], [.5, .5, 2.])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_temp = tempfile.TemporaryDirectory(prefix='star_ate_fixture_')
        cls.addClassCleanup(cls.fixture_temp.cleanup)
        cls.fixture = Path(cls.fixture_temp.name)/'run'
        with patch.object(adapter, 'load_methods', return_value=dict.fromkeys(adapter.METHODS)), \
             patch.object(adapter, 'predict_method', side_effect=fake_predictor), \
             patch.object(adapter, 'predict_intervals', side_effect=fake_interval):
            adapter.run_experiment(cls.fixture)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='star_ate_acceptance_')
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)/'run'
        shutil.copytree(self.fixture, self.output)

    def resign(self, name):
        path = self.output/'run_manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['output_sha256'][name] = adapter.sha256(self.output/name)
        adapter.write_json(path, manifest)

    def test_full_acceptance_and_upstream_readers(self):
        report = adapter.validate_result(self.output)
        self.assertEqual((report['contexts'], report['method_contexts'], report['evaluation_rows']), (2, 6, 18))
        for center in (0, 1):
            rows, _ = read_results(self.output/'report_views'/f'center_t{center}')
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(np.isnan(r['variance_unbiased']) for r in ate_table(rows, 50)))
        rows, _ = read_results(self.output)
        fig, source = build_figure(rows)
        import matplotlib.pyplot as plt
        plt.close(fig)
        self.assertTrue(source)

    def test_receipt_pair_metadata_rejected_by_strict_acceptance(self):
        original = (self.output/'execution.json').read_text(encoding='utf-8')
        for field, value in [('pair_repeat_id', None), ('pair_repeat_id', 2),
                             ('pair_repeat_id', True), ('center_t', None), ('center_t', 2)]:
            with self.subTest(field=field, value=value):
                execution = json.loads(original)
                if value is None:
                    del execution['receipts'][0][field]
                else:
                    execution['receipts'][0][field] = value
                adapter.write_json(self.output/'execution.json', execution)
                self.resign('execution.json')  # Only this disposable fake artifact.
                with self.assertRaisesRegex(ValueError, 'receipt|Receipt'):
                    adapter.validate_result(self.output)

    def test_exported_pair_metadata_rejected_by_strict_acceptance(self):
        for name in ('evaluation/context_metrics.csv',
                     'report_views/center_t0/evaluation/context_metrics.csv'):
            original = (self.output/name).read_bytes()
            for field in ('repeat', 'pair_repeat_id', 'center_t'):
                for missing in (False, True):
                    with self.subTest(path=name, field=field, missing=missing):
                        with (self.output/name).open(encoding='utf-8', newline='') as handle:
                            rows = list(csv.DictReader(handle))
                        if missing:
                            for row in rows:
                                del row[field]
                        else:
                            rows[0][field] = '999'
                        adapter.write_csv(self.output/name, rows)
                        self.resign(name)  # Test that hashes alone cannot hide invalid metadata.
                        with self.assertRaisesRegex(ValueError, 'CSV'):
                            adapter.validate_result(self.output)
                        (self.output/name).write_bytes(original)
                        self.resign(name)

    def test_hash_tampering(self):
        with (self.output/'execution.json').open('a') as handle:
            handle.write(' ')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            adapter.validate_result(self.output)

    def test_missing_method_receipt_even_if_rehashed(self):
        path = self.output/'execution.json'
        execution = json.loads(path.read_text())
        execution['receipts'].pop()
        adapter.write_json(path, execution)
        self.resign('execution.json')
        with self.assertRaisesRegex(ValueError, 'method-contexts'):
            adapter.validate_result(self.output)

    def test_wrong_shared_input_even_if_rehashed(self):
        name = 'factual/'+adapter.context_id(900, 0, 1)+'.npz'
        with np.load(self.output/name, allow_pickle=False) as saved:
            inputs = {k: saved[k] for k in saved.files}
        inputs['treatment'][1] = 1-inputs['treatment'][1]
        np.savez_compressed(self.output/name, **inputs)
        self.resign(name)
        with self.assertRaisesRegex(ValueError, 'Factual/pair mismatch'):
            adapter.validate_result(self.output)

    def test_wrong_csv_even_if_rehashed(self):
        name = 'evaluation/context_metrics.csv'
        with (self.output/name).open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        rows[0]['covered'] = '99'
        adapter.write_csv(self.output/name, rows)
        self.resign(name)
        with self.assertRaisesRegex(ValueError, 'CSV'):
            adapter.validate_result(self.output)

    def test_failure_artifacts_retained_and_counted(self):
        destination = Path(self.temporary.name)/'failed_calls'
        def fail(*args, **kwargs):
            raise RuntimeError('Synthetic failure, do not retry')
        with patch.object(adapter, 'load_methods', return_value=dict.fromkeys(adapter.METHODS)), \
             patch.object(adapter, 'predict_method', side_effect=fail), \
             patch.object(adapter, 'predict_intervals', side_effect=fail):
            adapter.run_experiment(destination)
        report = adapter.validate_result(destination)
        self.assertEqual(report['point_failure_calls'], 6)
        self.assertEqual(report['interval_failure_calls'], 2)
        with (destination/'evaluation/summary.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.assertEqual(row['point_requested'], '2')
            self.assertEqual(row['point_valid'], '0')
            if row['method'] == 'j':
                self.assertEqual(row['coverage_including_failures'], '0.0')
                self.assertEqual(row['interval_requested'], '2')
            else:
                self.assertEqual(row['coverage_including_failures'], 'nan')

    def test_existing_directory_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            adapter.run_experiment(self.output)


if __name__ == '__main__':
    unittest.main()
