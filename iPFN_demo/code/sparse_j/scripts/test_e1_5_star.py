"""Small checks of the paired design and estimands; no model inference."""
import unittest
import csv
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from scripts.e1_5_star import (DEFAULT_SEED, N, M, ARMS, EFFECTS, build_world, effects, make_pair,
                               pair_rng, score, sha256, verify_assets, source_snapshot, write_csv, write_json)
from scripts.e1_5_star_analysis import (support_reason, summarize, validate_run, STRUCTURAL,
                                       ASSIGNMENT, SAMPLING, COVERED)
from scripts.e1_5_star_resources import memory_trend
from pfn_social.phase3_hierarchical import structural_outcome


class StarDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.world, cls.truth = build_world(0)

    def test_star_and_truth(self):
        a = self.world.adjacency
        np.testing.assert_array_equal(a, a.T)
        np.testing.assert_array_equal(np.diag(a), 0)
        np.testing.assert_array_equal(a.sum(1), [199] + [1]*199)
        self.assertEqual(self.truth.shape, (200, 4))
        e = effects(self.truth)
        np.testing.assert_allclose(e[:, 2], e[:, 0] + e[:, 1], atol=1e-12)

    def test_pair_shares_noise_assignment_and_mask(self):
        pair, _ = make_pair(self.base, self.world, 0, 0, DEFAULT_SEED)
        left, right = pair
        np.testing.assert_array_equal(left['observed_indices'], right['observed_indices'])
        np.testing.assert_array_equal(left['treatment'][1:], right['treatment'][1:])
        residuals = []
        for inputs in pair:
            ids = inputs['observed_indices']
            residuals.append(inputs['outcomes_observed'] - structural_outcome(self.world, inputs['treatment'])[ids])
        np.testing.assert_allclose(*residuals, atol=5e-7, rtol=0)

    def test_repeat_zero_matches_legacy_and_other_repeats_differ(self):
        legacy = np.random.default_rng(DEFAULT_SEED)
        rng, entropy = pair_rng(DEFAULT_SEED, 0, 0)
        self.assertEqual(entropy, DEFAULT_SEED)
        np.testing.assert_array_equal(legacy.integers(0, 2**31, 100), rng.integers(0, 2**31, 100))
        first, _ = pair_rng(DEFAULT_SEED, 0, 1)
        again, _ = pair_rng(DEFAULT_SEED, 0, 1)
        other, _ = pair_rng(DEFAULT_SEED, 0, 2)
        a, b = first.integers(0, 2**31, 100), other.integers(0, 2**31, 100)
        np.testing.assert_array_equal(a, again.integers(0, 2**31, 100))
        self.assertFalse(np.array_equal(a, b))

    def test_support_follows_hub_and_uses_only_observed_outcomes(self):
        pair, _ = make_pair(self.base, self.world, 0, 0, DEFAULT_SEED)
        for center_t, inputs in enumerate(pair):
            metrics, support = score(inputs, self.truth, self.truth, 0, 0, center_t)
            self.assertTrue(all(row['arm_rmse'] == 0 for row in metrics))
            leaves = [row for row in support if row['category'] == 'leaf']
            self.assertEqual(sum(row['observed_count'] for row in support), 50)
            for row in leaves:
                if int(row['arm'][1]) != center_t:
                    self.assertEqual(row['observed_count'], 0)
                    self.assertFalse(row['empirical_support'])

    def test_all_support_reasons_and_hub_observation_are_distinct(self):
        self.assertEqual(support_reason('leaf', 0, 1, 0, 0, 0), STRUCTURAL)
        self.assertEqual(support_reason('leaf', 1, 0, 0, 0, 0), ASSIGNMENT)
        self.assertEqual(support_reason('leaf', 1, 0, 0, 1, 0), SAMPLING)
        self.assertEqual(support_reason('leaf', 1, 0, 0, 1, 1), COVERED)
        self.assertEqual(support_reason('center', 1, 0, 0, 0, 0), STRUCTURAL)
        with self.assertRaises(ValueError):
            support_reason('leaf', 0, 1, 0, 1, 0)
        pair, _ = make_pair(self.base, self.world, 0, 0, DEFAULT_SEED)
        inputs = pair[0]
        inputs['observed_indices'] = np.arange(50)
        _, support = score(inputs, self.truth, self.truth, 0, 0, 0)
        self.assertTrue(all(r['center_observed'] for r in support))
        self.assertEqual(sum(r['observed_count'] for r in support if r['category'] == 'center'), 1)

    def test_asset_tampering_is_not_bypassed(self):
        original = sha256
        with patch('scripts.e1_5_star.sha256', side_effect=lambda p: '0'*64 if Path(p).name == 'iPFN_J.pt' else original(p)):
            with self.assertRaisesRegex(ValueError, 'Integrity mismatch'):
                verify_assets()

    def test_original_smoke_factual_inputs_and_truth_unchanged(self):
        from scripts.e1_5_star import ROOT
        directory = ROOT.parents[1]/'results/e1_5_smoke_20261025'
        if not directory.is_dir():
            self.skipTest('Historical local smoke artifacts are not distributed with source')
        pair, _ = make_pair(self.base, self.world, 0, 0, DEFAULT_SEED)
        for center_t, inputs in enumerate(pair):
            with np.load(directory/f'm0_r0_t{center_t}_input.npz', allow_pickle=False) as saved:
                for key, value in inputs.items():
                    np.testing.assert_array_equal(value, saved[key])
            with np.load(directory/f'm0_r0_t{center_t}_evaluation.npz', allow_pickle=False) as saved:
                np.testing.assert_allclose(self.truth, saved['truth_mu'], atol=1e-12)


class SummaryTests(unittest.TestCase):
    def test_macos_memory_resources_are_explicitly_unavailable(self):
        from scripts.e1_5_star import task_memory_usage
        with patch('scripts.e1_5_star.sys.platform', 'darwin'), \
                patch('scripts.e1_5_star.memory_usage') as native:
            resources = task_memory_usage()
        native.assert_not_called()
        self.assertEqual(resources['status'], 'unavailable')
        self.assertIn('macOS', resources['reason'])
        self.assertEqual({k: v for k, v in resources.items() if k not in {'status', 'reason'}},
                         dict(peak_working_set_bytes=None, working_set_bytes=None,
                              private_bytes=None, process_pid=None))
        self.assertEqual(json.loads(json.dumps(resources)), resources)

    def test_windows_memory_resources_preserve_native_result(self):
        from scripts.e1_5_star import task_memory_usage
        measured = dict(peak_working_set_bytes=200, working_set_bytes=100,
                        private_bytes=150, process_pid=123)
        with patch('scripts.e1_5_star.sys.platform', 'win32'), \
                patch('scripts.e1_5_star.memory_usage', return_value=measured) as native:
            self.assertIs(task_memory_usage(), measured)
        native.assert_called_once_with()

    def test_native_memory_errors_are_not_hidden(self):
        from scripts.e1_5_star import task_memory_usage
        with patch('scripts.e1_5_star.sys.platform', 'win32'), \
                patch('scripts.e1_5_star.memory_usage', side_effect=RuntimeError('native failure')):
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                task_memory_usage()

    def test_mechanism_blocks_equal_weights_and_no_dropping_uncovered_arms(self):
        support, metrics = [], []
        for mid in (1, 3):
            for rid in (0, 1):
                for c in (0, 1):
                    for category in ('center', 'leaf'):
                        metrics.append(dict(mechanism_id=mid, pair_repeat_id=rid, center_t=c,
                                            category=category, arm_rmse=float(mid)))
                    for arm in ARMS:
                        blocked = c != int(arm[1])
                        support.append(dict(mechanism_id=mid, pair_repeat_id=rid, center_t=c,
                                            category='leaf', arm=arm, empirical_support=False,
                                            support_reason=STRUCTURAL if blocked else ASSIGNMENT,
                                            arm_rmse=10.+mid+rid if blocked else 10.))
        _, mechanisms, _, summary = summarize(support, metrics, [1, 3], [0, 1])
        self.assertEqual([r['primary_delta'] for r in mechanisms], [1.5, 3.5])
        self.assertEqual(summary['mean_delta'], 2.5)
        self.assertEqual(summary['independent_units'], 2)
        self.assertEqual(summary['ci95'], [1.5, 3.5])
        self.assertEqual(summary['bootstrap_seed'], 2026102502)
        self.assertEqual(summary['bootstrap_draws'], 10000)
        self.assertEqual(summary, summarize(support, metrics, [1, 3], [0, 1])[3])
        with self.assertRaises(ValueError):
            summarize(support[:-1], metrics, [1, 3], [0, 1])

    def test_memory_growth_diagnostic(self):
        stable = [dict(resources=dict(private_bytes=100*1024**2)) for _ in range(40)]
        growth = [dict(resources=dict(private_bytes=(100+2*i)*1024**2)) for i in range(40)]
        self.assertEqual(memory_trend(stable)['status'], 'no_sustained_growth_detected_at_this_scale')
        self.assertEqual(memory_trend(growth)['status'], 'investigate_growth')

    @unittest.skipUnless(os.name == 'nt', 'Resource measurement requires Windows')
    def test_native_windows_resource_measurements(self):
        from scripts.e1_5_star_resources import memory_usage, physical_memory, cpu_seconds
        usage = memory_usage()
        self.assertGreater(usage['private_bytes'], 0)
        self.assertGreaterEqual(usage['peak_working_set_bytes'], usage['working_set_bytes'])
        self.assertGreater(physical_memory()['total_bytes'], 0)
        self.assertGreaterEqual(cpu_seconds(-1)['cpu_seconds'], 0)


def acceptance_fixture(directory):
    """Synthetic prediction fixture, NOT a PFN experiment or scientific result."""
    base, world, truth = build_world(0)
    from dataclasses import asdict
    assets = verify_assets()
    manifest = dict(schema_version=2, status='complete', purpose='smoke', n=N, m=M, family_id=0, seed=DEFAULT_SEED,
                    mechanism_ids=[0], pair_repeat_ids=[0, 1], arm_order=list(ARMS), effect_order=list(EFFECTS),
                    cpu_threads=1, assets_sha256=assets,
                    model_identity=dict(sha256=assets['models/iPFN_J.pt']),
                    model_config=dict(architecture='J', raw_p=20, hidden=64),
                    source_sha256=source_snapshot(directory),
                    mechanisms=[dict(mechanism_id=0, config=asdict(base.config),
                                     parent_mechanism_sha256=base.mechanism_sha256, noise_sd=base.setting.noise_sd)],
                    pairs=[], tasks=[])
    metrics, support = [], []
    for rid in (0, 1):
        pair, entropy = make_pair(base, world, 0, rid, DEFAULT_SEED)
        manifest['pairs'].append(dict(mechanism_id=0, pair_repeat_id=rid, rng_entropy=entropy, bit_generator='PCG64'))
        for c, inputs in enumerate(pair):
            pred = truth + .1*c  # Deliberately synthetic; tests exercise serialization/acceptance only.
            name = f'm0_r{rid}_t{c}'
            np.savez_compressed(directory / (name+'_input.npz'), **inputs)
            np.savez_compressed(directory / (name+'_evaluation.npz'), truth_mu=truth, pred_mu=pred,
                                true_effects=effects(truth), pred_effects=effects(pred),
                                arm_errors=pred-truth, effect_errors=effects(pred)-effects(truth))
            met, sup = score(inputs, truth, pred, 0, rid, c)
            metrics.extend(met)
            support.extend(sup)
            manifest['tasks'].append(dict(mechanism_id=0, pair_repeat_id=rid, center_t=c, task_id=name,
                                          observed_indices=inputs['observed_indices'].tolist(),
                                          center_observed=bool(0 in inputs['observed_indices'])))
    write_csv(directory/'metrics.csv', metrics)
    write_csv(directory/'support.csv', support)
    contrasts, primary, secondary, summary = summarize(support, metrics, [0], [0, 1])
    write_csv(directory/'paired_contrasts.csv', contrasts)
    write_csv(directory/'mechanism_primary.csv', primary)
    write_csv(directory/'mechanism_metrics.csv', secondary)
    write_json(directory/'summary.json', summary)
    manifest['output_sha256'] = {p.name: sha256(p) for p in directory.iterdir()}
    write_json(directory/'manifest.json', manifest)
    return manifest


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='e1_5_unit_')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.manifest = acceptance_fixture(self.directory)

    def resign(self, filename):
        # Simulate internally inconsistent data even when its file hash is updated.
        self.manifest['output_sha256'][filename] = sha256(self.directory/filename)
        write_json(self.directory/'manifest.json', self.manifest)

    def change_archive(self, name, mutate):
        path = self.directory/name
        with np.load(path, allow_pickle=False) as saved:
            values = {k: saved[k] for k in saved.files}
        mutate(values)
        np.savez_compressed(path, **values)
        self.resign(name)

    def test_valid_artifacts_and_repeat_truth(self):
        result = validate_run(self.directory)
        self.assertEqual(result['tasks'], 4)
        self.assertEqual(result['support_rows'], 32)
        self.assertTrue(result['passed'])

    def test_hash_failure(self):
        with (self.directory/'metrics.csv').open('a') as handle:
            handle.write('corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_run(self.directory)

    def test_duplicate_task_failure(self):
        self.manifest['tasks'].append(self.manifest['tasks'][0])
        write_json(self.directory/'manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'Duplicate task'):
            validate_run(self.directory)

    def test_incomplete_status_failure(self):
        self.manifest['status'] = 'running'
        write_json(self.directory/'manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'completion state'):
            validate_run(self.directory)

    def test_frozen_protocol_mismatch_failure(self):
        self.manifest['purpose'] = 'timing_pilot'
        write_json(self.directory/'manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'Frozen protocol'):
            validate_run(self.directory)

    def test_nonfinite_prediction_failure(self):
        self.change_archive('m0_r0_t0_evaluation.npz', lambda v: v['pred_mu'].__setitem__((0, 0), np.nan))
        with self.assertRaisesRegex(ValueError, 'Invalid prediction'):
            validate_run(self.directory)

    def test_duplicate_observation_index_failure(self):
        self.change_archive('m0_r0_t0_input.npz', lambda v: v['observed_indices'].__setitem__(0, v['observed_indices'][1]))
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_run(self.directory)

    def test_pair_random_quantity_failure(self):
        self.change_archive('m0_r0_t1_input.npz', lambda v: v['outcomes_observed'].__setitem__(0, v['outcomes_observed'][0]+1))
        with self.assertRaisesRegex(ValueError, 'shared random quantities'):
            validate_run(self.directory)

    def test_truth_inconsistency_failure(self):
        self.change_archive('m0_r1_t0_evaluation.npz', lambda v: v['truth_mu'].__setitem__((0, 0), v['truth_mu'][0, 0]+1))
        with self.assertRaisesRegex(ValueError, 'mechanism truth'):
            validate_run(self.directory)

    def test_csv_mismatch_even_with_valid_hash(self):
        path = self.directory/'metrics.csv'
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        rows[0]['direct_rmse'] = '99'
        write_csv(path, rows)
        self.resign('metrics.csv')
        with self.assertRaisesRegex(ValueError, 'metrics.csv'):
            validate_run(self.directory)

    def test_bootstrap_summary_tampering(self):
        path = self.directory/'summary.json'
        summary = json.loads(path.read_text())
        summary['bootstrap_seed'] += 1
        write_json(path, summary)
        self.resign('summary.json')
        with self.assertRaisesRegex(ValueError, 'bootstrap summary'):
            validate_run(self.directory)

    def test_existing_output_rejected_before_inference(self):
        from scripts.e1_5_star import main
        with patch('sys.argv', ['e1_5_star.py', '--output', str(self.directory)]), self.assertRaises(SystemExit) as caught:
            main()
        self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
