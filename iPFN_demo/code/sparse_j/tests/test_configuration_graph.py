from dataclasses import asdict, replace
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from pfn_social.configuration_graph import configuration_graph, degree_sequence, degree_statistics, _join_components
from pfn_social.phase3_hierarchical import SETTINGS, CONFIGURATION_BASE_IDS, Setting, connected_graph, _hash_arrays
from pfn_social.phase5_large_network import make_repeat
from pfn_social.phase6_limited_outcomes import BASE_IDS, configuration as legacy_configuration, make_observed_draw
from pfn_social.e12_configuration import FAMILY_BASE_IDS, configuration, generate_mechanism as e12_mechanism


@pytest.mark.parametrize('n', [50, 100, 200])
@pytest.mark.parametrize('base', CONFIGURATION_BASE_IDS)
def test_registered_graph_invariants(n, base):
    setting = replace(SETTINGS[base], n=n)
    for seed in (0, 1, 20261028):
        a = connected_graph(setting, np.random.default_rng(seed))
        stats = degree_statistics(a)
        assert a.dtype == np.uint8 and a.shape == (n, n)
        assert stats['connected']
        assert stats['degree_mean'] == setting.degree
        assert stats['degree_sd'] == setting.degree_spread
        np.testing.assert_array_equal(np.sort(a.sum(1)),
            degree_sequence(n, setting.degree, setting.degree_spread))


def test_seed_reproducibility_and_variation():
    a, b, c = [configuration_graph(50, 4, 2, np.random.default_rng(s)) for s in (9, 9, 10)]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize('args', [(3, 2, 0), (51, 4, 1), (50, 4, -1),
    (50, 4, 3), (50, 49, 1), (50, 4.5, 0), (True, 4, 0), (10, 5, 3)])
def test_invalid_or_non_graphical_sequence_rejected(args):
    with pytest.raises(ValueError):
        degree_sequence(*args)


def test_component_join_preserves_degrees():
    graph = nx.disjoint_union_all([nx.cycle_graph(4), nx.complete_graph(5), nx.cycle_graph(6)])
    before = dict(graph.degree)
    _join_components(graph)
    assert nx.is_connected(graph)
    assert dict(graph.degree) == before
    assert not list(nx.selfloop_edges(graph))


def test_dense_and_cycle_boundary():
    for n, k in [(4, 2), (4, 3), (10, 2), (10, 9)]:
        a = configuration_graph(n, k, 0, np.random.default_rng(0))
        assert degree_statistics(a)['connected']
        assert np.all(a.sum(1) == k)


def test_registry_and_legacy_identity_unchanged():
    assert BASE_IDS == (0, 1, 3, 4, 5, 6, 7)
    assert list(asdict(Setting('old', 50, 'ws', 4, 1))) == [
        'name', 'n', 'graph', 'degree', 'noise_sd', 'representation']
    assert len(SETTINGS) == 14
    for f, base in enumerate(BASE_IDS):
        cfg = legacy_configuration(f)
        assert asdict(cfg) == dict(base_setting_id=base, n=200, setting_id=f,
                                   split='dev', split_seed=331029, role='main')
    assert not set(FAMILY_BASE_IDS).intersection(range(7))
    for f, base in FAMILY_BASE_IDS.items():
        assert configuration(f).base_setting_id == base
    with pytest.raises(ValueError):
        legacy_configuration(100)


def test_legacy_graph_fingerprints():
    from pfn_social.phase6_limited_outcomes import generate_mechanism as legacy_mechanism
    expected = json.loads(Path(__file__).with_name('legacy_graph_hashes.json').read_text())
    for family in range(7):
        assert _hash_arrays(legacy_mechanism(family, 0).parent.adjacency) == expected[str(family)]


def test_new_dgp_truth_and_factual_contract():
    torch.set_num_threads(1)
    m = e12_mechanism(102, 0)
    assert np.isfinite(m.parent.truth).all() and m.parent.truth.shape == (200, 4)
    assert m.parent.setting.graph == 'configuration'
    rows, _ = make_observed_draw(m, 0)
    assert [r['context_id'] for r in rows] == [f'f102_m0_r0_observed{k}' for k in (20, 50, 100)]
    assert [r['m_observed'] for r in rows] == [20, 50, 100]
    draw0, draw1 = make_repeat(m.parent, 0), make_repeat(m.parent, 1)
    assert not torch.equal(draw0['y_obs'], draw1['y_obs'])
    assert torch.equal(draw0['adjacency'], draw1['adjacency'])
    assert all('truth' not in r['batch'] for r in rows)


def test_original_method_interfaces_and_evaluation():
    from benchmark import configuration as model_configuration, load_methods, predict_method, CLASSICS
    from pfn_social.intervals import predict_intervals
    from evaluate import evaluate_context
    from make_table import ate_table
    torch.set_num_threads(1)
    methods = load_methods(['j', 'ht'], model_configuration())
    assert CLASSICS['ht'] == 'HT' and CLASSICS['f-z2'] == 'F-z2'
    m = e12_mechanism(100, 0)
    records = []
    for repeat in range(2):
        rows, _ = make_observed_draw(m, repeat)
        row = rows[1]
        inputs = {k: v.numpy().copy() for k, v in row['batch'].items()}
        for name, model in methods.items():
            pred = predict_method(name, model, inputs, 0)
            receipt = dict(context_id=row['context_id'], method=name, interval_level=.95,
                           ate_supported=pred['ate_supported'], ate_interval_supported=[name == 'j']*3)
            if name == 'j':
                pred.update(predict_intervals(model, **inputs))
                assert np.all(pred['ate_lower'] <= pred['ate_upper'])
            assert np.isfinite(pred['ate_effects']).all()
            records.extend(evaluate_context(m.parent.truth, pred, receipt))
    table = ate_table(records, 50)
    assert all(np.isfinite(r['variance_unbiased']) for r in table)
    assert all(r['interval_requested'] == (2 if r['method'] == 'j' else 0) for r in table)


def test_metrics_do_not_hide_failed_intervals():
    from evaluate import evaluate_context, summarize
    from make_table import ate_table
    rows = []
    for repeat, error in enumerate((1., 3.)):
        pred = dict(ate_effects=np.full(3, error), ate_lower=np.full(3, -1.),
                    ate_upper=np.full(3, 1.))
        receipt = dict(context_id=f'f100_m0_r{repeat}_observed50', method='j',
                       ate_supported=[True]*3, ate_interval_supported=[True]*3,
                       interval_failure={'message': 'test'} if repeat else None)
        rows.extend(evaluate_context(np.zeros((2, 4)), pred, receipt))
    for row in ate_table(rows, 50):
        assert row['absolute_conditional_bias'] == 2
        assert row['variance_unbiased'] == 2
        assert row['rmse'] == pytest.approx(np.sqrt(5))
        assert row['coverage'] == 1
        assert row['coverage_including_failures'] == .5
    assert all(r['bias'] == 2 for r in summarize(rows))
