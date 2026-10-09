"""Catch skipped methods or lost unit/effect rows before reporting success."""
import pytest


def complete_report():
    ite = ['PFN with interference', 'Localized DR-Lasso', 'NSI', 'HyperSCI', 'Tnet', 'causalPFN']
    ate = ['PFN with interference', 'HT', 'Haj', 'F', 'L', 'reg-net', 'Tnet', 'causalPFN']
    result = {}
    for level, methods, count, key in [('ite', ite, 3, 'ite_unit_results'),
                                      ('ate', ate, 1, 'ate_graph_results')]:
        result[key] = [dict(dataset_id=1, unit_id=i+1, method=m, effect=e,
                            estimate=float('nan') if m == 'NSI' else 1.,
                            supported=m != 'NSI')
                       for m in methods
                       for e in (['direct'] if m == 'causalPFN' else ['direct', 'spillover', 'total'])
                       for i in range(count)]
    return result


def test_complete_inventory_keeps_unsupported_predictions_and_direct_only_causalpfn():
    from pfn_interference.baselines.validation import validate_baseline_results
    inventory = validate_baseline_results(complete_report(), n_units=3)
    assert len(inventory) == 38  # 16 ITE method/effect pairs + 22 ATE pairs.
    nsi = [r for r in inventory if r['method'] == 'NSI']
    assert len(nsi) == 3 and all(r['n_rows'] == 3 and r['n_supported'] == 0 for r in nsi)
    assert {r['effect'] for r in inventory if r['method'] == 'causalPFN'} == {'direct'}


@pytest.mark.parametrize('damage', ['missing_method', 'missing_effect', 'missing_unit', 'duplicate_unit'])
def test_incomplete_baseline_results_fail_instead_of_silent_success(damage):
    from pfn_interference.baselines.validation import validate_baseline_results
    report = complete_report()
    if damage == 'missing_method':
        report['ate_graph_results'] = [r for r in report['ate_graph_results'] if r['method'] != 'HT']
    elif damage == 'missing_effect':
        report['ite_unit_results'] = [r for r in report['ite_unit_results']
                                     if not (r['method'] == 'Tnet' and r['effect'] == 'total')]
    elif damage == 'missing_unit':
        report['ite_unit_results'].pop(0)
    else:
        report['ite_unit_results'][0] = dict(report['ite_unit_results'][1])
    with pytest.raises(ValueError, match='baseline|Baseline'):
        validate_baseline_results(report, n_units=3)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_supported_nonfinite_estimates_are_computation_failures(value):
    from pfn_interference.baselines.validation import validate_baseline_results
    report = complete_report()
    report['ite_unit_results'][0]['estimate'] = value
    assert report['ite_unit_results'][0]['supported'] is True
    with pytest.raises(ValueError, match='Baseline.*non-finite'):
        validate_baseline_results(report, n_units=3)
