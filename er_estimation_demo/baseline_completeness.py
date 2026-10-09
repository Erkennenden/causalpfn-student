"""Require every intended baseline/effect/unit result, including unsupported rows."""
from collections import defaultdict
import math
from baselines.localized_config import LOCALIZED_METHOD_NAME

ITE_METHODS = ('PFN with interference', LOCALIZED_METHOD_NAME, 'NSI', 'HyperSCI', 'Tnet', 'causalPFN')
ATE_METHODS = ('PFN with interference', 'HT', 'Haj', 'F', 'L', 'reg-net', 'Tnet', 'causalPFN')
EFFECTS = ('direct', 'spillover', 'total')


def validate_baseline_results(report, *, n_units):
    """Validate one episode's raw rows before pooling or publishing a success report.

    Absence of statistical support is valid and counted; missing computations,
    duplicate units and silently skipped methods are errors. causalPFN implements
    only the direct effect and is never filled in for spillover/total.
    """
    if n_units <= 0:
        raise ValueError('Baseline validation requires a positive unit count.')
    inventory = []
    for level, methods, key in (('ite', ITE_METHODS, 'ite_unit_results'),
                                ('ate', ATE_METHODS, 'ate_graph_results')):
        expected_pairs = {(m, e) for m in methods
                          for e in (('direct',) if m == 'causalPFN' else EFFECTS)}
        groups = defaultdict(list)
        for row in report.get(key, []):
            groups[(row['method'], row['effect'])].append(row)
        if set(groups) != expected_pairs:
            raise ValueError(f'Baseline {level} method/effect inventory mismatch: '
                             f'missing={sorted(expected_pairs-set(groups))}, '
                             f'unexpected={sorted(set(groups)-expected_pairs)}')
        for method, effect in sorted(expected_pairs):
            rows = groups[(method, effect)]
            expected = n_units if level == 'ite' else 1
            if len(rows) != expected or {r['dataset_id'] for r in rows} != {1}:
                raise ValueError(f'Baseline {level}/{method}/{effect}: incomplete episode rows')
            if level == 'ite' and {r['unit_id'] for r in rows} != set(range(1, n_units+1)):
                raise ValueError(f'Baseline {method}/{effect}: missing or duplicate unit IDs')
            if any(bool(r.get('supported', True)) and
                   not math.isfinite(float(r['estimate'])) for r in rows):
                raise ValueError(f'Baseline {level}/{method}/{effect}: non-finite '
                                 'estimate marked supported; computation failed')
            n_supported = sum(bool(r.get('supported', True)) for r in rows)
            inventory.append(dict(level=level, method=method, effect=effect,
                                  n_expected=expected, n_rows=len(rows),
                                  n_supported=n_supported))
    return inventory
